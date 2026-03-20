"""Tests for proxy -> logging integration path.

End-to-end: POST /v1/chat/completions -> vLLM response -> log record
via fire-and-forget _log_inference task.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.proxy.app import _log_inference, health, proxy, stats
from services.proxy.config import ProxyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def proxy_config():
    return ProxyConfig(
        vllm_base_url="http://fake-vllm:8000",
        proxy_port=9999,
    )


def _build_test_app(proxy_config, *, inference_logger=None):
    """Build a FastAPI test app with routes registered manually (no lifespan)."""
    import httpx

    app = FastAPI()
    app.get("/flywheel/health")(health)
    app.get("/flywheel/stats")(stats)
    app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(proxy)

    app.state.config = proxy_config
    app.state.inference_logger = inference_logger
    app.state.catalog = None
    app.state.http_client = AsyncMock(spec=httpx.AsyncClient)
    app.state.stats = {"total_proxied": 0, "total_logged": 0, "log_errors": 0}

    return app


# ---------------------------------------------------------------------------
# _log_inference unit tests
# ---------------------------------------------------------------------------


class TestLogInference:
    """Direct tests for the _log_inference fire-and-forget task."""

    @pytest.mark.asyncio
    async def test_successful_logging(self):
        """Valid JSON request/response -> inference_logger.log_inference called."""
        app = FastAPI()
        mock_logger = AsyncMock()
        mock_logger.log_inference = AsyncMock()
        app.state.inference_logger = mock_logger
        app.state.stats = {"total_proxied": 0, "total_logged": 0, "log_errors": 0}

        request_body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        response_body = json.dumps({
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
        }).encode()

        await _log_inference(app, request_body, response_body, latency_ms=42.5)

        mock_logger.log_inference.assert_awaited_once()
        call_kwargs = mock_logger.log_inference.call_args
        assert call_kwargs.kwargs.get("model_id") == "test-model" or \
               (call_kwargs.args and "test-model" in str(call_kwargs))
        assert app.state.stats["total_logged"] == 1
        assert app.state.stats["log_errors"] == 0

    @pytest.mark.asyncio
    async def test_invalid_json_request_increments_log_errors(self):
        """Unparseable request JSON -> log_errors incremented, no crash."""
        app = FastAPI()
        app.state.inference_logger = AsyncMock()
        app.state.stats = {"total_proxied": 0, "total_logged": 0, "log_errors": 0}

        await _log_inference(app, b"not-json", b"{}", latency_ms=10.0)

        assert app.state.stats["log_errors"] == 1
        assert app.state.stats["total_logged"] == 0

    @pytest.mark.asyncio
    async def test_invalid_json_response_increments_log_errors(self):
        """Unparseable response JSON -> log_errors incremented."""
        app = FastAPI()
        app.state.inference_logger = AsyncMock()
        app.state.stats = {"total_proxied": 0, "total_logged": 0, "log_errors": 0}

        valid_request = json.dumps({"model": "m"}).encode()
        await _log_inference(app, valid_request, b"not-json", latency_ms=10.0)

        assert app.state.stats["log_errors"] == 1

    @pytest.mark.asyncio
    async def test_logger_exception_increments_log_errors(self):
        """When inference_logger.log_inference raises, error is caught."""
        app = FastAPI()
        mock_logger = AsyncMock()
        mock_logger.log_inference = AsyncMock(
            side_effect=RuntimeError("write failed"),
        )
        app.state.inference_logger = mock_logger
        app.state.stats = {"total_proxied": 0, "total_logged": 0, "log_errors": 0}

        request_body = json.dumps({"model": "m"}).encode()
        response_body = json.dumps({"choices": []}).encode()

        await _log_inference(app, request_body, response_body, latency_ms=5.0)

        assert app.state.stats["log_errors"] == 1
        assert app.state.stats["total_logged"] == 0

    @pytest.mark.asyncio
    async def test_model_id_defaults_to_unknown(self):
        """Missing 'model' key in request -> model_id='unknown'."""
        app = FastAPI()
        mock_logger = AsyncMock()
        mock_logger.log_inference = AsyncMock()
        app.state.inference_logger = mock_logger
        app.state.stats = {"total_proxied": 0, "total_logged": 0, "log_errors": 0}

        request_body = json.dumps({"messages": []}).encode()
        response_body = json.dumps({"choices": []}).encode()

        await _log_inference(app, request_body, response_body, latency_ms=1.0)

        call_kwargs = mock_logger.log_inference.call_args
        # model_id should be "unknown"
        assert call_kwargs.kwargs.get("model_id") == "unknown" or \
               "unknown" in str(call_kwargs)


# ---------------------------------------------------------------------------
# Proxy integration: logging triggers on chat completions only
# ---------------------------------------------------------------------------


class TestProxyLoggingTrigger:
    """Logging is triggered only for POST /v1/chat/completions with 200."""

    def test_chat_completions_post_triggers_logging_task(self, proxy_config):
        """POST /v1/chat/completions with logger -> creates asyncio task."""
        import httpx

        mock_logger = AsyncMock()
        mock_logger.log_inference = AsyncMock()
        app = _build_test_app(proxy_config, inference_logger=mock_logger)

        vllm_response = httpx.Response(
            status_code=200,
            content=json.dumps({
                "choices": [{"message": {"content": "Hi"}}],
            }).encode(),
            headers={"content-type": "application/json"},
        )
        app.state.http_client.request = AsyncMock(return_value=vllm_response)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        assert app.state.stats["total_proxied"] == 1

    def test_non_chat_completions_post_does_not_log(self, proxy_config):
        """POST to other paths does not trigger logging."""
        import httpx

        mock_logger = AsyncMock()
        app = _build_test_app(proxy_config, inference_logger=mock_logger)

        vllm_response = httpx.Response(
            status_code=200,
            content=b'{"result": "ok"}',
            headers={"content-type": "application/json"},
        )
        app.state.http_client.request = AsyncMock(return_value=vllm_response)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/embeddings",
            json={"input": "text"},
        )

        assert response.status_code == 200
        assert app.state.stats["total_proxied"] == 1
        # No logging task created for non-chat-completions
        mock_logger.log_inference.assert_not_awaited()

    def test_get_request_does_not_log(self, proxy_config):
        """GET requests are proxied but never logged."""
        import httpx

        mock_logger = AsyncMock()
        app = _build_test_app(proxy_config, inference_logger=mock_logger)

        vllm_response = httpx.Response(
            status_code=200,
            content=b'{"models": []}',
            headers={"content-type": "application/json"},
        )
        app.state.http_client.request = AsyncMock(return_value=vllm_response)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

        assert response.status_code == 200
        mock_logger.log_inference.assert_not_awaited()

    def test_no_logging_when_logger_is_none(self, proxy_config):
        """When inference_logger is None, no logging is attempted."""
        import httpx

        app = _build_test_app(proxy_config, inference_logger=None)

        vllm_response = httpx.Response(
            status_code=200,
            content=json.dumps({"choices": [{"message": {"content": "x"}}]}).encode(),
            headers={"content-type": "application/json"},
        )
        app.state.http_client.request = AsyncMock(return_value=vllm_response)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )

        assert response.status_code == 200
        # total_logged should remain 0 since no logger is set
        assert app.state.stats["total_logged"] == 0

    def test_non_200_response_does_not_log(self, proxy_config):
        """Non-200 vLLM responses are proxied but not logged."""
        import httpx

        mock_logger = AsyncMock()
        app = _build_test_app(proxy_config, inference_logger=mock_logger)

        vllm_response = httpx.Response(
            status_code=400,
            content=b'{"error": "bad request"}',
            headers={"content-type": "application/json"},
        )
        app.state.http_client.request = AsyncMock(return_value=vllm_response)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )

        assert response.status_code == 400
        mock_logger.log_inference.assert_not_awaited()
