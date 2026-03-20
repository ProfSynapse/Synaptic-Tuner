"""Tests for services.proxy.app — FastAPI logging proxy."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.proxy.config import ProxyConfig


@pytest.fixture
def proxy_config():
    """Proxy config pointing at a fake vLLM backend."""
    return ProxyConfig(
        vllm_base_url="http://fake-vllm:8000",
        proxy_port=9999,
    )


@pytest.fixture
def mock_app(proxy_config):
    """Create a test app with mocked lifespan dependencies."""
    import httpx
    from fastapi import FastAPI

    from services.proxy.app import health, proxy, stats

    app = FastAPI()

    # Register routes manually to avoid lifespan initialization
    app.get("/flywheel/health")(health)
    app.get("/flywheel/stats")(stats)
    app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(proxy)

    app.state.config = proxy_config
    app.state.inference_logger = None
    app.state.catalog = None
    app.state.http_client = AsyncMock(spec=httpx.AsyncClient)
    app.state.stats = {"total_proxied": 0, "total_logged": 0, "log_errors": 0}

    return app


class TestHealthEndpoint:
    """Proxy health check endpoint."""

    def test_health_returns_200(self, mock_app):
        client = TestClient(mock_app, raise_server_exceptions=False)
        response = client.get("/flywheel/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["logging_enabled"] is False  # No inference_logger set

    def test_health_shows_logging_enabled(self, mock_app):
        mock_app.state.inference_logger = MagicMock()
        client = TestClient(mock_app, raise_server_exceptions=False)
        response = client.get("/flywheel/health")
        data = response.json()
        assert data["logging_enabled"] is True


class TestStatsEndpoint:
    """Proxy stats endpoint."""

    def test_stats_returns_counters(self, mock_app):
        mock_app.state.stats = {"total_proxied": 5, "total_logged": 3, "log_errors": 1}
        client = TestClient(mock_app, raise_server_exceptions=False)
        response = client.get("/flywheel/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_proxied"] == 5
        assert data["total_logged"] == 3


class TestProxyForwarding:
    """Proxy forwards requests to vLLM backend."""

    def test_forwards_get_request(self, mock_app):
        import httpx

        mock_response = httpx.Response(
            status_code=200,
            content=b'{"models": []}',
            headers={"content-type": "application/json"},
        )
        mock_app.state.http_client.request = AsyncMock(return_value=mock_response)

        client = TestClient(mock_app, raise_server_exceptions=False)
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert response.json() == {"models": []}

    def test_forwards_post_request(self, mock_app):
        import httpx

        vllm_response_data = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(vllm_response_data).encode(),
            headers={"content-type": "application/json"},
        )
        mock_app.state.http_client.request = AsyncMock(return_value=mock_response)

        client = TestClient(mock_app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 200

    def test_503_when_vllm_down(self, mock_app):
        import httpx

        mock_app.state.http_client.request = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        client = TestClient(mock_app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
        assert response.status_code == 503
        assert "unavailable" in response.json()["error"]

    def test_504_when_vllm_timeout(self, mock_app):
        import httpx

        mock_app.state.http_client.request = AsyncMock(
            side_effect=httpx.TimeoutException("Timed out"),
        )

        client = TestClient(mock_app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
        assert response.status_code == 504

    def test_increments_proxied_counter(self, mock_app):
        import httpx

        mock_response = httpx.Response(
            status_code=200,
            content=b"{}",
            headers={"content-type": "application/json"},
        )
        mock_app.state.http_client.request = AsyncMock(return_value=mock_response)

        client = TestClient(mock_app, raise_server_exceptions=False)
        client.get("/v1/models")

        assert mock_app.state.stats["total_proxied"] == 1


class TestProxyConfig:
    """ProxyConfig construction."""

    def test_from_env_defaults(self):
        cfg = ProxyConfig.from_env()
        assert cfg.vllm_base_url == "http://localhost:8000"
        assert cfg.proxy_port == 8080

    def test_from_flywheel_config(self):
        flywheel_cfg = MagicMock()
        flywheel_cfg.vllm_host = "gpu-server"
        flywheel_cfg.vllm_port = 8001
        flywheel_cfg.catalog_backend = "postgres"
        flywheel_cfg.catalog_path = "/data/flywheel.db"
        flywheel_cfg.catalog_url = "postgres://localhost/db"
        flywheel_cfg.tenant_id = "tenant1"
        flywheel_cfg.log_dir = "/logs"
        flywheel_cfg.proxy_port = 9090
        flywheel_cfg.proxy_timeout_seconds = 60.0
        flywheel_cfg.flush_interval_seconds = 2.0

        cfg = ProxyConfig.from_flywheel_config(flywheel_cfg)
        assert cfg.vllm_base_url == "http://gpu-server:8001"
        assert cfg.catalog_backend == "postgres"
        assert cfg.proxy_port == 9090
