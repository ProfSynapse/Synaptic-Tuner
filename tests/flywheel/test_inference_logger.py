"""Tests for shared.flywheel.inference_logger — InferenceLogger."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from shared.flywheel.catalog import InferenceLogRecord
from shared.flywheel.config import FlywheelConfig
from shared.flywheel.inference_logger import InferenceLogger


def _make_request(*, tools: list | None = None) -> dict:
    """Build a minimal OpenAI chat completion request."""
    req = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
        "max_tokens": 512,
    }
    if tools:
        req["tools"] = tools
    return req


def _make_response(*, content: str = "Hi there", tool_calls: list | None = None) -> dict:
    """Build a minimal OpenAI chat completion response."""
    message = {"content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.fixture
def mock_catalog():
    """Mock LogCatalog with async methods."""
    cat = AsyncMock()
    cat.insert_logs_batch = AsyncMock(return_value=1)
    return cat


@pytest.fixture
def flywheel_config():
    return FlywheelConfig(flush_interval_seconds=0.1)


class TestInferenceLoggerBuildRecord:
    """InferenceLogger._build_record captures fields correctly."""

    def test_basic_record_fields(self, mock_catalog, flywheel_config, tmp_path):
        logger = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        record = logger._build_record(
            request=_make_request(),
            response=_make_response(),
            latency_ms=42.5,
            model_id="test-model",
            adapter_name="lora-v1",
        )
        assert record.model_id == "test-model"
        assert record.adapter_name == "lora-v1"
        assert record.latency_ms == 42.5
        assert record.response_content == "Hi there"
        assert record.prompt_tokens == 10
        assert record.completion_tokens == 5
        assert record.temperature == 0.5
        assert record.max_tokens == 512

    def test_tools_requested_true_when_tools_in_request(self, mock_catalog, flywheel_config, tmp_path):
        logger = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        tools = [{"type": "function", "function": {"name": "search"}}]
        record = logger._build_record(
            request=_make_request(tools=tools),
            response=_make_response(),
            latency_ms=10,
            model_id="m",
            adapter_name=None,
        )
        assert record.tools_requested is True
        assert record.tools == tools

    def test_tools_requested_false_when_no_tools(self, mock_catalog, flywheel_config, tmp_path):
        logger = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        record = logger._build_record(
            request=_make_request(),
            response=_make_response(),
            latency_ms=10,
            model_id="m",
            adapter_name=None,
        )
        assert record.tools_requested is False
        assert record.tools == []

    def test_tool_calls_captured_from_response(self, mock_catalog, flywheel_config, tmp_path):
        logger = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        tc = [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}]
        record = logger._build_record(
            request=_make_request(tools=[{"type": "function"}]),
            response=_make_response(tool_calls=tc),
            latency_ms=10,
            model_id="m",
            adapter_name=None,
        )
        assert len(record.tool_calls) == 1
        assert record.tool_calls[0]["function"]["name"] == "search"

    def test_empty_response_handled(self, mock_catalog, flywheel_config, tmp_path):
        logger = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        record = logger._build_record(
            request=_make_request(),
            response={"choices": []},
            latency_ms=10,
            model_id="m",
            adapter_name=None,
        )
        assert record.response_content == ""
        assert record.tool_calls == []


@pytest.mark.asyncio
class TestInferenceLoggerAsync:
    """Async behavior of InferenceLogger."""

    async def test_log_returns_immediately(self, mock_catalog, flywheel_config, tmp_path):
        """log_inference should return a log_id without blocking."""
        logger = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        await logger.start()
        try:
            start = time.monotonic()
            log_id = await logger.log_inference(
                request=_make_request(),
                response=_make_response(),
                latency_ms=10,
                model_id="m",
            )
            elapsed = time.monotonic() - start
            assert log_id != ""
            assert elapsed < 1.0  # Should be nearly instant
        finally:
            await logger.stop()

    async def test_disabled_logger_returns_empty(self, mock_catalog, flywheel_config, tmp_path):
        logger = InferenceLogger(
            tmp_path / "logs", mock_catalog, flywheel_config, enabled=False,
        )
        await logger.start()
        log_id = await logger.log_inference(
            request=_make_request(),
            response=_make_response(),
            latency_ms=10,
            model_id="m",
        )
        assert log_id == ""
        await logger.stop()

    async def test_writes_to_jsonl_file(self, mock_catalog, flywheel_config, tmp_path):
        log_dir = tmp_path / "logs"
        logger = InferenceLogger(log_dir, mock_catalog, flywheel_config)
        await logger.start()

        await logger.log_inference(
            request=_make_request(),
            response=_make_response(content="test output"),
            latency_ms=15,
            model_id="test-model",
        )

        # Wait for the writer loop to flush
        await asyncio.sleep(0.3)
        await logger.stop()

        # Check that a JSONL file was written
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        lines = jsonl_files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["model_id"] == "test-model"
        assert data["response_content"] == "test output"

    async def test_indexes_in_catalog(self, mock_catalog, flywheel_config, tmp_path):
        logger = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        await logger.start()

        await logger.log_inference(
            request=_make_request(),
            response=_make_response(),
            latency_ms=10,
            model_id="m",
        )

        await asyncio.sleep(0.3)
        await logger.stop()

        mock_catalog.insert_logs_batch.assert_called()

    async def test_total_logged_counter(self, mock_catalog, flywheel_config, tmp_path):
        logger = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        await logger.start()

        for _ in range(3):
            await logger.log_inference(
                request=_make_request(),
                response=_make_response(),
                latency_ms=10,
                model_id="m",
            )

        await asyncio.sleep(0.3)
        await logger.stop()

        assert logger.total_logged == 3
