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
from shared.flywheel.inference_logger import (
    InferenceLogger,
    _scrub_credentials,
    _scrub_messages,
)


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


# ---------------------------------------------------------------------------
# Credential scrubbing tests
# ---------------------------------------------------------------------------


class TestScrubCredentials:
    """Verify _scrub_credentials replaces sensitive patterns in text."""

    def test_scrubs_openai_style_api_key(self):
        text = "Using key sk-abc123def456ghi789jkl012mno"
        result = _scrub_credentials(text)
        assert "sk-abc123" not in result
        assert "[REDACTED_API_KEY]" in result

    def test_scrubs_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature"
        result = _scrub_credentials(text)
        assert "eyJhbGci" not in result
        assert "Bearer [REDACTED]" in result

    def test_scrubs_password_in_json(self):
        text = '{"username": "admin", "password": "supersecret123"}'
        result = _scrub_credentials(text)
        assert "supersecret123" not in result
        assert '"password": "[REDACTED]"' in result

    def test_preserves_non_sensitive_text(self):
        text = "Hello world, this is a normal message with no secrets."
        result = _scrub_credentials(text)
        assert result == text

    def test_scrubs_multiple_patterns_in_same_text(self):
        text = (
            "key=sk-abcdefghijklmnopqrstuvwxyz "
            "auth=Bearer eyJtoken.payload.sig "
            '"password": "mypass1234"'
        )
        result = _scrub_credentials(text)
        assert "sk-abcdef" not in result
        assert "eyJtoken" not in result
        assert "mypass1234" not in result

    def test_short_password_not_scrubbed(self):
        """Passwords shorter than 4 chars should NOT be matched by the regex."""
        text = '"password": "abc"'
        result = _scrub_credentials(text)
        assert result == text

    def test_empty_string_returns_empty(self):
        assert _scrub_credentials("") == ""


class TestScrubMessages:
    """Verify _scrub_messages deep-copies and scrubs message content."""

    def test_scrubs_content_field(self):
        messages = [
            {"role": "user", "content": "My API key is sk-abcdefghijklmnopqrstuv12345"},
        ]
        result = _scrub_messages(messages)
        assert "[REDACTED_API_KEY]" in result[0]["content"]
        assert "sk-abcdef" not in result[0]["content"]

    def test_preserves_non_string_content(self):
        messages = [
            {"role": "system", "content": None},
            {"role": "assistant"},
        ]
        result = _scrub_messages(messages)
        assert result[0]["content"] is None
        assert "content" not in result[1]

    def test_does_not_mutate_original(self):
        original_content = "Bearer eyJsecrettoken.payload.sig"
        messages = [{"role": "user", "content": original_content}]
        result = _scrub_messages(messages)
        assert messages[0]["content"] == original_content
        assert "Bearer [REDACTED]" in result[0]["content"]

    def test_empty_messages_list(self):
        assert _scrub_messages([]) == []

    def test_preserves_role_and_other_fields(self):
        messages = [
            {"role": "assistant", "content": "safe text", "tool_calls": [{"id": "1"}]},
        ]
        result = _scrub_messages(messages)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "safe text"
        assert result[0]["tool_calls"] == [{"id": "1"}]


class TestBuildRecordScrubbing:
    """Verify that _build_record applies scrubbing to messages and response content."""

    def test_messages_scrubbed_in_record(self, mock_catalog, flywheel_config, tmp_path):
        logger_inst = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        request = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Use key sk-abcdefghijklmnopqrstuv12345"},
            ],
            "temperature": 0.5,
            "max_tokens": 512,
        }
        record = logger_inst._build_record(
            request=request,
            response=_make_response(),
            latency_ms=10,
            model_id="m",
            adapter_name=None,
        )
        assert "sk-abcdef" not in record.messages[0]["content"]
        assert "[REDACTED_API_KEY]" in record.messages[0]["content"]

    def test_response_content_scrubbed_in_record(self, mock_catalog, flywheel_config, tmp_path):
        logger_inst = InferenceLogger(tmp_path / "logs", mock_catalog, flywheel_config)
        record = logger_inst._build_record(
            request=_make_request(),
            response=_make_response(content="Bearer eyJsecrettoken.payload.sig"),
            latency_ms=10,
            model_id="m",
            adapter_name=None,
        )
        assert "eyJsecrettoken" not in record.response_content
        assert "Bearer [REDACTED]" in record.response_content
