from __future__ import annotations

import pytest

from shared.llm.exceptions import LLMResponseError
from shared.llm.providers.openrouter import OpenRouterClient


def test_openrouter_structured_output_parse_failure_includes_raw_response():
    client = OpenRouterClient(api_key="test-key", model="test-model")

    def fake_make_request(payload):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"environment": {"fixture": {"files": [\n  {"path": "foo.md", "content": "bar"}\n'
                    }
                }
            ]
        }

    client._make_request = fake_make_request  # type: ignore[method-assign]

    with pytest.raises(LLMResponseError) as exc_info:
        client.structured_output(
            messages=[{"role": "user", "content": "Generate JSON"}],
            schema={"name": "response", "type": "object"},
        )

    err = exc_info.value
    assert err.raw_response is not None
    assert '"environment"' in err.raw_response
    assert "Response excerpt:" in str(err)


def test_openrouter_chat_sends_reasoning_effort_object():
    client = OpenRouterClient(
        api_key="test-key",
        model="test-model",
        thinking_effort="HIGH",
    )
    captured = {}

    def fake_make_request(payload):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    client._make_request = fake_make_request  # type: ignore[method-assign]

    assert client.chat([{"role": "user", "content": "Hello"}]) == "ok"
    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in captured["payload"]


def test_openrouter_structured_output_sends_reasoning_effort_object():
    client = OpenRouterClient(
        api_key="test-key",
        model="test-model",
        thinking_effort="minimal",
    )
    captured = {}

    def fake_make_request(payload):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    client._make_request = fake_make_request  # type: ignore[method-assign]

    assert client.structured_output(
        messages=[{"role": "user", "content": "Generate JSON"}],
        schema={"name": "response", "type": "object"},
    ) == {"ok": True}
    assert captured["payload"]["reasoning"] == {"effort": "minimal"}
    assert "reasoning_effort" not in captured["payload"]
