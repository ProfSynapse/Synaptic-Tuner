from __future__ import annotations

import pytest
import requests

from shared.llm.exceptions import LLMResponseError
from shared.llm.providers.openrouter import (
    OpenRouterBatchRejectedError,
    OpenRouterBatchSubmissionAmbiguousError,
    OpenRouterClient,
)


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

    assert client.chat([{"role": "user", "content": "Hello"}]).text == "ok"
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
    ).value == {"ok": True}
    assert captured["payload"]["reasoning"] == {"effort": "minimal"}
    assert "reasoning_effort" not in captured["payload"]


class _FakeResponse:
    def __init__(self, payload, *, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError("failed")

    def json(self):
        return self.payload


class _FakeResponseWithInvalidJSON(_FakeResponse):
    def __init__(self):
        super().__init__(None, status_code=202)

    def json(self):
        raise ValueError("invalid JSON")


def test_openrouter_batch_submit_uses_beta_endpoint_and_outer_and_body_model(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return _FakeResponse({"id": "batch_123", "status": "validating"}, status_code=202)

    monkeypatch.setattr("shared.llm.providers.openrouter.requests.post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="openai/model", timeout_seconds=17)

    result = client.submit_batch(
        [
            {
                "custom_id": "doc-a-123",
                "body": {
                    "model": "openai/model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ]
    )

    assert result == {"id": "batch_123", "status": "validating"}
    assert captured["url"] == "https://openrouter.ai/api/beta/batches"
    assert captured["timeout"] == 17.0
    assert captured["payload"] == {
        "endpoint": "/v1/chat/completions",
        "model": "openai/model",
        "requests": [
            {
                "custom_id": "doc-a-123",
                "body": {
                    "model": "openai/model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ],
    }


def test_openrouter_batch_submit_rejects_duplicate_ids_and_body_model_mismatch():
    client = OpenRouterClient(api_key="test-key", model="openai/model")
    item = {"custom_id": "same", "body": {"model": "openai/model", "messages": []}}

    with pytest.raises(ValueError, match="duplicate.*custom_id"):
        client.submit_batch([item, item])
    with pytest.raises(ValueError, match="model must equal outer model"):
        client.submit_batch([{"custom_id": "one", "body": {"model": "other/model"}}])


def test_openrouter_batch_observe_uses_persisted_id_and_returns_inline_results(monkeypatch):
    captured = {}
    payload = {
        "id": "batch_123",
        "status": "completed",
        "results": [{"custom_id": "one", "response": {"status_code": 200, "body": {}}, "error": None}],
    }

    def fake_get(url, *, headers, timeout):
        captured.update(url=url, timeout=timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("shared.llm.providers.openrouter.requests.get", fake_get)
    client = OpenRouterClient(api_key="test-key", model="openai/model")

    assert client.observe_batch("batch_123") == payload
    assert captured["url"] == "https://openrouter.ai/api/beta/batches/batch_123"


def test_openrouter_batch_http_rejection_omits_provider_controlled_message(monkeypatch):
    source_prompt = "PRIVATE SOURCE PROMPT MUST NOT LEAK"

    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(
            {"error": {"code": "unsupported", "message": source_prompt, "metadata": {"secret": "omit"}}},
            status_code=400,
        )

    monkeypatch.setattr("shared.llm.providers.openrouter.requests.post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="openai/model")

    with pytest.raises(OpenRouterBatchRejectedError) as exc_info:
        client.submit_batch([{"custom_id": "one", "body": {"model": "openai/model"}}])

    assert "HTTP 400" in str(exc_info.value)
    assert "code=unsupported" in str(exc_info.value)
    assert source_prompt not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


def test_openrouter_batch_rejection_omits_non_identifier_error_code(monkeypatch):
    private_code = "PRIVATE PROMPT / inject into logs"

    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(
            {"error": {"code": private_code, "message": "also private"}},
            status_code=429,
        )

    monkeypatch.setattr("shared.llm.providers.openrouter.requests.post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="openai/model")

    with pytest.raises(OpenRouterBatchRejectedError) as exc_info:
        client.submit_batch([{"custom_id": "one", "body": {"model": "openai/model"}}])

    assert exc_info.value.status_code == 429
    assert exc_info.value.code is None
    assert private_code not in str(exc_info.value)
    assert "also private" not in str(exc_info.value)


@pytest.mark.parametrize(
    "response_factory",
    [
        lambda: (_ for _ in ()).throw(requests.exceptions.Timeout("late")),
        lambda: _FakeResponseWithInvalidJSON(),
        lambda: _FakeResponse("not-an-object", status_code=202),
        lambda: _FakeResponse({"status": "validating"}, status_code=202),
        lambda: _FakeResponse({"id": "batch_1", "status": "validating", "bad": float("nan")}, status_code=202),
    ],
)
def test_openrouter_batch_ambiguous_submit_outcomes_are_never_retryable_rejections(monkeypatch, response_factory):
    def fake_post(url, *, headers, json, timeout):
        return response_factory()

    monkeypatch.setattr("shared.llm.providers.openrouter.requests.post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="openai/model")

    with pytest.raises(OpenRouterBatchSubmissionAmbiguousError):
        client.submit_batch([{"custom_id": "one", "body": {"model": "openai/model"}}])
