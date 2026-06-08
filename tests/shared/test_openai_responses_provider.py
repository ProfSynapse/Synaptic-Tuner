from __future__ import annotations

import pytest

from shared.llm.config import LLMConfig
from shared.llm.exceptions import LLMResponseError
from shared.llm.factory import create_client, list_providers
from shared.llm.providers.openai_responses import OpenAIResponsesClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_openai_responses_chat_sends_responses_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"output_text": "hello"})

    monkeypatch.setattr("shared.llm.providers.openai_responses.requests.post", fake_post)

    client = OpenAIResponsesClient(
        api_key="test-key",
        model="gpt-test",
        base_url="https://example.test/v1/",
        timeout_seconds=12,
    )
    result = client.chat(
        [{"role": "user", "content": "Hello"}],
        temperature=0.4,
        max_tokens=77,
    )

    assert result == "hello"
    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 12
    assert captured["json"] == {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "Hello"}],
        "temperature": 0.4,
        "max_output_tokens": 77,
        "store": False,
    }


def test_openai_responses_chat_omits_default_temperature(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": "hello"})

    monkeypatch.setattr("shared.llm.providers.openai_responses.requests.post", fake_post)

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-test")

    assert client.chat([{"role": "user", "content": "Hello"}]) == "hello"
    assert "temperature" not in captured["json"]


def test_openai_responses_chat_extracts_typed_output(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "part one"},
                            {"type": "output_text", "text": " part two"},
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("shared.llm.providers.openai_responses.requests.post", fake_post)

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-test")

    assert client.chat([{"role": "user", "content": "Hello"}]) == "part one part two"


def test_openai_responses_structured_output_uses_text_format(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": '{"answer": "yes"}'})

    monkeypatch.setattr("shared.llm.providers.openai_responses.requests.post", fake_post)

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-test")
    schema = {
        "name": "judge_result",
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    result = client.structured_output(
        [{"role": "user", "content": "Return JSON"}],
        schema=schema,
        max_tokens=123,
    )

    assert result == {"answer": "yes"}
    assert captured["json"]["store"] is False
    assert "temperature" not in captured["json"]
    assert captured["json"]["max_output_tokens"] == 123
    assert captured["json"]["text"]["format"] == {
        "type": "json_schema",
        "name": "judge_result",
        "strict": False,
        "schema": schema,
    }


def test_openai_responses_structured_output_sends_explicit_temperature(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": '{"answer": "yes"}'})

    monkeypatch.setattr("shared.llm.providers.openai_responses.requests.post", fake_post)

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-test")

    result = client.structured_output(
        [{"role": "user", "content": "Return JSON"}],
        schema={"name": "response", "type": "object"},
        temperature=0.2,
    )

    assert result == {"answer": "yes"}
    assert captured["json"]["temperature"] == 0.2


def test_openai_responses_structured_output_strict_opt_in(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": '{"answer": "yes"}'})

    monkeypatch.setattr("shared.llm.providers.openai_responses.requests.post", fake_post)

    client = OpenAIResponsesClient(
        api_key="test-key",
        model="gpt-test",
        structured_output_strict=True,
    )

    client.structured_output(
        [{"role": "user", "content": "Return JSON"}],
        schema={"name": "response", "type": "object"},
    )

    assert captured["json"]["text"]["format"]["strict"] is True

    client.structured_output(
        [{"role": "user", "content": "Return JSON"}],
        schema={"name": "response", "type": "object"},
        strict=False,
    )

    assert captured["json"]["text"]["format"]["strict"] is False


@pytest.mark.parametrize("field", ["store", "previous_response_id", "conversation"])
def test_openai_responses_rejects_stateful_kwargs(field):
    client = OpenAIResponsesClient(api_key="test-key", model="gpt-test")

    with pytest.raises(LLMResponseError, match=field):
        client.chat([{"role": "user", "content": "Hello"}], **{field: "unsafe"})


def test_openai_responses_structured_parse_failure_includes_raw_response(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return _FakeResponse({"output_text": '{"answer": '})

    monkeypatch.setattr("shared.llm.providers.openai_responses.requests.post", fake_post)

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-test")

    with pytest.raises(LLMResponseError) as exc_info:
        client.structured_output(
            [{"role": "user", "content": "Return JSON"}],
            schema={"name": "response", "type": "object"},
        )

    err = exc_info.value
    assert err.raw_response == '{"answer": '
    assert "Response excerpt:" in str(err)


def test_openai_responses_factory_registration():
    config = LLMConfig(
        provider="openai_responses",
        model="gpt-test",
        openai_api_key="test-key",
        openai_responses_base_url="https://example.test/v1",
        openai_responses_timeout_seconds=5,
        openai_responses_store=True,
        openai_responses_structured_output_strict=True,
    )

    client = create_client(config=config)

    assert isinstance(client, OpenAIResponsesClient)
    assert client.provider_name == "openai_responses"
    assert client.model_name == "gpt-test"
    assert client.timeout_seconds == 5
    assert client.store is True
    assert client.structured_output_strict is True
    assert "openai_responses" in list_providers()


def test_openai_responses_config_requires_openai_api_key():
    config = LLMConfig(provider="openai_responses", model="gpt-test")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        config.validate()


def test_evaluator_accepts_openai_responses_backend_and_judge_provider():
    from Evaluator.cli import parse_args
    from Evaluator.client_factory import create_settings, get_supported_backends
    from Evaluator.config import OpenAIResponsesSettings
    from Evaluator.enums import BackendType

    settings = create_settings("openai_responses", model="gpt-test")

    assert isinstance(settings, OpenAIResponsesSettings)
    assert BackendType.OPENAI_RESPONSES in get_supported_backends()

    args = parse_args(
        [
            "--backend",
            "openai_responses",
            "--model",
            "gpt-test",
            "--judge",
            "--judge-provider",
            "openai_responses",
        ]
    )

    assert args.backend == "openai_responses"
    assert args.judge_provider == "openai_responses"


def test_evaluator_openai_responses_adapter_passes_timeout_to_shared_client(monkeypatch):
    from Evaluator.config import OpenAIResponsesSettings
    from Evaluator.shared_llm_adapters import SharedOpenAIResponsesAdapter

    captured = {}

    class FakeClient:
        provider_name = "openai_responses"

        def chat(self, messages, temperature=0.7, max_tokens=1024):
            return "ok"

        def test_connection(self):
            return True

        def list_models(self):
            return []

    def fake_create_client(provider=None, model=None, config_defaults=None):
        captured["provider"] = provider
        captured["model"] = model
        captured["config_defaults"] = config_defaults
        return FakeClient()

    monkeypatch.setattr("Evaluator.shared_llm_adapters.create_client", fake_create_client)

    adapter = SharedOpenAIResponsesAdapter(
        settings=OpenAIResponsesSettings(model="gpt-test"),
        timeout=17.5,
    )

    assert adapter.client.provider_name == "openai_responses"
    assert captured == {
        "provider": "openai_responses",
        "model": "gpt-test",
        "config_defaults": {"timeout_seconds": 17.5},
    }
