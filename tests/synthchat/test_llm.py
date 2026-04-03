"""Tests for SynthChat.llm — client pool and caller modules."""
from __future__ import annotations

import pytest
from SynthChat.llm.client_pool import LLMClientPool
from SynthChat.llm.caller import call_llm, call_llm_structured


class _FakeClient:
    """Minimal fake LLM client for pool tests."""
    def __init__(self, provider_name="openrouter", model_name="fake/model"):
        self._provider_name = provider_name
        self._model_name = model_name
        self.provider = None
        self.timeout_seconds = 60.0
        self.default_max_tokens = None

    @property
    def provider_name(self):
        return self._provider_name

    @property
    def model_name(self):
        return self._model_name

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        return "fake response"

    def structured_output(self, messages, schema, temperature=0.3, max_tokens=2048):
        return {"result": "ok"}


class _FailClient(_FakeClient):
    """Client that always raises."""
    def chat(self, **kwargs):
        raise RuntimeError("always fails")

    def structured_output(self, **kwargs):
        raise RuntimeError("always fails")


# ---- LLMClientPool ----

class TestLLMClientPool:
    def test_default_client_returned_for_none_config(self):
        client = _FakeClient()
        pool = LLMClientPool(client)
        result = pool.get_stage_clients(None)
        assert result == [client]

    def test_default_client_returned_for_empty_config(self):
        client = _FakeClient()
        pool = LLMClientPool(client)
        result = pool.get_stage_clients({})
        assert result == [client]

    def test_normalize_stage_spec_string(self):
        result = LLMClientPool.normalize_stage_spec("gpt-4")
        assert result == {"model": "gpt-4"}

    def test_normalize_stage_spec_empty_string(self):
        result = LLMClientPool.normalize_stage_spec("")
        assert result is None

    def test_normalize_stage_spec_dict(self):
        result = LLMClientPool.normalize_stage_spec({"model": "gpt-4", "provider": "OpenRouter"})
        assert result == {"model": "gpt-4", "provider": "openrouter"}

    def test_normalize_stage_spec_empty_dict(self):
        result = LLMClientPool.normalize_stage_spec({})
        assert result is None

    def test_normalize_stage_spec_non_string_non_dict(self):
        assert LLMClientPool.normalize_stage_spec(42) is None
        assert LLMClientPool.normalize_stage_spec(None) is None

    def test_get_or_create_returns_default_for_matching_spec(self):
        client = _FakeClient(provider_name="openrouter", model_name="fake/model")
        pool = LLMClientPool(client)
        result = pool.get_or_create({"model": "fake/model", "provider": "openrouter"})
        assert result is client

    def test_get_or_create_caches_new_client(self):
        default = _FakeClient()
        created_clients = []
        def factory(provider, model, config_defaults=None):
            c = _FakeClient(provider_name=provider, model_name=model)
            created_clients.append(c)
            return c

        pool = LLMClientPool(default, client_factory=factory)
        c1 = pool.get_or_create({"model": "other/model"})
        c2 = pool.get_or_create({"model": "other/model"})
        assert c1 is c2
        assert len(created_clients) == 1

    def test_different_specs_create_different_clients(self):
        default = _FakeClient()
        def factory(provider, model, config_defaults=None):
            return _FakeClient(provider_name=provider, model_name=model)

        pool = LLMClientPool(default, client_factory=factory)
        c1 = pool.get_or_create({"model": "model_a"})
        c2 = pool.get_or_create({"model": "model_b"})
        assert c1 is not c2

    def test_fallback_models_in_chain(self):
        default = _FakeClient()
        def factory(provider, model, config_defaults=None):
            return _FakeClient(provider_name=provider, model_name=model)

        pool = LLMClientPool(default, client_factory=factory)
        config = {
            "model": "primary/model",
            "fallback_models": [
                {"model": "fallback_a"},
                {"model": "fallback_b"},
            ]
        }
        chain = pool.get_stage_clients(config)
        assert len(chain) == 3
        models = [c.model_name for c in chain]
        assert "primary/model" in models
        assert "fallback_a" in models
        assert "fallback_b" in models

    def test_duplicate_fallback_deduplicated(self):
        default = _FakeClient()
        def factory(provider, model, config_defaults=None):
            return _FakeClient(provider_name=provider, model_name=model)

        pool = LLMClientPool(default, client_factory=factory)
        config = {
            "model": "primary/model",
            "fallback_models": [
                {"model": "primary/model"},
            ]
        }
        chain = pool.get_stage_clients(config)
        assert len(chain) == 1


# ---- call_llm ----

class TestCallLlm:
    def test_returns_response(self):
        client = _FakeClient()
        result = call_llm(prompt="hello", default_client=client, randomize=False)
        assert result == "fake response"

    def test_retries_on_empty(self):
        call_count = 0
        class EmptyThenOk:
            provider_name = "test"
            model_name = "test"
            default_max_tokens = None
            def chat(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    return ""
                return "ok"

        result = call_llm(prompt="hello", default_client=EmptyThenOk(), randomize=False, max_retries=3)
        assert result == "ok"
        assert call_count == 2

    def test_falls_back_to_next_client(self):
        fail_client = _FailClient()
        ok_client = _FakeClient()
        result = call_llm(
            prompt="hello",
            default_client=fail_client,
            llm_clients=[fail_client, ok_client],
            randomize=False,
            max_retries=1,
        )
        assert result == "fake response"

    def test_raises_after_exhausting_retries(self):
        client = _FailClient()
        with pytest.raises(RuntimeError, match="always fails"):
            call_llm(prompt="hello", default_client=client, randomize=False, max_retries=2)


# ---- call_llm_structured ----

class TestCallLlmStructured:
    def test_returns_structured_response(self):
        client = _FakeClient()
        result = call_llm_structured(
            prompt="hello",
            schema={"type": "object"},
            default_client=client,
            randomize=False,
        )
        assert result == {"result": "ok"}

    def test_falls_back_to_chat_when_no_structured_output(self):
        class ChatOnlyClient:
            provider_name = "test"
            model_name = "test"
            default_max_tokens = None
            def chat(self, **kwargs):
                return '{"key": "value"}'

        result = call_llm_structured(
            prompt="hello",
            schema={"type": "object"},
            default_client=ChatOnlyClient(),
            randomize=False,
        )
        assert result == {"key": "value"}

    def test_raises_on_non_json_from_chat_fallback(self):
        class NonJsonClient:
            provider_name = "test"
            model_name = "test"
            default_max_tokens = None
            def chat(self, **kwargs):
                return "not json at all"

        with pytest.raises(ValueError, match="non-JSON"):
            call_llm_structured(
                prompt="hello",
                schema={"type": "object"},
                default_client=NonJsonClient(),
                randomize=False,
            )

    def test_system_prompt_included_in_messages(self):
        captured = {}
        class CapturingClient:
            provider_name = "test"
            model_name = "test"
            default_max_tokens = None
            def structured_output(self, messages, schema, **kwargs):
                captured["messages"] = messages
                return {"ok": True}

        call_llm_structured(
            prompt="question",
            schema={"type": "object"},
            default_client=CapturingClient(),
            system_prompt="Be helpful",
            randomize=False,
        )
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][0]["content"] == "Be helpful"
        assert captured["messages"][1]["role"] == "user"
