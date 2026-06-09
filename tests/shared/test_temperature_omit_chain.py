"""Chain-level regression tests for temperature omit-by-default (gpt-5 compat).

Location: tests/shared/test_temperature_omit_chain.py

These tests guard the end-to-end propagation of a *None* temperature from the
upstream callers down to the OpenAI Responses provider payload, for BOTH the
eval/target path and the judge path. gpt-5-family reasoning models reject the
`temperature` parameter entirely, so when no temperature is explicitly
provided it must be OMITTED from the request (not sent as null or a default
float).

The provider-level omission guard itself
(`if temperature is not None: payload["temperature"] = temperature`) is
covered by tests/shared/test_openai_responses_provider.py. These tests cover
the CALLER chains that must not force a float before the value reaches that
guard:

  - EVAL path:  create_settings(...) -> OpenAIResponsesSettings.temperature
                -> SharedLLMAdapter.chat passes settings.temperature to
                   client.chat(...) -> provider omission guard.
  - JUDGE path: JudgeConfig.temperature -> JudgeService.judge passes
                judge_config.temperature to client.structured_output(...)
                -> provider omission guard.
"""

from __future__ import annotations

import pytest

from Evaluator.client_factory import create_settings
from Evaluator.config import OpenAIResponsesSettings
from shared.judge.judge_service import JudgeService
from shared.judge.models import JudgeConfig, RubricDef
from shared.llm.providers.openai_responses import OpenAIResponsesClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# Structural hops: defaults stay None from CLI args down to the settings/config
# ---------------------------------------------------------------------------

def test_create_settings_defaults_temperature_to_none():
    """No explicit temperature -> settings.temperature is None (eval hop).

    Mirrors cli.py passing args.temperature (now default None) into
    create_settings; None must survive into the settings dataclass so the
    adapter forwards None to the provider.
    """
    settings = create_settings(backend="openai_responses", model="gpt-5-nano")
    assert settings.temperature is None


def test_create_settings_preserves_explicit_temperature():
    """An explicitly-passed temperature still flows through unchanged."""
    settings = create_settings(
        backend="openai_responses", model="gpt-4o-mini", temperature=0.2
    )
    assert settings.temperature == 0.2


def test_judge_config_defaults_temperature_to_none():
    """JudgeConfig default temperature is None (judge hop)."""
    assert JudgeConfig().temperature is None


# ---------------------------------------------------------------------------
# EVAL chain: settings(None) -> adapter call shape -> provider omits temperature
# ---------------------------------------------------------------------------

def test_eval_chain_omits_temperature_when_unset(monkeypatch):
    """Settings default (None) reaches the provider and is omitted from chat().

    Reproduces the adapter's exact call shape
    (SharedLLMAdapter.chat -> client.chat(temperature=self.settings.temperature))
    using a real OpenAIResponsesClient with monkeypatched transport.
    """
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": "ok"})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    settings = OpenAIResponsesSettings(model="gpt-5-nano")
    client = OpenAIResponsesClient(api_key="test-key", model=settings.model)

    # Exact forwarding shape used by Evaluator/shared_llm_adapters.py::chat
    result = client.chat(
        [{"role": "user", "content": "hi"}],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    assert result == "ok"
    assert "temperature" not in captured["json"]


def test_eval_chain_sends_explicit_temperature(monkeypatch):
    """An explicit temperature is still sent (non-gpt-5 backends rely on it)."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": "ok"})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    settings = OpenAIResponsesSettings(model="gpt-4o-mini", temperature=0.2)
    client = OpenAIResponsesClient(api_key="test-key", model=settings.model)

    client.chat(
        [{"role": "user", "content": "hi"}],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    assert captured["json"]["temperature"] == 0.2


# ---------------------------------------------------------------------------
# JUDGE chain: JudgeConfig(None) -> JudgeService -> provider omits temperature
# ---------------------------------------------------------------------------

def _minimal_rubric() -> RubricDef:
    return RubricDef(
        key="quality",
        name="Quality",
        description="Quality of the response",
        scope="response",
        pass_threshold=0.5,
        judge_prompt="Judge this: {response}",
        output_schema={
            "type": "object",
            "properties": {"quality_score": {"type": "number"}},
        },
    )


def test_judge_chain_omits_temperature_when_unset(monkeypatch):
    """JudgeConfig default (None) reaches structured_output and is omitted."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": '{"quality_score": 1.0}'})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-5-mini")
    service = JudgeService(llm_client=client, judge_config=JudgeConfig())

    service.judge(prompt="Judge this: hello", rubrics=[_minimal_rubric()])

    assert "temperature" not in captured["json"]


def test_judge_chain_sends_explicit_temperature(monkeypatch):
    """An explicit judge temperature is still sent."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": '{"quality_score": 1.0}'})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-4o-mini")
    service = JudgeService(llm_client=client, judge_config=JudgeConfig(temperature=0.3))

    service.judge(prompt="Judge this: hello", rubrics=[_minimal_rubric()])

    assert captured["json"]["temperature"] == 0.3
