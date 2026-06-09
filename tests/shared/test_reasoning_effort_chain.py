"""Chain-level regression tests for the gpt-5 reasoning-effort knob.

Location: tests/shared/test_reasoning_effort_chain.py

gpt-5-family reasoning models default to medium effort, which can consume the
entire max_output_tokens budget and return an empty message item. Reasoning
effort is plumbed to the OpenAI Responses provider for BOTH the eval/target
path (chat) and the judge path (structured_output). When set it is sent as
``reasoning: {effort: ...}``; when None it is OMITTED entirely so non-gpt-5
backends are unaffected.

ENGINE-KNOB RECONCILE (upstream #98): the canonical engine knob is
``thinking_effort`` on the cloud settings classes, default ``None`` (effort
omitted unless a caller opts in — this sidesteps the per-model
'minimal'-unsupported gap, e.g. gpt-5.4-mini). Our former separate
``reasoning_effort`` settings field was dropped; ``create_settings`` still
accepts a deprecated ``reasoning_effort`` alias that folds into
``thinking_effort``. The provider keeps a per-call ``reasoning_effort``
argument that OVERRIDES the instance ``thinking_effort`` default.

The JUDGE-side knob (JudgeConfig / EvalJudgeConfig.reasoning_effort) is our
additive judge work — unchanged: it still defaults to "minimal", is validated,
and is threaded per-call to the provider (overriding the instance default).

These tests mirror tests/shared/test_temperature_omit_chain.py:
  - provider sends/omits the reasoning field based on the resolved effort;
  - engine settings default thinking_effort to None; the reasoning_effort alias folds in;
  - the judge config defaults stay "minimal" and are validated;
  - the adapter / judge_service call shapes propagate the value.
"""

from __future__ import annotations

import pytest

from Evaluator.client_factory import create_settings
from Evaluator.config import EvalJudgeConfig, OpenAIResponsesSettings
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
# Engine-knob structural hops: thinking_effort default None; reasoning_effort
# alias folds into it; judge configs still default "minimal" and validate.
# ---------------------------------------------------------------------------

def test_create_settings_defaults_thinking_effort_to_none():
    # Post-#98 reconcile: the engine no longer forces "minimal"; effort is
    # omitted unless a caller opts in (closes the per-model 'minimal' gap).
    settings = create_settings(backend="openai_responses", model="gpt-5-nano")
    assert settings.thinking_effort is None


def test_create_settings_honors_explicit_thinking_effort():
    settings = create_settings(
        backend="openai_responses", model="gpt-5-nano", thinking_effort="low"
    )
    assert settings.thinking_effort == "low"


def test_create_settings_reasoning_effort_alias_folds_into_thinking_effort():
    # The deprecated reasoning_effort kwarg still works, folded into the
    # canonical thinking_effort knob.
    settings = create_settings(
        backend="openai_responses", model="gpt-5-nano", reasoning_effort="low"
    )
    assert settings.thinking_effort == "low"


def test_create_settings_explicit_thinking_effort_wins_over_alias():
    settings = create_settings(
        backend="openai_responses",
        model="gpt-5-nano",
        thinking_effort="high",
        reasoning_effort="low",
    )
    assert settings.thinking_effort == "high"


def test_eval_judge_config_defaults_reasoning_effort_to_minimal():
    assert EvalJudgeConfig().reasoning_effort == "minimal"


def test_judge_config_defaults_reasoning_effort_to_minimal():
    assert JudgeConfig().reasoning_effort == "minimal"


def test_invalid_reasoning_effort_rejected_at_judge_config():
    with pytest.raises(ValueError):
        JudgeConfig(reasoning_effort="extreme")


def test_none_thinking_effort_allowed_at_settings_layer():
    settings = OpenAIResponsesSettings(model="gpt-5-nano", thinking_effort=None)
    assert settings.thinking_effort is None


# ---------------------------------------------------------------------------
# EVAL chain: settings.thinking_effort -> client instance -> provider payload.
# Post-#98 the adapter does NOT pass effort per-call; the client carries it as
# the instance thinking_effort (set at construction from settings).
# ---------------------------------------------------------------------------

def test_eval_chain_includes_reasoning_when_instance_effort_set(monkeypatch):
    """The client's instance thinking_effort is applied to the chat payload."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": "ok"})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    # Mirror the adapter: settings.thinking_effort -> client constructor.
    settings = OpenAIResponsesSettings(model="gpt-5-nano", thinking_effort="minimal")
    client = OpenAIResponsesClient(
        api_key="test-key",
        model=settings.model,
        thinking_effort=settings.thinking_effort,
    )

    result = client.chat(
        [{"role": "user", "content": "hi"}],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    assert result == "ok"
    assert captured["json"]["reasoning"] == {"effort": "minimal"}


def test_eval_chain_omits_reasoning_when_none(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": "ok"})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    # Default thinking_effort is None -> no reasoning field on the wire.
    settings = OpenAIResponsesSettings(model="gpt-4o-mini")
    client = OpenAIResponsesClient(
        api_key="test-key",
        model=settings.model,
        thinking_effort=settings.thinking_effort,
    )

    client.chat(
        [{"role": "user", "content": "hi"}],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    assert "reasoning" not in captured["json"]


def test_per_call_reasoning_effort_overrides_instance_thinking_effort(monkeypatch):
    """The provider's per-call reasoning_effort wins over the instance default.

    This is the judge path's mechanism: judge_service passes
    reasoning_effort=judge_config.reasoning_effort per call.
    """
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": "ok"})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    client = OpenAIResponsesClient(
        api_key="test-key", model="gpt-5-nano", thinking_effort="high"
    )
    client.chat(
        [{"role": "user", "content": "hi"}],
        reasoning_effort="low",  # per-call override
    )

    assert captured["json"]["reasoning"] == {"effort": "low"}


# ---------------------------------------------------------------------------
# JUDGE chain: JudgeConfig.reasoning_effort -> JudgeService -> provider payload
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


def test_judge_chain_includes_reasoning_when_set(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": '{"quality_score": 1.0}'})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-5-mini")
    service = JudgeService(llm_client=client, judge_config=JudgeConfig())  # minimal

    service.judge(prompt="Judge this: hello", rubrics=[_minimal_rubric()])

    assert captured["json"]["reasoning"] == {"effort": "minimal"}


def test_judge_chain_omits_reasoning_when_none(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse({"output_text": '{"quality_score": 1.0}'})

    monkeypatch.setattr(
        "shared.llm.providers.openai_responses.requests.post", fake_post
    )

    client = OpenAIResponsesClient(api_key="test-key", model="gpt-4o-mini")
    service = JudgeService(
        llm_client=client, judge_config=JudgeConfig(reasoning_effort=None)
    )

    service.judge(prompt="Judge this: hello", rubrics=[_minimal_rubric()])

    assert "reasoning" not in captured["json"]
