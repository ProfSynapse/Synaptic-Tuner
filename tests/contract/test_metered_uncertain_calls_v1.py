"""Unknown paid requests stop later calls; CPU clients have no provider effects."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shared.llm.metering import MeteredCallRefused, MeteredLLMClient
from shared.llm.usage import (
    LLMCompletionV1,
    LLMStructuredV1,
    UsageAccumulator,
    measured_usage,
)
from synaptic_tuner.api.v1.reference.evaluation import _MeteredBackendClient


class Client:
    provider_name = "openrouter"
    model_name = "fixture"

    def __init__(self, mode="measured"):
        self.mode, self.calls = mode, []

    def _usage(self, method):
        self.calls.append(method)
        if self.mode == "raise":
            raise RuntimeError("private provider failure")
        return None if self.mode == "missing" else measured_usage(3, 5)

    def chat(self, *args, **kwargs):
        usage = self._usage("chat")
        return object() if self.mode == "wrong" else LLMCompletionV1("ok", usage)

    def structured_output(self, *args, **kwargs):
        usage = self._usage("structured")
        return (
            object()
            if self.mode == "wrong"
            else LLMStructuredV1({"quality_score": 0.95}, usage)
        )


def invoke(client, method):
    return client.chat([]) if method == "chat" else client.structured_output([], {})


@pytest.mark.parametrize("mode", ["raise", "wrong", "missing"])
@pytest.mark.parametrize("first", ["chat", "structured"])
def test_unknown_request_blocks_both_methods_and_sibling_wrappers(mode, first):
    meter = UsageAccumulator()
    original, sibling = Client(mode), Client()
    wrapped, other = (
        MeteredLLMClient(original, meter, stop_on_unknown=True),
        MeteredLLMClient(sibling, meter, stop_on_unknown=True),
    )
    if mode == "missing":
        invoke(wrapped, first)
    else:
        with pytest.raises((RuntimeError, TypeError)):
            invoke(wrapped, first)
    for client in (wrapped, other):
        for method in ("chat", "structured"):
            with pytest.raises(MeteredCallRefused, match="^metered_run_indeterminate$"):
                invoke(client, method)
    assert original.calls == [first] and sibling.calls == []
    assert meter.calls == meter.unmeasured == 1


@pytest.mark.parametrize("measured", [False, True])
def test_actual_engine_retry_loop_preserves_measured_retry_only(
    tmp_path, monkeypatch, measured
):
    from SynthChat.engine import ImprovementEngine
    import SynthChat.engine as engine_module

    monkeypatch.setattr(engine_module, "FALLBACK_PROVIDERS", [])
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)
    inner, meter = Client("measured" if measured else "raise"), UsageAccumulator()
    client = MeteredLLMClient(inner, meter, stop_on_unknown=True)
    engine = ImprovementEngine(
        client, tmp_path, logger=Mock(), enable_interactions=False
    )
    attempts = []

    def operation():
        attempts.append(1)
        result = client.chat([])
        if measured and len(attempts) == 1:
            raise ValueError("known response failed declarative validation")
        return result

    if measured:
        assert engine._call_with_retry(operation, "improve").text == "ok"
        assert len(inner.calls) == meter.calls == len(attempts) == 2
        assert not meter.indeterminate
    else:
        with pytest.raises(MeteredCallRefused):
            engine._call_with_retry(operation, "improve")
        assert len(attempts) == engine_module.MAX_RETRIES
        assert inner.calls == ["chat"] and meter.calls == 1


def test_judge_service_swallow_blocks_shared_paid_backend():
    from shared.judge import JudgeConfig, RubricDef
    from shared.judge.judge_service import JudgeService

    meter, inner = UsageAccumulator(), Client("raise")
    judge = JudgeService(
        MeteredLLMClient(inner, meter, stop_on_unknown=True), JudgeConfig()
    )
    rubric = RubricDef("quality", "Quality", "fixture", "response", 0.8, "Score.", {})
    result = judge.judge("response", [rubric])
    assert result.error is not None
    backend = Mock()
    wrapped = _MeteredBackendClient(backend, meter, stop_on_unknown=True)
    with pytest.raises(MeteredCallRefused):
        wrapped.chat([])
    assert backend.mock_calls == [] and inner.calls == ["structured"]
    assert meter.calls == meter.unmeasured == 1


def test_backend_unknown_blocks_judge_and_separate_local_meter_remains_usable():
    shared, local = UsageAccumulator(), UsageAccumulator()
    backend = Mock()
    backend.chat.return_value = SimpleNamespace(usage=None)
    _MeteredBackendClient(backend, shared, stop_on_unknown=True).chat([])
    judge_inner = Client()
    with pytest.raises(MeteredCallRefused):
        MeteredLLMClient(judge_inner, shared, stop_on_unknown=True).structured_output(
            [], {}
        )
    assert judge_inner.calls == [] and shared.calls == 1
    assert MeteredLLMClient(judge_inner, local).chat([]).text == "ok"
    assert local.measured and judge_inner.calls == ["chat"]


def test_unwrapped_legacy_client_behavior_is_unchanged():
    inner = Client("raise")
    for _ in range(2):
        with pytest.raises(RuntimeError):
            inner.chat([])
    assert inner.calls == ["chat", "chat"]


@pytest.mark.parametrize("flag", [1, 0, None, "yes"])
def test_opt_in_requires_exact_bool(flag):
    with pytest.raises(TypeError):
        MeteredLLMClient(Client(), UsageAccumulator(), stop_on_unknown=flag)
    with pytest.raises(TypeError):
        _MeteredBackendClient(Client(), UsageAccumulator(), stop_on_unknown=flag)


def test_default_metered_local_client_keeps_legacy_missing_usage_calls():
    inner, meter = Client("missing"), UsageAccumulator()
    wrapped = MeteredLLMClient(inner, meter)
    wrapped.chat([])
    wrapped.structured_output([], {})
    assert inner.calls == ["chat", "structured"] and meter.calls == 2
    assert meter.indeterminate


def test_paid_backend_raising_usage_accessor_latches_without_swallowing():
    class UsageFailure(RuntimeError):
        pass

    class Response:
        @property
        def usage(self):
            raise UsageFailure("private accessor failure")

    backend, meter = Mock(), UsageAccumulator()
    backend.chat.return_value = Response()
    client = _MeteredBackendClient(backend, meter, stop_on_unknown=True)
    with pytest.raises(UsageFailure):
        client.chat([])
    with pytest.raises(MeteredCallRefused):
        client.chat([])
    assert backend.chat.call_count == meter.calls == meter.unmeasured == 1


def test_default_backend_raising_usage_accessor_preserves_legacy_behavior():
    class Response:
        @property
        def usage(self):
            raise ValueError("legacy accessor failure")

    backend, meter = Mock(), UsageAccumulator()
    backend.chat.return_value = Response()
    client = _MeteredBackendClient(backend, meter)
    for _ in range(2):
        with pytest.raises(ValueError):
            client.chat([])
    assert backend.chat.call_count == 2 and meter.calls == 0
