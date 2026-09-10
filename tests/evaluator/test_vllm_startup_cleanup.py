"""Cleanup regressions migrated from the retired global vLLM launcher."""

import pytest

from Evaluator import vllm_runtime, vllm_setup


class _Process:
    cleanup_pending = True

    def __init__(self, *, result=True, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def close(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        self.cleanup_pending = not self.result
        return self.result


@pytest.mark.parametrize(
    "failure",
    [OSError("closed"), RuntimeError("closed"), KeyboardInterrupt(), SystemExit(7)],
)
def test_startup_preserves_original_failure_when_cleanup_raises(
    tmp_path, monkeypatch, failure
):
    process = _Process(error=OSError("cleanup"))
    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)
    monkeypatch.setattr(vllm_runtime, "_spawn", lambda *args, **kwargs: process)
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda process: True)

    def fail(*args):
        raise failure

    monkeypatch.setattr(vllm_runtime, "_ready", fail)
    spec = vllm_runtime.VLLMStartupSpec(
        source=vllm_runtime.ExplicitNetworkVLLMSource("fixture/model"),
        served_model_name="fixture",
    )
    with pytest.raises(type(failure)) as caught:
        vllm_runtime.start_vllm_runtime(spec, cwd=tmp_path, environment={})
    assert caught.value is failure
    assert caught.value.cleanup_lease is process
    assert process.calls == 1


def test_runtime_context_closes_successful_owner():
    process = _Process()
    with vllm_runtime.VLLMRuntimeLease(
        process, host="127.0.0.1", port=8000, served_model_name="fixture"
    ):
        pass
    assert process.calls == 1
    assert not process.cleanup_pending


def test_runtime_context_reports_unresolved_cleanup_without_active_error():
    process = _Process(result=False)
    with pytest.raises(vllm_runtime.VLLMRuntimeError):
        with vllm_runtime.VLLMRuntimeLease(
            process, host="127.0.0.1", port=8000, served_model_name="fixture"
        ):
            pass
    assert process.cleanup_pending


def test_runtime_context_preserves_interrupt_during_unresolved_cleanup():
    process = _Process(result=False)
    failure = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with vllm_runtime.VLLMRuntimeLease(
            process, host="127.0.0.1", port=8000, served_model_name="fixture"
        ):
            raise failure
    assert caught.value is failure
    assert process.cleanup_pending


def test_retired_global_lifecycle_has_no_compatibility_exports():
    for name in (
        "start_vllm_server",
        "stop_vllm_server",
        "is_server_managed",
        "_server_process",
        "_wait_for_server",
        "_stop_vllm_process",
    ):
        assert not hasattr(vllm_setup, name)
