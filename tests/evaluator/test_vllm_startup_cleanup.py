"""Cleanup regressions migrated from the retired global vLLM launcher."""

import math
from types import SimpleNamespace

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


def _spec():
    return vllm_runtime.VLLMStartupSpec(
        source=vllm_runtime.ExplicitNetworkVLLMSource("fixture/model"),
        served_model_name="fixture",
    )


def _projection(*, startup=10.0, probe=5.0):
    return SimpleNamespace(
        argv=("python",),
        cwd=None,
        environment={},
        host="127.0.0.1",
        port=8000,
        served_model_name="fixture",
        expected_model_names=("fixture",),
        startup_timeout_s=startup,
        readiness_request_timeout_s=probe,
    )


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


@pytest.mark.parametrize(
    "deadline",
    [True, False, "1", object(), float("nan"), float("inf"), -float("inf"), 10**10000],
    ids=("true", "false", "text", "object", "nan", "inf", "negative-inf", "huge-int"),
)
def test_absolute_deadline_requires_a_finite_exact_number_before_projection(
    tmp_path, monkeypatch, deadline
):
    calls = []
    monkeypatch.setattr(
        vllm_runtime, "_projection", lambda *args, **kwargs: calls.append(1)
    )
    with pytest.raises((TypeError, ValueError)):
        vllm_runtime.start_vllm_runtime(
            _spec(), cwd=tmp_path, environment={}, deadline=deadline
        )
    assert calls == []


def test_elapsed_deadline_denies_before_projection_port_or_spawn(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_runtime, "_monotonic", lambda: 5.0)
    monkeypatch.setattr(
        vllm_runtime, "_projection", lambda *args, **kwargs: calls.append("projection")
    )
    monkeypatch.setattr(
        vllm_runtime, "_port_available", lambda *args: calls.append("port")
    )
    monkeypatch.setattr(
        vllm_runtime, "_spawn", lambda *args, **kwargs: calls.append("spawn")
    )
    with pytest.raises(vllm_runtime.VLLMRuntimeError, match="deadline expired"):
        vllm_runtime.start_vllm_runtime(
            _spec(), cwd=tmp_path, environment={}, deadline=5
        )
    assert calls == []


@pytest.mark.parametrize("phase", ["projection", "port"])
def test_deadline_elapsed_during_preparation_denies_before_spawn(
    tmp_path, monkeypatch, phase
):
    now = [0.0]
    calls = []
    monkeypatch.setattr(vllm_runtime, "_monotonic", lambda: now[0])

    def projection(*args, **kwargs):
        calls.append("projection")
        if phase == "projection":
            now[0] = 2.0
        return _projection()

    def port(*args):
        calls.append("port")
        if phase == "port":
            now[0] = 2.0
        return True

    monkeypatch.setattr(vllm_runtime, "_projection", projection)
    monkeypatch.setattr(vllm_runtime, "_port_available", port)
    monkeypatch.setattr(
        vllm_runtime, "_spawn", lambda *args, **kwargs: calls.append("spawn")
    )
    with pytest.raises(vllm_runtime.VLLMRuntimeError, match="deadline expired"):
        vllm_runtime.start_vllm_runtime(
            _spec(), cwd=tmp_path, environment={}, deadline=1.0
        )
    assert calls == (
        ["projection"] if phase == "projection" else ["projection", "port"]
    )


def test_deadline_elapsed_during_spawn_denies_before_probe_and_cleans(
    tmp_path, monkeypatch
):
    process = _Process()
    now = [0.0]
    ready_calls = []
    monkeypatch.setattr(vllm_runtime, "_monotonic", lambda: now[0])
    monkeypatch.setattr(
        vllm_runtime, "_projection", lambda *args, **kwargs: _projection()
    )
    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)

    def spawn(*args, **kwargs):
        now[0] = 2.0
        return process

    monkeypatch.setattr(vllm_runtime, "_spawn", spawn)
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: True)
    monkeypatch.setattr(vllm_runtime, "_ready", lambda *args: ready_calls.append(args))
    with pytest.raises(vllm_runtime.VLLMRuntimeError, match="deadline expired"):
        vllm_runtime.start_vllm_runtime(
            _spec(), cwd=tmp_path, environment={}, deadline=1.0
        )
    assert ready_calls == []
    assert process.calls == 1


@pytest.mark.parametrize(
    ("deadline", "expected_timeout"), [(3.0, 3.0), (10.0, 5.0), (20.0, 5.0)]
)
def test_absolute_deadline_clamps_existing_startup_and_probe_limits(
    tmp_path, monkeypatch, deadline, expected_timeout
):
    process = _Process()
    timeouts = []
    monkeypatch.setattr(vllm_runtime, "_monotonic", lambda: 0.0)
    monkeypatch.setattr(
        vllm_runtime, "_projection", lambda *args, **kwargs: _projection()
    )
    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)
    monkeypatch.setattr(vllm_runtime, "_spawn", lambda *args, **kwargs: process)
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: True)

    def ready(host, port, names, timeout):
        timeouts.append(timeout)
        return True

    monkeypatch.setattr(vllm_runtime, "_ready", ready)
    lease = vllm_runtime.start_vllm_runtime(
        _spec(), cwd=tmp_path, environment={}, deadline=deadline
    )
    assert timeouts == [expected_timeout]
    assert lease.close() is True


@pytest.mark.parametrize("failure", [None, KeyboardInterrupt()])
def test_readiness_crossing_absolute_deadline_cleans_once_and_preserves_control(
    tmp_path, monkeypatch, failure
):
    process = _Process(result=False)
    now = [0.0]
    monkeypatch.setattr(vllm_runtime, "_monotonic", lambda: now[0])
    monkeypatch.setattr(
        vllm_runtime, "_projection", lambda *args, **kwargs: _projection()
    )
    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)
    monkeypatch.setattr(vllm_runtime, "_spawn", lambda *args, **kwargs: process)
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: True)

    def ready(*args):
        now[0] = 2.0
        if failure is not None:
            raise failure
        return True

    monkeypatch.setattr(vllm_runtime, "_ready", ready)
    expected = type(failure) if failure is not None else vllm_runtime.VLLMRuntimeError
    with pytest.raises(expected) as caught:
        vllm_runtime.start_vllm_runtime(
            _spec(), cwd=tmp_path, environment={}, deadline=1.0
        )
    assert process.calls == 1
    assert caught.value.cleanup_lease is process
    if failure is None:
        assert isinstance(caught.value, vllm_runtime.VLLMRuntimeError)
    else:
        assert caught.value is failure


def test_nonfinite_monotonic_clock_is_closed_before_projection(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_runtime, "_monotonic", lambda: math.nan)
    monkeypatch.setattr(
        vllm_runtime, "_projection", lambda *args, **kwargs: calls.append(1)
    )
    with pytest.raises(vllm_runtime.VLLMRuntimeError, match="clock is invalid"):
        vllm_runtime.start_vllm_runtime(
            _spec(), cwd=tmp_path, environment={}, deadline=1.0
        )
    assert calls == []


def test_nonfinite_clock_after_ready_cannot_bypass_deadline_and_cleans(
    tmp_path, monkeypatch
):
    process = _Process()
    now = [0.0]
    monkeypatch.setattr(vllm_runtime, "_monotonic", lambda: now[0])
    monkeypatch.setattr(
        vllm_runtime, "_projection", lambda *args, **kwargs: _projection()
    )
    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)
    monkeypatch.setattr(vllm_runtime, "_spawn", lambda *args, **kwargs: process)
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: True)

    def ready(*args):
        now[0] = math.nan
        return True

    monkeypatch.setattr(vllm_runtime, "_ready", ready)
    with pytest.raises(vllm_runtime.VLLMRuntimeError, match="clock is invalid"):
        vllm_runtime.start_vllm_runtime(
            _spec(), cwd=tmp_path, environment={}, deadline=1.0
        )
    assert process.calls == 1


class _ClockFloat(float):
    pass


@pytest.mark.parametrize(
    "bad", (True, 10**400, _ClockFloat(0)), ids=("bool", "overflow", "subclass")
)
@pytest.mark.parametrize("phase", ("initial", "port", "ready"))
def test_nonexact_clock_fails_closed_at_every_effect_boundary(
    tmp_path, monkeypatch, phase, bad
):
    process = _Process()
    now = [bad if phase == "initial" else 0.0]
    calls = []
    monkeypatch.setattr(vllm_runtime, "_monotonic", lambda: now[0])

    def projection(*args, **kwargs):
        calls.append("projection")
        return _projection()

    def port(*args):
        calls.append("port")
        if phase == "port":
            now[0] = bad
        return True

    def ready(*args):
        calls.append("ready")
        now[0] = bad
        return True

    monkeypatch.setattr(vllm_runtime, "_projection", projection)
    monkeypatch.setattr(vllm_runtime, "_port_available", port)
    monkeypatch.setattr(
        vllm_runtime, "_spawn", lambda *args, **kwargs: calls.append("spawn") or process
    )
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: True)
    monkeypatch.setattr(vllm_runtime, "_ready", ready)
    with pytest.raises(vllm_runtime.VLLMRuntimeError, match="clock is invalid"):
        vllm_runtime.start_vllm_runtime(
            _spec(), cwd=tmp_path, environment={}, deadline=1.0
        )
    assert (
        calls
        == {
            "initial": [],
            "port": ["projection", "port"],
            "ready": ["projection", "port", "spawn", "ready"],
        }[phase]
    )
    assert process.calls == (1 if phase == "ready" else 0)
