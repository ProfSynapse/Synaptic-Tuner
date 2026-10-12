"""CPU-only checks for the optional SFT numeric-observation callback seam."""

from __future__ import annotations

import builtins
import json
import sys
import types
import warnings
from types import SimpleNamespace

import pytest

# The capacity helper's POSIX resource import is irrelevant to these CPU tests.
# Windows lacks that standard-library module; never replace it on Linux.
_added_resource_stub = sys.platform == "win32" and "resource" not in sys.modules
if _added_resource_stub:
    sys.modules["resource"] = types.ModuleType("resource")
try:
    from Trainers.sft.src.training_callbacks import MetricsTableCallback
    from Trainers.shared.callbacks import base
finally:
    if _added_resource_stub:
        del sys.modules["resource"]
from synaptic_tuner.api.v1.observations import TrainingMetricObservedPayloadV1


def _state(step: int, *, initial: int = 0, primary: bool = True):
    return SimpleNamespace(global_step=step, max_steps=100, is_world_process_zero=primary)


def _args():
    return SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=3,
                           world_size=1, max_grad_norm=1.0, output_dir="unused")


def _begin(callback, *, initial: int = 0):
    callback.on_train_begin(_args(), _state(initial), SimpleNamespace())


def test_opt_in_sft_sink_has_numeric_payload_cadence_and_resume_rate(tmp_path, monkeypatch) -> None:
    points: list[TrainingMetricObservedPayloadV1] = []
    callback = MetricsTableCallback(5, str(tmp_path), metric_sink=points.append)
    ticks = iter((10.0, 15.0))
    monkeypatch.setattr(base.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(base, "_observed_gpu_bytes", lambda: (101, 202, 303, 404))
    _begin(callback, initial=20)
    callback.on_log(_args(), _state(24), SimpleNamespace(), logs={"loss": 1.0})
    assert points == []
    callback.on_log(_args(), _state(25), SimpleNamespace(), logs={
        "loss": -0.5, "learning_rate": 0.001, "epoch": 0.25,
    })
    assert len(points) == 1
    point = points[0]
    assert type(point) is TrainingMetricObservedPayloadV1
    assert (point.step, point.total_steps, point.loss, point.learning_rate, point.epoch) == (
        25, 100, -0.5, 0.001, 0.25,
    )
    assert (point.train_elapsed_seconds, point.job_elapsed_seconds) == (5.0, None)
    assert point.estimated_samples_per_second == 6.0  # (25 - 20) * 2 * 3 / 5; per device
    assert (point.gpu_current_allocated_bytes, point.gpu_current_reserved_bytes,
            point.gpu_peak_allocated_bytes, point.gpu_peak_reserved_bytes) == (101, 202, 303, 404)
    assert point.gpu_utilization_percent is None and point.tokens_per_second is None


def test_sink_skips_nonprimary_nonoptimizer_and_invalid_steps(tmp_path, monkeypatch) -> None:
    points: list[TrainingMetricObservedPayloadV1] = []
    callback = MetricsTableCallback(5, str(tmp_path), metric_sink=points.append)
    monkeypatch.setattr(base.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(base, "_observed_gpu_bytes", lambda: (None,) * 4)
    _begin(callback)
    callback.on_log(_args(), _state(5, primary=False), SimpleNamespace(), logs={"loss": 1.0})
    callback.on_log(_args(), _state(5), SimpleNamespace(), logs={"eval_loss": 1.0, "epoch": 1.0})
    callback._emit_metric(state=_state(0), args=_args(), logs={"loss": 1.0})
    callback._emit_metric(state=_state(True), args=_args(), logs={"loss": 1.0})
    callback._emit_metric(state=_state(2**63), args=_args(), logs={"loss": 1.0})
    assert points == []


def test_distributed_sample_rate_is_unknown_without_global_scope(tmp_path, monkeypatch) -> None:
    points: list[TrainingMetricObservedPayloadV1] = []
    callback = MetricsTableCallback(5, str(tmp_path), metric_sink=points.append)
    ticks = iter((10.0, 15.0))
    monkeypatch.setattr(base.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(base, "_observed_gpu_bytes", lambda: (None,) * 4)
    _begin(callback)
    args = _args()
    args.world_size = 2
    callback._emit_metric(state=_state(5), args=args, logs={"loss": 1.0})
    assert points[0].estimated_samples_per_second is None


def test_invalid_measurements_are_null_and_sink_failure_is_closed(tmp_path, monkeypatch, capsys) -> None:
    seen: list[TrainingMetricObservedPayloadV1] = []

    def failing_sink(payload: TrainingMetricObservedPayloadV1) -> None:
        seen.append(payload)
        raise RuntimeError("secret private failure text")

    callback = MetricsTableCallback(5, str(tmp_path), metric_sink=failing_sink)
    ticks = iter((10.0, 9.0))
    monkeypatch.setattr(base.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(base, "_observed_gpu_bytes", lambda: (None,) * 4)
    _begin(callback)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        callback._emit_metric(state=_state(5), args=_args(), logs={
            "loss": -0.2, "learning_rate": float("nan"), "epoch": -1.0,
        })
    assert len(seen) == 1
    assert seen[0].loss == -0.2
    assert (seen[0].learning_rate, seen[0].epoch, seen[0].train_elapsed_seconds,
            seen[0].estimated_samples_per_second) == (None, None, None, None)
    assert callback.metric_sink is None
    output = capsys.readouterr().out
    assert output.count("Optional training metric observation sink disabled") == 1
    assert "secret private failure text" not in output
    callback._emit_metric(state=_state(10), args=_args(), logs={"loss": 1.0})
    assert len(seen) == 1


def test_direct_cuda_bytes_are_exact_or_unknown(monkeypatch) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 1,
        memory_allocated=lambda device: 11 if device == 1 else -1,
        memory_reserved=lambda device: 22,
        max_memory_allocated=lambda device: 33,
        max_memory_reserved=lambda device: 44,
    )
    monkeypatch.setattr(base.torch, "cuda", cuda)
    assert base._observed_gpu_bytes() == (11, 22, 33, 44)
    cuda.current_device = lambda: -1
    assert base._observed_gpu_bytes() == (None, None, None, None)
    cuda.current_device = lambda: 1
    cuda.memory_allocated = lambda device: True
    cuda.max_memory_reserved = lambda device: 2**63
    assert base._observed_gpu_bytes() == (None, 22, 33, None)


def test_sink_argument_is_optional_but_must_be_callable(tmp_path) -> None:
    callback = MetricsTableCallback(5, str(tmp_path))
    assert callback.metric_sink is None
    with pytest.raises(TypeError, match="metric_sink must be callable"):
        MetricsTableCallback(5, str(tmp_path), metric_sink=1)


def test_optional_clock_setup_failure_disables_sink_without_stopping_training(tmp_path, monkeypatch, capsys) -> None:
    seen: list[TrainingMetricObservedPayloadV1] = []
    callback = MetricsTableCallback(5, str(tmp_path), metric_sink=seen.append)

    def failed_clock() -> float:
        raise RuntimeError("secret clock failure")

    monkeypatch.setattr(base.time, "monotonic", failed_clock)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _begin(callback)
    assert callback.metric_sink is None
    assert seen == []
    output = capsys.readouterr().out
    assert output.count("Optional training metric observation sink disabled") == 1
    assert "secret clock failure" not in output


def test_default_callback_keeps_legacy_jsonl_and_never_loads_metric_contract(tmp_path, monkeypatch) -> None:
    callback = MetricsTableCallback(5, str(tmp_path))
    monkeypatch.setattr(base.time, "monotonic", lambda: pytest.fail("no-sink monotonic call"))
    monkeypatch.setattr(base, "_observed_gpu_bytes", lambda: pytest.fail("no-sink GPU call"))
    monkeypatch.setattr(base, "capture_runtime_capacity_snapshot", lambda torch: {})
    real_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "synaptic_tuner.api.v1.observations":
            pytest.fail("no-sink observation import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    _begin(callback)
    callback.on_log(_args(), _state(5), SimpleNamespace(), logs={"loss": 1.0})
    rows = [json.loads(line) for line in callback.log_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["step"] == 5 and rows[0]["loss"] == 1.0
