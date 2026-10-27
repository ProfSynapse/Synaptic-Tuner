"""No ML imports; actual nonblocking numeric pipe callbacks and early budgets."""

from dataclasses import replace
import json
import os
from types import SimpleNamespace

import pytest

from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
from tuner.runtime.native_lifecycle_protocol import NativeLifecyclePolicyV1
from tuner.runtime.native_worker_progress import NativeWorkerProgress, NativeSFTProgressCallback


@pytest.fixture
def progress():
    read, write = os.pipe()
    launch = NativeLifecycleLaunchV1(NativeLifecyclePolicyV1(32, 65536), "a" * 64)
    emitter = NativeWorkerProgress(write, launch)
    try:
        yield read, emitter
    finally:
        emitter.close()
        os.close(read)


def projection(method="sft"):
    return SimpleNamespace(workload=SimpleNamespace(method=method), controls=SimpleNamespace(
        iterations=3, steps_per_report=1, steps_per_eval=1, logging_steps=1))


def test_sft_callbacks_emit_exact_numeric_metrics_and_distinct_reload(progress):
    read, emitter = progress
    emitter.bind(projection())
    callback = NativeSFTProgressCallback(emitter.emit)
    emitter.emit("preparation", 0, {})
    callback.on_val_loss_report({"iteration": 0, "val_loss": 3.5, "private_extra": "ignored"})
    callback.on_train_loss_report({"iteration": 3, "train_loss": 2.25, "learning_rate": .001,
                                   "trained_tokens": 17, "private_extra": "ignored"})
    emitter.emit("reload", 3, {"loss": 1.75})
    emitter.emit("publication", 3, {})
    values = [json.loads(raw) for raw in os.read(read, 4096).splitlines()]
    assert [value["phase"] for value in values] == ["preparation", "validation", "training", "reload", "publication"]
    assert values[2]["metrics"] == {"loss": 2.25, "learning_rate": .001, "tokens": 17}
    assert "private" not in str(values)
    assert os.get_inheritable(emitter._fd) is False


@pytest.mark.parametrize("method", ["sft", "grpo"])
def test_insufficient_declared_budget_rejects_before_any_progress(progress, method):
    read, emitter = progress
    emitter._launch = replace(emitter._launch, policy=NativeLifecyclePolicyV1(1, 8192))
    with pytest.raises(ValueError, match="cannot cover"):
        emitter.bind(projection(method))
    os.set_blocking(read, False)
    with pytest.raises(BlockingIOError): os.read(read, 4096)


@pytest.mark.parametrize("phase,step,metrics", [
    ("private text", 1, {}), ("training", 4, {}), ("training", True, {}),
    ("training", 1, {"tokens": 1.0}), ("training", 1, {"loss": float("nan")}),
    ("training", 1, {"path": "private"}),
])
def test_malformed_or_out_of_range_callback_never_enters_pipe(progress, phase, step, metrics):
    read, emitter = progress
    emitter.bind(projection())
    with pytest.raises(ValueError): emitter.emit(phase, step, metrics)
    os.set_blocking(read, False)
    with pytest.raises(BlockingIOError): os.read(read, 4096)


def test_progress_cannot_move_backwards_or_rebind(progress):
    _, emitter = progress
    emitter.bind(projection())
    emitter.emit("training", 2, {"loss": 2.5})
    with pytest.raises(ValueError, match="update"):
        emitter.emit("training", 1, {})
    with pytest.raises(ValueError, match="already bound"):
        emitter.bind(projection())


def test_regular_file_or_readable_pipe_is_not_a_progress_writer(tmp_path):
    launch = NativeLifecycleLaunchV1(NativeLifecyclePolicyV1(32, 65536), "a" * 64)
    with open(tmp_path / "file", "wb") as regular:
        with pytest.raises(ValueError): NativeWorkerProgress(regular.fileno(), launch)
    read, write = os.pipe()
    try:
        with pytest.raises(ValueError): NativeWorkerProgress(read, launch)
    finally:
        os.close(read);os.close(write)


def test_dead_guardian_poisoned_delivery_does_not_retry(progress):
    read, emitter = progress
    emitter.bind(projection())
    os.close(read)
    try:
        with pytest.raises(BrokenPipeError): emitter.emit("preparation", 0, {})
        with pytest.raises(ValueError, match="unavailable"): emitter.emit("preparation", 0, {})
    finally:
        # Keep fixture ownership consistent without exposing another descriptor.
        replacement = os.open(os.devnull, os.O_RDONLY)
        if replacement != read:
            os.dup2(replacement, read)
            os.close(replacement)
