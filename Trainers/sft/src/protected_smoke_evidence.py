"""Fail-closed semantic evidence for the one-step protected SFT smoke.

The module imports heavyweight ML dependencies lazily so contract tests and
operator tooling can inspect it without importing the training runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "synaptic-protected-sft-evidence/v1"


class ProtectedSmokeEvidenceError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _tensor_bytes(tensor: Any) -> bytes:
    value = tensor.detach().cpu().contiguous()
    try:
        return value.numpy().tobytes(order="C")
    except TypeError:
        # bfloat16 is not supported by every NumPy build. Viewing as bytes keeps
        # the exact storage identity without changing numeric values.
        return value.view(-1).view(getattr(__import__("torch"), "uint8")).numpy().tobytes()


def capture_trainable_snapshot(model: Any) -> dict[str, Any]:
    """Clone the exact trainable tensor set before the protected update."""

    snapshot: dict[str, Any] = {}
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if parameter.requires_grad:
            snapshot[name] = parameter.detach().cpu().clone()
    if not snapshot:
        raise ProtectedSmokeEvidenceError("Protected smoke found no trainable tensors")
    return snapshot


def tensor_set_identity(tensors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        raw = _tensor_bytes(tensor)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def compare_trainable_snapshot(before: Mapping[str, Any], model: Any) -> dict[str, object]:
    after = {
        name: parameter.detach().cpu()
        for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0])
        if parameter.requires_grad
    }
    if set(before) != set(after):
        raise ProtectedSmokeEvidenceError("Trainable tensor set changed during protected training")
    squared_delta = 0.0
    changed = 0
    for name in before:
        delta = (after[name].to(dtype=__import__("torch").float64) - before[name].to(dtype=__import__("torch").float64))
        value = float((delta * delta).sum().item())
        if not math.isfinite(value):
            raise ProtectedSmokeEvidenceError("Protected adapter delta is non-finite")
        squared_delta += value
        if value > 0:
            changed += 1
    delta_l2 = math.sqrt(squared_delta)
    if not math.isfinite(delta_l2) or delta_l2 <= 0 or changed <= 0:
        raise ProtectedSmokeEvidenceError("Protected adapter delta must be finite and nonzero")
    return {
        "pre_step_identity": tensor_set_identity(before),
        "post_step_identity": tensor_set_identity(after),
        "delta_l2": delta_l2,
        "changed_tensor_count": changed,
        "trainable_tensor_count": len(after),
    }


class ProtectedOptimizerBoundaryCallback:
    """Transformers callback-compatible observer for actual optimizer steps."""

    _PASSTHROUGH_EVENTS = frozenset({
        "on_epoch_begin", "on_epoch_end", "on_evaluate", "on_init_end",
        "on_pre_optimizer_step", "on_predict", "on_prediction_step", "on_save",
        "on_step_begin", "on_step_end", "on_substep_end", "on_train_begin",
        "on_train_end",
    })

    def __init__(self) -> None:
        self.optimizer_boundaries = 0
        self.step_one_losses: list[float] = []

    @staticmethod
    def _passthrough(args, state, control, **kwargs):  # noqa: ANN001
        return control

    def __getattr__(self, name: str):
        if name in self._PASSTHROUGH_EVENTS:
            return self._passthrough
        raise AttributeError(name)

    def on_optimizer_step(self, args, state, control, **kwargs):  # noqa: ANN001
        self.optimizer_boundaries += 1
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if int(getattr(state, "global_step", -1)) == 1 and isinstance(logs, Mapping) and "loss" in logs:
            loss = float(logs["loss"])
            if math.isfinite(loss):
                self.step_one_losses.append(loss)
        return control


def _read_json_regular(path: Path, *, maximum: int = 1024 * 1024) -> object:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ProtectedSmokeEvidenceError("Protected evidence input is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        content = os.read(descriptor, maximum + 1)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(content) > maximum or not stat.S_ISREG(opened.st_mode):
        raise ProtectedSmokeEvidenceError("Protected evidence input exceeds its bound")
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtectedSmokeEvidenceError("Protected evidence JSON is invalid") from exc


def load_optimizer_state_weights_only(path: Path) -> object:
    """The sole permitted optimizer pickle load, confined to the pinned job."""

    import torch

    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > 512 * 1024 * 1024:
        raise ProtectedSmokeEvidenceError("Optimizer state is not a bounded regular file")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ProtectedSmokeEvidenceError("Runtime lacks required weights_only optimizer loading") from exc


def _optimizer_steps(state: object) -> list[int]:
    if not isinstance(state, Mapping) or not isinstance(state.get("state"), Mapping):
        raise ProtectedSmokeEvidenceError("Optimizer state has an invalid shape")
    steps: list[int] = []
    for entry in state["state"].values():
        if not isinstance(entry, Mapping) or "step" not in entry:
            continue
        step = entry["step"]
        if hasattr(step, "item"):
            step = step.item()
        if type(step) not in {int, float} or int(step) != step:
            raise ProtectedSmokeEvidenceError("Optimizer step counter is invalid")
        steps.append(int(step))
    if not steps or any(step != 1 for step in steps):
        raise ProtectedSmokeEvidenceError("Optimizer counters do not prove exactly one update")
    return steps


def finalize_protected_evidence(
    *, model: Any, trainer: Any, callback: ProtectedOptimizerBoundaryCallback,
    before: Mapping[str, Any], checkpoint_dir: Path, output_path: Path,
) -> dict[str, object]:
    """Validate semantic evidence and persist one canonical JSON record."""

    if callback.optimizer_boundaries != 1:
        raise ProtectedSmokeEvidenceError("Protected smoke requires exactly one optimizer boundary")
    global_step = int(getattr(trainer.state, "global_step", -1))
    if global_step != 1:
        raise ProtectedSmokeEvidenceError("Protected smoke requires trainer global_step == 1")
    if len(callback.step_one_losses) != 1 or not math.isfinite(callback.step_one_losses[0]):
        raise ProtectedSmokeEvidenceError("Protected smoke requires one finite step-one loss")
    trainer_state = _read_json_regular(checkpoint_dir / "trainer_state.json")
    if not isinstance(trainer_state, Mapping) or trainer_state.get("global_step") != 1:
        raise ProtectedSmokeEvidenceError("Checkpoint trainer_state does not prove step one")
    optimizer_state = load_optimizer_state_weights_only(checkpoint_dir / "optimizer.pt")
    optimizer_steps = _optimizer_steps(optimizer_state)
    scheduler_state = load_optimizer_state_weights_only(checkpoint_dir / "scheduler.pt")
    if not isinstance(scheduler_state, Mapping) or scheduler_state.get("last_epoch") != 1:
        raise ProtectedSmokeEvidenceError("Scheduler state is inconsistent with one update")
    delta = compare_trainable_snapshot(before, model)
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "global_step": 1,
        "optimizer_boundaries": 1,
        "optimizer_parameter_count": len(optimizer_steps),
        "optimizer_steps": sorted(optimizer_steps),
        "scheduler_last_epoch": int(scheduler_state["last_epoch"]),
        "step_one_loss": callback.step_one_losses[0],
        **delta,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(payload))
    return payload


__all__ = [
    "ProtectedOptimizerBoundaryCallback", "ProtectedSmokeEvidenceError",
    "capture_trainable_snapshot", "compare_trainable_snapshot",
    "finalize_protected_evidence", "load_optimizer_state_weights_only",
    "tensor_set_identity",
]
