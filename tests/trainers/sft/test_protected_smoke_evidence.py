from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from Trainers.sft.src.protected_smoke_evidence import (  # noqa: E402
    ProtectedOptimizerBoundaryCallback, ProtectedSmokeEvidenceError,
    capture_trainable_snapshot, compare_trainable_snapshot,
)


class TinyModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def named_parameters(self):
        return [("adapter.weight", self.weight)]


def test_requires_finite_nonzero_trainable_delta() -> None:
    model = TinyModel()
    before = capture_trainable_snapshot(model)
    with pytest.raises(ProtectedSmokeEvidenceError, match="nonzero"):
        compare_trainable_snapshot(before, model)
    with torch.no_grad():
        model.weight.add_(0.25)
    result = compare_trainable_snapshot(before, model)
    assert result["delta_l2"] > 0
    assert result["changed_tensor_count"] == 1


def test_callback_counts_real_optimizer_boundaries_and_one_finite_loss() -> None:
    callback = ProtectedOptimizerBoundaryCallback()
    control = object()
    assert callback.on_optimizer_step(None, SimpleNamespace(global_step=0), control) is control
    callback.on_log(None, SimpleNamespace(global_step=1), control, logs={"loss": 1.25})
    callback.on_log(None, SimpleNamespace(global_step=2), control, logs={"loss": 1.0})
    assert callback.optimizer_boundaries == 1
    assert callback.step_one_losses == [1.25]
