"""CPU tests for the batched-vs-unbatched equivalence self-check."""

import pytest
import torch

from MechInterp.intervention.equivalence import (
    max_abs_divergence,
    relative_tolerance,
    equivalence_ok,
)


def test_max_abs_divergence_zero_for_identical():
    a = torch.randn(3, 8)
    div = max_abs_divergence(a, a.clone())
    assert div["max_abs"] == 0.0
    assert div["per_row_max_abs"] == [0.0, 0.0, 0.0]


def test_max_abs_divergence_shape_mismatch():
    with pytest.raises(ValueError):
        max_abs_divergence(torch.randn(2, 4), torch.randn(3, 4))


def test_relative_tolerance_scales_and_floors():
    big = torch.tensor([[100.0, -50.0]])
    small = torch.tensor([[0.0, 0.0]])
    assert relative_tolerance(big, rel=0.01) == pytest.approx(1.0)
    assert relative_tolerance(small, rel=0.01, floor=1e-3) == pytest.approx(1e-3)


def test_equivalence_ok_passes_at_numeric_floor():
    ref = torch.randn(4, 16) * 10.0
    perturbed = ref + torch.randn(4, 16) * 1e-4  # tiny numeric noise
    res = equivalence_ok(ref + 0, perturbed, rel=1e-2, floor=1e-3)
    assert res["passed"]


def test_equivalence_ok_fails_on_real_divergence():
    ref = torch.zeros(2, 4)
    diverged = ref.clone()
    diverged[0, 0] = 5.0  # a real edit, far above the floor
    res = equivalence_ok(ref, diverged, rel=1e-2, floor=1e-3)
    assert not res["passed"]
    assert res["max_abs"] == pytest.approx(5.0)
