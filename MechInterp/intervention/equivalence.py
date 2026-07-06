"""
Batched-vs-unbatched equivalence self-check for the intervention engine.

When an intervention is applied inside a padded batch, the edited hidden state
for each row should match the edit computed for that row on its own (unpadded).
Any divergence should sit at the model's own batched-vs-unbatched numeric floor
(a few parts in the low thousandths in bf16), which is orders of magnitude below
any useful intervention magnitude. This check computes that divergence so a
caller can confirm the batched path did not introduce a position or masking bug.

The comparison is a relative one: the maximum absolute divergence is reported
alongside a tolerance derived from the compared magnitudes, so the same check
works whether the hidden states are large or small. Pass/fail is left to the
caller, which typically requires the divergence to be far below the intervention
strength.
"""

from __future__ import annotations

import torch


def max_abs_divergence(batched_final: torch.Tensor, unbatched_final: torch.Tensor) -> dict:
    """Compare per-row final-token hidden states from batched vs unbatched runs.

    Both inputs are (n_rows, hidden_dim). Returns the per-row max-absolute
    divergence and the overall maximum, computed in float32.
    """
    if batched_final.shape != unbatched_final.shape:
        raise ValueError("batched and unbatched tensors must have the same shape")
    a = batched_final.detach().float()
    b = unbatched_final.detach().float()
    diff = (a - b).abs()
    per_row = diff.amax(dim=1)
    return {
        "max_abs": float(diff.max()),
        "per_row_max_abs": per_row.tolist(),
        "mean_abs": float(diff.mean()),
    }


def relative_tolerance(
    reference: torch.Tensor, rel: float = 1e-2, floor: float = 1e-3
) -> float:
    """Return a tolerance scaled to the reference magnitude, with a floor.

    The tolerance is rel times the reference's max absolute value, but never
    below floor, so a near-zero reference still admits the numeric floor.
    """
    scale = float(reference.detach().float().abs().max())
    return max(rel * scale, floor)


def equivalence_ok(
    batched_final: torch.Tensor,
    unbatched_final: torch.Tensor,
    rel: float = 1e-2,
    floor: float = 1e-3,
) -> dict:
    """Judge batched-vs-unbatched equivalence against a relative tolerance."""
    div = max_abs_divergence(batched_final, unbatched_final)
    tol = relative_tolerance(unbatched_final, rel=rel, floor=floor)
    return {
        **div,
        "tolerance": tol,
        "passed": div["max_abs"] <= tol,
    }
