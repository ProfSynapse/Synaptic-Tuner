"""
Decision loss: soft cross-entropy + Brier, plus an optional frozen-readout KL.

Location: Trainers/decision/decision_core/losses.py
Used by:  trainer.py.

- Cross-entropy against the (ordinal-smoothed) target distribution.
- ``brier_weight`` * multi-class Brier against the same target. Open-Jev trains
  with 0.1; it sharpens calibration at almost no accuracy cost.
- ``kl_frozen_weight`` * KL(frozen || student), where "frozen" is the base
  torso's own option-marker readout with the adapter disabled. strands-decider
  uses 0.3 so the trained readout cannot drift away from what the pretrained LM
  already knows. Only rows whose option count fits the single-token markers are
  eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .readouts import masked_log_softmax


@dataclass
class LossConfig:
    brier_weight: float = 0.0
    kl_frozen_weight: float = 0.0


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


def decision_loss(
    logits: torch.Tensor,
    batch: dict[str, Any],
    config: LossConfig,
    frozen_logits_fn: Any = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (loss, parts) for masked logits [B, K] and a collated batch."""
    n_options = batch["n_options"]
    target = batch["target"].to(logits.device)
    weights = batch["weights"].to(logits.device)

    log_probs = masked_log_softmax(logits, n_options)
    ce = -(target * log_probs).sum(dim=-1)
    loss = weighted_mean(ce, weights)
    parts = {"ce": float(loss.detach())}

    if config.brier_weight > 0:
        brier = ((log_probs.exp() - target) ** 2).sum(dim=-1)
        brier_mean = weighted_mean(brier, weights)
        loss = loss + config.brier_weight * brier_mean
        parts["brier"] = float(brier_mean.detach())

    if config.kl_frozen_weight > 0 and frozen_logits_fn is not None:
        kl = frozen_kl(log_probs, batch, frozen_logits_fn)
        if kl is not None:
            loss = loss + config.kl_frozen_weight * kl
            parts["kl_frozen"] = float(kl.detach())

    return loss, parts


def frozen_kl(
    student_log_probs: torch.Tensor, batch: dict[str, Any], frozen_logits_fn: Any
) -> torch.Tensor | None:
    """Mean KL(frozen || student) over rows the frozen marker readout can score."""
    n_options = batch["n_options"]
    max_markers = int(frozen_logits_fn.max_options)
    eligible = n_options <= max_markers
    if not bool(eligible.any()):
        return None
    rows = eligible.nonzero(as_tuple=True)[0]
    width = int(n_options[rows].max())
    ref_logits = frozen_logits_fn(
        input_ids=batch["input_ids"][rows],
        attention_mask=batch["attention_mask"][rows],
        option_index=batch["option_index"][rows, :width],
        answer_index=batch["answer_index"][rows],
        n_options=n_options[rows],
    )
    ref_log_probs = masked_log_softmax(ref_logits, n_options[rows])
    student = student_log_probs[rows, :width]
    kl = F.kl_div(student, ref_log_probs, log_target=True, reduction="none").sum(dim=-1)
    return kl.mean()
