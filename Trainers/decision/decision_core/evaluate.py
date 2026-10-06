"""
Prediction, temperature calibration, and evaluation for decision models.

Location: Trainers/decision/decision_core/evaluate.py
Used by:  train_decision.py (calibrate + evaluate stages), inference.py.

All logits are mapped back to the *canonical* option order before metrics, so
a shuffled rendering and a canonical one are directly comparable. Metric math
lives in shared/ml/calibration_metrics.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from shared.ml.calibration_metrics import (
    calibration_report,
    fit_temperature,
    grouped_reports,
    softmax,
)

from .collate import CollatorConfig, DecisionCollator
from .examples import KINDS, DecisionExample


@dataclass
class Predictions:
    logits: list[np.ndarray]  # canonical order, uncalibrated, one row per example
    labels: list[int]
    kinds: list[str]
    tasks: list[str]


def _canonical(slot_logits: np.ndarray, order: Sequence[int]) -> np.ndarray:
    out = np.empty(len(order), dtype=np.float64)
    for slot, canonical in enumerate(order):
        out[canonical] = slot_logits[slot]
    return out


@torch.no_grad()
def predict(
    model: Any,
    examples: list[DecisionExample],
    *,
    max_length: int,
    batch_size: int = 8,
    orders: list[Sequence[int]] | None = None,
    device: torch.device | str | None = None,
) -> Predictions:
    """Uncalibrated canonical-order logits for each example.

    ``orders`` renders each example under a given permutation; ``None`` is canonical.
    """
    cfg = model.decision_config
    collator = DecisionCollator(
        model.tokenizer,
        CollatorConfig(max_length=max_length, marker_style=cfg.marker_style, headers=dict(cfg.headers)),
        train=False,
    )
    device = device or next(model.parameters()).device
    model.eval()
    logits_out: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        chunk_orders = None if orders is None else orders[start:start + batch_size]
        batch = collator(chunk, orders=chunk_orders)
        tensors = {k: batch[k].to(device) for k in
                   ("input_ids", "attention_mask", "option_index", "answer_index", "n_options")}
        logits = model(**tensors).float().cpu().numpy()
        for i, ex in enumerate(chunk):
            logits_out.append(_canonical(logits[i, : ex.n_options], batch["orders"][i]))
    return Predictions(
        logits=logits_out,
        labels=[ex.label for ex in examples],
        kinds=[ex.kind for ex in examples],
        tasks=[ex.task for ex in examples],
    )


def fit_temperatures(pred: Predictions, *, min_rows_per_kind: int = 50) -> dict[str, float]:
    """One NLL-optimal temperature per question kind.

    A single global temperature cannot serve all three: noul, choice and score sit
    at very different accuracies, so a value good for one over-softens another.
    """
    temps: dict[str, float] = {}
    for kind in KINDS:
        idx = [i for i, k in enumerate(pred.kinds) if k == kind]
        if len(idx) < min_rows_per_kind:
            continue
        temps[kind] = fit_temperature([pred.logits[i] for i in idx], [pred.labels[i] for i in idx])
    return temps


def calibrated_probs(pred: Predictions, temps: dict[str, float]) -> list[np.ndarray]:
    return [softmax(row, temps.get(kind, 1.0)) for row, kind in zip(pred.logits, pred.kinds)]


def random_orders(examples: list[DecisionExample], rng: random.Random) -> list[list[int]]:
    """A non-identity option order per example (score rows: reversed)."""
    orders = []
    for ex in examples:
        n = ex.n_options
        identity = list(range(n))
        if ex.kind == "score":
            orders.append(identity[::-1])
            continue
        order = identity[:]
        while n > 1 and order == identity:
            rng.shuffle(order)
        orders.append(order)
    return orders


def evaluate(
    model: Any,
    examples: list[DecisionExample],
    temps: dict[str, float],
    *,
    max_length: int,
    batch_size: int = 8,
    shuffle_trials: int = 2,
    ece_bins: int = 15,
    seed: int = 0,
) -> dict[str, Any]:
    """Accuracy / NLL / Brier / ECE overall, per kind and per task, plus order consistency."""
    base = predict(model, examples, max_length=max_length, batch_size=batch_size)
    raw = [softmax(r) for r in base.logits]
    cal = calibrated_probs(base, temps)
    report: dict[str, Any] = {
        "n_examples": len(examples),
        "temperatures": dict(temps),
        "uncalibrated": calibration_report(raw, base.labels, ece_bins).as_dict(),
        "calibrated": calibration_report(cal, base.labels, ece_bins).as_dict(),
        "by_kind": grouped_reports(cal, base.labels, base.kinds, ece_bins),
        "by_task": grouped_reports(cal, base.labels, base.tasks, ece_bins),
    }

    if shuffle_trials > 0 and examples:
        rng = random.Random(seed)
        canonical_answer = [int(np.argmax(r)) for r in base.logits]
        changed = 0
        total = 0
        for _ in range(shuffle_trials):
            shuffled = predict(model, examples, max_length=max_length, batch_size=batch_size,
                               orders=random_orders(examples, rng))
            for a, row in zip(canonical_answer, shuffled.logits):
                changed += int(a != int(np.argmax(row)))
                total += 1
        report["order_consistency"] = {
            "trials": shuffle_trials,
            "answer_change_rate": changed / total if total else float("nan"),
        }
    return report
