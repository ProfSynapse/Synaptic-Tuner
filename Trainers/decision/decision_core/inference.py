"""
Answer a Jev-shaped request with a trained decision model.

Location: Trainers/decision/decision_core/inference.py
Used by:  train_decision.py (post-train smoke), notebooks, future serving.

    model = DecisionModel.load("decision_output/<run>/final_model", device="cuda")
    decide(model, "Help! My payouts have been failing for 3 days.", {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "...", "technical": "..."}},
    })

Returns ``{name: {"type": ..., ...}}`` with ``noul`` (P(true)), ``choice``
(best option + probabilities) or ``score`` (expected level + probabilities),
using the calibrated per-kind temperatures saved with the checkpoint.
"""

from __future__ import annotations

from typing import Any

from shared.ml.calibration_metrics import softmax

from .evaluate import predict
from .examples import examples_from_jev_row


def decide(model: Any, state: Any, questions: dict[str, dict[str, Any]], *,
           batch_size: int = 8) -> dict[str, dict[str, Any]]:
    cfg = model.decision_config
    names = list(questions)
    # Placeholder gold answers so the rows validate; they are never read.
    placeholder = {}
    for name, q in questions.items():
        kind = str(q.get("type", "")).lower()
        if kind == "noul":
            placeholder[name] = False
        elif kind == "score":
            placeholder[name] = 0
        else:
            criteria = q.get("criteria") or {}
            placeholder[name] = next(iter(criteria))
    examples = examples_from_jev_row({"state": state, "questions": questions, "answers": placeholder})
    pred = predict(model, examples, max_length=cfg.max_length, batch_size=batch_size)

    answers: dict[str, dict[str, Any]] = {}
    for name, ex, logits in zip(names, examples, pred.logits):
        probs = softmax(logits, cfg.temperature_by_kind.get(ex.kind, 1.0))
        if ex.kind == "noul":
            answers[name] = {"type": "noul", "noul": float(probs[1])}
        elif ex.kind == "choice":
            dist = {opt[0]: float(p) for opt, p in zip(ex.options, probs)}
            answers[name] = {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist}
        else:
            dist = {str(i): float(p) for i, p in enumerate(probs)}
            answers[name] = {
                "type": "score",
                "score": float(sum(i * p for i, p in enumerate(probs))),
                "probabilities": dist,
            }
    return answers
