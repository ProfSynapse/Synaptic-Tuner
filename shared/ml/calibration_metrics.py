"""
shared/ml/calibration_metrics.py

Pure-function metrics for calibrated categorical predictions: accuracy, NLL,
multi-class Brier score, expected calibration error (ECE), and post-hoc
temperature fitting. Inputs are per-row probability vectors (or logits for
temperature fitting) whose length may differ row to row -- a decision model
answers a 2-option yes/no and a 24-option choice in the same batch.

Sibling to ``shared/ml/retrieval_metrics.py``: numpy only, imports nothing from
``Evaluator/`` or ``Trainers/``.

Also: rank AUROC, AURC, the confident-wrong / underconfident-right pair, and
split-conformal LAC prediction sets + shortest ordinal intervals -- the
"confidence interval" tools for categorical decisions.

Used by:
- ``Trainers/decision/decision_core/evaluate.py`` (calibrate + evaluate stages).
- ``Trainers/decision/decision_core/confidence_analysis.py`` (confidence analysis).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

_EPS = 1e-12


def _as_rows(rows: Iterable[Sequence[float]]) -> list[np.ndarray]:
    out = [np.asarray(r, dtype=np.float64) for r in rows]
    for r in out:
        if r.ndim != 1 or r.size == 0:
            raise ValueError("each row must be a non-empty 1-D probability vector")
    return out


def softmax(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax of one row, after dividing by ``temperature``."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def accuracy(probs: Iterable[Sequence[float]], labels: Sequence[int]) -> float:
    rows = _as_rows(probs)
    if not rows:
        return float("nan")
    return float(np.mean([int(np.argmax(p)) == int(y) for p, y in zip(rows, labels)]))


def nll(probs: Iterable[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean negative log-likelihood of the gold label."""
    rows = _as_rows(probs)
    if not rows:
        return float("nan")
    return float(np.mean([-math.log(max(float(p[int(y)]), _EPS)) for p, y in zip(rows, labels)]))


def brier(probs: Iterable[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean multi-class Brier score: sum_k (p_k - onehot_k)^2, averaged over rows.

    Ranges over [0, 2]. For a two-option row this is twice the binary Brier score
    of P(gold).
    """
    rows = _as_rows(probs)
    if not rows:
        return float("nan")
    total = 0.0
    for p, y in zip(rows, labels):
        target = np.zeros_like(p)
        target[int(y)] = 1.0
        total += float(np.sum((p - target) ** 2))
    return total / len(rows)


def expected_calibration_error(
    probs: Iterable[Sequence[float]], labels: Sequence[int], n_bins: int = 15
) -> float:
    """Top-label ECE with equal-width confidence bins over [0, 1]."""
    rows = _as_rows(probs)
    if not rows:
        return float("nan")
    conf = np.array([float(p.max()) for p in rows])
    correct = np.array([int(np.argmax(p)) == int(y) for p, y in zip(rows, labels)], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin i holds (edges[i], edges[i+1]]; confidence 0 cannot occur for a softmax row.
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    accuracy: float
    nll: float
    brier: float
    ece: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "nll": self.nll,
            "brier": self.brier,
            "ece": self.ece,
        }


def calibration_report(
    probs: Iterable[Sequence[float]], labels: Sequence[int], n_bins: int = 15
) -> CalibrationReport:
    rows = _as_rows(probs)
    labels = list(labels)
    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} probability rows but {len(labels)} labels")
    return CalibrationReport(
        n=len(rows),
        accuracy=accuracy(rows, labels),
        nll=nll(rows, labels),
        brier=brier(rows, labels),
        ece=expected_calibration_error(rows, labels, n_bins=n_bins),
    )


def fit_temperature(
    logits: Iterable[Sequence[float]],
    labels: Sequence[int],
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    iters: int = 80,
) -> float:
    """Temperature T minimising the NLL of softmax(logits / T) on held-out rows.

    NLL is unimodal in log T for a fixed set of logits, so a golden-section search
    over log T is exact to the tolerance and needs no autograd. Returns 1.0 when
    there are no rows.
    """
    rows = _as_rows(logits)
    labels = [int(y) for y in labels]
    if not rows:
        return 1.0
    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} logit rows but {len(labels)} labels")

    def objective(log_t: float) -> float:
        t = math.exp(log_t)
        return nll([softmax(r, t) for r in rows], labels)

    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = objective(d)
    return float(math.exp((a + b) / 2.0))


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUROC (ties count half); NaN when one class is absent."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def aurc(confidence: Sequence[float], correct: Sequence[int]) -> float:
    """Area under the risk-coverage curve (lower is better).

    Rows are ranked by confidence; risk at coverage k/n is the error rate of the
    k most confident rows. A diagnostic of confidence ranking -- not a licence to
    abstain.
    """
    c = np.asarray(confidence, dtype=np.float64)
    err = 1.0 - np.asarray(correct, dtype=np.float64)
    if c.size == 0:
        return float("nan")
    order = np.argsort(-c, kind="mergesort")
    cum_err = np.cumsum(err[order])
    risks = cum_err / np.arange(1, c.size + 1)
    return float(risks.mean())


def confident_wrong_rate(confidence: Sequence[float], correct: Sequence[int], tau: float = 0.8) -> float:
    """Share of all rows that are confident-wrong: answered wrongly with confidence >= tau."""
    c = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    return float(np.mean((c >= tau) & ~ok)) if c.size else float("nan")


def underconfident_right_rate(confidence: Sequence[float], correct: Sequence[int], tau: float = 0.5) -> float:
    """Share of all rows that are underconfident-right: answered correctly with confidence < tau."""
    c = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    return float(np.mean((c < tau) & ok)) if c.size else float("nan")


def conformal_lac_qhat(true_label_probs: Sequence[float], alpha: float) -> float:
    """Split-conformal threshold for LAC sets from calibration rows.

    Nonconformity is 1 - p(true label). Sets {k : 1 - p_k <= qhat} then cover the
    true label with probability >= 1 - alpha on exchangeable test rows.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = 1.0 - np.asarray(true_label_probs, dtype=np.float64)
    n = scores.size
    if n == 0:
        return 1.0
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def lac_set(probs: Sequence[float], qhat: float, *, include_argmax: bool = True) -> list[int]:
    """Option indices in the LAC prediction set.

    ``include_argmax`` keeps the chosen answer in the set so a set is never empty:
    the model always commits to a choice (slightly conservative coverage).
    """
    p = np.asarray(probs, dtype=np.float64)
    members = [int(k) for k in np.flatnonzero(1.0 - p <= qhat)]
    best = int(np.argmax(p))
    if include_argmax and best not in members:
        members.append(best)
    return sorted(members)


def shortest_ordinal_interval(probs: Sequence[float], mass: float) -> tuple[int, int]:
    """Shortest contiguous level range [lo, hi] holding >= mass of an ordinal distribution.

    Ties in width break toward more mass. Levels are in canonical (ascending) order.
    """
    p = np.asarray(probs, dtype=np.float64)
    n = p.size
    best: tuple[int, int] | None = None
    best_mass = -1.0
    for width in range(1, n + 1):
        for lo in range(0, n - width + 1):
            m = float(p[lo:lo + width].sum())
            if m >= mass - 1e-12 and m > best_mass:
                best, best_mass = (lo, lo + width - 1), m
        if best is not None:
            return best
    return (0, n - 1)


def grouped_reports(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[str],
    n_bins: int = 15,
) -> Mapping[str, dict[str, float]]:
    """One :func:`calibration_report` per distinct group key (task, kind, ...)."""
    by: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(str(g), []).append(i)
    return {
        g: calibration_report([probs[i] for i in idx], [labels[i] for i in idx], n_bins).as_dict()
        for g, idx in sorted(by.items())
    }
