"""
Gate primitives for intervention-cell adjudication.

All randomness is seeded explicitly so a gate verdict is reproducible from its
recorded seed. The primitives operate on per-row outcome arrays so they are
agnostic to the project's notion of what a "flip" or a "kill" means; the caller
supplies boolean/integer indicators.

  count_flips(before, after, target)
      Count rows whose outcome moved from one state to another. Generic enough
      to express "monitored predicate cleared" (True -> False on a target
      predicate) or "collateral" (a desirable state that flipped).

  kill_diff_vs_control(primary_ind, control_ind, seed, n_boot)
      Difference in per-row positive-indicator counts between a primary arm and
      a count-matched control, with a seeded row-bootstrap confidence interval.

  permutation_p(primary_ind, pool_ind, n_primary, seed, n_perm)
      One-sided permutation p-value: how often a random count-matched draw from
      the pool matches or beats the primary arm's positive count.

  auroc_floor(labels, scores, seed, n_boot)
      Point AUROC plus a Hanley-McNeil analytic standard error and a seeded
      bootstrap lower confidence bound.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _as_bool_array(x: Sequence) -> np.ndarray:
    return np.asarray(x, dtype=bool)


def count_flips(
    before: Sequence,
    after: Sequence,
    from_state: bool = True,
    to_state: bool = False,
) -> int:
    """Count rows that moved from from_state to to_state.

    before / after are aligned per-row boolean indicators of some predicate.
    The default (True -> False) counts rows where the predicate held before the
    intervention and no longer holds after it.
    """
    b = _as_bool_array(before)
    a = _as_bool_array(after)
    if b.shape != a.shape:
        raise ValueError("before and after must be the same length")
    moved = (b == from_state) & (a == to_state)
    return int(moved.sum())


def kill_diff_vs_control(
    primary_ind: Sequence,
    control_ind: Sequence,
    seed: int,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Positive-count difference between primary and control, with bootstrap CI.

    primary_ind and control_ind are aligned per-row 0/1 indicators over the same
    universe of rows (for example, 1 if the row was a positive event in that arm,
    else 0). The point estimate is sum(primary) - sum(control). The CI resamples
    row indices with replacement n_boot times and reports the diff-of-sums
    percentile interval.

    Passing convention is left to the caller (typically diff >= floor AND the CI
    lower bound excludes zero).
    """
    p = np.asarray(primary_ind, dtype=float)
    c = np.asarray(control_ind, dtype=float)
    if p.shape != c.shape:
        raise ValueError("primary_ind and control_ind must be the same length")
    n = p.shape[0]
    diff = float(p.sum() - c.sum())
    per_row = p - c
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = per_row[idx].sum()
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    ci_lo = float(np.quantile(boots, lo_q))
    ci_hi = float(np.quantile(boots, hi_q))
    return {
        "diff": diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_level": ci,
        "n_boot": n_boot,
        "seed": seed,
        "n_rows": n,
    }


def permutation_p(
    primary_positive: int,
    pool_ind: Sequence,
    n_primary: int,
    seed: int,
    n_perm: int = 1000,
) -> dict:
    """One-sided permutation p-value for a count-matched positive count.

    Draw n_primary rows without replacement from the pool n_perm times and count
    how often the drawn positive count is >= the observed primary positive count.
    The p-value is (hits + 1) / (n_perm + 1) (add-one smoothing so it is never 0).
    """
    pool = _as_bool_array(pool_ind)
    if n_primary > pool.shape[0]:
        raise ValueError("n_primary exceeds pool size")
    rng = np.random.default_rng(seed)
    idx_all = np.arange(pool.shape[0])
    hits = 0
    null_counts = np.empty(n_perm, dtype=int)
    for k in range(n_perm):
        draw = rng.choice(idx_all, size=n_primary, replace=False)
        cnt = int(pool[draw].sum())
        null_counts[k] = cnt
        if cnt >= primary_positive:
            hits += 1
    p_value = (hits + 1) / (n_perm + 1)
    return {
        "primary_positive": int(primary_positive),
        "null_mean": float(null_counts.mean()),
        "null_max": int(null_counts.max()),
        "p_value": float(p_value),
        "n_perm": n_perm,
        "seed": seed,
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC with tie handling (Mann-Whitney U / (n_pos*n_neg))."""
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s_sorted = scores[order]
    # average ranks for ties
    i = 0
    r = 1
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        r += (j - i) + 1
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil analytic standard error of an AUROC estimate."""
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    num = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    )
    return float(np.sqrt(num / (n_pos * n_neg)))


def auroc_floor(
    labels: Sequence,
    scores: Sequence,
    seed: int,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> dict:
    """AUROC point estimate with Hanley-McNeil SE and a seeded bootstrap CI-LB.

    labels are 0/1; scores are real-valued (higher = more positive). Returns the
    point AUROC, the analytic SE, and the bootstrap lower confidence bound (the
    "floor"), which the caller compares against a threshold.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("labels and scores must be the same length")
    auc = _roc_auc(y, s)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    boots = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if (yb == 1).sum() == 0 or (yb == 0).sum() == 0:
            continue
        boots.append(_roc_auc(yb, sb))
    lo_q = (1.0 - ci) / 2.0
    ci_lb = float(np.quantile(boots, lo_q)) if boots else float("nan")
    return {
        "auroc": auc,
        "hanley_mcneil_se": se,
        "ci_lb": ci_lb,
        "ci_level": ci,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_boot": n_boot,
        "n_boot_used": len(boots),
        "seed": seed,
    }
