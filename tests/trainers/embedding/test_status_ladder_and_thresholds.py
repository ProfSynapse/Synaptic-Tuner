"""Status-ladder branch (R4) + threshold gating.

CONTEXT: ``Evaluator/runner.py`` adds an optional ``EvaluationRecord.retrieval``
field and a status branch placed BEFORE the correctness branch so a retrieval
scenario is scored on its own continuous-metric ladder (CONTRACTS §4.3). The
``RetrievalValidationResult.passed/warned`` flags come from
``retrieval_verifier._apply_thresholds`` (the R4 ``min`` + ``warn_margin``
ladder). This suite exercises:

1. EvaluationRecord.status across the pass/warn/fail boundary, with
   error-precedence FIRST and the retrieval branch BEFORE correctness.
2. _apply_thresholds: pass/warn/fail across the threshold boundary, including
   the "failing result is never reported as a passing warning" invariant.

These are the load-bearing decision branches: a regression that mis-orders the
ladder (e.g. lets a retrieval scenario fall through to correctness, or reports a
hard-fail as a warn) would silently mis-grade eval runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Evaluator.runner import EvaluationRecord  # noqa: E402
from shared.verifiers.builtins.retrieval_verifier import (  # noqa: E402
    RetrievalThresholds,
    RetrievalValidationResult,
    _apply_thresholds,
)


def _retrieval_result(*, passed: bool, warned: bool) -> RetrievalValidationResult:
    return RetrievalValidationResult(
        metrics={"ndcg@10": 0.5},
        passed=passed,
        warned=warned,
        primary_metric_name="ndcg@10",
        primary_metric=0.5,
    )


def _record(**kwargs) -> EvaluationRecord:
    """Build an EvaluationRecord with only the fields a status test needs."""
    base = dict(
        case=None,
        response_text=None,
        validator=None,
        latency_s=None,
    )
    base.update(kwargs)
    return EvaluationRecord(**base)


# ---------------------------------------------------------------------------
# EvaluationRecord.status — retrieval ladder
# ---------------------------------------------------------------------------

def test_status_retrieval_pass():
    rec = _record(retrieval=_retrieval_result(passed=True, warned=False))
    assert rec.status == "pass"


def test_status_retrieval_warn():
    rec = _record(retrieval=_retrieval_result(passed=True, warned=True))
    assert rec.status == "warn"


def test_status_retrieval_fail():
    rec = _record(retrieval=_retrieval_result(passed=False, warned=False))
    assert rec.status == "fail"


def test_status_error_precedence_over_retrieval():
    """error is checked FIRST: even a 'passed' retrieval result yields fail."""
    rec = _record(
        error="boom",
        retrieval=_retrieval_result(passed=True, warned=False),
    )
    assert rec.status == "fail"


def test_status_retrieval_branch_short_circuits_correctness():
    """When retrieval is set, the correctness ladder is NOT consulted.

    We attach a 'failing' correctness sentinel; if the retrieval branch did not
    short-circuit, status would read fail. It must read pass (retrieval passed),
    proving retrieval is evaluated on its own ladder BEFORE correctness.
    """

    class _FailingCorrectness:
        passed = False

    rec = _record(
        retrieval=_retrieval_result(passed=True, warned=False),
        correctness=_FailingCorrectness(),
    )
    assert rec.status == "pass"


# ---------------------------------------------------------------------------
# _apply_thresholds — R4 pass/warn/fail across the boundary
# ---------------------------------------------------------------------------

def test_thresholds_pass_above_min():
    metrics = {"ndcg@10": 0.50, "recall@10": 0.60}
    thr = RetrievalThresholds(min={"ndcg@10": 0.30, "recall@10": 0.40}, warn_margin=0.05)
    passed, warned, gating = _apply_thresholds(metrics, thr)
    assert passed is True
    assert warned is False
    assert gating["ndcg@10"]["ok"] is True


def test_thresholds_fail_below_min():
    metrics = {"ndcg@10": 0.25}  # below min 0.30
    thr = RetrievalThresholds(min={"ndcg@10": 0.30}, warn_margin=0.05)
    passed, warned, _ = _apply_thresholds(metrics, thr)
    assert passed is False
    assert warned is False  # a failing result is never a passing warning


def test_thresholds_warn_in_band():
    # value 0.32 is in [0.30, 0.35) -> passes but warns.
    metrics = {"ndcg@10": 0.32}
    thr = RetrievalThresholds(min={"ndcg@10": 0.30}, warn_margin=0.05)
    passed, warned, gating = _apply_thresholds(metrics, thr)
    assert passed is True
    assert warned is True
    assert gating["ndcg@10"]["warn"] is True


def test_thresholds_exact_min_boundary_passes():
    # value == min is OK (>=), and within the warn band -> warn.
    metrics = {"ndcg@10": 0.30}
    thr = RetrievalThresholds(min={"ndcg@10": 0.30}, warn_margin=0.05)
    passed, warned, _ = _apply_thresholds(metrics, thr)
    assert passed is True
    assert warned is True


def test_thresholds_just_above_warn_band_no_warn():
    # value 0.35 == min+margin -> NOT < min+margin -> passes WITHOUT warn.
    metrics = {"ndcg@10": 0.35}
    thr = RetrievalThresholds(min={"ndcg@10": 0.30}, warn_margin=0.05)
    passed, warned, _ = _apply_thresholds(metrics, thr)
    assert passed is True
    assert warned is False


def test_thresholds_one_fail_one_warn_overall_fail():
    """If ANY gated metric fails, the overall result fails and warn is cleared,
    even if another metric was in the warn band."""
    metrics = {"ndcg@10": 0.20, "recall@10": 0.31}  # ndcg fails; recall would warn
    thr = RetrievalThresholds(
        min={"ndcg@10": 0.30, "recall@10": 0.30}, warn_margin=0.05
    )
    passed, warned, _ = _apply_thresholds(metrics, thr)
    assert passed is False
    assert warned is False


def test_thresholds_metric_without_min_does_not_gate():
    """A metric absent from `min` is reported but never gates."""
    metrics = {"ndcg@10": 0.50, "mrr@10": 0.01}  # mrr has no min
    thr = RetrievalThresholds(min={"ndcg@10": 0.30}, warn_margin=0.0)
    passed, warned, gating = _apply_thresholds(metrics, thr)
    assert passed is True
    assert "mrr@10" not in gating  # only gated metrics appear


def test_thresholds_missing_metric_treated_as_zero_fails():
    """A gated metric absent from `metrics` is treated as 0.0 -> fails."""
    metrics: dict[str, float] = {}
    thr = RetrievalThresholds(min={"ndcg@10": 0.30}, warn_margin=0.05)
    passed, _, gating = _apply_thresholds(metrics, thr)
    assert passed is False
    assert gating["ndcg@10"]["value"] == 0.0
