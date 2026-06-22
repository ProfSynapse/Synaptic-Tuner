"""Tests for the WORST-CASE quality-gated selection metric (topic-diversity pressure).

Location: tests/shared/test_quality_gated_min_selection.py
Summary: Regression coverage for ``_quality_gated_min_normalized_score`` — the
         run-level selection metric an optimizer run can select on
         (``objective.metric: stats.quality_gated_min_normalized_score``). Its
         sibling MEAN metric is already covered in test_quality_gate_scoring.py;
         the MIN was shipped (worst-case hardening) with ZERO direct tests.

         The MIN exists for ONE reason: a candidate that aces one case but drifts
         off another must be scored by its WEAKEST case, so a strong case can NOT
         mask a weak one (which the mean allows). These tests lock that property
         in as a permanent guard, plus the None / default-off contract and the
         floor-breach interaction (a hard-eliminated case drives the min to 0.0).

         The engine is generic: this metric knows none of the domain dimension
         names; it operates purely on per-record gated composites. The fixtures
         here use minimal stand-in records mirroring the reporting helper's shape.
"""
from __future__ import annotations

import pytest

from Evaluator.reporting import (
    _quality_gated_min_normalized_score,
    _quality_gated_normalized_score,
)


# --- Minimal stand-in records mirroring the reporting helper's read shape ----
# _judge_case_quality_gated(record) reads record.judge.judge_result.scores[*]
# .quality_gated_score and returns the FIRST non-None gated score for the case
# (or None when the case has no judge / no gated score).


class _Score:
    def __init__(self, gated):
        self.quality_gated_score = gated


class _Judge:
    def __init__(self, gated_scores):
        self.judge_result = type("R", (), {"scores": [_Score(g) for g in gated_scores]})()


class _Record:
    def __init__(self, gated_scores=None):
        # gated_scores=None => a non-judge case (record.judge is None)
        self.judge = _Judge(gated_scores) if gated_scores is not None else None


# --- The load-bearing property: min, not mean -------------------------------


def test_min_is_the_worst_case_not_the_average():
    """A strong case must NOT mask a weak case — the metric is the WORST case.

    This is the entire point of the worst-case hardening: a candidate aced one
    case (~0.92) but drifted on another (~0.05); the 0.85 MEAN hid that. The MIN
    surfaces the drifted case. Same records, mean vs min diverge sharply.
    """
    records = [
        _Record([0.92]),   # strong case
        _Record([0.05]),   # drifted case — the weakness that must dominate
        _Record([0.80]),
    ]
    assert _quality_gated_min_normalized_score(records) == pytest.approx(0.05)
    # And the mean would have MASKED it (sits up in the "passing" band):
    assert _quality_gated_normalized_score(records) == pytest.approx((0.92 + 0.05 + 0.80) / 3)
    # The divergence is the guard: if someone "simplifies" min->mean, this breaks.
    assert _quality_gated_min_normalized_score(records) < _quality_gated_normalized_score(records)


def test_min_rewards_consistent_across_topics_over_spiky():
    """A prompt that clears every topic decently beats one that spikes then drifts.

    Consistent candidate (all ~0.70) must outrank a spiky one (0.95 + 0.10) on the
    MIN, even though the spiky one wins on the mean. This is the selection-pressure
    inversion the min was introduced to create.
    """
    # Spiky is deliberately chosen so its MEAN beats consistent's, but its MIN
    # (the drifted case) is far below — the exact mask the min must defeat.
    consistent = [_Record([0.70]), _Record([0.70]), _Record([0.70])]   # mean 0.70, min 0.70
    spiky = [_Record([0.99]), _Record([0.15]), _Record([0.99])]        # mean 0.71, min 0.15
    min_consistent = _quality_gated_min_normalized_score(consistent)
    min_spiky = _quality_gated_min_normalized_score(spiky)
    assert min_consistent > min_spiky, "min must prefer the topic-consistent candidate"
    # The mean would have INVERTED this preference (spiky edges ahead on the mean),
    # so selecting on the mean would pick the drift-prone prompt — which is exactly
    # why selection moved to the min.
    assert _quality_gated_normalized_score(spiky) > _quality_gated_normalized_score(consistent)


def test_floor_breach_case_drives_the_min_to_zero():
    """A hard-eliminated case (gated 0.0 from a floor breach) makes the run min 0.0.

    A floor breach on ANY single topic zeroes that case's gated score; the min then
    hard-eliminates the whole candidate regardless of how strong its other topics
    are. That is the floors-as-gates contract composing with worst-case selection.
    """
    records = [_Record([0.91]), _Record([0.0]), _Record([0.88])]
    assert _quality_gated_min_normalized_score(records) == 0.0


# --- None / default-off contract (byte-identical to the mean's condition) ----


def test_min_is_none_when_no_case_carries_a_gated_score():
    """Default-off: no rubric had a quality_gate => no case has a gated score =>
    the min stat is absent (None), exactly like the mean variant."""
    records = [
        _Record(None),            # non-judge case
        _Record([None, None]),    # judged but no gated score on any dim
    ]
    assert _quality_gated_min_normalized_score(records) is None
    # Same None-condition as the mean — they must agree on absence.
    assert _quality_gated_normalized_score(records) is None


def test_min_ignores_non_judge_cases_but_counts_judged_ones():
    """Non-judge cases (record.judge is None) are skipped; the min is over the
    judged cases only — a non-judge case must not be treated as a 0.0 floor."""
    records = [
        _Record(None),       # skipped, NOT a 0.0
        _Record([0.40]),     # the only gated case
        _Record(None),       # skipped
    ]
    assert _quality_gated_min_normalized_score(records) == pytest.approx(0.40)


def test_single_judged_case_min_equals_that_case():
    records = [_Record([0.63])]
    assert _quality_gated_min_normalized_score(records) == pytest.approx(0.63)
