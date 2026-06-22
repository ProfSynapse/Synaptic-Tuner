"""Tests for the optional floors-as-gates "quality gated" judge composite.

Location: tests/shared/test_quality_gate_scoring.py
Summary: Unit tests for the generic, config-driven quality_gate feature added to
         the dimensioned-rubric judge path. Covers:
           * DEFAULT-OFF: a rubric with no quality_gate produces a None gated
             score and leaves the weighted composite byte-identical.
           * RENORMALIZED quality composite when all floor dims pass.
           * HARD-ELIMINATE (gated score 0.0) on a floor-dim breach.
           * FAIL-CLOSED ValueError when a configured floor or quality dim is
             absent from the judge's per-dimension output (and that this surfaces
             as a failed JudgeResult through judge(), never a silent pass).
           * The reporting helper _quality_gated_normalized_score: None on the
             default-off path, mean across cases when present.

         The engine is generic: every domain value (which dims are floors, the
         threshold, the quality weights) comes from the rubric config here, never
         from the engine. These fixtures use an example rubric shape (6 dims: 2
         safety floors + 4 quality dims) but the engine knows none of those names.
"""
from __future__ import annotations

import pytest

from shared.judge.judge_service import JudgeService
from shared.judge.models import JudgeConfig, RubricDef


class _NoopClient:
    """Minimal stand-in: the dimensioned-scoring path never calls the client."""


# --- Fixtures -------------------------------------------------------------

# The 6 dimensions are an example shape: 2 safety floors (specificity,
# openness_neutrality) + 4 quality dims. Weights here are the ACTIVE
# weighted-composite weights (sum 1.0); the quality_gate config below carries the
# SEPARATE renormalized quality-only weights.
_DIMENSIONS = [
    {"key": "clarity", "name": "Clarity", "weight": 0.15},
    {"key": "relevance", "name": "Relevance", "weight": 0.25},
    {"key": "openness_neutrality", "name": "Openness & Neutrality", "weight": 0.20},
    {"key": "specificity", "name": "Specificity", "weight": 0.15},
    {"key": "actionability", "name": "Actionability", "weight": 0.15},
    {"key": "fidelity_to_intent", "name": "Fidelity to Intent", "weight": 0.10},
]

# Floors-as-gates config: the two safety dims are hard gates at 0.60; the four
# quality dims carry renormalized weights (relevance 0.45 / clarity 0.30 /
# action 0.15 / fidelity 0.10, sum 1.0).
_QUALITY_GATE = {
    "floor_dims": ["specificity", "openness_neutrality"],
    "floor_threshold": 0.60,
    "quality_weights": {
        "relevance": 0.45,
        "clarity": 0.30,
        "actionability": 0.15,
        "fidelity_to_intent": 0.10,
    },
}


def _rubric(*, with_gate: bool, key: str = "example_rubric") -> RubricDef:
    return RubricDef(
        key=key,
        name="Example Rubric",
        description="test rubric",
        scope="response",
        pass_threshold=0.85,
        judge_prompt="Judge: {response}",
        output_schema={"type": "object"},
        dimensions=_DIMENSIONS,
        quality_gate=_QUALITY_GATE if with_gate else None,
    )


def _raw(scores: dict) -> dict:
    """Build a raw_output dict in the judge's per-dimension {reasoning, score} shape."""
    return {
        key: {"reasoning": "because", "score": value}
        for key, value in scores.items()
    }


def _service() -> JudgeService:
    return JudgeService(llm_client=_NoopClient(), judge_config=JudgeConfig())


# Per-dim scores where BOTH floors pass (>= 0.60) and quality dims vary.
_FLOORS_PASS = {
    "clarity": 0.80,
    "relevance": 0.70,
    "openness_neutrality": 0.90,        # floor, passes
    "specificity": 0.95,    # floor, passes
    "actionability": 0.60,
    "fidelity_to_intent": 0.50,
}


# --- Default-off (byte-identical) ----------------------------------------

def test_no_quality_gate_yields_none_gated_score():
    """A rubric without a quality_gate config never produces a gated score."""
    service = _service()
    result = service._score_dimensioned_rubric(_raw(_FLOORS_PASS), _rubric(with_gate=False), None)
    assert result.quality_gated_score is None


def test_no_quality_gate_leaves_weighted_composite_unchanged():
    """The existing weighted composite is identical whether or not a gate is set.

    Same per-dim scores, gate on vs off -> the .score (weighted composite) must be
    byte-identical; only the optional gated field differs.
    """
    service = _service()
    raw = _raw(_FLOORS_PASS)
    without = service._score_dimensioned_rubric(raw, _rubric(with_gate=False), None)
    with_ = service._score_dimensioned_rubric(raw, _rubric(with_gate=True), None)
    assert without.score == with_.score
    assert without.quality_gated_score is None
    assert with_.quality_gated_score is not None


# --- Renormalized quality composite (floors pass) ------------------------

def test_gated_score_is_renormalized_quality_composite_when_floors_pass():
    service = _service()
    result = service._score_dimensioned_rubric(_raw(_FLOORS_PASS), _rubric(with_gate=True), None)
    # relevance 0.45*0.70 + clarity 0.30*0.80 + action 0.15*0.60 + fidelity 0.10*0.50
    expected = 0.45 * 0.70 + 0.30 * 0.80 + 0.15 * 0.60 + 0.10 * 0.50
    assert result.quality_gated_score == pytest.approx(expected)


def test_gated_score_excludes_floor_dims_from_the_quality_sum():
    """Changing a FLOOR dim (while it still passes) must NOT move the gated score.

    The floors are gates, not weighted contributors; only quality dims feed the
    gated composite.
    """
    service = _service()
    base = service._score_dimensioned_rubric(_raw(_FLOORS_PASS), _rubric(with_gate=True), None)
    bumped = dict(_FLOORS_PASS)
    bumped["openness_neutrality"] = 0.61   # still passes the 0.60 floor
    bumped["specificity"] = 0.62
    other = service._score_dimensioned_rubric(_raw(bumped), _rubric(with_gate=True), None)
    assert base.quality_gated_score == pytest.approx(other.quality_gated_score)


# --- Hard-eliminate on floor breach --------------------------------------

def test_floor_breach_hard_eliminates_to_zero():
    service = _service()
    breached = dict(_FLOORS_PASS)
    breached["openness_neutrality"] = 0.40  # below the 0.60 floor
    result = service._score_dimensioned_rubric(_raw(breached), _rubric(with_gate=True), None)
    assert result.quality_gated_score == 0.0


def test_floor_breach_does_not_zero_the_weighted_composite():
    """A floor breach zeroes ONLY the gated selection score; the audit/weighted
    composite still reflects the actual per-dim scores."""
    service = _service()
    breached = dict(_FLOORS_PASS)
    breached["specificity"] = 0.10  # below floor
    result = service._score_dimensioned_rubric(_raw(breached), _rubric(with_gate=True), None)
    assert result.quality_gated_score == 0.0
    assert result.score > 0.0  # weighted composite is untouched


def test_floor_dim_exactly_at_threshold_passes():
    """floor_threshold is a strict-less-than gate: score == threshold passes."""
    service = _service()
    edge = dict(_FLOORS_PASS)
    edge["openness_neutrality"] = 0.60  # exactly at the threshold -> passes
    result = service._score_dimensioned_rubric(_raw(edge), _rubric(with_gate=True), None)
    assert result.quality_gated_score is not None
    assert result.quality_gated_score > 0.0


# --- Fail-closed ----------------------------------------------------------

def test_missing_floor_dim_fails_closed_with_valueerror():
    """A configured floor dim absent from the judge output raises (never silently
    passes the gate)."""
    service = _service()
    missing = dict(_FLOORS_PASS)
    del missing["openness_neutrality"]  # drop a configured floor dim
    # The dimensioned-rubric loop reads from rubric.dimensions, so to truly drop a
    # dim from per_dimension we score it against a rubric whose dimension list also
    # omits it but whose quality_gate still references it.
    rubric = RubricDef(
        key="example_rubric",
        name="x", description="x", scope="response", pass_threshold=0.85,
        judge_prompt="{response}", output_schema={"type": "object"},
        dimensions=[d for d in _DIMENSIONS if d["key"] != "openness_neutrality"],
        quality_gate=_QUALITY_GATE,  # still names openness_neutrality as a floor
    )
    with pytest.raises(ValueError, match="floor dim 'openness_neutrality'"):
        service._score_dimensioned_rubric(_raw(missing), rubric, None)


def test_missing_quality_dim_fails_closed_with_valueerror():
    service = _service()
    missing = dict(_FLOORS_PASS)
    del missing["relevance"]
    rubric = RubricDef(
        key="example_rubric",
        name="x", description="x", scope="response", pass_threshold=0.85,
        judge_prompt="{response}", output_schema={"type": "object"},
        dimensions=[d for d in _DIMENSIONS if d["key"] != "relevance"],
        quality_gate=_QUALITY_GATE,  # still names relevance as a quality dim
    )
    with pytest.raises(ValueError, match="quality dim 'relevance'"):
        service._score_dimensioned_rubric(_raw(missing), rubric, None)


def test_empty_quality_weights_fails_closed():
    service = _service()
    rubric = _rubric(with_gate=True)
    rubric.quality_gate = {"floor_dims": ["openness_neutrality"], "floor_threshold": 0.6, "quality_weights": {}}
    with pytest.raises(ValueError, match="quality_weights is empty"):
        service._score_dimensioned_rubric(_raw(_FLOORS_PASS), rubric, None)


def test_fail_closed_surfaces_as_failed_judge_result_not_a_crash(monkeypatch):
    """A fail-closed ValueError in scoring is caught by judge() and returned as a
    failed JudgeResult (passed=False) -- never an uncaught crash, never a pass."""
    service = _service()
    rubric = RubricDef(
        key="example_rubric",
        name="x", description="x", scope="response", pass_threshold=0.85,
        judge_prompt="{response}", output_schema={"type": "object"},
        dimensions=[d for d in _DIMENSIONS if d["key"] != "openness_neutrality"],
        quality_gate=_QUALITY_GATE,
    )
    # Stub the LLM call so judge() reaches _parse_scores with our crafted output.
    monkeypatch.setattr(
        service.llm_client,
        "structured_output",
        lambda **kwargs: _raw({k: v for k, v in _FLOORS_PASS.items() if k != "openness_neutrality"}),
        raising=False,
    )
    result = service.judge(prompt="x", rubrics=[rubric])
    assert result.passed is False
    assert result.error is not None
    assert "openness_neutrality" in result.error


# --- Reporting helper -----------------------------------------------------

def test_reporting_quality_gated_normalized_score_none_when_absent():
    """_quality_gated_normalized_score returns None when no record carries a gated
    score (default-off path), so the stat is simply absent."""
    from Evaluator.reporting import _quality_gated_normalized_score

    class _Judge:
        def __init__(self, scores):
            self.judge_result = type("R", (), {"scores": scores})()

    class _Score:
        def __init__(self, gated):
            self.quality_gated_score = gated

    class _Record:
        def __init__(self, judge):
            self.judge = judge

    # No judge at all, and a judge whose scores all have gated=None.
    records = [
        _Record(None),
        _Record(_Judge([_Score(None), _Score(None)])),
    ]
    assert _quality_gated_normalized_score(records) is None


def test_reporting_quality_gated_normalized_score_means_across_cases():
    from Evaluator.reporting import _quality_gated_normalized_score

    class _Judge:
        def __init__(self, scores):
            self.judge_result = type("R", (), {"scores": scores})()

    class _Score:
        def __init__(self, gated):
            self.quality_gated_score = gated

    class _Record:
        def __init__(self, judge):
            self.judge = judge

    records = [
        _Record(_Judge([_Score(0.8)])),
        _Record(_Judge([_Score(0.0)])),  # a hard-eliminated case pulls the mean down
        _Record(None),                   # non-judge case ignored
    ]
    assert _quality_gated_normalized_score(records) == pytest.approx(0.4)
