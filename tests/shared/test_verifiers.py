"""Tests for the shared verifier package: CompositeVerifier, builtins, registry.

These tests prove:
- CompositeVerifier reproduces each of the five combine semantics exactly.
- The three builtins (substring/structure/llm_judge) behave per contract and
  hold parity with the existing source logic they consolidate:
    * structure  -> parity with FitnessEvaluator directly.
    * llm_judge  -> aggregate() pins the canonical aggregation scalars (the
      GRPO copy ``judge_reward._aggregate`` was deleted in Phase 3).
- The registry builds verifiers by spec and rejects unknown types.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.verifiers import (
    CompositeVerifier,
    VerifierInput,
    VerifierOutput,
    build_composite,
    build_verifier,
)
from shared.verifiers.builtins.llm_judge import aggregate as canonical_aggregate
from shared.validation.fitness import create_fitness_evaluator
from shared.judge.models import JudgeResult, JudgeScore, RubricDef


# ---------------------------------------------------------------------------
# Stub children
# ---------------------------------------------------------------------------

class StubVerifier:
    """A fixed-output child verifier for composite tests."""

    def __init__(self, name: str, score: float, passed: bool):
        self.name = name
        self._score = score
        self._passed = passed

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        return VerifierOutput(score=self._score, passed=self._passed)


def _sample(text: str = "x", ground_truth=None, prompt: str = "") -> VerifierInput:
    return VerifierInput(
        completion_text=text,
        parsed=None,
        prompt_text=prompt,
        ground_truth=ground_truth or {},
    )


# ---------------------------------------------------------------------------
# CompositeVerifier — one test per combine mode
# ---------------------------------------------------------------------------

class TestCompositeVerifier:

    def test_weighted_sum(self):
        # Σ wᵢ·sᵢ = 1.0*0.5 + 0.3*1.0 + 0.5*0.2 = 0.5 + 0.3 + 0.1 = 0.9
        children = [
            (StubVerifier("a", 0.5, True), 1.0),
            (StubVerifier("b", 1.0, True), 0.3),
            (StubVerifier("c", 0.2, False), 0.5),
        ]
        cv = CompositeVerifier("cw", children, combine="weighted_sum", pass_threshold=0.5)
        out = cv.verify(_sample())
        assert out.score == pytest.approx(0.9)
        assert out.passed is True  # 0.9 >= 0.5
        assert out.detail["combine"] == "weighted_sum"
        assert len(out.detail["children"]) == 3

    def test_weighted_sum_below_threshold(self):
        children = [(StubVerifier("a", 0.1, True), 1.0)]
        cv = CompositeVerifier("cw", children, combine="weighted_sum", pass_threshold=0.5)
        out = cv.verify(_sample())
        assert out.score == pytest.approx(0.1)
        assert out.passed is False

    def test_max_tier_skips_non_passing_highest(self):
        # Highest score (0.9) has passed=False, so it must be skipped.
        # Among passing children {0.7, 0.4}, max is 0.7.
        children = [
            (StubVerifier("hi", 0.9, False), 1.0),
            (StubVerifier("mid", 0.7, True), 1.0),
            (StubVerifier("lo", 0.4, True), 1.0),
        ]
        cv = CompositeVerifier("ct", children, combine="max_tier")
        out = cv.verify(_sample())
        assert out.score == pytest.approx(0.7)
        assert out.passed is True

    def test_max_tier_none_passed(self):
        children = [
            (StubVerifier("a", 0.9, False), 1.0),
            (StubVerifier("b", 0.8, False), 1.0),
        ]
        cv = CompositeVerifier("ct", children, combine="max_tier")
        out = cv.verify(_sample())
        assert out.score == 0.0
        assert out.passed is False

    def test_mean(self):
        children = [
            (StubVerifier("a", 0.2, True), 1.0),
            (StubVerifier("b", 0.8, True), 1.0),
        ]
        cv = CompositeVerifier("cm", children, combine="mean", pass_threshold=0.5)
        out = cv.verify(_sample())
        assert out.score == pytest.approx(0.5)
        assert out.passed is True  # 0.5 >= 0.5

    def test_min(self):
        children = [
            (StubVerifier("a", 0.3, True), 1.0),
            (StubVerifier("b", 0.8, True), 1.0),
        ]
        cv = CompositeVerifier("cmin", children, combine="min", pass_threshold=0.5)
        out = cv.verify(_sample())
        assert out.score == pytest.approx(0.3)
        assert out.passed is False  # 0.3 < 0.5

    def test_all_pass_true(self):
        children = [
            (StubVerifier("a", 0.6, True), 1.0),
            (StubVerifier("b", 0.9, True), 1.0),
        ]
        cv = CompositeVerifier("cap", children, combine="all_pass")
        out = cv.verify(_sample())
        assert out.score == 1.0
        assert out.passed is True

    def test_all_pass_false(self):
        children = [
            (StubVerifier("a", 0.6, True), 1.0),
            (StubVerifier("b", 0.9, False), 1.0),
        ]
        cv = CompositeVerifier("cap", children, combine="all_pass")
        out = cv.verify(_sample())
        assert out.score == 0.0
        assert out.passed is False

    def test_unknown_combine_raises(self):
        with pytest.raises(ValueError):
            CompositeVerifier("bad", [], combine="nonsense")


# ---------------------------------------------------------------------------
# substring builtin
# ---------------------------------------------------------------------------

class TestSubstringVerifier:

    def test_contained(self):
        v = build_verifier({"type": "substring", "target": "hello"})
        out = v.verify(_sample("well hello there"))
        assert out.score == 1.0
        assert out.passed is True

    def test_not_contained(self):
        v = build_verifier({"type": "substring", "target": "absent"})
        out = v.verify(_sample("nothing here"))
        assert out.score == 0.0
        assert out.passed is False

    def test_empty_needle_is_vacuous(self):
        v = build_verifier({"type": "substring", "target": ""})
        out = v.verify(_sample("anything at all"))
        assert out.score == 1.0
        assert out.passed is True

    def test_case_insensitive_default(self):
        v = build_verifier({"type": "substring", "target": "HELLO"})
        out = v.verify(_sample("say hello now"))
        assert out.score == 1.0

    def test_case_sensitive_flag(self):
        v = build_verifier(
            {"type": "substring", "target": "HELLO", "case_sensitive": True}
        )
        out = v.verify(_sample("say hello now"))
        assert out.score == 0.0

    def test_target_field_from_ground_truth(self):
        v = build_verifier({"type": "substring", "target_field": "answer"})
        out = v.verify(_sample("the value is 42", ground_truth={"answer": "42"}))
        assert out.score == 1.0


# ---------------------------------------------------------------------------
# structure builtin — parity with FitnessEvaluator
# ---------------------------------------------------------------------------

VALID_TOOL_CALL = (
    '<tool_call>\n{"name": "useTools", "arguments": {"a": 1}}\n</tool_call>'
)
INVALID_COMPLETION = "just plain text, no tool call here"


class TestStructureVerifier:

    @pytest.mark.parametrize("completion", [VALID_TOOL_CALL, INVALID_COMPLETION])
    def test_parity_with_fitness_evaluator(self, completion):
        # Expected: compute via FitnessEvaluator directly with the same config.
        ref = create_fitness_evaluator(validations=[], scoring_method="binary")
        ref_result = ref.evaluate(completion)

        v = build_verifier(
            {
                "type": "structure",
                "validations": [],
                "scoring_method": "binary",
            }
        )
        out = v.verify(_sample(completion))

        assert out.score == ref_result.score
        assert out.passed == ref_result.is_valid

    def test_valid_vs_invalid_differ(self):
        v = build_verifier(
            {"type": "structure", "validations": [], "scoring_method": "binary"}
        )
        valid = v.verify(_sample(VALID_TOOL_CALL))
        invalid = v.verify(_sample(INVALID_COMPLETION))
        assert valid.score == 1.0 and valid.passed is True
        assert invalid.score == 0.0 and invalid.passed is False


# ---------------------------------------------------------------------------
# llm_judge builtin — aggregation parity with judge_reward._aggregate
# ---------------------------------------------------------------------------

class FakeJudgeService:
    """A judge service stub returning a fixed JudgeResult (no network)."""

    def __init__(self, result: JudgeResult):
        self._result = result
        self.calls = []

    def judge(self, prompt, rubrics, system_prompt=None):
        self.calls.append((prompt, rubrics))
        return self._result


def _make_judge_result() -> JudgeResult:
    return JudgeResult(
        passed=False,
        scores=[
            JudgeScore(
                rubric_key="r1",
                rubric_name="R1",
                score=0.8,
                passed=True,
                pass_threshold=0.5,
            ),
            JudgeScore(
                rubric_key="r2",
                rubric_name="R2",
                score=0.2,
                passed=False,
                pass_threshold=0.5,
            ),
            JudgeScore(
                rubric_key="r3",
                rubric_name="R3",
                score=0.6,
                passed=True,
                pass_threshold=0.5,
            ),
        ],
    )


class TestLLMJudgeAggregationParity:
    """Canonical aggregate() must produce the expected per-strategy scalars.

    Phase 3 deleted ``judge_reward._aggregate`` (its math now lives only in the
    canonical ``shared.verifiers.builtins.llm_judge.aggregate``). The expected
    values below are the SAME numbers the old parity assertion checked against
    that now-removed copy, hardcoded to pin canonical behavior.
    """

    # _make_judge_result() scores: 0.8 (pass), 0.2 (fail), 0.6 (pass).
    @pytest.mark.parametrize(
        "strategy, expected",
        [
            ("mean_score", (0.8 + 0.2 + 0.6) / 3),  # 0.5333...
            ("mean_passed", 2.0 / 3.0),
            ("min_score", 0.2),
            ("all_pass", 0.0),
        ],
    )
    def test_aggregate_values(self, strategy, expected):
        result = _make_judge_result()
        assert canonical_aggregate(result, strategy) == expected

    def test_aggregate_empty_scores(self):
        empty = JudgeResult(passed=False, scores=[])
        for strategy in ("mean_score", "mean_passed", "min_score", "all_pass"):
            assert canonical_aggregate(empty, strategy) == 0.0

    def test_unknown_aggregation_raises(self):
        with pytest.raises(ValueError):
            canonical_aggregate(_make_judge_result(), "bogus")


class TestLLMJudgeVerifier:

    def _rubric(self, key):
        return RubricDef(
            key=key,
            name=key.upper(),
            description="d",
            scope="response",
            pass_threshold=0.5,
            judge_prompt="judge {response}",
            output_schema={},
        )

    def test_injected_judge_service(self):
        result = _make_judge_result()
        fake = FakeJudgeService(result)

        v = build_verifier(
            {
                "type": "llm_judge",
                "params": {
                    "judge_service": fake,
                    "rubrics": [self._rubric("r1"), self._rubric("r2"), self._rubric("r3")],
                    "aggregation": "mean_passed",
                    "pass_threshold": 0.5,
                },
            }
        )
        out = v.verify(_sample("a response", prompt="a prompt"))
        # mean_passed over [True, False, True] = 2/3
        assert out.score == pytest.approx(2.0 / 3.0)
        assert out.passed is True  # 0.666 >= 0.5
        assert len(fake.calls) == 1

    def test_mean_score_aggregation(self):
        result = _make_judge_result()
        fake = FakeJudgeService(result)
        v = build_verifier(
            {
                "type": "llm_judge",
                "params": {
                    "judge_service": fake,
                    "rubrics": [self._rubric("r1"), self._rubric("r2"), self._rubric("r3")],
                    "aggregation": "min_score",
                    "pass_threshold": 0.5,
                },
            }
        )
        out = v.verify(_sample("resp"))
        assert out.score == pytest.approx(0.2)  # min of [0.8, 0.2, 0.6]
        assert out.passed is False

    def test_missing_judge_service_raises(self):
        with pytest.raises(ValueError):
            build_verifier(
                {"type": "llm_judge", "params": {"rubrics": [self._rubric("r1")]}}
            )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_build_substring_works(self):
        v = build_verifier({"type": "substring", "target": "ok"})
        out = v.verify(_sample("looks ok to me"))
        assert out.score == 1.0

    def test_unknown_type_raises(self):
        with pytest.raises(KeyError):
            build_verifier({"type": "does_not_exist"})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError):
            build_verifier({"no_type": True})

    def test_build_composite_via_specs(self):
        cv = build_composite(
            specs=[
                {"type": "substring", "target": "a", "weight": 1.0},
                {"type": "substring", "target": "b", "weight": 0.5},
            ],
            combine="weighted_sum",
            pass_threshold=0.5,
        )
        # "a" present (1.0*1.0), "b" absent (0.5*0.0) -> 1.0
        out = cv.verify(_sample("only a here"))
        assert out.score == pytest.approx(1.0)
        assert out.passed is True
