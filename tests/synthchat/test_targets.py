"""Tests for SynthChat.targets — target spec normalization and stage review."""
from __future__ import annotations

import pytest
from SynthChat.targets import (
    _apply_stage_review_result,
    _extract_shared_seed_spec,
    _normalize_target_spec,
)


# ---- _normalize_target_spec ----

class TestNormalizeTargetSpec:
    def test_integer_input(self):
        assert _normalize_target_spec(5) == {"seed_count": 5, "rollouts_per_seed": 1}

    def test_zero_input(self):
        assert _normalize_target_spec(0) == {"seed_count": 0, "rollouts_per_seed": 1}

    def test_dict_with_count(self):
        result = _normalize_target_spec({"count": 3})
        assert result == {"seed_count": 3, "rollouts_per_seed": 1}

    def test_dict_with_seed_and_rollouts(self):
        result = _normalize_target_spec({"seed_count": 2, "rollouts_per_seed": 4})
        assert result == {"seed_count": 2, "rollouts_per_seed": 4}

    def test_dict_seed_overrides_count(self):
        result = _normalize_target_spec({"count": 10, "seed_count": 3})
        assert result["seed_count"] == 3

    def test_boolean_raises(self):
        with pytest.raises(ValueError, match="Boolean"):
            _normalize_target_spec(True)

    def test_negative_integer_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _normalize_target_spec(-1)

    def test_negative_rollout_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _normalize_target_spec({"seed_count": 1, "rollouts_per_seed": -1})

    def test_string_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _normalize_target_spec("five")

    def test_empty_dict_defaults(self):
        result = _normalize_target_spec({})
        assert result == {"seed_count": 1, "rollouts_per_seed": 1}


# ---- _extract_shared_seed_spec ----

class TestExtractSharedSeedSpec:
    def test_no_shared_seed(self):
        spec, cleaned = _extract_shared_seed_spec({"a": 1, "b": 2})
        assert spec is None
        assert cleaned == {"a": 1, "b": 2}

    def test_shared_seed_extracted(self):
        targets = {
            "scenario_a": 3,
            "_shared_seed": {
                "scenario": "env_gen",
                "seed_count": 2,
                "targets": ["scenario_a", "scenario_b"],
            },
        }
        spec, cleaned = _extract_shared_seed_spec(targets)
        assert spec is not None
        assert spec["scenario"] == "env_gen"
        assert spec["seed_count"] == 2
        assert spec["targets"] == ["scenario_a", "scenario_b"]
        assert "_shared_seed" not in cleaned

    def test_shared_seed_missing_scenario_raises(self):
        with pytest.raises(ValueError, match="scenario"):
            _extract_shared_seed_spec({"_shared_seed": {"seed_count": 1}})

    def test_non_dict_targets_raises(self):
        with pytest.raises(ValueError, match="dictionary"):
            _extract_shared_seed_spec("not a dict")

    def test_non_dict_shared_seed_raises(self):
        with pytest.raises(ValueError, match="object"):
            _extract_shared_seed_spec({"_shared_seed": "bad"})

    def test_scenario_key_alias(self):
        spec, _ = _extract_shared_seed_spec({
            "_shared_seed": {"scenario_key": "env_gen", "targets": []}
        })
        assert spec["scenario"] == "env_gen"

    def test_scenarios_alias_for_targets(self):
        spec, _ = _extract_shared_seed_spec({
            "_shared_seed": {"scenario": "env_gen", "scenarios": ["a", "b"]}
        })
        assert spec["targets"] == ["a", "b"]

    def test_negative_seed_count_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _extract_shared_seed_spec({
                "_shared_seed": {"scenario": "env_gen", "seed_count": -1}
            })


# ---- _apply_stage_review_result ----

class TestApplyStageReviewResult:
    def test_none_review_does_nothing(self):
        failures = []
        reviews = {}
        _apply_stage_review_result(failures, reviews, "stage_a", None)
        assert failures == []
        assert reviews == {}

    def test_failed_review_adds_failure(self):
        failures = []
        reviews = {}
        _apply_stage_review_result(
            failures, reviews, "stage_a",
            {"passed": False, "enforce": True},
        )
        assert "stage_a" in failures
        assert "stage_a" in reviews

    def test_passed_review_clears_prior_failure(self):
        failures = ["stage_a"]
        reviews = {}
        _apply_stage_review_result(
            failures, reviews, "stage_a",
            {"passed": True},
        )
        assert "stage_a" not in failures
        assert "stage_a" in reviews

    def test_failed_but_not_enforced(self):
        failures = []
        reviews = {}
        _apply_stage_review_result(
            failures, reviews, "stage_a",
            {"passed": False, "enforce": False},
        )
        assert "stage_a" not in failures
        assert "stage_a" in reviews

    def test_no_duplicate_failures(self):
        failures = ["stage_a"]
        reviews = {}
        _apply_stage_review_result(
            failures, reviews, "stage_a",
            {"passed": False, "enforce": True},
        )
        assert failures.count("stage_a") == 1

    def test_default_enforce_is_true(self):
        failures = []
        reviews = {}
        _apply_stage_review_result(
            failures, reviews, "stage_a",
            {"passed": False},
        )
        assert "stage_a" in failures
