"""Tests for shared.validation.rollout_filters — generic config-driven engine.

Covers: dot-path resolution (incl. missing), every operator, on_missing
keep/drop, applies_to target scoping, empty-set no-op, eager spec validation,
and the stats accumulator breakdown.
"""
from __future__ import annotations

import pytest

from shared.validation.rollout_filters import (
    MISSING,
    FilterDecision,
    FilterStats,
    RolloutFilter,
    RolloutFilterSet,
    get_path,
)


class TestGetPath:
    def test_top_level(self):
        assert get_path({"a": 1}, "a") == 1

    def test_nested(self):
        rec = {"a": {"b": {"c": 42}}}
        assert get_path(rec, "a.b.c") == 42

    def test_missing_leaf(self):
        assert get_path({"a": {"b": {}}}, "a.b.c") is MISSING

    def test_missing_intermediate(self):
        assert get_path({"a": 1}, "a.b.c") is MISSING

    def test_non_dict_intermediate(self):
        # 'a' resolves to a list, so descending further yields MISSING.
        assert get_path({"a": [1, 2]}, "a.b") is MISSING

    def test_value_none_is_not_missing(self):
        # A stored None is a real value, distinct from MISSING.
        assert get_path({"a": None}, "a") is None
        assert get_path({"a": None}, "a") is not MISSING

    def test_empty_path(self):
        assert get_path({"a": 1}, "") is MISSING


class TestOperators:
    def _match(self, op, field_value, value, **extra):
        filt = RolloutFilter.from_spec({"field": "x", "op": op, "value": value, **extra})
        return filt.matches({"x": field_value})

    def test_eq(self):
        assert self._match("eq", 5, 5) is True
        assert self._match("eq", 5, 6) is False

    def test_ne(self):
        assert self._match("ne", 5, 6) is True
        assert self._match("ne", 5, 5) is False

    def test_gt(self):
        assert self._match("gt", 6, 5) is True
        assert self._match("gt", 5, 5) is False

    def test_gte(self):
        assert self._match("gte", 5, 5) is True
        assert self._match("gte", 4, 5) is False

    def test_lt(self):
        assert self._match("lt", 4, 5) is True
        assert self._match("lt", 5, 5) is False

    def test_lte(self):
        assert self._match("lte", 5, 5) is True
        assert self._match("lte", 6, 5) is False

    def test_numeric_comparison_fails_safe_on_incomparable(self):
        # "abc" > 5 would raise TypeError; engine treats it as non-match.
        assert self._match("gt", "abc", 5) is False
        assert self._match("lte", "abc", 5) is False

    def test_in(self):
        assert self._match("in", "b", ["a", "b", "c"]) is True
        assert self._match("in", "z", ["a", "b", "c"]) is False

    def test_not_in(self):
        assert self._match("not_in", "z", ["a", "b"]) is True
        assert self._match("not_in", "a", ["a", "b"]) is False

    def test_exists(self):
        filt = RolloutFilter.from_spec({"field": "x", "op": "exists"})
        assert filt.matches({"x": 1}) is True
        assert filt.matches({"x": None}) is True  # present-but-None still exists
        assert filt.matches({"y": 1}) is False

    def test_missing(self):
        filt = RolloutFilter.from_spec({"field": "x", "op": "missing"})
        assert filt.matches({"y": 1}) is True
        assert filt.matches({"x": 1}) is False


class TestOnMissing:
    def test_keep_default(self):
        # Default on_missing=keep -> a missing field MATCHES (record retained).
        filt = RolloutFilter.from_spec({"field": "a.b", "op": "gte", "value": 5})
        assert filt.matches({"a": {}}) is True

    def test_drop(self):
        filt = RolloutFilter.from_spec(
            {"field": "a.b", "op": "gte", "value": 5, "on_missing": "drop"}
        )
        assert filt.matches({"a": {}}) is False

    def test_present_field_ignores_on_missing(self):
        filt = RolloutFilter.from_spec(
            {"field": "a.b", "op": "gte", "value": 5, "on_missing": "drop"}
        )
        assert filt.matches({"a": {"b": 7}}) is True
        assert filt.matches({"a": {"b": 2}}) is False


class TestAppliesToScoping:
    def test_filter_scoped_to_grpo_does_not_affect_kto_positive(self):
        fs = RolloutFilterSet(
            filters=[
                {"field": "n", "op": "gte", "value": 5, "applies_to": ["grpo"]},
            ],
            default_targets=["grpo", "kto_positive", "sft"],
        )
        rec = {"n": 2}  # would fail the gte 5 predicate
        assert fs.apply(rec, "grpo").passed is False
        # Scoped only to grpo -> kto_positive is unaffected (passes).
        assert fs.apply(rec, "kto_positive").passed is True

    def test_omitted_applies_to_uses_default_set(self):
        fs = RolloutFilterSet(
            filters=[{"field": "n", "op": "gte", "value": 5}],
            default_targets=["grpo", "kto_positive"],  # excludes kto_negative
        )
        rec = {"n": 2}
        assert fs.apply(rec, "grpo").passed is False
        assert fs.apply(rec, "kto_positive").passed is False
        # kto_negative not in default set -> filter does not apply -> passes.
        assert fs.apply(rec, "kto_negative").passed is True

    def test_explicit_kto_negative_targeting(self):
        fs = RolloutFilterSet(
            filters=[{"field": "n", "op": "gte", "value": 5, "applies_to": ["kto_negative"]}],
            default_targets=["grpo", "kto_positive"],
        )
        rec = {"n": 2}
        assert fs.apply(rec, "kto_negative").passed is False
        assert fs.apply(rec, "grpo").passed is True


class TestEmptySetNoOp:
    def test_none_filters(self):
        fs = RolloutFilterSet(filters=None, default_targets=["grpo"])
        assert fs.is_empty is True
        assert fs.apply({"anything": 1}, "grpo").passed is True

    def test_empty_list(self):
        fs = RolloutFilterSet(filters=[], default_targets=["grpo"])
        assert fs.is_empty is True
        assert fs.apply({}, "grpo").passed is True


class TestAndSemantics:
    def test_all_must_match(self):
        fs = RolloutFilterSet(
            filters=[
                {"field": "n", "op": "gte", "value": 5},
                {"field": "fam", "op": "in", "value": ["tools"]},
            ],
            default_targets=["grpo"],
        )
        assert fs.apply({"n": 6, "fam": "tools"}, "grpo").passed is True
        assert fs.apply({"n": 6, "fam": "other"}, "grpo").passed is False
        assert fs.apply({"n": 2, "fam": "tools"}, "grpo").passed is False

    def test_decision_reports_drop_cause(self):
        fs = RolloutFilterSet(
            filters=[{"field": "n", "op": "gte", "value": 5}],
            default_targets=["grpo"],
        )
        decision = fs.apply({"n": 2}, "grpo")
        assert isinstance(decision, FilterDecision)
        assert decision.passed is False
        assert decision.dropped_by is not None
        assert decision.dropped_by.field == "n"
        assert decision.dropped_on_missing is False

    def test_decision_flags_on_missing_drop(self):
        fs = RolloutFilterSet(
            filters=[{"field": "n", "op": "gte", "value": 5, "on_missing": "drop"}],
            default_targets=["grpo"],
        )
        decision = fs.apply({}, "grpo")
        assert decision.passed is False
        assert decision.dropped_on_missing is True


class TestInvalidSpecRaises:
    def test_unknown_op(self):
        with pytest.raises(ValueError, match="op"):
            RolloutFilterSet(filters=[{"field": "x", "op": "bogus", "value": 1}])

    def test_missing_field(self):
        with pytest.raises(ValueError, match="field"):
            RolloutFilterSet(filters=[{"op": "eq", "value": 1}])

    def test_missing_value_for_value_op(self):
        with pytest.raises(ValueError, match="value"):
            RolloutFilterSet(filters=[{"field": "x", "op": "eq"}])

    def test_bad_on_missing(self):
        with pytest.raises(ValueError, match="on_missing"):
            RolloutFilterSet(filters=[{"field": "x", "op": "eq", "value": 1, "on_missing": "nope"}])

    def test_in_requires_list(self):
        with pytest.raises(ValueError, match="list"):
            RolloutFilterSet(filters=[{"field": "x", "op": "in", "value": "notalist"}])

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="unknown"):
            RolloutFilterSet(filters=[{"field": "x", "op": "eq", "value": 1, "applies": ["grpo"]}])

    def test_presence_op_needs_no_value(self):
        # Should NOT raise — exists/missing ignore value.
        fs = RolloutFilterSet(filters=[{"field": "x", "op": "exists"}])
        assert fs.apply({"x": 1}, "anything").passed is True


class TestFilterStats:
    def test_single_filter_breakdown(self):
        fs = RolloutFilterSet(
            filters=[{"field": "n", "op": "gte", "value": 5}],
            default_targets=["grpo"],
        )
        stats = FilterStats()
        for rec in [{"n": 6}, {"n": 2}, {"n": 1}]:
            stats.record("grpo", fs, fs.apply(rec, "grpo"))
        # one passes, two dropped by predicate
        breakdown = stats.as_dict()
        assert breakdown["grpo"]["passed"] == 1
        assert breakdown["grpo"]["dropped_by_predicate"] == 2
        assert breakdown["grpo"]["dropped_on_missing"] == 0

    def test_on_missing_drop_counted_separately(self):
        fs = RolloutFilterSet(
            filters=[{"field": "n", "op": "gte", "value": 5, "on_missing": "drop"}],
            default_targets=["grpo"],
        )
        stats = FilterStats()
        for rec in [{"n": 6}, {}, {"n": 2}]:
            stats.record("grpo", fs, fs.apply(rec, "grpo"))
        breakdown = stats.as_dict()
        assert breakdown["grpo"]["passed"] == 1
        assert breakdown["grpo"]["dropped_by_predicate"] == 1
        assert breakdown["grpo"]["dropped_on_missing"] == 1

    def test_per_filter_detail_present(self):
        fs = RolloutFilterSet(
            filters=[{"field": "n", "op": "gte", "value": 5}],
            default_targets=["grpo"],
        )
        stats = FilterStats()
        stats.record("grpo", fs, fs.apply({"n": 2}, "grpo"))
        detail = stats.as_dict()["grpo"]["filters"]
        assert len(detail) == 1
        assert detail[0]["field"] == "n"
        assert detail[0]["op"] == "gte"
        assert detail[0]["dropped_by_predicate"] == 1
