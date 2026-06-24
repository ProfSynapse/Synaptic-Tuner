"""Integration tests for filter wiring in project_rollout_datasets.

Verifies that a `total_turns gte 5` filter:
  * drops a 2-turn record from grpo and kto_positive,
  * passes a record lacking total_turns under on_missing: keep,
  * leaves kto_negative untouched unless explicitly targeted,
and that total_turns is surfaced into projected GRPO rows.
"""
from __future__ import annotations

from pathlib import Path

from SynthChat.scripts import project_rollout_datasets as proj
from shared.validation import FilterStats, RolloutFilterSet


def _positive_record(total_turns=None, *, include_turns=True, scenario="s1"):
    """A record that passes _is_quality_positive with one executed tool."""
    episode_trace = {
        "total_tool_calls": 1,
        "stop_reason": "completed",
    }
    if include_turns:
        episode_trace["total_turns"] = total_turns
    return {
        "conversations": [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "done"},
        ],
        "metadata": {
            "scenario": scenario,
            "environment": {
                "passed": True,
                "executed_tools": [{"name": "useTools", "arguments": {"cmd": "x"}}],
                "episode_trace": episode_trace,
            },
            "stage_reviews": {},
        },
    }


def _negative_record(total_turns=2, scenario="sneg"):
    """A record that passes _is_meaningful_negative and is a KTO-False candidate."""
    return {
        "conversations": [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "partial"},
        ],
        "metadata": {
            "scenario": scenario,
            "labels": {"filter": {"kto_candidate_label": False}},
            "environment": {
                "passed": False,
                "executed_tools": [{"name": "useTools", "arguments": {"cmd": "x"}}],
                "episode_trace": {
                    "total_tool_calls": 1,
                    "total_turns": total_turns,
                    "stop_reason": "max_tool_steps_exceeded",
                },
            },
            "stage_reviews": {},
        },
    }


def _positive_kto_record(total_turns=2, scenario="spos"):
    """A KTO-True candidate that also passes _is_quality_positive."""
    rec = _positive_record(total_turns=total_turns, scenario=scenario)
    rec["metadata"]["labels"] = {"filter": {"kto_candidate_label": True}}
    return rec


def _make_filter_set():
    return RolloutFilterSet(
        filters=[{"field": "metadata.environment.episode_trace.total_turns", "op": "gte", "value": 5}],
        default_targets=proj.DEFAULT_FILTER_TARGETS,
    )


class TestGrpoFiltering:
    def test_short_record_dropped_from_grpo(self):
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _positive_record(total_turns=2)
        row = proj._build_grpo_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is None
        assert stats.as_dict()["grpo"]["dropped_by_predicate"] == 1

    def test_long_record_kept_and_surfaces_total_turns(self):
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _positive_record(total_turns=7)
        row = proj._build_grpo_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is not None
        assert row["total_turns"] == 7
        assert stats.as_dict()["grpo"]["passed"] == 1

    def test_missing_total_turns_passes_under_keep(self):
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _positive_record(include_turns=False)
        row = proj._build_grpo_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is not None  # on_missing defaults to keep
        assert row["total_turns"] == 0  # surfaced field defaults to 0 when absent

    def test_no_filter_set_is_noop(self):
        rec = _positive_record(total_turns=1)
        row = proj._build_grpo_row(Path("a.jsonl"), 0, rec, None, None)
        assert row is not None
        assert row["total_turns"] == 1


class TestKtoPositiveFiltering:
    def test_short_kto_positive_dropped(self):
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _positive_kto_record(total_turns=2)
        row = proj._build_kto_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is None
        assert stats.as_dict()["kto_positive"]["dropped_by_predicate"] == 1

    def test_long_kto_positive_kept(self):
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _positive_kto_record(total_turns=8)
        row = proj._build_kto_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is not None
        assert row["label"] is True


class TestKtoNegativeUnaffected:
    def test_negative_untouched_by_default_filter(self):
        # total_turns=2 would fail gte 5, but kto_negative is excluded from the
        # default set, so the negative survives.
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _negative_record(total_turns=2)
        row = proj._build_kto_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is not None
        assert row["label"] is False

    def test_negative_dropped_when_explicitly_targeted(self):
        fs = RolloutFilterSet(
            filters=[{
                "field": "metadata.environment.episode_trace.total_turns",
                "op": "gte",
                "value": 5,
                "applies_to": ["kto_negative"],
            }],
            default_targets=proj.DEFAULT_FILTER_TARGETS,
        )
        stats = FilterStats()
        rec = _negative_record(total_turns=2)
        row = proj._build_kto_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is None
        assert stats.as_dict()["kto_negative"]["dropped_by_predicate"] == 1


class TestFilterConfigLoading:
    def test_load_filter_set_from_yaml(self, tmp_path):
        cfg = tmp_path / "filters.yaml"
        cfg.write_text(
            "projection:\n"
            "  filters:\n"
            "    - field: metadata.environment.episode_trace.total_turns\n"
            "      op: gte\n"
            "      value: 5\n",
            encoding="utf-8",
        )
        fs = proj._load_filter_set(str(cfg))
        assert fs is not None
        assert not fs.is_empty
        assert list(fs.default_targets) == list(proj.DEFAULT_FILTER_TARGETS)

    def test_load_filter_set_none_when_unset(self):
        assert proj._load_filter_set(None) is None
