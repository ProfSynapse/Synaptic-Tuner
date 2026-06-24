"""Tests for the full multi-turn SFT projection in project_rollout_datasets.

Covers:
  * multi-turn reconstruction order [system, user, assistant(tool_calls), tool, ...]
  * OpenAI tool_calls shape preservation
  * dropping judge/validation/final_text_request scaffolding turns
  * label/total_turns/provenance fields
  * non-positive rollout -> None
  * total_turns gte 5 filter drops short SFT rows but keeps long ones, and a
    record missing total_turns passes under on_missing: keep.
"""
from __future__ import annotations

import json
from pathlib import Path

from SynthChat.scripts import project_rollout_datasets as proj
from shared.validation import FilterStats, RolloutFilterSet


def _tool_call(name="useTools", args=None):
    return {
        "id": "1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args or {"cmd": "x"}),
        },
    }


def _multi_turn_record(total_turns=2, *, include_turns=True, scenario="s1"):
    """A quality-positive record with a full multi-turn conversation_trace.

    Trace order intentionally interleaves trainable turns with scaffolding turns
    (judge/validation/final_text_request) to prove the scaffolding is dropped.
    """
    episode_trace = {"total_tool_calls": 1, "stop_reason": "completed"}
    if include_turns:
        episode_trace["total_turns"] = total_turns
    return {
        "conversations": [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "done"},
        ],
        "conversation_trace": [
            {"role": "system", "kind": "prompt_message", "content": "you are helpful"},
            {"role": "user", "kind": "prompt_message", "content": "please act"},
            {
                "role": "assistant",
                "kind": "assistant_response",
                "content": "Tool calls: [display repr]",
                "raw": {"role": "assistant", "content": None, "tool_calls": [_tool_call()]},
            },
            {"role": "user", "kind": "tool_feedback", "content": "Tool execution results:\nOK"},
            {"role": "user", "kind": "judge_feedback", "content": "looks good"},
            {
                "role": "assistant",
                "kind": "assistant_response",
                "content": "All set.",
                "raw": {"role": "assistant", "content": "All set."},
            },
            {"role": "user", "kind": "final_text_request", "content": "give final text"},
        ],
        "metadata": {
            "scenario": scenario,
            "environment": {
                "passed": True,
                "executed_tools": [{"name": "useTools", "arguments": {"cmd": "x"}}],
                "episode_trace": episode_trace,
            },
            "environment_seed": {"seed_id": "seed-123"},
            "stage_reviews": {},
        },
    }


def _non_positive_record():
    rec = _multi_turn_record()
    rec["metadata"]["environment"]["passed"] = False
    return rec


def _make_filter_set():
    return RolloutFilterSet(
        filters=[{"field": "metadata.environment.episode_trace.total_turns", "op": "gte", "value": 5}],
        default_targets=proj.DEFAULT_FILTER_TARGETS,
    )


class TestSftReconstruction:
    def test_full_multi_turn_order_and_roles(self):
        rec = _multi_turn_record(total_turns=2)
        row = proj._build_sft_row(Path("a.jsonl"), 0, rec)
        assert row is not None
        roles = [m["role"] for m in row["conversations"]]
        # scaffolding (judge_feedback, final_text_request) dropped
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_assistant_tool_calls_openai_shape_preserved(self):
        rec = _multi_turn_record()
        row = proj._build_sft_row(Path("a.jsonl"), 0, rec)
        tool_call_msg = row["conversations"][2]
        assert tool_call_msg["role"] == "assistant"
        # pure tool-call turn keeps content None (from raw.content, not display repr)
        assert tool_call_msg["content"] is None
        tc = tool_call_msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["id"] == "1"
        assert tc["function"]["name"] == "useTools"
        # arguments stays a JSON string (OpenAI shape)
        assert isinstance(tc["function"]["arguments"], str)

    def test_tool_result_becomes_tool_role(self):
        rec = _multi_turn_record()
        row = proj._build_sft_row(Path("a.jsonl"), 0, rec)
        tool_msg = row["conversations"][3]
        assert tool_msg["role"] == "tool"
        assert "Tool execution results" in tool_msg["content"]
        assert "tool_calls" not in tool_msg

    def test_final_assistant_text_turn_kept(self):
        rec = _multi_turn_record()
        row = proj._build_sft_row(Path("a.jsonl"), 0, rec)
        final = row["conversations"][-1]
        assert final["role"] == "assistant"
        assert final["content"] == "All set."
        assert "tool_calls" not in final

    def test_provenance_and_label_fields(self):
        rec = _multi_turn_record(total_turns=4, scenario="my_scenario")
        row = proj._build_sft_row(Path("agg.jsonl"), 7, rec)
        assert row["label"] is True
        assert row["total_turns"] == 4
        assert row["scenario_id"] == "my_scenario"
        assert row["source_example_id"] == "agg.jsonl:7"
        assert row["metadata"]["source_artifact"] == "agg.jsonl"
        assert row["metadata"]["seed_id"] == "seed-123"
        assert row["metadata"]["stop_reason"] == "completed"

    def test_non_positive_returns_none(self):
        rec = _non_positive_record()
        assert proj._build_sft_row(Path("a.jsonl"), 0, rec) is None

    def test_degenerate_no_user_returns_none(self):
        rec = _multi_turn_record()
        # strip the only user prompt_message turn (keep tool_feedback user turns)
        rec["conversation_trace"] = [
            t for t in rec["conversation_trace"]
            if not (t.get("role") == "user" and t.get("kind") == "prompt_message")
        ]
        assert proj._build_sft_row(Path("a.jsonl"), 0, rec) is None


class TestSftFiltering:
    def test_short_record_dropped(self):
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _multi_turn_record(total_turns=2)
        row = proj._build_sft_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is None
        assert stats.as_dict()["sft"]["dropped_by_predicate"] == 1

    def test_long_record_kept(self):
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _multi_turn_record(total_turns=7)
        row = proj._build_sft_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is not None
        assert row["total_turns"] == 7
        assert stats.as_dict()["sft"]["passed"] == 1

    def test_missing_total_turns_passes_under_keep(self):
        fs = _make_filter_set()
        stats = FilterStats()
        rec = _multi_turn_record(include_turns=False)
        row = proj._build_sft_row(Path("a.jsonl"), 0, rec, fs, stats)
        assert row is not None  # on_missing defaults to keep
        assert row["total_turns"] == 0  # surfaced field defaults to 0 when absent

    def test_no_filter_set_is_noop(self):
        rec = _multi_turn_record(total_turns=1)
        row = proj._build_sft_row(Path("a.jsonl"), 0, rec, None, None)
        assert row is not None
