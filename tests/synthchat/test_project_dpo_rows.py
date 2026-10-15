"""Tests for DPO pair projection from harness-review replacements."""
from __future__ import annotations

import json
from pathlib import Path

from SynthChat.scripts import project_rollout_datasets as proj


def _call(command):
    return {"id": "c", "type": "function", "function": {"name": "useTools", "arguments": json.dumps({"tool": command})}}


def _assistant(turn_index, content=None, tool_calls=None):
    raw = {"role": "assistant", "content": content}
    if tool_calls:
        raw["tool_calls"] = tool_calls
    return {"role": "assistant", "kind": "assistant_response", "content": "display", "raw": raw, "turn_index": turn_index}


def _record(turns, *, passed=True):
    return {
        "conversations": [],
        "conversation_trace": [
            {"role": "system", "kind": "prompt_message", "content": "sys"},
            {"role": "user", "kind": "prompt_message", "content": "find the note"},
            _assistant(1, tool_calls=[_call('search content "password"')]),
            {"role": "user", "kind": "tool_feedback", "content": "Tool execution results: [\"docs/a.md\"]", "turn_index": 1},
            _assistant(2, tool_calls=[_call('content read "docs/a.md" 1')]),
            {"role": "user", "kind": "tool_feedback", "content": "Tool execution results: 1: text", "turn_index": 2},
            _assistant(3, content="The note says text."),
        ],
        "metadata": {
            "scenario": "s1",
            "environment": {"passed": passed, "harness_review": {"turns": turns}},
        },
    }


_REPLACE_2 = {
    "turn_index": 2,
    "decision": "REPLACE",
    "proposal": {"role": "assistant", "content": "It is probably in docs/a.md.", "tool_calls": None},
    "accepted": {"role": "assistant", "content": None, "tool_calls": [_call('content read "docs/a.md" 1')]},
    "reason": "read first",
    "components_used": ["read-before-answering"],
}


def test_replace_becomes_pair_at_the_same_state():
    rows = proj._build_dpo_rows(Path("run.jsonl"), 0, _record([{"turn_index": 1, "decision": "PASS"}, _REPLACE_2]))
    assert len(rows) == 1
    row = rows[0]
    assert [m["role"] for m in row["prompt"]] == ["system", "user", "assistant", "tool"]
    assert row["chosen"][0]["tool_calls"][0]["function"]["arguments"] == json.dumps({"tool": 'content read "docs/a.md" 1'})
    assert row["rejected"] == [{"role": "assistant", "content": "It is probably in docs/a.md."}]
    assert row["metadata"]["turn_index"] == 2
    assert row["metadata"]["components_used"] == ["read-before-answering"]
    assert row["source_example_id"] == "run.jsonl:0:turn2"


def test_reviewer_voice_empty_proposals_and_failed_rollouts_are_skipped():
    voiced = {**_REPLACE_2, "reviewer_voice": True}
    empty = {**_REPLACE_2, "proposal": {"role": "assistant", "content": ""}}
    assert proj._build_dpo_rows(Path("r"), 0, _record([voiced])) == []
    assert proj._build_dpo_rows(Path("r"), 0, _record([empty])) == []
    assert proj._build_dpo_rows(Path("r"), 0, _record([_REPLACE_2], passed=False)) == []


def test_rollouts_without_harness_review_produce_no_pairs():
    record = _record([])
    record["metadata"]["environment"].pop("harness_review")
    assert proj._build_dpo_rows(Path("r"), 0, record) == []


def test_per_turn_sft_rows_end_at_each_assistant_turn_and_skip_reviewer_voice():
    record = _record([_REPLACE_2])
    record["metadata"]["environment"]["harness_review"]["reviewer_voice_turns"] = [2]
    rows = proj._build_sft_turn_rows(Path("run.jsonl"), 0, record)
    assert [r["metadata"]["turn_index"] for r in rows] == [1, 3]
    assert [m["role"] for m in rows[0]["conversations"]] == ["system", "user", "assistant"]
    assert rows[1]["conversations"][-1] == {"role": "assistant", "content": "The note says text."}
    assert rows[1]["source_example_id"] == "run.jsonl:0:turn3"
