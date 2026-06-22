"""Tests for Tools/materialize_toolcall_eval.py.

These exercise the materializer's PURE functions against a tiny SYNTHETIC,
NON-PERSONAL parquet built in-process (no HF download, no real holdout). They
pin the emitted JSONL case schema, the system-turn preservation invariant, the
one-case-per-tool-call-turn invariant, JSON-string->dict argument normalization,
stratified per-tool cap + round-robin fill, drop-report counts, and the privacy
path guard.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Tools import materialize_toolcall_eval as M


# ---------------------------------------------------------------------------
# Synthetic source builders (NON-PERSONAL — toy data only)
# ---------------------------------------------------------------------------

def _tool_call(name: str, arguments: dict) -> dict:
    """A source-shaped tool_call: {type, function:{name, arguments(JSON STRING)}}."""
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _turn(role: str, content: str = "", tool_calls=None, reasoning=None) -> dict:
    return {
        "role": role,
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
    }


def _synthetic_conversations() -> list:
    """Three conversations:

    conv A: system + user + assistant(single tool call 'search')
    conv B: system + user + assistant(MULTI tool call: 'lookup' then 'fetch')
            + user + assistant(single tool call 'search')  -> two tool-call turns
    conv C: user + assistant(single tool call 'search')   -> NO system turn
    """
    sys_a = "You are assistant A with tools."
    sys_b = "You are assistant B with tools."
    conv_a = [
        _turn("system", sys_a),
        _turn("user", "find cats"),
        _turn("assistant", "", tool_calls=[_tool_call("search", {"q": "cats"})]),
    ]
    conv_b = [
        _turn("system", sys_b),
        _turn("user", "look it up then fetch"),
        _turn(
            "assistant",
            "",
            tool_calls=[
                _tool_call("lookup", {"id": 1}),
                _tool_call("fetch", {"url": "http://x"}),
            ],
        ),
        _turn("user", "now search dogs"),
        _turn("assistant", "", tool_calls=[_tool_call("search", {"q": "dogs"})]),
    ]
    conv_c = [
        _turn("user", "find birds"),
        _turn("assistant", "", tool_calls=[_tool_call("search", {"q": "birds"})]),
    ]
    return [conv_a, conv_b, conv_c]


def _write_parquet(path: Path, conversations: list) -> None:
    """Write the conversations into a single-column parquet (column 'conversations')."""
    table = pa.table({"conversations": conversations})
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# read_conversations + build_candidates
# ---------------------------------------------------------------------------

def test_read_conversations_roundtrips_synthetic_parquet(tmp_path):
    p = tmp_path / "syn.parquet"
    _write_parquet(p, _synthetic_conversations())
    convs = M.read_conversations(p)
    assert len(convs) == 3
    # Each turn coerced to the expected plain-dict shape.
    assert convs[0][0]["role"] == "system"
    assert convs[0][0]["content"] == "You are assistant A with tools."


def _build_all_candidates():
    stats = defaultdict(int)
    convs = _synthetic_conversations()
    candidates = M.build_candidates(convs, stats, max_prompt_tokens=10_000)
    return candidates, stats


def test_one_case_per_assistant_tool_call_turn():
    candidates, _ = _build_all_candidates()
    # conv A: 1 turn, conv B: 2 turns, conv C: 1 turn -> 4 tool-call turns total.
    assert len(candidates) == 4
    tools = sorted(c["first_tool"] for c in candidates)
    # First-call tool of each turn: search (A), lookup (B multi), search (B), search (C).
    assert tools == ["lookup", "search", "search", "search"]


def test_system_turn_preserved_verbatim_in_every_case():
    candidates, _ = _build_all_candidates()
    by_tool_sys = defaultdict(set)
    for c in candidates:
        by_tool_sys[c["system"]].add(c["first_tool"])
    # conv A + B cases carry their system content verbatim; conv C has none ("").
    assert "You are assistant A with tools." in by_tool_sys
    assert "You are assistant B with tools." in by_tool_sys
    # conv C produced a case with empty system (no system turn).
    assert "" in by_tool_sys
    # Now render to the emitted case schema and confirm the system survives there.
    for i, cand in enumerate(candidates):
        case = M.build_case(i, cand)
        assert case["system"] == cand["system"]


def test_messages_exclude_target_turn_and_carry_history():
    candidates, _ = _build_all_candidates()
    # conv B's SECOND tool-call turn (search dogs) must see the prior turns,
    # including the assistant multi-call turn, as history.
    search_dogs = [
        c for c in candidates
        if c["first_tool"] == "search" and c["question"] == "now search dogs"
    ]
    assert len(search_dogs) == 1
    msgs = search_dogs[0]["messages"]
    roles = [m["role"] for m in msgs]
    # history = system, user, assistant(multi-call), user  (4 prior turns)
    assert roles == ["system", "user", "assistant", "user"]
    # Target turn itself is NOT in messages.
    assert all(m["content"] != "" or m["role"] != "assistant" for m in msgs[-1:])


def test_ground_truth_arguments_is_dict_not_json_string():
    candidates, _ = _build_all_candidates()
    for c in candidates:
        args = c["ground_truth"]["arguments"]
        assert isinstance(args, dict), f"arguments must be normalized to dict, got {type(args)}"
    # And the rendered case keeps it a dict.
    case = M.build_case(0, candidates[0])
    assert isinstance(case["ground_truth"]["arguments"], dict)


def test_ground_truth_all_calls_captures_multi_call_turn():
    candidates, _ = _build_all_candidates()
    multi = [c for c in candidates if c["first_tool"] == "lookup"]
    assert len(multi) == 1
    all_calls = multi[0]["ground_truth"]["all_calls"]
    assert [c["tool_name"] for c in all_calls] == ["lookup", "fetch"]
    # first-call tool_name + arguments are the scored reference.
    assert multi[0]["ground_truth"]["tool_name"] == "lookup"
    assert multi[0]["ground_truth"]["arguments"] == {"id": 1}


def test_no_system_turn_counted():
    _, stats = _build_all_candidates()
    # conv C lacks a system turn.
    assert stats["no_system_turn"] == 1


def test_unparseable_arguments_counted():
    # A tool_call whose arguments string is NOT valid JSON -> counted + kept raw.
    bad = [
        _turn("system", "S"),
        _turn("user", "u"),
        _turn(
            "assistant",
            "",
            tool_calls=[{"type": "function", "function": {"name": "search", "arguments": "{not json"}}],
        ),
    ]
    stats = defaultdict(int)
    cands = M.build_candidates([bad], stats, max_prompt_tokens=10_000)
    assert len(cands) == 1
    assert stats["unparseable_arguments"] == 1
    # raw unparseable string kept; not None.
    assert cands[0]["ground_truth"]["arguments"] == "{not json"


def test_oversize_candidate_dropped():
    big_content = "x" * 100_000  # ~27k tokens at 3.6 chars/token
    conv = [
        _turn("system", "S"),
        _turn("user", big_content),
        _turn("assistant", "", tool_calls=[_tool_call("search", {"q": "x"})]),
    ]
    stats = defaultdict(int)
    cands = M.build_candidates([conv], stats, max_prompt_tokens=16384)
    assert cands == []
    assert stats["oversize_dropped"] == 1


# ---------------------------------------------------------------------------
# stratified_sample: per-tool cap + round-robin fill + report counts
# ---------------------------------------------------------------------------

def _candidates_for_tool(tool: str, n: int) -> list:
    return [
        {
            "first_tool": tool,
            "system": "S",
            "question": f"{tool}-{i}",
            "messages": [{"role": "user", "content": f"{tool}-{i}"}],
            "ground_truth": {"tool_name": tool, "arguments": {"i": i}, "all_calls": []},
        }
        for i in range(n)
    ]


def test_per_tool_cap_honored():
    # tool 'a' has 10 candidates, cap=3 -> at most 3 kept, 7 capped.
    cands = _candidates_for_tool("a", 10)
    selected, report = M.stratified_sample(cands, sample_budget=100, per_tool_cap=3, seed=7)
    assert len(selected) == 3
    assert report["a"]["n_available"] == 10
    assert report["a"]["n_capped"] == 7
    assert report["a"]["n_kept"] == 3


def test_round_robin_spreads_across_tools():
    # tool 'a' frequent (10), tool 'rare' has 1. Budget 4, cap 10.
    # Round-robin (sorted tool order: a, rare) must include the rare tool, not
    # crowd it out: a,rare,a,a -> rare appears.
    cands = _candidates_for_tool("a", 10) + _candidates_for_tool("rare", 1)
    selected, report = M.stratified_sample(cands, sample_budget=4, per_tool_cap=10, seed=1)
    assert len(selected) == 4
    picked_tools = [c["first_tool"] for c in selected]
    assert "rare" in picked_tools
    assert report["rare"]["n_kept"] == 1
    assert report["a"]["n_kept"] == 3


def test_round_robin_stops_when_groups_exhausted_before_budget():
    # Total available (2+1=3) < budget (10): selection caps at availability.
    cands = _candidates_for_tool("a", 2) + _candidates_for_tool("b", 1)
    selected, report = M.stratified_sample(cands, sample_budget=10, per_tool_cap=10, seed=3)
    assert len(selected) == 3
    assert report["a"]["n_kept"] == 2
    assert report["b"]["n_kept"] == 1


def test_sampling_is_deterministic_for_seed():
    cands = _candidates_for_tool("a", 10) + _candidates_for_tool("b", 10)
    s1, _ = M.stratified_sample(cands, sample_budget=6, per_tool_cap=5, seed=99)
    s2, _ = M.stratified_sample(cands, sample_budget=6, per_tool_cap=5, seed=99)
    assert [c["question"] for c in s1] == [c["question"] for c in s2]


# ---------------------------------------------------------------------------
# Emitted case schema
# ---------------------------------------------------------------------------

def test_build_case_emits_exact_schema():
    candidates, _ = _build_all_candidates()
    case = M.build_case(0, candidates[0])
    assert set(case.keys()) == {
        "id", "question", "tags", "system", "messages", "ground_truth", "verifiers"
    }
    assert case["id"] == "tc_000000"
    # tag carries the reference tool so the run can be grouped by tool:<name>.
    assert f"tool:{candidates[0]['first_tool']}" in case["tags"]
    assert "toolcall_holdout" in case["tags"]
    # verifier spec is the exact args_match overlap contract.
    spec = case["verifiers"][0]
    assert spec["type"] == "args_match"
    assert spec["params"] == {
        "scheme": "overlap",
        "gt_tool_field": "tool_name",
        "gt_args_field": "arguments",
        "pass_threshold": 0.5,
    }


# ---------------------------------------------------------------------------
# Privacy path guard
# ---------------------------------------------------------------------------

def test_privacy_guard_rejects_non_gitignored_out(tmp_path):
    # A path NOT under personal_finetune/ or scratch/ must be refused.
    bad = tmp_path / "leaky" / "cases.jsonl"
    with pytest.raises(SystemExit):
        M.assert_gitignored_path(bad, arg_name="--out")


def test_privacy_guard_allows_personal_finetune(tmp_path):
    ok = tmp_path / "personal_finetune" / "eval" / "cases.jsonl"
    resolved = M.assert_gitignored_path(ok, arg_name="--out")
    assert "personal_finetune" in resolved.parts


def test_privacy_guard_allows_scratch(tmp_path):
    ok = tmp_path / "scratch" / "fixtures" / "cases.jsonl"
    resolved = M.assert_gitignored_path(ok, arg_name="--out")
    assert "scratch" in resolved.parts
