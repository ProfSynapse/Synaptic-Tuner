"""Unit tests for the native tool-trajectory emit module.

The transcript-distillation skill's ``trajectory.py`` turns normalized adapter
events into native OpenAI-style ``messages`` (structured tool_calls + real
scrubbed/truncated tool outputs + per-trajectory inferred schemas), windowed at
answer boundaries. These tests pin the data-shape contract the SFT trainer
consumes (``loss_mask_mode=assistant_only, tool_call_mode=native``):

  - argument normalization: Codex JSON-string -> object, Claude object passes
    through, degenerate/non-JSON -> a single ``_raw`` key (never a template
    crash);
  - output truncation: verbatim under the keep threshold, head+tail+marker over;
  - schema inference: a key is ``required`` only if present in EVERY call;
    apply_patch uses its fixed patch-blob schema;
  - tool blacklist: ceremony-only assistant turns are dropped AND their orphaned
    tool results are dropped too (pairing stays coherent); an UNKNOWN tool name
    (likely a user MCP) is KEPT, not dropped;
  - reasoning_content appears ONLY on turns that had reasoning (no synthesis);
  - windowing splits an over-length trajectory into <= budget windows, each with
    the system message replayed, without dropping content;
  - the require_tool_call gate skips conversations with no kept (non-ceremony)
    tool call, and segment-level filtering drops windows with no kept call;
  - the hard schema invariant: every tool a window calls has its schema in the
    window's own system prompt.

Loaded by file path (the canonical .skills tree is not an importable package).
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".skills" / "transcript-distillation" / "scripts" / "trajectory.py"
_SPEC = spec_from_file_location("trajectory_skill_module", SCRIPT_PATH)
T = module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
# register before exec so @dataclass can resolve cls.__module__ in sys.modules.
sys.modules[_SPEC.name] = T
_SPEC.loader.exec_module(T)


def _cfg(**kw):
    return T.TrajectoryConfig.from_dict(kw)


# ---------------------------------------------------------------------------
# argument normalization
# ---------------------------------------------------------------------------
def test_normalize_arguments_claude_object_passthrough():
    assert T.normalize_arguments({"file_path": "x.py"}) == {"file_path": "x.py"}


def test_normalize_arguments_codex_json_string_to_object():
    assert T.normalize_arguments('{"cmd": "ls"}') == {"cmd": "ls"}


def test_normalize_arguments_degenerate_string_becomes_raw_key():
    # the corrupt-rollout hazard: a non-JSON blob must not crash; it lands in
    # a single inspectable key.
    out = T.normalize_arguments("to=functions.shell to=functions.shell")
    assert out == {"_raw": "to=functions.shell to=functions.shell"}


def test_normalize_arguments_json_non_object_becomes_raw():
    assert T.normalize_arguments("[1, 2, 3]") == {"_raw": "[1, 2, 3]"}
    assert T.normalize_arguments("") == {}
    assert T.normalize_arguments(None) == {}


# ---------------------------------------------------------------------------
# output truncation
# ---------------------------------------------------------------------------
def test_truncate_output_keeps_short_verbatim():
    text, trunc = T.truncate_output("short", keep=2000, head=1200, tail=600)
    assert text == "short" and trunc is False


def test_truncate_output_head_tail_with_marker():
    body = "H" * 3000 + "T" * 3000  # 6000 chars
    text, trunc = T.truncate_output(body, keep=2000, head=1200, tail=600)
    assert trunc is True
    assert text.startswith("H" * 1200)
    assert text.endswith("T" * 600)
    assert "[truncated" in text
    # honest elision count = 6000 - 1200 - 600
    assert "[truncated 4200 chars]" in text


# ---------------------------------------------------------------------------
# schema inference
# ---------------------------------------------------------------------------
def test_infer_schemas_required_only_when_in_every_call():
    events = [
        {"role": "assistant", "tool_calls": [
            {"name": "Read", "input": {"file_path": "a.py"}}]},
        {"role": "assistant", "tool_calls": [
            {"name": "Read", "input": {"file_path": "b.py", "limit": 10}}]},
    ]
    schemas, stats = T.infer_tool_schemas(events)
    assert stats["schema_source"] == "infer"
    read = next(s for s in schemas if s["name"] == "Read")
    props = read["parameters"]["properties"]
    assert set(props) == {"file_path", "limit"}
    assert props["limit"]["type"] == "integer"
    # file_path in both calls -> required; limit only in one -> optional.
    assert read["parameters"]["required"] == ["file_path"]


def test_infer_schemas_apply_patch_uses_fixed_patch_schema():
    events = [{"role": "assistant", "tool_calls": [
        {"name": "apply_patch", "input": '{"input": "*** Begin Patch\\n..."}'}]}]
    schemas, _ = T.infer_tool_schemas(events)
    ap = next(s for s in schemas if s["name"] == "apply_patch")
    assert ap["parameters"]["properties"]["input"]["type"] == "string"
    assert ap["parameters"]["required"] == ["input"]


# ---------------------------------------------------------------------------
# message building: blacklist, reasoning, structured calls
# ---------------------------------------------------------------------------
def _coding_traj_events():
    return [
        {"role": "human", "text": "fix the bug"},
        {"role": "assistant", "text": "", "reasoning": "read the file first",
         "tool_calls": [{"name": "Read", "input": {"file_path": "x.py"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "def f(): pass"},
        {"role": "assistant", "text": "found it", "tool_calls": [
            {"name": "Edit", "input": {"file_path": "x.py", "old": "pass", "new": "return 1"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "edited"},
        {"role": "assistant", "text": "fixed the bug"},  # traceless final answer
    ]


def test_build_messages_structured_calls_and_reasoning_only_where_present():
    msgs, stats = T.build_messages(_coding_traj_events(), _cfg(), schemas=[])
    # system + user + 3 assistant + 2 tool
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    first_asst = msgs[2]
    assert first_asst["tool_calls"][0]["type"] == "function"
    assert first_asst["tool_calls"][0]["function"]["name"] == "Read"
    assert first_asst["tool_calls"][0]["function"]["arguments"] == {"file_path": "x.py"}
    assert first_asst["reasoning_content"] == "read the file first"
    # traceless final answer carries NO reasoning_content (no empty-think later)
    assert "reasoning_content" not in msgs[-1]
    assert stats["assistant_with_reasoning"] == 1
    assert stats["tool_turns"] == 2


def test_build_messages_drops_ceremony_turn_and_orphan_tool_result():
    events = [
        {"role": "human", "text": "go"},
        # ceremony-only assistant turn (blacklisted, no text) -> dropped
        {"role": "assistant", "text": "", "tool_calls": [
            {"name": "SendMessage", "input": {"to": "lead", "message": "hi"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "ack"},  # orphan -> dropped
        {"role": "assistant", "text": "now the real work", "tool_calls": [
            {"name": "Bash", "input": {"command": "ls"}}]},
        {"role": "tool", "tool_error": False, "command": "ls", "output": "x.py"},
        {"role": "assistant", "text": "done"},
    ]
    msgs, stats = T.build_messages(events, _cfg(), schemas=[])
    roles = [m["role"] for m in msgs]
    # the SendMessage turn and its ack tool result are both gone.
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert stats["tool_calls_dropped"] == 1
    assert stats["tool_calls_kept"] == 1
    # the surviving assistant tool turn is Bash, and the ack output never appears
    assert all("ack" not in (m.get("content") or "") for m in msgs)


def test_reasoning_scope_latest_keeps_only_last_reasoning_turn():
    """Default scope='latest': stale reasoning on earlier assistant turns is
    dropped; only the LAST reasoning-bearing assistant turn keeps it. Never adds
    reasoning (a traceless final turn stays traceless)."""
    events = [
        {"role": "human", "text": "go"},
        {"role": "assistant", "text": "", "reasoning": "EARLY think",
         "tool_calls": [{"name": "Read", "input": {"file_path": "a.py"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "body"},
        {"role": "assistant", "text": "mid", "reasoning": "LATE think",
         "tool_calls": [{"name": "Edit", "input": {"file_path": "a.py", "o": "x", "n": "y"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "ok"},
        {"role": "assistant", "text": "done"},  # traceless final
    ]
    msgs, stats = T.build_messages(events, _cfg(reasoning_scope="latest"), schemas=[])
    reasoned = [m for m in msgs if m.get("role") == "assistant" and m.get("reasoning_content")]
    assert len(reasoned) == 1
    assert reasoned[0]["reasoning_content"] == "LATE think"
    # the EARLY think turn lost its reasoning but kept its tool_call.
    early = msgs[2]
    assert early["tool_calls"][0]["function"]["name"] == "Read"
    assert "reasoning_content" not in early
    # traceless final still traceless (no synthesis)
    assert "reasoning_content" not in msgs[-1]
    assert stats["assistant_with_reasoning"] == 1


def test_reasoning_scope_all_keeps_every_reasoning_turn():
    events = [
        {"role": "human", "text": "go"},
        {"role": "assistant", "text": "", "reasoning": "EARLY think",
         "tool_calls": [{"name": "Read", "input": {"file_path": "a.py"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "body"},
        {"role": "assistant", "text": "done", "reasoning": "LATE think"},
    ]
    msgs, stats = T.build_messages(events, _cfg(reasoning_scope="all"), schemas=[])
    reasoned = [m["reasoning_content"] for m in msgs
                if m.get("role") == "assistant" and m.get("reasoning_content")]
    assert reasoned == ["EARLY think", "LATE think"]
    assert stats["assistant_with_reasoning"] == 2


def test_reasoning_scope_invalid_rejected():
    import pytest
    with pytest.raises(ValueError, match="reasoning_scope"):
        T.TrajectoryConfig.from_dict({"reasoning_scope": "bogus"})


def test_codex_json_string_arguments_normalized_in_messages():
    events = [
        {"role": "human", "text": "run it"},
        {"role": "assistant", "text": "", "tool_calls": [
            {"name": "shell_command", "input": '{"command": "pytest"}'}]},
        {"role": "tool", "tool_error": False, "command": "pytest", "output": "passed"},
        {"role": "assistant", "text": "tests pass"},
    ]
    msgs, _ = T.build_messages(events, _cfg(), schemas=[])
    call = msgs[2]["tool_calls"][0]["function"]
    assert call["arguments"] == {"command": "pytest"}  # object, not the raw string


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------
def test_window_short_trajectory_is_single_window():
    msgs, _ = T.build_messages(_coding_traj_events(), _cfg(), schemas=[])
    windows, stats = T.window_messages(msgs, _cfg(max_seq_tokens=16384))
    assert stats["split"] is False
    assert len(windows) == 1


def test_window_long_trajectory_splits_and_replays_system():
    # tiny budget forces splitting; build a long alternating trajectory.
    events = [{"role": "human", "text": "start"}]
    for i in range(12):
        events.append({"role": "assistant", "text": f"step {i} " + "x" * 200,
                       "tool_calls": [{"name": "Bash", "input": {"command": f"cmd{i}"}}]})
        events.append({"role": "tool", "tool_error": False, "command": f"cmd{i}",
                       "output": "y" * 200})
        events.append({"role": "assistant", "text": f"answer {i} " + "z" * 200})  # boundary
    msgs, _ = T.build_messages(events, _cfg(), schemas=[{"name": "Bash",
                               "description": "d", "parameters": {"type": "object",
                               "properties": {}, "required": []}}])
    windows, stats = T.window_messages(
        msgs, _cfg(max_seq_tokens=400, chars_per_token=4.0))
    assert stats["split"] is True
    assert len(windows) > 1
    # every window replays the system message at its head.
    assert all(w[0]["role"] == "system" for w in windows)
    # TEMPLATE CONTRACT: every window must contain at least one user turn, else
    # the stock Qwen3.5 template raises "No user query found in messages". Windows
    # after the first carry the most recent prior user turn into their head.
    assert all(any(m["role"] == "user" for m in w) for w in windows)
    # carried context is real prior content (not synthesized): the carried user
    # turn equals the conversation's earlier user turn.
    assert stats["user_carried_windows"] >= 1
    # no NON-USER content is lost or duplicated: concatenated non-system,
    # non-user turns equal the original (user turns may be carried/duplicated).
    orig_nonuser = [m for m in msgs if m["role"] not in ("system", "user")]
    recombined_nonuser = [m for w in windows for m in w
                          if m["role"] not in ("system", "user")]
    assert recombined_nonuser == orig_nonuser
    # the single original user turn is present in every window (carried forward).
    assert all(any(m["role"] == "user" and m["content"] == "start" for m in w)
               for w in windows)


def test_window_carries_user_so_no_window_starts_userless():
    """Regression: the stock Qwen3.5 template raises "No user query found in
    messages" for a window whose first non-system turn is assistant/tool. Every
    window must therefore carry a user turn. Build a trajectory with an EARLY
    user turn then a long tool-only run so later windows would otherwise be
    user-less."""
    events = [{"role": "human", "text": "kick it off"}]
    for i in range(15):
        events.append({"role": "assistant", "text": f"working {i} " + "x" * 250,
                       "tool_calls": [{"name": "Bash", "input": {"command": f"c{i}"}}]})
        events.append({"role": "tool", "tool_error": False, "command": f"c{i}",
                       "output": "z" * 250})
        events.append({"role": "assistant", "text": f"step done {i} " + "q" * 250})
    msgs, _ = T.build_messages(events, _cfg(), schemas=[])
    windows, stats = T.window_messages(
        msgs, _cfg(max_seq_tokens=350, chars_per_token=4.0))
    assert len(windows) > 1
    # no window's first post-system turn is assistant/tool with no user anywhere.
    for w in windows:
        body = [m for m in w if m["role"] != "system"]
        assert body[0]["role"] == "user", (
            "window starts with a non-user turn -> stock template would crash")
    assert stats["user_carried_windows"] >= 1


# ---------------------------------------------------------------------------
# top-level emit + gate
# ---------------------------------------------------------------------------
def test_emit_trajectory_rows_happy_path():
    rows, stats = T.emit_trajectory_rows(
        _coding_traj_events(), source_kind="codex", project="p", rel_id="r.jsonl",
        cfg=_cfg())
    assert stats["emitted"] == 1
    row = rows[0]
    assert row["metadata"]["schema_source"] == "infer"
    assert row["metadata"]["tool_names"] == ["Edit", "Read"]
    # the system message carries the inferred per-trajectory schemas.
    sysmsg = row["messages"][0]["content"]
    assert "# Available tools" in sysmsg and "Read" in sysmsg and "Edit" in sysmsg


def test_emit_skips_conversation_with_no_kept_tool_call():
    events = [
        {"role": "human", "text": "hi"},
        {"role": "assistant", "text": "hello", "tool_calls": [
            {"name": "SendMessage", "input": {"to": "x", "message": "y"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "ok"},
        {"role": "assistant", "text": "bye"},
    ]
    rows, stats = T.emit_trajectory_rows(
        events, source_kind="claude_main", project="p", rel_id="r.jsonl", cfg=_cfg())
    assert rows == []
    assert stats["skipped"] == "no_kept_tool_call"


# ---------------------------------------------------------------------------
# blacklist semantics: unknown name (likely MCP) is KEPT, schema invariant,
# segment-level window filtering
# ---------------------------------------------------------------------------
def test_unknown_tool_name_is_kept_as_likely_mcp():
    """A blacklist KEEPS unrecognized tool names — they are far more likely to be
    the user's custom MCP tools than ceremony, and a whitelist would silently
    drop them. The MCP call survives into the structured tool_calls AND earns an
    inferred schema in the system prompt."""
    events = [
        {"role": "human", "text": "search my vault"},
        {"role": "assistant", "text": "", "tool_calls": [
            {"name": "nexus-synaptic-labs:vaultLibrarian",
             "input": {"query": "PACT notes"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "3 notes found"},
        {"role": "assistant", "text": "here are your notes"},
    ]
    rows, stats = T.emit_trajectory_rows(
        events, source_kind="claude_ai_export", project="p", rel_id="r.json", cfg=_cfg())
    assert stats["emitted"] == 1
    row = rows[0]
    # the MCP call survived (unknown name kept, not dropped)
    assert row["metadata"]["tool_names"] == ["nexus-synaptic-labs:vaultLibrarian"]
    # and its inferred schema is present in the window's own system prompt.
    sysmsg = row["messages"][0]["content"]
    assert "nexus-synaptic-labs:vaultLibrarian" in sysmsg
    assert '"query"' in sysmsg


def test_keep_pattern_force_keeps_mcp_even_if_blacklisted():
    """KEEP-PATTERN insurance: an MCP/namespaced name is force-kept even if it is
    (mis)placed in the blacklist, so a ceremony-set edit can never swallow an
    MCP. Plain ceremony names (no namespace marker) are still dropped."""
    bl = set(T.DEFAULT_TOOL_BLACKLIST)
    kp = T.DEFAULT_TOOL_KEEP_PATTERNS
    # all three MCP forms kept
    assert T._is_kept_tool("mcp__workspace__bash", bl, kp)
    assert T._is_kept_tool("nexus-synaptic-labs:toolManager_useTools", bl, kp)
    assert T._is_kept_tool("hubspot:hubspotContact", bl, kp)
    # bare user-MCP name kept (not blacklisted, not namespaced -> unknown-kept)
    assert T._is_kept_tool("noteEditor", bl, kp)
    # even if an MCP-shaped name were wrongly blacklisted, the keep-pattern wins
    assert T._is_kept_tool("server:SendMessage", bl, kp)
    # plain ceremony still dropped
    assert not T._is_kept_tool("SendMessage", bl, kp)
    assert not T._is_kept_tool("Skill", bl, kp)
    assert not T._is_kept_tool("TodoWrite", bl, kp)
    # no ceremony default name accidentally matches a keep-pattern
    assert not any(any(p in c for p in kp) for c in bl)


def test_skill_is_ceremony_and_dropped():
    """`Skill` (claude.ai skill-invocation) is platform ceremony per the scope
    ruling — a Skill-only assistant turn is dropped along with its tool result."""
    events = [
        {"role": "human", "text": "go"},
        {"role": "assistant", "text": "", "tool_calls": [
            {"name": "Skill", "input": {"skill": "PACT:bootstrap"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "loaded"},
        {"role": "assistant", "text": "now real work", "tool_calls": [
            {"name": "Bash", "input": {"command": "ls"}}]},
        {"role": "tool", "tool_error": False, "command": "ls", "output": "x.py"},
        {"role": "assistant", "text": "done"},
    ]
    msgs, stats = T.build_messages(events, _cfg(), schemas=[])
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert stats["tool_calls_dropped"] == 1  # the Skill call
    assert all("loaded" not in (m.get("content") or "") for m in msgs)


def test_hubspot_is_hard_dropped_in_every_naming_form():
    """HARD-DROP overrides keep-patterns: the user ruled ALL hubspot:* tool uses
    out of training (third-party CRM contact PII). Every naming form must drop —
    namespaced (``hubspot:hubspotContact``, ``local__hubspot__*``,
    ``mcp__claude_ai_Hubspot__*``) AND the marker-less forms (bare
    ``hubspotContact``, prefixed ``hubspot_create_deal``) that a keep-pattern
    would otherwise FORCE-KEEP. Non-hubspot MCPs are unaffected."""
    bl = set(T.DEFAULT_TOOL_BLACKLIST)
    kp = T.DEFAULT_TOOL_KEEP_PATTERNS
    dp = T.DEFAULT_TOOL_DROP_PATTERNS
    for nm in ("hubspot:hubspotContact", "local__hubspot__hubspotContact",
               "mcp__claude_ai_Hubspot__hubspot_useTools", "hubspotContact",
               "hubspot_create_deal", "Hubspot:hubspot_getTools", "hubspot"):
        # COUNTER-TEST: WITHOUT drop_patterns every form is KEPT (keep-pattern or
        # unknown-kept) — exactly the leak the drop closes. WITH drop_patterns
        # every form is DROPPED.
        assert T._is_kept_tool(nm, bl, kp), f"{nm} should be kept pre-drop (the leak)"
        assert not T._is_kept_tool(nm, bl, kp, dp), f"{nm} should hard-drop"
    # a genuine non-hubspot MCP is still kept under the same drop_patterns
    assert T._is_kept_tool("noteEditor", bl, kp, dp)
    assert T._is_kept_tool("nexus-synaptic-labs:vaultLibrarian", bl, kp, dp)


def test_hubspot_turn_and_matching_tool_result_dropped():
    """A hubspot tool call + its tool_result are BOTH removed from the rendered
    messages (the orphan-result pairing stays coherent), and no hubspot schema
    leaks into the system prompt. The surrounding non-hubspot work survives."""
    events = [
        {"role": "human", "text": "look up the contact then read the file"},
        {"role": "assistant", "text": "", "tool_calls": [
            {"name": "hubspot:hubspotContact", "input": {"email": "a@b.com"}}]},
        {"role": "tool", "tool_error": False, "command": "",
         "output": "Contact: Jane Doe, phone 555-1234, deal $40k"},
        {"role": "assistant", "text": "", "tool_calls": [
            {"name": "Read", "input": {"file_path": "/x.py"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "print('hi')"},
        {"role": "assistant", "text": "done"},
    ]
    cfg = _cfg()
    schemas, _ = T.infer_tool_schemas(
        events, set(cfg.tool_blacklist), cfg.tool_keep_patterns, cfg.tool_drop_patterns)
    msgs, stats = T.build_messages(events, cfg, schemas)
    # the hubspot call was dropped, the Read survived
    all_tool_names = [
        c["function"]["name"] for m in msgs for c in (m.get("tool_calls") or [])]
    assert all_tool_names == ["Read"]
    # its tool_result (the CRM PII) is gone from every message body
    blob = " ".join(m.get("content") or "" for m in msgs)
    assert "Jane Doe" not in blob and "hubspot" not in blob.lower()
    # and no hubspot schema in the system prompt
    assert "hubspot" not in msgs[0]["content"].lower()
    # role sequence stays coherent (no orphan tool turn)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant"]


def test_client_excluded_conversation_is_dropped_when_term_in_body_only():
    """CONTENT-LEVEL client exclusion: a conversation whose TITLE/slug is clean
    but whose BODY mentions a third-party client term (project-zephyr / acme-corp /
    client-x) is dropped ENTIRELY — slug-level scope filtering misses these.
    COUNTER-TEST: the identical conversation WITHOUT the client term emits."""
    # Self-contained placeholder client terms (the real canonical patterns live
    # in the gitignored personal_finetune/client_exclude module; these neutral
    # stand-ins exercise the identical drop behaviour without baking real names).
    client_terms = ["acme-corp", "project-zephyr", "client-x"]
    base = [
        {"role": "human", "text": "help me with the project"},
        {"role": "assistant", "text": "", "tool_calls": [
            {"name": "Read", "input": {"file_path": "/notes.md"}}]},
        {"role": "tool", "tool_error": False, "command": "", "output": "PLACEHOLDER"},
        {"role": "assistant", "text": "done"},
    ]
    # clean conversation EMITS
    clean = [dict(e) for e in base]
    clean[2] = {**base[2], "output": "weekly standup notes for the api team"}
    rows, stats = T.emit_trajectory_rows(
        clean, source_kind="claude_ai_export", project="weekly sync", rel_id="r.json",
        cfg=_cfg(client_exclude=client_terms))
    assert stats["emitted"] == 1, "clean conv must emit"

    # SAME conv but a client term hidden in the tool OUTPUT (not the title) -> drop
    leaky = [dict(e) for e in base]
    leaky[2] = {**base[2], "output": "pricing model for the client-x client engagement"}
    rows, stats = T.emit_trajectory_rows(
        leaky, source_kind="claude_ai_export", project="weekly sync", rel_id="r.json",
        cfg=_cfg(client_exclude=client_terms))
    assert rows == [] and stats["skipped"] == "client_excluded", \
        "client term in BODY must drop the whole conversation"

    # term in a USER turn and in a tool-call INPUT both also trigger
    for inject in (
        [{"role": "human", "text": "summarize the acme-corp meeting"}, base[1], base[2], base[3]],
        [base[0], {"role": "assistant", "text": "", "tool_calls": [
            {"name": "Write", "input": {"path": "/x", "content": "project-zephyr roadmap"}}]},
         base[2], base[3]],
    ):
        rows, stats = T.emit_trajectory_rows(
            inject, source_kind="claude_ai_export", project="ok", rel_id="r.json",
            cfg=_cfg(client_exclude=client_terms))
        assert rows == [] and stats["skipped"] == "client_excluded"

    # disabling client_exclude ([]), the leaky conv emits again (knob works)
    rows, stats = T.emit_trajectory_rows(
        leaky, source_kind="claude_ai_export", project="x", rel_id="r.json",
        cfg=_cfg(client_exclude=[]))
    assert stats["emitted"] == 1


def test_schema_invariant_holds_for_every_called_tool():
    """Every tool a row calls MUST have its schema in that row's system prompt —
    the generalizability bar (schema-conditioned tool use, not name memory)."""
    rows, _ = T.emit_trajectory_rows(
        _coding_traj_events(), source_kind="codex", project="p", rel_id="r.jsonl",
        cfg=_cfg())
    for row in rows:
        sysmsg = row["messages"][0]["content"]
        for name in row["metadata"]["tool_names"]:
            assert f"## {name}" in sysmsg, f"{name} called but no schema in system"


def test_segment_filtering_drops_windows_with_no_kept_tool_call():
    """When a long trajectory windows into segments, a window that contains NO
    kept tool call teaches no tool use (require_tool_call) and is dropped, while
    tool-bearing windows survive. No content within a kept window is lost."""
    # build a trajectory whose TAIL is a long chatty answer-only run (no tools)
    events = [{"role": "human", "text": "start"}]
    # tool-bearing head
    for i in range(3):
        events.append({"role": "assistant", "text": f"step {i} " + "x" * 200,
                       "tool_calls": [{"name": "Bash", "input": {"command": f"c{i}"}}]})
        events.append({"role": "tool", "tool_error": False, "command": f"c{i}",
                       "output": "y" * 200})
        events.append({"role": "assistant", "text": f"answer {i} " + "z" * 200})
    # long tool-free chatty tail (will window into its own no-tool segment)
    for i in range(8):
        events.append({"role": "human", "text": f"followup {i} " + "q" * 200})
        events.append({"role": "assistant", "text": f"reply {i} " + "w" * 400})
    rows, stats = T.emit_trajectory_rows(
        events, source_kind="claude_main", project="p", rel_id="r.jsonl",
        cfg=_cfg(max_seq_tokens=400, chars_per_token=4.0))
    # at least one no-tool window was dropped by the segment filter
    assert stats["windows_dropped_no_tool"] >= 1
    # every emitted row contains at least one kept tool call
    for row in rows:
        assert row["metadata"]["tool_names"], "emitted a window with no tool call"
    # window_count metadata reflects SURVIVING windows
    assert all(r["metadata"]["window_count"] == len(rows) for r in rows)


def _tool_conv_with_chat_tail_events():
    """A long tool conversation: a tool-bearing head then a long tool-free chatty
    tail that windows into its own no-tool segment(s). Shared by the per-window
    gate tests below."""
    events = [{"role": "human", "text": "start"}]
    for i in range(3):
        events.append({"role": "assistant", "text": f"step {i} " + "x" * 200,
                       "tool_calls": [{"name": "Bash", "input": {"command": f"c{i}"}}]})
        events.append({"role": "tool", "tool_error": False, "command": f"c{i}",
                       "output": "y" * 200})
        events.append({"role": "assistant", "text": f"answer {i} " + "z" * 200})
    for i in range(8):
        events.append({"role": "human", "text": f"followup {i} " + "q" * 200})
        events.append({"role": "assistant", "text": f"reply {i} " + "w" * 400})
    return events


def test_per_window_relax_keeps_chat_windows_of_a_tool_conversation():
    """With require_tool_call_per_window=False, a QUALIFYING tool conversation
    emits its no-tool (pure chat/reasoning) windows too — they are clean native
    prose (tools are never folded into prose), so keeping them re-introduces no
    anti-pattern. The conversation-level require_tool_call gate is unchanged."""
    events = _tool_conv_with_chat_tail_events()
    cfg = _cfg(max_seq_tokens=400, chars_per_token=4.0,
               require_tool_call_per_window=False)
    rows, stats = T.emit_trajectory_rows(
        events, source_kind="claude_main", project="p", rel_id="r.jsonl", cfg=cfg)
    # no no-tool window is dropped now; at least one chat window is KEPT.
    assert stats["windows_dropped_no_tool"] == 0
    assert stats["chat_windows_kept"] >= 1
    # both a tool-bearing window AND a tool-free chat window are present.
    assert any(r["metadata"]["tool_names"] for r in rows), "expected a tool window"
    assert any(not r["metadata"]["tool_names"] for r in rows), \
        "expected a recovered chat (no-tool) window"
    # window_count metadata reflects ALL surviving windows.
    assert all(r["metadata"]["window_count"] == len(rows) for r in rows)


def test_per_window_relax_still_excludes_pure_chat_conversation():
    """The per-window relax does NOT turn the emit into a global
    require_tool_call=False: a PURE-CHAT conversation (zero kept tool calls
    anywhere) is still skipped entirely by the conversation-level gate, so it is
    never double-represented (it flows through the chat path instead)."""
    events = [
        {"role": "human", "text": "explain monads"},
        {"role": "assistant", "text": "a monad is a monoid in the category of endofunctors"},
        {"role": "human", "text": "thanks"},
        {"role": "assistant", "text": "you're welcome"},
    ]
    cfg = _cfg(require_tool_call_per_window=False)
    rows, stats = T.emit_trajectory_rows(
        events, source_kind="claude_ai_export", project="p", rel_id="r.json", cfg=cfg)
    assert rows == []
    assert stats["skipped"] == "no_kept_tool_call"


# ---------------------------------------------------------------------------
# real-tokenizer length guard
# ---------------------------------------------------------------------------
def _fake_real_len(messages, tokenizer):
    """Deterministic stand-in for the real-tokenizer length: token count ==
    #whitespace-words across every message's content + tool-call names. Lets the
    guard's RE-WINDOWING LOGIC be tested hermetically without a model download or
    the trainer's tokenization plumbing (exercised separately by render/rebuild)."""
    words = 0
    for m in messages:
        words += len((m.get("content") or "").split())
        for c in (m.get("tool_calls") or []):
            words += len(((c.get("function") or {}).get("name") or "").split())
    return words


class _FakeTok:
    """Marker tokenizer; never actually invoked because _real_token_len is
    monkeypatched to _fake_real_len in these tests."""


def _guarded(row, cfg, schema_names, monkeypatch):
    monkeypatch.setattr(T, "_real_token_len", _fake_real_len)
    return T.guard_row_real_length(
        row, cfg=cfg, schema_names=schema_names, tokenizer=_FakeTok())


def _row(messages):
    names = sorted({(c.get("function") or {}).get("name", "")
                    for m in messages if m.get("role") == "assistant"
                    for c in (m.get("tool_calls") or [])})
    return {"messages": messages,
            "metadata": {"source_kind": "s", "project": "p", "rel_id": "r",
                         "window_index": 0, "window_count": 1,
                         "tool_names": names, "schema_source": "infer",
                         "n_messages": len(messages), "est_tokens": 0}}


def test_length_guard_noop_when_row_within_budget(monkeypatch):
    # a tiny row tokenizes well under budget -> returned unchanged, no rewindow.
    row = _row([
        {"role": "system", "content": "sys ## Bash"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok",
         "tool_calls": [{"function": {"name": "Bash", "arguments": {}}}]},
        {"role": "user", "content": "<tool_response>done"},
        {"role": "assistant", "content": "answer"},
    ])
    out, stats = _guarded(row, _cfg(max_seq_tokens=10_000), {"Bash"}, monkeypatch)
    assert out == [row]
    assert stats["retokenized_over"] == 0
    assert stats["rewindowed"] == 0


def test_length_guard_rewindows_oversize_multimessage_row(monkeypatch):
    # a multi-segment row that tokenizes OVER a tiny budget must be re-windowed
    # into >1 sub-rows, each within budget, each retaining its tool call + schema.
    msgs = [{"role": "system", "content": "sys ## Bash"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"q{i} " + "word " * 40})
        msgs.append({"role": "assistant", "content": f"a{i} " + "word " * 40,
                     "tool_calls": [{"function": {"name": "Bash",
                                                  "arguments": {"command": f"c{i}"}}}]})
        msgs.append({"role": "user", "content": "<tool_response> " + "word " * 40})
        msgs.append({"role": "assistant", "content": f"final{i} " + "word " * 40})
    row = _row(msgs)
    cfg = _cfg(max_seq_tokens=300, chars_per_token=4.0)
    out, stats = _guarded(row, cfg, {"Bash"}, monkeypatch)
    assert stats["retokenized_over"] == 1
    assert stats["rewindowed"] >= 1
    assert len(out) > 1
    for r in out:
        # real length of every emitted sub-row is within budget (the guarantee)
        assert _fake_real_len(r["messages"], None) <= cfg.max_seq_tokens
        # schema invariant still holds: every called tool's schema in system
        assert r["metadata"]["tool_names"], "sub-row lost its tool call"
        assert r["metadata"]["window_count"] == len(out)


def test_length_guard_drops_unsplittable_oversize_singleton(monkeypatch):
    # one giant user+assistant exchange that alone exceeds budget and cannot be
    # split -> dropped (fail-closed: never ship an over-budget row), counted as
    # such. A user turn is present so this exercises the oversize-singleton path
    # rather than the no-user-turn fail-closed drop (covered separately below).
    row = _row([
        {"role": "system", "content": "sys ## Bash"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "x " * 5000,
         "tool_calls": [{"function": {"name": "Bash", "arguments": {}}}]},
    ])
    out, stats = _guarded(row, _cfg(max_seq_tokens=100, chars_per_token=4.0),
                          {"Bash"}, monkeypatch)
    assert out == []
    assert stats["dropped_oversize_singleton"] == 1


def test_length_guard_drops_userless_row_failclosed(monkeypatch):
    # a row with NO user turn cannot be rendered by the Qwen chat template (it
    # raises "No user query found in messages") and the trainer rejects it
    # identically. The guard must drop it fail-closed BEFORE measuring length,
    # counting it as a no-user fragment (not as an oversize singleton).
    row = _row([
        {"role": "system", "content": "sys ## Bash"},
        {"role": "assistant", "content": "x " * 5000,
         "tool_calls": [{"function": {"name": "Bash", "arguments": {}}}]},
    ])
    out, stats = _guarded(row, _cfg(max_seq_tokens=100, chars_per_token=4.0),
                          {"Bash"}, monkeypatch)
    assert out == []
    assert stats["dropped_no_user_fragment"] == 1
    assert stats["dropped_oversize_singleton"] == 0


def test_public_fallback_client_exclude_ships_empty():
    """SECURITY INVARIANT (public contract): no real client/personal terms are
    baked into the public source tree. The in-tree fallback list MUST ship empty
    so the shipped repo carries no third-party names; the real terms live only in
    the gitignored ``personal_finetune`` canonical module (imported when present)
    or are supplied explicitly via config ``client_exclude``.

    Guard this going forward: if someone reintroduces literal client terms into
    the fallback, this test fails."""
    assert T._FALLBACK_CLIENT_EXCLUDE_PATTERNS == []
    # With no canonical module importable and an empty fallback, the resolved
    # default is also empty -> no content-level client exclusion is hardcoded.
    # (When the gitignored canonical module IS present it may be non-empty; we do
    # not assert on that here so the public suite is deterministic without it.)
    assert isinstance(T.DEFAULT_CLIENT_EXCLUDE_PATTERNS, list)


def test_client_exclude_override_is_honored():
    """Override mechanism: when an explicit ``client_exclude`` source is provided
    via config, the resolved patterns reflect it (and actually drive the
    content-level scan). Uses a synthetic, non-real placeholder term so no real
    client names appear in the public source/tests."""
    placeholder = "synthetic-placeholder-client"
    cfg = _cfg(client_exclude=[placeholder])
    # config carries the override verbatim
    assert cfg.client_exclude == [placeholder]
    # and it compiles into a usable matcher that flags conversations mentioning it
    client_re = T._compile_client_re(cfg.client_exclude)
    assert client_re is not None
    events = [{"role": "user", "text": f"please contact {placeholder} today"}]
    assert T._events_have_client_term(events, client_re) is True
    # an unrelated conversation is not flagged
    clean = [{"role": "user", "text": "no third-party names here"}]
    assert T._events_have_client_term(clean, client_re) is False


def test_canonical_client_exclude_override_when_present():
    """If the gitignored canonical ``personal_finetune`` module IS present in the
    checkout, the resolved default prefers it (SINGLE SOURCE OF TRUTH). Skipped
    when the module is absent so the public suite stays green without it."""
    import sys as _sys
    from pathlib import Path as _Path
    pf = _Path(__file__).resolve().parents[2] / "personal_finetune" / "scripts"
    if not (pf / "client_exclude.py").exists():
        import pytest
        pytest.skip("canonical client_exclude.py not present in this checkout")
    _sys.path.insert(0, str(pf))
    from client_exclude import _CLIENT_TERMS  # noqa: E402
    # the canonical patterns resolver prefers the canonical import when available
    assert T._canonical_client_patterns() == list(_CLIENT_TERMS)
