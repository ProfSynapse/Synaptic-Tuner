"""Exact raw retention, normalized journal, boundaries, receipts and independent replay."""

from __future__ import annotations

import base64
import json
import re
import shutil

import pytest

from synaptic_tuner.experimental.weight_editor.canon import sha256_hex

from .conftest import BAD_PATCH, OK_PATCH, QUERY12, final, of_type, patch, req, script, types

ODD_RAW = '{"type":"assistant_message",   "text":"spacing\\u00e9  \\n kept"}'          # odd whitespace/escapes
MALFORMED = '{"type": "assistant_message", "text": '                                     # truncated JSON
FUTURE = '{"type":"future_event","payload":{"k":[1,2,3]}}'
BINARY = b"\xff\xfe\x00 not utf8"


def test_exact_raw_events_and_unknown_events_are_retained(make):
    fx = make(script=script(
        [{"raw": ODD_RAW}, {"raw": MALFORMED}, {"raw": FUTURE}, {"raw_b64": base64.b64encode(BINARY).decode()},
         {"type": "reasoning_summary", "text": "visible summary only"}, req("inspect_weights", tensor="w.layer0")],
        [final()]))
    code, out, _ = fx.run()
    assert code == 4 and out["halt"]["code"] == "capture_incomplete"
    assert fx.effect_lines() == []
    code, out, _ = fx.resume()
    assert code == 6 and not out["capture_qualification"]["qualified"]
    raw = fx.raw()
    emitted = [ODD_RAW.encode(), MALFORMED.encode(), FUTURE.encode(), BINARY]
    assert [base64.b64decode(r["data"]["raw_b64"]) for r in raw[:4]] == emitted   # byte-exact
    assert all(r["data"]["raw_sha256"] == sha256_hex(base64.b64decode(r["data"]["raw_b64"])) for r in raw)
    events = of_type(fx.journal(), "provider_event")
    kinds = [e["data"]["kind"] for e in events]
    assert kinds[:5] == ["message", "unknown", "unknown", "unknown", "reasoning_summary"]
    for e in events[1:4]:
        assert e["data"]["capture"]["complete"] is False and e["data"]["raw"]["seq"] in (2, 3, 4)
    assert out["unknown_events"] == 3 and out["capture_incomplete"] is True
    term = json.loads((fx.run_dir() / "terminal_manifest.json").read_text())
    assert term["capture_incomplete"] is True and term["hidden_reasoning_captured"] is False
    text = (fx.run_dir() / "transcript.md").read_text()
    assert "UNKNOWN/UNPARSED EVENT" in text and FUTURE in text and "visible summary only" in text
    assert "base64] " + base64.b64encode(BINARY).decode() in text


def test_model_visible_messages_and_provider_metadata_are_recorded(make):
    fx = make(script=script(
        [{"type": "session_started", "session_id": "S1", "model": "fake-editor-1"},
         {"type": "usage", "input_tokens": 12, "output_tokens": 3},
         req("inspect_weights", tensor="w.layer0"), {"type": "turn_finished", "finish_reason": "tool_use"}],
        [final()]))
    assert fx.run()[0] == 0
    j = fx.journal()
    starts = of_type(j, "turn_start")
    assert starts[0]["data"]["messages"][0] == {"role": "system", "content": "Synthetic editor task: propose a small safe edit.\n"}
    assert starts[0]["data"]["messages"][-1]["content"].startswith("BROKER CATALOG")
    result = of_type(j, "dispatch_result")[0]["data"]["message"]
    assert starts[1]["data"]["messages"] == [result]          # exactly what the model next saw
    assert starts[1]["data"]["session_id"] == "S1"
    payloads = {e["data"]["kind"]: e["data"]["payload"] for e in of_type(j, "provider_event")}
    assert payloads["usage"]["output_tokens"] == 3 and payloads["finish"]["finish_reason"] == "tool_use"
    reg = of_type(j, "run_registered")[0]["data"]
    assert reg["model_requested"] == {"provider": "fake", "model": "fake-editor-1", "settings": {"effort": "fixture"}}


def test_returned_model_identity_violation_halts_and_discards_that_turns_requests(make):
    fx = make(script=script(
        [{"type": "session_started", "session_id": "S1", "model": "a-different-model"},
         req("inspect_weights", tensor="w.layer0")], [final()]))
    code, out, _ = fx.run()
    assert code == 4 and out["halt"]["code"] == "model_drift"
    assert fx.effect_lines() == []
    assert [r["data"]["reason"] for r in of_type(fx.journal(), "request_denied")] == ["model_drift"]


@pytest.mark.parametrize("fail,cls", [("cancelled", "cancelled"), ("timeout", "timeout"), ("provider_error", "provider_error")])
def test_adapter_failures_are_recorded_halt_and_are_not_success(make, fail, cls):
    fx = make(script=script(
        [req("inspect_weights", tensor="w.layer0")],
        {"events": [{"type": "assistant_message", "text": "partial before failure"}], "fail": fail},
        [final()]))
    code, out, _ = fx.run()
    assert code == 4 and out["status"] == "incomplete" and out["halt"]["code"] == "adapter_failure"
    assert not (fx.run_dir() / "terminal_manifest.json").exists()
    err = of_type(fx.journal(), "adapter_error")[0]["data"]
    assert err["classification"] == cls
    assert any(r["data"]["kind"] == "message" for r in of_type(fx.journal(), "provider_event"))   # partial traffic kept
    code, out, _ = fx.resume()          # explicit operator resume starts a fresh, flagged turn
    assert code == 6 and not out["capture_qualification"]["qualified"]
    assert [s["data"]["resumed"] for s in of_type(fx.journal(), "turn_start")] == [False, False, True]
    assert len(fx.effect_lines()) == 1


def test_final_assistant_text_alone_is_never_success(make):
    fx = make(script=script([{"type": "assistant_message", "text": "I am done, the edit is great"},
                             {"type": "turn_finished", "finish_reason": "stop"}]))
    code, out, _ = fx.run()
    assert code == 4 and out["status"] == "incomplete" and out["halt"]["code"] == "model_ended_without_choice"
    assert not (fx.run_dir() / "final_selection.json").exists()
    assert not (fx.run_dir() / "terminal_manifest.json").exists()
    assert fx.cli("verify", "--run-dir", fx.run_dir())[1]["status"] == "incomplete"


def test_handoff_and_compaction_capsules_are_captured_exactly(make):
    cap = {"summary": "synthetic compaction", "continuation_offset": 7, "kept": ["a", "b"]}
    fx = make(script=script(
        [{"type": "session_started", "session_id": "S1", "model": "fake-editor-1"},
         {"type": "handoff", "new_session_id": "S2", "capsule": cap},
         {"type": "truncation", "capsule": {"dropped_messages": 4}, "new_session_id": "S2"},
         req("inspect_weights", tensor="w.layer0")], [final()]))
    assert fx.run()[0] == 4
    assert fx.resume()[0] == 6
    j = fx.journal()
    hs = of_type(j, "handoff")
    assert hs[0]["data"]["capsule"] == cap and hs[0]["data"]["from_session"] == "S1" and hs[0]["data"]["to_session"] == "S2"
    assert hs[1]["data"]["kind"] == "truncation"
    starts = of_type(j, "turn_start")
    assert starts[1]["data"]["session_id"] == "S2"
    ctx = [m["content"] for m in starts[1]["data"]["messages"]]
    assert any(c.startswith("HANDOFF CAPSULE") and '"continuation_offset":7' in c for c in ctx)
    assert any(c.startswith("TRUNCATION CAPSULE") for c in ctx)


def test_valid_and_invalid_patch_receipts_remain_in_the_ledger_and_only_valid_is_selectable(make):
    fx = make(script=script(
        [patch("good"), patch("bad", BAD_PATCH), patch("wrong", json.dumps({"tensor": "w.layer1", "deltas": [0.1]})),
         patch("nan", '{"tensor": "w.layer0", "deltas": [NaN]}')],
        [final("patch", "req-0002", "choose the oversized one")],       # invalid -> must be refused
        [final("patch", "req-0001", "choose the valid one")]))
    code, out, _ = fx.run()
    assert code == 0
    statuses = {r["data"]["request_id"]: r["data"]["receipt"]["status"] for r in of_type(fx.journal(), "dispatch_result")}
    assert statuses == {"req-0001": "ok", "req-0002": "invalid", "req-0003": "invalid", "req-0004": "invalid"}
    for rid in statuses:                              # each receipt and its artifact are retained on disk
        doc = json.loads((fx.run_dir() / "requests" / rid / "receipt.json").read_text())
        assert doc["receipt"]["status"] == statuses[rid]
        art = doc["artifacts"][0]
        assert sha256_hex((fx.run_dir() / art["relpath"]).read_bytes()) == art["sha256"]
    denied = of_type(fx.journal(), "final_choice_denied")
    assert [d["data"]["reason"] for d in denied] == ["not_selectable"]
    sel = json.loads((fx.run_dir() / "final_selection.json").read_text())
    assert sel["proposal_request_id"] == "req-0001"
    frozen = of_type(fx.journal(), "final_patch_frozen")[0]
    assert sel["journal_head"] == fx.journal()[frozen["seq"] - 2]["digest"]    # record just before the freeze
    assert sel["raw_head"] == fx.raw()[-1]["digest"]
    assert sel["proposal_artifacts"][0]["sha256"] == sha256_hex(OK_PATCH.encode())
    assert sel["rationale"] == "choose the valid one"
    assert sha256_hex((fx.run_dir() / "final_selection.json").read_bytes()) == frozen["data"]["final_selection_sha256"]
    term = json.loads((fx.run_dir() / "terminal_manifest.json").read_text())
    assert term["final_selection_sha256"] == frozen["data"]["final_selection_sha256"]
    assert term["frozen_record_digest"] == frozen["digest"]


def test_final_choice_requires_visible_rationale_and_known_choice(make):
    fx = make(script=script(
        [patch("good")],
        [{"type": "final_choice", "choice": "patch", "proposal": "req-0001"}],
        [{"type": "final_choice", "choice": "patch", "proposal": "req-0001", "rationale": "ok", "command": "x"}],
        [{"type": "final_choice", "choice": "ship_it", "rationale": "ok"}],
        [{"type": "final_choice", "choice": "no_edit", "proposal": "req-0001", "rationale": "ok"}],
        [final("patch", "req-9999")],
        [final("patch", "req-0001")]))
    code, out, _ = fx.run()
    assert code == 0
    assert [d["data"]["reason"] for d in of_type(fx.journal(), "final_choice_denied")] == [
        "rationale_required", "malformed", "malformed", "not_allowed", "not_selectable"]


def test_no_discovery_after_freeze_or_evaluation(make):
    fx = make(script=script([patch("good")], [final("patch", "req-0001")], [QUERY12]))
    assert fx.run()[0] == 0
    turns = len(of_type(fx.journal(), "turn_start"))
    code, out, _ = fx.resume()                 # idempotent on a complete run; the model is not consulted
    assert code == 0 and out["status"] == "complete" and len(of_type(fx.journal(), "turn_start")) == turns
    assert fx.cli("advance", "--run-dir", fx.run_dir(), "--to", "closed", "--reference", "x")[1]["error"]["code"] == "bad_transition"
    anchor = fx.paths["root"] / "freeze-anchor.json"
    assert fx.cli("anchor", "--run-dir", fx.run_dir(), "--output", anchor)[0] == 0
    code, out, _ = fx.cli("advance", "--run-dir", fx.run_dir(), "--to", "independent_evaluation", "--reference", "eval-lane-1", "--anchor-file", anchor)
    assert code == 0 and out["phase"] == "independent_evaluation"
    code, out, _ = fx.resume()
    assert code == 1 and out["error"]["code"] == "phase_closed"
    assert fx.effect_lines().__len__() == 1 and len(of_type(fx.journal(), "turn_start")) == turns
    assert types(fx.journal())[-1] == "resume_refused"
    assert fx.cli("verify", "--run-dir", fx.run_dir())[0] == 0
    assert fx.cli("advance", "--run-dir", fx.run_dir(), "--to", "closed", "--reference", "done", "--anchor-file", anchor)[0] == 0


def test_advance_refuses_an_incomplete_run(make):
    fx = make(script=script([{"type": "assistant_message", "text": "no choice"}]))
    assert fx.run()[0] == 4
    code, out, _ = fx.cli("advance", "--run-dir", fx.run_dir(), "--to", "independent_evaluation", "--reference", "x")
    assert code == 1 and out["error"]["code"] == "bad_transition"


def test_independent_replay_links_exact_code_tool_output_and_artifacts(make, workdir):
    code_with_fences = json.dumps({"tensor": "w.layer0", "deltas": [0.1], "note": "```python\nprint(1)\n```"})
    fx = make(script=script(
        [req("inspect_weights", tensor="w.layer0"), patch("edit-a", code_with_fences)],
        [final("patch", "req-0002")]))
    assert fx.run()[0] == 0
    # Move the retained run elsewhere and delete the original inputs: replay must need nothing else.
    moved = workdir / "moved-run"
    shutil.copytree(fx.run_dir(), moved)
    shutil.rmtree(fx.paths["root"])
    code, out, _ = fx.cli("verify", "--run-dir", moved)
    assert code == 0 and out["ok"] and out["status"] == "complete"
    exported = workdir / "export.md"
    code, out, _ = fx.cli("replay", "--run-dir", moved, "--output", exported)
    assert code == 0
    text = exported.read_text()
    assert text == (moved / "transcript.md").read_text()
    # exact code fragment appears inside a fence longer than any backtick run in it
    fence = re.search(r"Argument `patch` \(exact text\):\n(`{4,})\n(.*?)\n\1\n", text, re.S)
    assert fence and fence.group(2) == code_with_fences
    # exact tool output as shown to the model
    journal = [json.loads(x) for x in (moved / "journal" / "events.jsonl").read_text().splitlines()]
    results = of_type(journal, "dispatch_result")
    for r in results:
        assert r["data"]["message"]["content"] in text
    # artifact link resolves to bytes with the printed sha256
    m = re.search(r"- Artifact \[(artifacts/dsp-[0-9a-f]+/edit-a\.patch\.json)\]\(\1\) sha256 `([0-9a-f]{64})`", text)
    assert m and sha256_hex((moved / m.group(1)).read_bytes()) == m.group(2)
    assert (moved / m.group(1)).read_text() == code_with_fences
    # receipt file linked, and raw bytes shown match the retained raw log
    assert "requests/req-0002/receipt.json" in text
    raw = [json.loads(x) for x in (moved / "provider" / "events.jsonl").read_text().splitlines()]
    assert base64.b64decode(raw[0]["data"]["raw_b64"]).decode() in text
    # the derived view never silently replaces an existing different file
    exported.write_text("changed")
    code, out, _ = fx.cli("replay", "--run-dir", moved, "--output", exported)
    assert code == 1 and out["error"]["code"] == "output_exists"
    assert fx.cli("replay", "--run-dir", moved, "--output", exported, "--force")[0] == 0
    assert exported.read_text() == text


def test_replay_reconstructs_observation_without_running_any_port(make, workdir):
    fx = make(script=script([patch("p1")], [final("patch", "req-0001")]))
    assert fx.run()[0] == 0
    effects_before = fx.effect_lines()
    for _ in range(2):
        assert fx.cli("replay", "--run-dir", fx.run_dir(), "--output", workdir / "t.md", "--force")[0] == 0
    assert fx.effect_lines() == effects_before
