"""Real SIGKILL at the dispatch boundary, reconciliation and no duplicate effects."""

from __future__ import annotations

import json

import pytest

from .conftest import QUERY12, final, of_type, req, script, types

GPU_LIMITS = {"gpu_units": 10, "examples": 10}


def fixture_with_query(make, **kw):
    # turn 1 issues one 2-item "GPU" query; turn 2 freezes no_edit after seeing the result
    return make(limits=GPU_LIMITS, script=script([QUERY12], [final()]), **kw)


def kill_at(fx, stage, run_id="r1", op="query_development"):
    proc = fx.popen("run", run_id, pause_at=stage, pause_op=op)
    fx.wait_marker(stage, proc)
    fx.sigkill(proc)
    return proc


def dispatch_id(fx):
    return of_type(fx.journal(), "dispatch_intent")[0]["data"]["dispatch_id"]


def test_sigkill_after_effect_before_receipt_reconciles_complete_without_duplicate(make):
    fx = fixture_with_query(make)
    kill_at(fx, "after_effect")
    assert len(fx.effect_lines()) == 1                       # the effect really happened before the kill
    code, out, _ = fx.cli("verify", "--run-dir", fx.run_dir())
    assert code == 0 and out["status"] == "incomplete" and out["unresolved_dispatches"] == [dispatch_id(fx)]
    assert not (fx.run_dir() / "terminal_manifest.json").exists()
    # the write-ahead intent and the reservation survived; no result was recorded
    assert types(fx.journal())[-1] == "dispatch_intent"
    code, out, _ = fx.resume()
    assert code == 0 and out["status"] == "complete"
    assert len(fx.effect_lines()) == 1                       # no duplicate GPU effect
    result = of_type(fx.journal(), "dispatch_result")[0]["data"]
    assert result["via"] == "reconciliation"
    assert [r["data"]["status"] for r in of_type(fx.journal(), "reconcile_result")] == ["complete"]
    gpu = out["budget"]["gpu_units"]
    assert gpu["committed"] == 1.0 and gpu["reserved_outstanding"] == 0       # charged exactly once, not lost
    assert fx.cli("verify", "--run-dir", fx.run_dir())[0] == 0
    # the model saw the real result on the next turn
    second = of_type(fx.journal(), "turn_start")[1]["data"]["messages"]
    assert json.loads(next(m["content"] for m in second if m["role"] == "tool"))["status"] == "ok"


def test_sigkill_before_executor_started_is_proven_unstarted_and_runs_exactly_once(make):
    fx = fixture_with_query(make)
    kill_at(fx, "before_started")
    assert fx.effect_lines() == []
    code, out, _ = fx.resume()
    assert code == 0 and out["status"] == "complete"
    assert len(fx.effect_lines()) == 1
    j = fx.journal()
    intents = of_type(j, "dispatch_intent")
    assert [(i["data"]["attempt"], i["data"]["basis"]) for i in intents] == [
        (1, "initial"), (2, "proven_unstarted_reconciliation")]
    assert intents[0]["data"]["dispatch_id"] == intents[1]["data"]["dispatch_id"]     # stable id across resume
    rec = of_type(j, "reconcile_result")[0]["data"]
    assert rec["status"] == "unstarted" and rec["authority"]["dispatch_id"] == dispatch_id(fx)
    assert out["budget"]["gpu_units"]["committed"] == 1.0 and out["budget"]["gpu_units"]["reserved_outstanding"] == 0


@pytest.mark.parametrize("stage,effects_after_kill", [("after_started", 0), ("before_done", 1)])
def test_sigkill_with_unknown_outcome_never_reexecutes_and_keeps_the_charge(make, stage, effects_after_kill):
    fx = fixture_with_query(make)
    kill_at(fx, stage)
    assert len(fx.effect_lines()) == effects_after_kill
    for attempt in range(3):                                  # repeated resumes stay safe
        code, out, _ = fx.resume()
        assert code == 3 and out["status"] == "incomplete" and out["halt"]["code"] == "reconciliation_needed"
        assert len(fx.effect_lines()) == effects_after_kill   # no implicit retry, ever
    assert out["dispatches"] == {dispatch_id(fx): "intent"}
    gpu = out["budget"]["gpu_units"]
    assert gpu["committed"] == 0 and gpu["reserved_outstanding"] == 1.0     # charge held, never reset or released
    assert len(of_type(fx.journal(), "dispatch_intent")) == 1
    assert [r["data"]["status"] for r in of_type(fx.journal(), "reconcile_result")] == ["unknown"] * 3
    code, out, _ = fx.cli("verify", "--run-dir", fx.run_dir())
    assert code == 0 and out["unresolved_dispatches"] == [dispatch_id(fx)] and out["status"] == "incomplete"
    assert not (fx.run_dir() / "final_selection.json").exists()
    assert "reconciliation_needed" in "".join(r["data"]["code"] for r in of_type(fx.journal(), "run_halted"))
    # reservation was never exceeded: a further query cannot spend the held units
    assert out["budget"]["gpu_units"]["charged"] == 1.0


def test_executor_exception_after_effect_is_ambiguous_and_never_retried(make):
    fx = fixture_with_query(make)
    code, out, _ = fx.run(fault="raise_after_effect")
    assert code == 3 and out["halt"]["code"] == "reconciliation_needed"
    assert len(fx.effect_lines()) == 1
    err = of_type(fx.journal(), "dispatch_error")[0]["data"]
    assert err["outcome"] == "unknown" and err["category"] == "RuntimeError"
    code, out, _ = fx.resume()                                # same executor, fault removed
    assert code == 3 and len(fx.effect_lines()) == 1          # reconcile saw started-but-not-done -> unknown
    assert out["budget"]["gpu_units"]["reserved_outstanding"] == 1.0


def test_executor_rejection_before_effect_releases_the_reservation_and_continues(make):
    fx = fixture_with_query(make)
    code, out, _ = fx.run(fault="reject_execute")
    assert code == 0 and fx.effect_lines() == []
    err = of_type(fx.journal(), "dispatch_error")[0]["data"]
    assert err["outcome"] == "not_started"
    assert out["budget"]["gpu_units"]["reserved_outstanding"] == 0 and out["budget"]["gpu_units"]["committed"] == 0
    assert out["dispatches"] == {dispatch_id(fx): "rejected"}


def test_executor_write_outside_artifact_namespace_is_refused_and_creates_nothing(make):
    fx = fixture_with_query(make)
    base_before = {p: p.read_bytes() for p in fx.capsule.rglob("*") if p.is_file()}
    code, out, _ = fx.run(fault="escape_write")
    assert code == 3                                           # boundary violation leaves outcome unknown by design
    err = of_type(fx.journal(), "dispatch_error")[0]["data"]
    assert err["boundary_violation"] is True and err["category"] == "WriteBoundaryViolation"
    assert not list(fx.paths["root"].rglob("escape.txt"))
    assert {p: p.read_bytes() for p in fx.capsule.rglob("*") if p.is_file()} == base_before   # base untouched
    assert fx.effect_lines() == []


def test_sigkill_mid_turn_discards_retained_requests_and_never_claims_full_capture(make):
    fx = make(limits=GPU_LIMITS, script=script(
        [{"type": "session_started", "session_id": "S1", "model": "fake-editor-1"}, QUERY12, {"self_sigkill": True},
         req("inspect_weights", tensor="w.layer0")],
        [final()], [final()]))
    code, out, proc = fx.run()
    assert code == -9                                          # the adapter really SIGKILLed the process mid-turn
    assert len(fx.raw()) == 2 and fx.effect_lines() == []
    assert "dispatch_intent" not in types(fx.journal())
    code, out, _ = fx.resume()
    assert code == 6 and out["status"] == "complete" and not out["capture_qualification"]["qualified"]
    j = fx.journal()
    ends = of_type(j, "turn_end")
    assert ends[0]["data"] == {"turn_id": "turn-0001", "status": "interrupted", "reason": "process_crash_or_kill"}
    assert fx.effect_lines() == []                           # interrupted requests have no authority
    assert of_type(j, "turn_start")[1]["data"]["resumed"] is True
    assert out["budget"].get("gpu_units", {}).get("committed", 0) == 0
    assert fx.cli("verify", "--run-dir", fx.run_dir())[0] == 6


def test_raw_event_fsynced_but_not_yet_normalized_is_recovered_not_lost(make):
    from synaptic_tuner.experimental.weight_editor.journal import HashChainLog, genesis_prev
    import base64, json as _json
    from synaptic_tuner.experimental.weight_editor.canon import sha256_hex

    fx = make(limits=GPU_LIMITS, script=script(
        [{"type": "session_started", "session_id": "S1", "model": "fake-editor-1"}, {"self_sigkill": True}],
        [final()], [final()]))
    assert fx.run()[0] == -9
    manifest = _json.loads((fx.run_dir() / "manifest.json").read_text())
    # simulate a crash between the raw-log fsync and the journal append of one more provider event
    raw_log = HashChainLog(fx.run_dir() / "provider" / "events.jsonl", "r1",
                           genesis_prev(manifest["binding_digest"], "provider"))
    ev = _json.dumps({"type": "tool_request", "operation": "inspect_weights", "arguments": {"tensor": "w.layer1"}}).encode()
    raw_log.append("provider_raw", {"turn_id": "turn-0001", "raw_b64": base64.b64encode(ev).decode(),
                                    "raw_sha256": sha256_hex(ev), "size": len(ev)})
    raw_log.close()
    code, out, _ = fx.resume()
    assert code == 6
    recovered = [e for e in of_type(fx.journal(), "provider_event") if e["data"]["recovered_from_raw"]]
    assert len(recovered) == 1 and recovered[0]["data"]["kind"] == "request"
    assert fx.effect_lines() == []


def test_single_writer_lock_refuses_a_second_writer_and_dies_with_the_process(make):
    fx = fixture_with_query(make)
    proc = fx.popen("run", "r1", pause_at="after_started", pause_op="query_development")
    fx.wait_marker("after_started", proc)
    code, out, _ = fx.resume()                                # second writer while the first is alive
    assert code == 1 and out["error"]["code"] == "lock_held"
    assert fx.effect_lines() == []
    fx.sigkill(proc)
    code, out, _ = fx.resume()                                # lock released by the kernel on SIGKILL
    assert code == 3 and out["halt"]["code"] == "reconciliation_needed"
