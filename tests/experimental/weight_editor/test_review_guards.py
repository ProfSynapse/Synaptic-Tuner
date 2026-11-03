"""Adversarial regression cases; operator actions still go through the real CLI."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from synaptic_tuner.experimental.weight_editor.canon import canonical_bytes, sha256_hex
from synaptic_tuner.experimental.weight_editor.journal import _digest

from .conftest import PKG, REPO, final, of_type, patch, req, script


@pytest.mark.parametrize("failure", ["provider_error", "timeout", "sigkill"])
def test_failed_turn_cannot_execute_its_patch_or_freeze_on_resume(make, failure):
    events = [patch("lost"), final("patch", "req-0001")]
    turn = {"events": events, "fail": failure}
    if failure == "sigkill":
        events.append({"self_sigkill": True})
    fx = make(script=script(turn, [final()]))
    assert fx.run()[0] == (-9 if failure == "sigkill" else 4)
    assert not (fx.run_dir() / "final_selection.json").exists()
    code, out, _ = fx.resume()
    assert code == 6 and not out["capture_qualification"]["qualified"]
    assert fx.effect_lines() == []
    assert json.loads((fx.run_dir() / "final_selection.json").read_text())["choice"] == "no_edit"
    assert of_type(fx.journal(), "final_choice_denied")[0]["data"]["reason"] in {
        "failed_turn", "interrupted_turn"}
    anchor = fx.paths["root"] / "anchor.json"
    assert fx.cli("anchor", "--run-dir", fx.run_dir(), "--output", anchor)[1]["error"]["code"] == "run_unqualified"


@pytest.mark.parametrize("event", ["usage", "turn_finished", "assistant_message"])
def test_model_drift_is_detected_in_any_normalized_event(make, event):
    fx = make(script=script([req("inspect_weights", tensor="w.layer0"),
                             {"type": event, "model": "other-model"}]))
    code, out, _ = fx.run()
    assert code == 4 and out["halt"]["code"] == "model_drift"
    assert fx.effect_lines() == []


def test_returned_identity_is_required_each_turn(make):
    fx = make(script=script([req("inspect_weights", tensor="w.layer0")],
                            {"events": [patch("unsafe"), final()], "model_identity": False}))
    code, out, _ = fx.run()
    assert code == 4 and out["halt"]["code"] == "model_identity_unverified"
    assert len(fx.effect_lines()) == 1  # only the earlier, identity-verified turn
    assert not (fx.run_dir() / "final_selection.json").exists()


def test_no_session_retry_gets_instructions_and_catalog(make):
    fx = make(script=script({"events": [], "fail": "timeout"}, [final()]))
    assert fx.run()[0] == 4
    assert fx.resume()[0] == 6
    starts = of_type(fx.journal(), "turn_start")
    assert starts[0]["data"]["messages"] == starts[1]["data"]["messages"]
    assert starts[1]["data"]["messages"][-1]["content"].startswith("BROKER CATALOG")


def test_final_choice_requires_receipt_delivery(make):
    fx = make(script=script([patch("first"), final("patch", "req-0001")],
                            [final("patch", "req-0001")]))
    assert fx.run()[0] == 0
    denials = of_type(fx.journal(), "final_choice_denied")
    assert [d["data"]["reason"] for d in denials] == ["receipt_not_seen"]
    assert len(of_type(fx.journal(), "turn_start")) == 2


def test_unmeasured_resource_limits_are_rejected(make):
    fx = make(limits={"tokens": 1}, script=script([final()]))
    code, out, _ = fx.run()
    assert code == 1 and out["error"]["code"] == "unmeasured_resource"
    assert fx.effect_lines() == []


def test_dispatch_attempt_limit_is_checked_before_redispatch(make):
    fx = make(limits={"dispatch_attempts": 1},
              script=script([req("query_development", items=["dev-001"])], [final()]))
    proc = fx.popen("run", pause_at="before_started", pause_op="query_development")
    fx.wait_marker("before_started", proc)
    fx.sigkill(proc)
    code, out, _ = fx.resume()
    assert code == 4 and out["halt"]["code"] == "budget_exhausted"
    assert out["budget"]["dispatch_attempts"]["committed"] == 1
    assert fx.effect_lines() == []


def test_executor_exception_text_is_not_model_visible(make):
    fx = make(script=script([patch("invalid", text=" ")],
                            [req("inspect_weights", tensor="w.layer0")], [final()]))
    assert fx.run(fault="reject_execute")[0] == 0
    denial = of_type(fx.journal(), "request_denied")[0]["data"]
    assert denial["detail"] == "patch must be non-empty"
    assert json.loads(denial["message"]["content"])["detail"] == "operation rejected by executor validation"
    rejection = of_type(fx.journal(), "dispatch_error")[0]["data"]
    assert json.loads(rejection["message"]["content"])["detail"] == "operation rejected by executor"


def test_package_helper_changes_refuse_resume(make, tmp_path):
    fx = make(script=script({"events": [], "fail": "timeout"}, [final()]))
    consumer = tmp_path / "consumer"
    shutil.copytree(REPO / "synaptic_tuner", consumer / "synaptic_tuner",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def call(verb):
        proc = subprocess.run([sys.executable, "-m", PKG, verb, "--config", str(fx.config),
                               "--run-id", "r1", *map(str, fx.ports_args())], cwd=consumer,
                              env=dict(os.environ, PYTHONPATH=str(consumer)), capture_output=True, text=True)
        return proc.returncode, json.loads(proc.stdout)

    assert call("run")[0] == 4
    helper = consumer / "synaptic_tuner/experimental/weight_editor/admission.py"
    helper.write_bytes(helper.read_bytes() + b"\n# changed helper\n")
    code, out = call("resume")
    assert code == 1 and out["error"]["code"] == "identity_drift"
    assert "broker_sources" in out["error"]["components"]


def _forge_rationale_and_rechain(run_dir):
    """Demonstrate the documented limit of a mutable, internally chained directory."""
    journal = run_dir / "journal/events.jsonl"
    records = [json.loads(x) for x in journal.read_text().splitlines()]
    prev = records[0]["prev"]
    frozen = None
    final_sha = None
    for rec in records:
        data = rec["data"]
        if rec["type"] == "provider_event" and data["kind"] == "final_choice":
            data["payload"]["rationale"] = "FORGED rationale"
        if rec["type"] == "final_choice_received":
            data["payload"]["rationale"] = "FORGED rationale"
        if rec["type"] == "final_patch_frozen":
            path = run_dir / "final_selection.json"
            sel = json.loads(path.read_text())
            sel.update(rationale="FORGED rationale", journal_head=prev)
            blob = canonical_bytes(sel) + b"\n"
            path.write_bytes(blob)
            final_sha = sha256_hex(blob)
            data.update(final_selection_sha256=final_sha, journal_head=prev)
        if rec["type"] == "editor_terminal":
            path = run_dir / "terminal_manifest.json"
            terminal = json.loads(path.read_text())
            terminal.update(frozen_record_digest=frozen, final_selection_sha256=final_sha)
            blob = canonical_bytes(terminal) + b"\n"
            path.write_bytes(blob)
            data["terminal_manifest_sha256"] = sha256_hex(blob)
        rec["prev"] = prev
        rec["digest"] = _digest(rec)
        if rec["type"] == "final_patch_frozen":
            frozen = rec["digest"]
        prev = rec["digest"]
    journal.write_bytes(b"".join(canonical_bytes(rec) + b"\n" for rec in records))


def test_external_anchor_catches_full_rewrite_and_blocks_evaluation(make):
    fx = make(script=script([final(rationale="ORIGINAL rationale")]))
    assert fx.run()[0] == 0
    anchor = fx.paths["root"] / "anchor.json"
    assert fx.cli("anchor", "--run-dir", fx.run_dir(), "--output", anchor)[0] == 0
    assert fx.cli("verify", "--run-dir", fx.run_dir(), "--anchor-file", anchor)[1]["external_anchor_checked"]
    _forge_rationale_and_rechain(fx.run_dir())
    # Internal consistency alone is honestly labelled; independent custody detects this rewrite.
    code, out, _ = fx.cli("verify", "--run-dir", fx.run_dir())
    assert code == 0 and "internal consistency" in out["integrity_scope"]
    code, out, _ = fx.cli("verify", "--run-dir", fx.run_dir(), "--anchor-file", anchor)
    assert code == 5 and out["problems"][0]["code"] == "external_anchor_mismatch"
    out = fx.cli("advance", "--run-dir", fx.run_dir(), "--to", "independent_evaluation",
                 "--reference", "evaluation", "--anchor-file", anchor)[1]
    assert out["error"]["code"] == "run_unqualified"


def test_advance_verifies_patch_artifacts(make):
    fx = make(script=script([patch("p")], [final("patch", "req-0001")]))
    assert fx.run()[0] == 0
    anchor = fx.paths["root"] / "anchor.json"
    assert fx.cli("anchor", "--run-dir", fx.run_dir(), "--output", anchor)[0] == 0
    artifact = next(fx.run_dir().glob("artifacts/dsp-*/*.patch.json"))
    artifact.write_text("changed patch")
    code, out, _ = fx.cli("advance", "--run-dir", fx.run_dir(), "--to", "independent_evaluation",
                          "--reference", "evaluation", "--anchor-file", anchor)
    assert code == 1 and out["error"]["code"] == "run_unqualified"
