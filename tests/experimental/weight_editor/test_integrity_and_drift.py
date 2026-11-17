"""Tamper evidence, audited tail recovery and immutable run binding."""

from __future__ import annotations

import json

import pytest

from .conftest import final, of_type, patch, req, script

TWO_STEP = None


def halted_fixture(make, **kw):
    """Run that stops after turn 2 on a provider timeout so it can be resumed."""
    fx = make(script=script([req("inspect_weights", tensor="w.layer0"), patch("p1")],
                            {"events": [], "fail": "timeout"},
                            [final("patch", "req-0002")]), **kw)
    code, out, _ = fx.run()
    assert code == 4 and out["halt"]["code"] == "adapter_failure"
    return fx


def complete_fixture(make):
    fx = make(script=script([req("inspect_weights", tensor="w.layer0"), patch("p1")], [final("patch", "req-0002")]))
    assert fx.run()[0] == 0
    return fx


def jfile(fx):
    return fx.run_dir() / "journal" / "events.jsonl"


def verify(fx, run_dir=None):
    return fx.cli("verify", "--run-dir", run_dir or fx.run_dir())


def test_clean_run_verifies(make):
    fx = complete_fixture(make)
    code, out, _ = verify(fx)
    assert code == 0 and out["ok"] and out["problems"] == []


def flip(path, needle, repl):
    data = path.read_bytes()
    assert needle in data
    path.write_bytes(data.replace(needle, repl, 1))


def _drop_line(fx):
    lines = jfile(fx).read_bytes().split(b"\n")
    del lines[5]
    jfile(fx).write_bytes(b"\n".join(lines))


def _swap_lines(fx):
    lines = jfile(fx).read_bytes().split(b"\n")
    lines[4], lines[5] = lines[5], lines[4]
    jfile(fx).write_bytes(b"\n".join(lines))


def _truncate_before_terminal(fx):
    lines = jfile(fx).read_bytes().split(b"\n")
    jfile(fx).write_bytes(b"\n".join(lines[:-2]) + b"\n")  # clean record boundary, valid prefix


def _forge_digest(fx):
    lines = jfile(fx).read_text().split("\n")
    rec = json.loads(lines[6])
    rec["data"]["tampered"] = True        # consistent JSON, digest not recomputed
    lines[6] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    jfile(fx).write_text("\n".join(lines))


@pytest.mark.parametrize("tamper,code", [
    (lambda fx: flip(jfile(fx), b"synthetic reason for p1", b"synthetic reason for p2"), "journal_corrupt"),
    (_drop_line, "journal_corrupt"),
    (_swap_lines, "journal_corrupt"),
    (_forge_digest, "journal_corrupt"),
    (lambda fx: jfile(fx).write_bytes(jfile(fx).read_bytes().replace(b",", b", ", 1)), "journal_corrupt"),  # whitespace
    (lambda fx: flip(fx.run_dir() / "provider" / "events.jsonl", b'"size":', b'"size": '), "journal_corrupt"),
    (_truncate_before_terminal, "journal_truncated_before_terminal"),
    (lambda fx: flip(fx.run_dir() / "manifest.json", b'"fake-editor-1"', b'"other-editor-1"'), "manifest_tampered"),
])
def test_chain_tampering_fails_closed(make, tamper, code):
    fx = complete_fixture(make)
    tamper(fx)
    exit_code, out, _ = verify(fx)
    assert exit_code == 5 and out["ok"] is False
    found = out["problems"][0]["code"] if "problems" in out else out["error"]["code"]
    assert found == code, out


def test_raw_event_alteration_is_detected_through_the_normalized_journal(make):
    fx = complete_fixture(make)
    raw = fx.run_dir() / "provider" / "events.jsonl"
    recs = [json.loads(x) for x in raw.read_text().splitlines()]
    import base64
    recs[0]["data"]["raw_b64"] = base64.b64encode(b'{"type":"assistant_message","text":"forged"}').decode()
    raw.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in recs) + "\n")
    assert verify(fx)[0] == 5


def test_artifact_receipt_and_final_selection_edits_are_detected(make):
    for victim in ("artifact", "receipt", "final", "terminal", "artifacts_manifest"):
        fx = complete_fixture(make)
        d = fx.run_dir()
        target = {
            "artifact": next(d.glob("artifacts/dsp-*/p1.patch.json")),
            "receipt": d / "requests" / "req-0002" / "receipt.json",
            "final": d / "final_selection.json",
            "terminal": d / "terminal_manifest.json",
            "artifacts_manifest": d / "artifacts" / "manifest.json",
        }[victim]
        target.write_bytes(target.read_bytes() + b" ")
        code, out, _ = verify(fx)
        assert code == 5 and not out["ok"], victim


def test_partial_line_fails_closed_then_audited_recovery_quarantines_it(make):
    fx = halted_fixture(make)
    jf = jfile(fx)
    good = jf.read_bytes()
    partial = b'{"data":{"half":"written"},"digest":"abc'
    jf.write_bytes(good + partial)
    code, out, _ = verify(fx)
    assert code == 5 and out["problems"][0]["code"] == "partial_tail"
    code, out, _ = fx.resume()
    assert code == 5 and out["error"]["code"] == "partial_tail"          # default: fail closed, nothing deleted
    assert jf.read_bytes() == good + partial
    code, out, _ = fx.resume(extra=["--recover-partial-tail"])
    assert code == 6 and out["status"] == "complete" and not out["capture_qualification"]["qualified"]
    rec = of_type(fx.journal(), "tail_recovery")[0]["data"]
    q = fx.run_dir() / rec["quarantined_file"]
    assert q.read_bytes() == partial                                       # preserved byte-for-byte
    assert rec["quarantined_bytes"] == len(partial)
    from synaptic_tuner.experimental.weight_editor.canon import sha256_hex
    assert rec["quarantined_sha256"] == sha256_hex(partial)
    assert verify(fx)[0] == 6
    assert "Audited tail recovery" in (fx.run_dir() / "transcript.md").read_text()


def test_corruption_is_not_hidden_by_tail_recovery(make):
    fx = halted_fixture(make)
    flip(jfile(fx), b"inspect_weights", b"inspect_weightz")
    code, out, _ = fx.resume(extra=["--recover-partial-tail"])
    assert code == 5 and out["error"]["code"] == "journal_corrupt"
    assert not (fx.run_dir() / "quarantine").exists()


# ---------------------------------------------------------------- drift


def refused_without_effects(fx, resume_result, code_name):
    code, out, _ = resume_result
    assert code in (1, 5), out
    assert out["error"]["code"] == code_name, out
    return out["error"]


def snapshot(fx):
    return (len(fx.effect_lines()), len(of_type(fx.journal(), "turn_start")), len(fx.raw()))


def test_config_drift_refuses_resume_and_is_audited(make):
    fx = halted_fixture(make)
    snap = snapshot(fx)
    cfg = json.loads(fx.config.read_text())
    cfg["final_choice"]["allow_no_edit"] = False
    fx.config.write_text(json.dumps(cfg))
    out = refused_without_effects(fx, fx.resume(), "identity_drift")
    assert "config_sha256" in out["components"]
    assert snapshot(fx) == snap
    refusal = of_type(fx.journal(), "resume_refused")[-1]["data"]
    assert refusal["reason"] == "identity_drift" and "config_sha256" in refusal["components"]


def test_whitespace_only_config_edit_still_refuses_because_bytes_are_bound(make):
    fx = halted_fixture(make)
    fx.config.write_text(fx.config.read_text() + "\n\n")
    refused_without_effects(fx, fx.resume(), "identity_drift")


def test_model_selection_drift_is_refused_never_substituted(make):
    fx = halted_fixture(make)
    snap = snapshot(fx)
    refused_without_effects(fx, fx.resume(model="other-model"), "model_mismatch")
    assert snapshot(fx) == snap
    cfg = json.loads(fx.config.read_text())
    cfg["model"]["model"] = "other-model"
    fx.config.write_text(json.dumps(cfg))
    refused_without_effects(fx, fx.resume(model="other-model"), "identity_drift")
    assert snapshot(fx) == snap


def test_model_settings_drift_is_refused(make):
    fx = halted_fixture(make)
    cfg = json.loads(fx.config.read_text())
    cfg["model"]["settings"]["effort"] = "different"
    fx.config.write_text(json.dumps(cfg))
    refused_without_effects(fx, fx.resume(), "identity_drift")


def test_capsule_byte_drift_is_refused_before_any_port_exists(make):
    fx = halted_fixture(make)
    snap = snapshot(fx)
    t = fx.capsule / "tensors" / "w.layer0.json"
    t.write_text(json.dumps({"values": [9, 9, 9]}))
    refused_without_effects(fx, fx.resume(), "capsule_hash_mismatch")
    assert snapshot(fx) == snap
    assert of_type(fx.journal(), "resume_refused")[-1]["data"]["reason"] == "config_or_capsule_invalid"


def test_capsule_manifest_drift_is_refused(make):
    fx = halted_fixture(make)
    p = fx.paths["capsule_manifest"]
    p.write_text(p.read_text() + "\n")
    refused_without_effects(fx, fx.resume(), "identity_drift")


def test_supplied_source_runtime_binding_is_part_of_identity(make, workdir):
    fx = make(script=script([req("inspect_weights", tensor="w.layer0")], {"events": [], "fail": "timeout"}, [final()]))
    b1, b2 = workdir / "b1.json", workdir / "b2.json"
    b1.write_text(json.dumps({"tuner_commit": "aaa", "runtime_lock_sha256": "1" * 64}))
    b2.write_text(json.dumps({"tuner_commit": "bbb", "runtime_lock_sha256": "1" * 64}))
    base = ["--config", fx.config, "--run-id", "r1", *fx.ports_args()]
    assert fx.cli("run", *base, "--binding-file", b1)[0] == 4
    res = fx.cli("resume", *base, "--binding-file", b2)
    assert res[0] == 1 and res[1]["error"]["code"] == "identity_drift" and res[1]["error"]["components"] == ["supplied"]
    assert fx.cli("resume", *base)[1]["error"]["code"] == "identity_drift"        # absent binding also differs
    assert fx.cli("resume", *base, "--binding-file", b1)[0] == 6


def test_port_implementation_drift_is_refused(make, workdir):
    fx = halted_fixture(make)
    script_path = fx.paths["script"]
    doc = json.loads(script_path.read_text())
    doc["turns"][2]["events"][0]["rationale"] = "changed model behavior"
    script_path.write_text(json.dumps(doc))                      # adapter identity binds the script bytes
    out = refused_without_effects(fx, fx.resume(), "identity_drift")
    assert "adapter" in out["components"]


def test_resume_under_identical_bytes_succeeds(make):
    fx = halted_fixture(make)
    code, out, _ = fx.resume()
    assert code == 6 and out["status"] == "complete" and out["resumes"] == 1
    assert len(fx.effect_lines()) == 2
