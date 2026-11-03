"""Smoke, validation and registration behavior through the public CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from synaptic_tuner.experimental.weight_editor.canon import sha256_hex

from .conftest import PKG, REPO, final, script


def test_synthetic_smoke_runs_verifies_and_exports(workdir):
    proc = subprocess.run([sys.executable, "-m", PKG, "smoke", "--workdir", str(workdir / "smoke")], cwd=REPO,
                          env=dict(os.environ, PYTHONPATH=str(REPO)), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ok"] and out["summary"]["status"] == "complete"
    assert out["effect_count"] == out["unique_dispatches"] == 4  # one effect per admitted dispatch
    assert out["summary"]["unknown_events"] == 0 and out["summary"]["capture_qualification"]["qualified"] is True
    assert out["summary"]["hidden_reasoning_captured"] is False
    # smoke never overwrites
    again = subprocess.run([sys.executable, "-m", PKG, "smoke", "--workdir", str(workdir / "smoke")], cwd=REPO,
                           env=dict(os.environ, PYTHONPATH=str(REPO)), capture_output=True, text=True)
    assert again.returncode == 1 and json.loads(again.stdout)["error"]["code"] == "run_exists"


def test_core_imports_without_torch_or_model_stack():
    code = ("import sys, synaptic_tuner.experimental.weight_editor.cli as c;"
            "print(sorted(m for m in ('torch','transformers','openai','anthropic','requests') if m in sys.modules))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)),
                          capture_output=True, text=True)
    assert proc.stdout.strip() == "[]", proc.stdout + proc.stderr


def test_validate_reports_without_effects_or_run(make):
    fx = make(script=script([final()]))
    code, out, _ = fx.cli("validate", "--config", fx.config, *fx.ports_args())
    assert code == 0 and out["ok"]
    assert out["operations"] == ["inspect_weights", "propose_patch", "query_development"]
    assert out["limits"].startswith("none configured")  # no invented default budget
    assert "not verified" in out["ports"]["isolation_note"]
    assert not fx.runs.exists() and fx.effect_lines() == []


def test_unsupported_operation_is_rejected_by_executor_port(make):
    fx = make(script=script([final()]), extra_ops={
        "analyze": {"arguments": {"tensor": {"type": "ref", "kind": "tensor"}}, "required": ["tensor"]}})
    code, out, _ = fx.cli("validate", "--config", fx.config, *fx.ports_args())
    assert code == 1 and out["error"]["code"] == "unsupported_capability"
    assert out["error"]["unsupported"] == ["analyze"]
    code, out, _ = fx.run()
    assert code == 1 and out["error"]["code"] == "unsupported_capability"
    assert not fx.runs.exists() or not list(fx.runs.iterdir())


def test_no_silent_model_fallback(make):
    fx = make(script=script([final()]))
    code, out, _ = fx.cli("validate", "--config", fx.config, *fx.ports_args(model="some-other-model"))
    assert code == 1 and out["error"]["code"] == "model_mismatch"
    code, out, _ = fx.run(model="some-other-model")
    assert code == 1 and out["error"]["code"] == "model_mismatch"
    assert fx.effect_lines() == [] and (not fx.runs.exists() or not list(fx.runs.iterdir()))


def _edit_config(fx, fn):
    cfg = json.loads(fx.config.read_text())
    fn(cfg)
    fx.config.write_text(json.dumps(cfg))


def _edit_capsule(fx, fn):
    path = fx.paths["capsule_manifest"]
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def _heldout(doc):
    doc["objects"][2]["split"] = "heldout"


def _traversal(doc):
    doc["objects"][2]["path"] = "../outside.json"


def _absolute(doc):
    doc["objects"][2]["path"] = "/etc/hostname"


def _symlink_escape(fx):
    outside = fx.paths["root"] / "outside.json"
    outside.write_text("{}")
    link = fx.capsule / "dev" / "link.json"
    link.symlink_to(outside)
    _edit_capsule(fx, lambda d: d["objects"].append(
        {"id": "dev-link", "kind": "dev_item", "path": "dev/link.json",
         "sha256": sha256_hex(b"{}"), "split": "development"}))


@pytest.mark.parametrize("mutate,code", [
    (lambda fx: _edit_capsule(fx, _heldout), "non_development_object"),
    (lambda fx: _edit_capsule(fx, _traversal), "path_traversal"),
    (lambda fx: _edit_capsule(fx, _absolute), "path_not_relative"),
    (_symlink_escape, "path_escape"),
    (lambda fx: _edit_capsule(fx, lambda d: d["objects"][2].update(sha256="0" * 64)), "capsule_hash_mismatch"),
    (lambda fx: _edit_capsule(fx, lambda d: d["objects"].append(dict(d["objects"][0]))), "capsule_invalid"),
    (lambda fx: _edit_config(fx, lambda c: c.update(policy={"budget": "unlimited"})), "config_invalid"),
    (lambda fx: _edit_config(fx, lambda c: c.update(shell="bash")), "config_invalid"),
    (lambda fx: _edit_config(fx, lambda c: c["model"].pop("model")), "config_invalid"),
    (lambda fx: _edit_config(fx, lambda c: c.update(resources={"limits": {"gpu_units": -1}})), "config_invalid"),
    (lambda fx: _edit_config(fx, lambda c: c["operations"]["query_development"]["cost"].update(
        gpu_units={"len_of": "items", "per_item": -2})), "config_invalid"),
    (lambda fx: _edit_config(fx, lambda c: c.update(operations={})), "config_invalid"),
    (lambda fx: _edit_config(fx, lambda c: c.update(output_root="capsule/out")), "output_root_overlap"),
    (lambda fx: _edit_config(fx, lambda c: c["operations"]["inspect_weights"]["arguments"].update(
        path={"type": "text", "max_length": 10})), "authority_argument_name"),
])
def test_config_and_capsule_rejections_happen_before_any_effect(make, mutate, code):
    fx = make(script=script([final()]))
    mutate(fx)
    ccode, out, _ = fx.cli("validate", "--config", fx.config, *fx.ports_args())
    assert ccode == 1 and out["error"]["code"] == code, out
    assert fx.run()[0] == 1
    assert fx.effect_lines() == [] and not (fx.runs.exists() and list(fx.runs.iterdir()))


def test_nan_and_duplicate_keys_in_config_are_not_strict_json(make):
    fx = make(script=script([final()]))
    fx.config.write_text('{"schema": "weight_editor_config/v1", "resources": {"limits": {"gpu_units": NaN}}}')
    code, out, _ = fx.cli("validate", "--config", fx.config, *fx.ports_args())
    assert code == 1 and "strict JSON" in out["error"]["message"]
    fx.config.write_text('{"schema": "a", "schema": "b"}')
    code, out, _ = fx.cli("validate", "--config", fx.config, *fx.ports_args())
    assert code == 1 and "duplicate key" in out["error"]["message"]


def test_runs_are_never_overwritten_and_run_ids_are_contained(make):
    fx = make(script=script([final()]))
    assert fx.run("r1")[0] == 0
    before = (fx.run_dir("r1") / "journal" / "events.jsonl").read_bytes()
    code, out, _ = fx.run("r1")
    assert code == 1 and out["error"]["code"] == "run_exists"
    assert (fx.run_dir("r1") / "journal" / "events.jsonl").read_bytes() == before
    for bad in ("../escape", "/abs/path", "A B", "a/b", ".hidden", ""):
        code, out, _ = fx.run(bad)
        assert code != 0 and (out is None or out["error"]["code"] in ("bad_run_id", "run_exists")), bad
    assert not (fx.paths["root"] / "escape").exists()


def test_symlinked_run_slot_cannot_redirect_outside_output_root(make):
    fx = make(script=script([final()]))
    outside = fx.paths["root"] / "outside-target"
    outside.mkdir()
    fx.runs.mkdir()
    (fx.runs / "evil").symlink_to(outside)
    code, out, _ = fx.run("evil")
    assert code == 1 and out["error"]["code"] == "run_exists"
    assert list(outside.iterdir()) == []
