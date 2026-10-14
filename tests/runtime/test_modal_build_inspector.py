"""Provider-free checks for the Modal build Sandbox identity guard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import sysconfig

import pytest

from tuner.runtime import modal_build_inspector as inspector
from tuner.runtime import packaged_training_worker


@pytest.fixture
def build_inputs(tmp_path, monkeypatch):
    python = {
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "executable": sys.executable,
        "executable_digest": "0" * 64,
    }
    root = str(Path(sys.executable).parent.parent)
    paths = sysconfig.get_paths(scheme="venv", vars={"base": root, "platbase": root})
    python.update({key: paths[key] for key in ("purelib", "platlib")})
    expected = {"python": python}
    path = tmp_path / "build-inputs.json"
    path.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(inspector, "_INPUT", path)
    monkeypatch.setattr(packaged_training_worker, "inspect_installed_runtime", lambda _value: {"accepted": True})
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-Exact123")
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    return expected, path


def test_inspector_accepts_sandbox_marker_without_function_marker(build_inputs, monkeypatch):
    expected, path = build_inputs
    expected["python"]["executable_digest"] = hashlib.sha256(Path(sys.executable).resolve(strict=True).read_bytes()).hexdigest()
    path.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setenv("MODAL_SANDBOX_ID", "sb-Exact123")

    report = inspector.inspect()

    assert report["image_id"] == "im-Exact123"
    assert report["measured"] == {"accepted": True}


@pytest.mark.parametrize("sandbox_id", [None, "", "sb-", "im-Exact123", "sb-abc/def", "sb-" + "a" * 65])
def test_inspector_rejects_missing_or_malformed_sandbox_marker(build_inputs, monkeypatch, sandbox_id):
    if sandbox_id is None:
        monkeypatch.delenv("MODAL_SANDBOX_ID", raising=False)
    else:
        monkeypatch.setenv("MODAL_SANDBOX_ID", sandbox_id)
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")

    with pytest.raises(inspector._InspectionFailure, match="MODAL_BUILD_INSPECTION_REJECTED") as caught:
        inspector.inspect()
    assert caught.value.phase == "RUNTIME_IDENTITY"


def test_inspector_still_requires_image_marker(build_inputs, monkeypatch):
    monkeypatch.setenv("MODAL_SANDBOX_ID", "sb-Exact123")
    monkeypatch.delenv("MODAL_IMAGE_ID")

    with pytest.raises(inspector._InspectionFailure, match="MODAL_BUILD_INSPECTION_REJECTED") as caught:
        inspector.inspect()
    assert caught.value.phase == "RUNTIME_IDENTITY"


@pytest.mark.parametrize("stage", ["PROVENANCE", "CLOSURE", "INVENTORY_ROOT_UNPROVEN"])
def test_main_emits_only_closed_nested_inventory_predicate(build_inputs, monkeypatch, capsys, stage):
    expected, path = build_inputs
    expected["python"]["executable_digest"] = hashlib.sha256(Path(sys.executable).resolve(strict=True).read_bytes()).hexdigest()
    path.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setenv("MODAL_SANDBOX_ID", "sb-Exact123")
    def reject(_expected):
        raise packaged_training_worker.PackagedInstalledRuntimeInspectionError(stage)
    monkeypatch.setattr(packaged_training_worker, "inspect_installed_runtime", reject)
    assert inspector.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "schema_version": "synaptic-modal-build-inspection-failure/v1",
        "phase": "INSTALLED_INVENTORY", "inventory_stage": stage,
    }


def test_main_does_not_emit_failure_exception_text(build_inputs, monkeypatch, capsys):
    _, path = build_inputs
    path.write_text("not-json-private-sentinel", encoding="utf-8")
    assert inspector.main() == 2
    captured = capsys.readouterr()
    assert "private-sentinel" not in captured.out + captured.err
    assert json.loads(captured.out)["phase"] == "INPUT"
