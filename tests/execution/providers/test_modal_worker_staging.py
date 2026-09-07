"""Real-file coverage for the exact prepared Modal worker boundary."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.execution.providers.modal.remote import ModalRemotePhaseError, _stage_runtime_worker
from tuner.runtime.offline_sft_worker import load_offline_sft_worker_closure


ROOT = Path(__file__).resolve().parents[3]


def checkout(tmp_path):
    manifest = ROOT / "tuner/runtime/manifests/offline-sft-worker-v1.json"
    payload = manifest.read_bytes()
    document = json.loads(payload)
    engine = tmp_path / "engine"
    for member in document["members"]:
        destination = engine / member["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / member["path"], destination)
    (engine / ".git").mkdir()
    (engine / ".git/HEAD").write_text("retained checkout\n")
    (engine / "extra.py").write_text("not a worker member\n")
    control_manifest = tmp_path / "control" / manifest.name
    control_manifest.parent.mkdir()
    control_manifest.write_bytes(payload)
    return engine, control_manifest, payload, document


def test_full_checkout_becomes_exact_worker_and_original_is_retained(tmp_path):
    engine, manifest, payload, document = checkout(tmp_path)
    source = SimpleNamespace(roots={"engine": str(engine)})
    _stage_runtime_worker(source, str(manifest), payload)
    closure = load_offline_sft_worker_closure(
        manifest, expected_digest=document["closure_digest"], engine_root=engine
    )
    assert len(closure.members) == document["member_count"]
    retained = tmp_path / "engine-source/checkout"
    assert (retained / ".git/HEAD").read_text() == "retained checkout\n"
    assert (retained / "extra.py").exists()
    assert not (engine / "extra.py").exists()
    assert not (engine / ".git").exists()


def test_retained_checkout_collision_leaves_source_untouched(tmp_path):
    engine, manifest, payload, _ = checkout(tmp_path)
    (tmp_path / "engine-source").mkdir()
    with pytest.raises(FileExistsError):
        _stage_runtime_worker(SimpleNamespace(roots={"engine": str(engine)}), str(manifest), payload)
    assert (engine / ".git/HEAD").exists()
    assert (engine / "extra.py").exists()


@pytest.mark.parametrize("mutation", ["changed", "missing", "symlink"])
def test_invalid_member_is_rejected_and_original_retained(tmp_path, mutation):
    engine, manifest, payload, document = checkout(tmp_path)
    member = engine / document["members"][0]["path"]
    if mutation == "changed":
        member.write_bytes(b"changed")
    else:
        member.unlink()
        if mutation == "symlink":
            member.symlink_to(ROOT / document["members"][0]["path"])
    with pytest.raises(ModalRemotePhaseError) as failure:
        _stage_runtime_worker(SimpleNamespace(roots={"engine": str(engine)}), str(manifest), payload)
    assert failure.value.diagnostic_code == "worker_source_copy_failed"
    assert (tmp_path / "engine-source/checkout/.git/HEAD").exists()


@pytest.mark.parametrize("aliased", ["engine", "control"])
def test_aliased_workspace_is_classified_before_moving_checkout(tmp_path, aliased):
    real = tmp_path / "real"
    real.mkdir()
    engine, manifest, payload, _ = checkout(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    selected_engine = alias / "engine" if aliased == "engine" else engine
    selected_manifest = alias / "control" / manifest.name if aliased == "control" else manifest
    with pytest.raises(ModalRemotePhaseError) as failure:
        _stage_runtime_worker(SimpleNamespace(roots={"engine": str(selected_engine)}), str(selected_manifest), payload)
    expected = "worker_source_path_noncanonical" if aliased == "engine" else "worker_control_path_noncanonical"
    assert failure.value.diagnostic_code == expected
    assert (engine / ".git/HEAD").exists()
    assert not (real / "engine-source").exists()
