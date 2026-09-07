"""Real-file coverage for the exact prepared Modal worker boundary."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.execution.providers.modal.remote import _stage_runtime_worker
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
    with pytest.raises(ValueError):
        _stage_runtime_worker(SimpleNamespace(roots={"engine": str(engine)}), str(manifest), payload)
    assert (tmp_path / "engine-source/checkout/.git/HEAD").exists()
