"""Hash-only maintenance of the fixed packaged-worker inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.regenerate_packaged_training_worker_closure import main
from tuner.runtime.packaged_worker_closure import PACKAGED_WORKER_MEMBERS


_ROOT = Path(__file__).parents[2]


def _copy_fixture(tmp_path: Path) -> Path:
    runtime = tmp_path / "tuner" / "runtime"
    (runtime / "manifests").mkdir(parents=True)
    source = _ROOT / "tuner" / "runtime"
    for name in PACKAGED_WORKER_MEMBERS:
        (runtime / name).write_bytes((source / name).read_bytes())
    (runtime / "manifests" / "packaged-training-worker-v1.json").write_bytes(
        (source / "manifests" / "packaged-training-worker-v1.json").read_bytes()
    )
    return runtime / "manifests" / "packaged-training-worker-v1.json"


def test_default_is_read_only_and_write_changes_only_existing_hashes(tmp_path: Path) -> None:
    manifest = _copy_fixture(tmp_path)
    member = tmp_path / "tuner" / "runtime" / "releases.py"
    member.write_bytes(member.read_bytes() + b"\n# fixture content drift\n")
    original = manifest.read_bytes()
    assert main([], root=tmp_path) == 3
    assert manifest.read_bytes() == original
    assert main(["--write"], root=tmp_path) == 0
    assert main([], root=tmp_path) == 0
    refreshed = json.loads(manifest.read_text())
    previous = json.loads(original)
    assert refreshed["schema_version"] == previous["schema_version"]
    assert tuple(item["path"] for item in refreshed["members"]) == PACKAGED_WORKER_MEMBERS
    assert [set(item) for item in refreshed["members"]] == [set(item) for item in previous["members"]]
    for item in refreshed["members"]:
        raw = (tmp_path / "tuner" / "runtime" / item["path"]).read_bytes()
        assert item["size_bytes"] == len(raw)
        assert item["sha256"] == hashlib.sha256(raw).hexdigest()


def test_inventory_widening_is_rejected_even_with_self_consistent_digest(tmp_path: Path) -> None:
    manifest = _copy_fixture(tmp_path)
    document = json.loads(manifest.read_text())
    document["members"].append({"path": "unreviewed.py", "size_bytes": 1, "sha256": "a" * 64})
    unsigned = {"schema_version": document["schema_version"], "members": document["members"]}
    document["closure_digest"] = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    tampered = (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    manifest.write_bytes(tampered)
    assert main(["--write"], root=tmp_path) == 125
    assert manifest.read_bytes() == tampered
