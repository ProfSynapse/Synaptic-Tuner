"""Maintenance rejects invalid metadata before touching retained manifests."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

from tuner.runtime.offline_sft_worker import _canonical_json
from tuner.runtime.packaged_grpo_worker_closure import (
    GRPO_OUTER_MEMBERS,
    GRPO_OUTER_SCHEMA,
    captured_grpo_outer_document,
)


def _tool():
    root = Path(__file__).absolute().parents[2]
    spec = importlib.util.spec_from_file_location(
        "grpo_closure_maintenance", root / "scripts/regenerate_grpo_worker_closures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document():
    value = {
        "schema_version": GRPO_OUTER_SCHEMA,
        "members": [
            {"path": name, "size_bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
            for name in GRPO_OUTER_MEMBERS
        ],
    }
    value["closure_digest"] = hashlib.sha256(_canonical_json(value)).hexdigest()
    return value


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("field,value", [("size_bytes", False), ("sha256", "not-a-digest")])
def test_invalid_previous_metadata_is_not_normalized(tmp_path, monkeypatch, write, field, value):
    tool = _tool()
    monkeypatch.setattr(tool, "_ROOT", tmp_path)
    monkeypatch.setattr(tool, "captured_grpo_outer_document", lambda root: _document())
    path = tmp_path / "tuner/runtime/manifests/packaged-grpo-worker-v1.json"
    path.parent.mkdir(parents=True)
    old = _document()
    old["members"][0][field] = value
    unsigned = {key: val for key, val in old.items() if key != "closure_digest"}
    old["closure_digest"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    raw = _canonical_json(old) + b"\n"
    path.write_bytes(raw)
    args = ["--kind", "outer"] + (["--write"] if write else [])
    assert tool.main(args) == 125
    assert path.read_bytes() == raw


def test_empty_outer_member_capture_rejects_before_initialization(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.setattr(tool, "_ROOT", tmp_path)
    for name in GRPO_OUTER_MEMBERS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (tmp_path / GRPO_OUTER_MEMBERS[0]).write_bytes(b"")
    destination = tmp_path / "tuner/runtime/manifests/packaged-grpo-worker-v1.json"
    with pytest.raises(ValueError):
        captured_grpo_outer_document(tmp_path)
    assert tool.main(["--kind", "outer", "--initialize"]) == 125
    assert not destination.exists()
