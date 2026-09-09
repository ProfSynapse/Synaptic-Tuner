from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/regenerate_modal_runtime_lock.py"
SPEC = importlib.util.spec_from_file_location("regenerate_modal_runtime_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    lock_source = ROOT / tool.LOCK_RELATIVE
    lock_target = root / tool.LOCK_RELATIVE
    lock_target.parent.mkdir(parents=True)
    lock_target.write_bytes(lock_source.read_bytes())
    for relative in tool.LOCKED_FILES.values():
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return root


def _document(root: Path) -> dict:
    return json.loads((root / tool.LOCK_RELATIVE).read_text(encoding="utf-8"))


def test_checked_in_modal_runtime_lock_is_current():
    assert tool.regenerate(ROOT) == 0


@pytest.mark.parametrize("member", ["modal_remote", "modal_worker_ports", "modal_worker_source"])
def test_write_changes_only_expected_hash_and_is_idempotent(tmp_path, member):
    root = _fixture(tmp_path)
    before = _document(root)
    relative = tool.LOCKED_FILES[member]
    changed = root / relative
    changed.write_bytes(changed.read_bytes() + b"\n")
    original_lock = (root / tool.LOCK_RELATIVE).read_bytes()

    assert tool.regenerate(root) == 3
    assert (root / tool.LOCK_RELATIVE).read_bytes() == original_lock
    assert tool.regenerate(root, write=True) == 0
    after = _document(root)
    expected = json.loads(json.dumps(before))
    expected["locked_files"][member]["sha256"] = hashlib.sha256(
        changed.read_bytes()
    ).hexdigest()
    assert after == expected
    written = (root / tool.LOCK_RELATIVE).read_bytes()
    assert tool.regenerate(root, write=True) == 0
    assert (root / tool.LOCK_RELATIVE).read_bytes() == written


@pytest.mark.parametrize(
    "fault", ["invalid", "missing", "escape", "source_symlink", "lock_symlink"]
)
def test_invalid_inputs_fail_without_changing_lock(tmp_path, fault):
    root = _fixture(tmp_path)
    lock = root / tool.LOCK_RELATIVE
    original = lock.read_bytes()
    source = root / tool.LOCKED_FILES["modal_remote"]
    if fault == "invalid":
        document = _document(root)
        document["locked_files"]["unexpected"] = {
            "path": tool.LOCKED_FILES["modal_remote"], "sha256": "0" * 64,
        }
        lock.write_bytes(tool._canonical(document))
        original = lock.read_bytes()
    elif fault == "missing":
        source.unlink()
    elif fault == "escape":
        document = _document(root)
        document["locked_files"]["modal_remote"]["path"] = "../outside.py"
        lock.write_bytes(tool._canonical(document))
        original = lock.read_bytes()
    elif fault == "source_symlink":
        payload = source.read_bytes()
        target = root / "ordinary.py"
        target.write_bytes(payload)
        source.unlink()
        source.symlink_to(target)
    else:
        target = root / "lock-copy.json"
        target.write_bytes(original)
        lock.unlink()
        lock.symlink_to(target)

    with pytest.raises(tool.LockRegenerationError):
        tool.regenerate(root, write=True)
    if fault == "lock_symlink":
        assert lock.is_symlink() and target.read_bytes() == original
    else:
        assert lock.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sdk_version", "evil"),
        ("registry_reference", "latest"),
        ("python.executable_sha256", "not-a-sha256"),
    ],
)
def test_invalid_runtime_pins_fail_without_changing_lock(tmp_path, field, value):
    root = _fixture(tmp_path)
    lock = root / tool.LOCK_RELATIVE
    document = _document(root)
    target = document
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    lock.write_bytes(tool._canonical(document))
    original = lock.read_bytes()

    with pytest.raises(tool.LockRegenerationError, match="LOCK_POLICY_INVALID"):
        tool.regenerate(root, write=True)
    assert lock.read_bytes() == original


def test_policy_valid_nonhash_change_is_preserved(tmp_path):
    root = _fixture(tmp_path)
    lock = root / tool.LOCK_RELATIVE
    document = _document(root)
    document["registry_reference"] = "example.invalid/reviewed@sha256:" + "a" * 64
    lock.write_bytes(tool._canonical(document))
    original = lock.read_bytes()

    assert tool.regenerate(root, write=True) == 0
    assert lock.read_bytes() == original


def test_reparse_source_is_refused_without_changing_lock(tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    lock = root / tool.LOCK_RELATIVE
    original = lock.read_bytes()
    source = root / tool.LOCKED_FILES["modal_remote"]
    real_lstat = Path.lstat

    def marked_lstat(path):
        observed = real_lstat(path)
        if path == source:
            values = dict((name, getattr(observed, name)) for name in dir(observed)
                          if name.startswith("st_") and not callable(getattr(observed, name)))
            values["st_file_attributes"] = tool._REPARSE_POINT
            return type("ReparseStat", (), values)()
        return observed

    monkeypatch.setattr(Path, "lstat", marked_lstat)
    with pytest.raises(tool.LockRegenerationError, match="REPARSE_POINT_REFUSED"):
        tool.regenerate(root, write=True)
    assert lock.read_bytes() == original


def test_symlink_repository_root_is_refused_before_resolution(tmp_path):
    root = _fixture(tmp_path)
    alias = tmp_path / "repository-alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(tool.LockRegenerationError, match="ROOT_IDENTITY_INVALID"):
        tool.regenerate(alias)
