from __future__ import annotations

import hashlib
import importlib.util
import json
import re
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
    for relative in [*tool.LOCKED_FILES.values(), tool.EXAMPLE_PIN_RELATIVE]:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return root


def _document(root: Path) -> dict:
    return json.loads((root / tool.LOCK_RELATIVE).read_text(encoding="utf-8"))


def _pin(root: Path) -> str:
    text = (root / tool.EXAMPLE_PIN_RELATIVE).read_text(encoding="utf-8")
    return re.search(r"^  expected_lock_sha256: ([0-9a-f]{64})$", text, re.MULTILINE).group(1)


def _lock_sha(root: Path) -> str:
    return hashlib.sha256((root / tool.LOCK_RELATIVE).read_bytes()).hexdigest()


def test_checked_in_modal_runtime_lock_is_current():
    assert tool.regenerate(ROOT) == 0


def test_checked_in_token_profile_example_pins_the_checked_in_lock():
    assert _pin(ROOT) == _lock_sha(ROOT)


def test_lock_refresh_updates_only_the_example_pin_and_reports_both(tmp_path, capsys):
    root = _fixture(tmp_path)
    example = root / tool.EXAMPLE_PIN_RELATIVE
    before_example = example.read_text(encoding="utf-8")
    old_pin = _pin(root)
    changed = root / tool.LOCKED_FILES["modal_worker_source"]
    changed.write_bytes(changed.read_bytes() + b"\n")

    assert tool.regenerate(root) == 3
    check_report = capsys.readouterr().err
    assert tool.LOCK_RELATIVE in check_report and tool.EXAMPLE_PIN_RELATIVE in check_report
    assert example.read_text(encoding="utf-8") == before_example

    assert tool.regenerate(root, write=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "REFRESHED"
    assert report["changed"] == [tool.LOCK_RELATIVE, tool.EXAMPLE_PIN_RELATIVE]
    new_pin = _lock_sha(root)
    assert new_pin != old_pin and _pin(root) == new_pin
    assert example.read_text(encoding="utf-8") == before_example.replace(old_pin, new_pin)
    assert tool.regenerate(root) == 0
    assert tool.regenerate(root, write=True) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "CURRENT"


def test_stale_example_alone_is_reported_and_repaired_without_touching_the_lock(tmp_path, capsys):
    root = _fixture(tmp_path)
    lock = root / tool.LOCK_RELATIVE
    original_lock = lock.read_bytes()
    example = root / tool.EXAMPLE_PIN_RELATIVE
    example.write_text(
        example.read_text(encoding="utf-8").replace(_pin(root), "0" * 64), encoding="utf-8",
    )

    assert tool.regenerate(root) == 3
    assert tool.EXAMPLE_PIN_RELATIVE in capsys.readouterr().err
    assert tool.regenerate(root, write=True) == 0
    assert json.loads(capsys.readouterr().out)["changed"] == [tool.EXAMPLE_PIN_RELATIVE]
    assert lock.read_bytes() == original_lock
    assert _pin(root) == _lock_sha(root)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text, pin: text.replace(pin, pin.upper()),
        lambda text, pin: text.replace("expected_lock_sha256", "expected_lock"),
        lambda text, pin: text + "# expected_lock_sha256: " + pin + "\n",
        lambda text, pin: text.replace("  expected_lock_sha256", "    expected_lock_sha256"),
    ],
)
def test_unrecognized_example_pin_fails_without_writing_anything(tmp_path, mutation):
    root = _fixture(tmp_path)
    changed = root / tool.LOCKED_FILES["modal_worker_source"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    lock = root / tool.LOCK_RELATIVE
    original_lock = lock.read_bytes()
    example = root / tool.EXAMPLE_PIN_RELATIVE
    example.write_text(mutation(example.read_text(encoding="utf-8"), _pin(root)), encoding="utf-8")
    original_example = example.read_bytes()

    with pytest.raises(tool.LockRegenerationError, match="EXAMPLE_PIN_INVALID"):
        tool.regenerate(root, write=True)
    assert lock.read_bytes() == original_lock and example.read_bytes() == original_example


def test_missing_example_fails_closed(tmp_path):
    root = _fixture(tmp_path)
    (root / tool.EXAMPLE_PIN_RELATIVE).unlink()
    with pytest.raises(tool.LockRegenerationError, match="FILE_MISSING"):
        tool.regenerate(root)


@pytest.mark.parametrize("member", ["deployment_wrapper", "modal_worker_ports", "modal_worker_source"])
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
    source = root / tool.LOCKED_FILES["deployment_wrapper"]
    if fault == "invalid":
        document = _document(root)
        document["locked_files"]["unexpected"] = {
            "path": tool.LOCKED_FILES["deployment_wrapper"], "sha256": "0" * 64,
        }
        lock.write_bytes(tool._canonical(document))
        original = lock.read_bytes()
    elif fault == "missing":
        source.unlink()
    elif fault == "escape":
        document = _document(root)
        document["locked_files"]["deployment_wrapper"]["path"] = "../outside.py"
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
    source = root / tool.LOCKED_FILES["deployment_wrapper"]
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
