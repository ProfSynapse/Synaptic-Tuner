"""Private runtime proposal and archive transport CPU contracts."""

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import tarfile
import zipfile

import pytest

from examples.training_qualification import native_runtime_preparation as preparation
from examples.training_qualification.bounded_command import BoundedCommandError


@pytest.fixture
def proposal():
    config = json.loads((preparation.BASE / "native-private-runtime.example.json").read_bytes())
    previous = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-runtime-r7-", dir=Path.home()) as name:
            parent = Path(name)
            config["private_root"] = str(parent / "new-runtime")
            path = parent / "proposal.json"
            path.write_bytes(preparation._canonical(config))
            yield config, path, parent
    finally:
        os.umask(previous)


def test_complete_wheel_selection_and_lock_are_exact(proposal, capsys):
    config, path, _ = proposal
    root, packages = preparation._config(config)
    assert root == Path(config["private_root"])
    assert len(packages) == 54
    assert len(config["wheels"]) == 54
    assert all(preparation._compatible_wheel(item["filename"], (27, 0, 1))
               for item in config["wheels"])
    assert not preparation._compatible_wheel("mlx-0.32.2-cp311-cp311-manylinux_x86_64.whl",
                                             (27, 0, 1))
    assert sum(item["size"] for item in config["wheels"]) == 146218037
    assert config["original_lock_sha256"] == preparation._ORIGINAL_SHA
    assert preparation.main(["plan", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_archive_bytes"] == 166269790
    assert result["runtime_lock_sha256"] == config["runtime_lock_sha256"]


@pytest.mark.parametrize("change", [
    lambda x: x["wheels"].pop(),
    lambda x: x["wheels"][0].update(size=x["wheels"][0]["size"] + 1),
    lambda x: x["wheels"][0].update(sha256="0" * 64),
    lambda x: x["wheels"][0].update(url="https://unreviewed.invalid/wheel.whl"),
    lambda x: x.update(uv_sha256="0" * 64),
])
def test_unreviewed_runtime_selection_rejects_before_claim(proposal, change):
    config, _, parent = proposal
    change(config)
    with pytest.raises(ValueError):
        preparation._config(config)
    assert not (parent / "new-runtime").exists()


def test_effectful_cli_rejects_wrong_host_before_private_claim(proposal, capsys, monkeypatch):
    _, path, parent = proposal
    monkeypatch.setattr(preparation.platform, "system", lambda: "Linux")
    assert preparation.main(["acquire", str(path)]) == 2
    assert capsys.readouterr().out == '{"diagnostic_code":"native_runtime_preparation_refused"}\n'
    assert not (parent / "new-runtime").exists()


class _Response(io.BytesIO):
    status = 200

    def __init__(self, raw):
        super().__init__(raw)
        self.headers = {}


def test_archive_stream_excess_byte_is_never_written(proposal, monkeypatch):
    _, _, parent = proposal
    target = parent / "wheelhouse"
    target.mkdir(mode=0o700)

    @contextmanager
    def response(_url, **_kwargs):
        with _Response(b"123456") as opened:
            yield opened

    monkeypatch.setattr(preparation, "_response", response)
    with pytest.raises(ValueError, match="transfer exceeds bound"):
        preparation._stream_archive("https://files.pythonhosted.org/packages/a.whl", target,
                                    "a.whl", 5, "0" * 64,
                                    hosts=frozenset({"files.pythonhosted.org"}), redirects=0)
    assert (target / "a.whl").read_bytes() == b""


def test_retained_archive_inventory_rejects_extra_wheel(proposal):
    config, _, parent = proposal
    root = parent / "new-runtime"
    preparation._root_layout(root)
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        item = tarfile.TarInfo("python/bin/python3.11")
        item.size = 1
        tar.addfile(item, io.BytesIO(b"p"))
    python = tar_buffer.getvalue()
    (root / "python-archive/20260211/python.tar.gz").write_bytes(python)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipped:
        zipped.writestr("package/module.py", "x=1")
    wheel = zip_buffer.getvalue()
    (root / "wheelhouse/package-1-py3-none-any.whl").write_bytes(wheel)
    config["python_archive"] = {"filename": "python.tar.gz", "size": len(python),
                                 "sha256": hashlib.sha256(python).hexdigest()}
    config["wheels"] = [{"filename": "package-1-py3-none-any.whl", "size": len(wheel),
                         "sha256": hashlib.sha256(wheel).hexdigest()}]
    config["wheel_archive_bytes"] = len(wheel)
    assert preparation._archives(config, root) == len(b"x=1")
    (root / "wheelhouse/extra.whl").write_bytes(b"extra")
    with pytest.raises(ValueError):
        preparation._archives(config, root)


def test_failed_acquisition_keeps_claimed_private_root(proposal, monkeypatch):
    config, _, parent = proposal
    calls = []

    def fail(*args, **kwargs):
        calls.append("transfer")
        raise ValueError("bounded transfer failed")

    monkeypatch.setattr(preparation, "_stream_archive", fail)
    with pytest.raises(ValueError, match="bounded transfer failed"):
        preparation._acquire(config, parent / "new-runtime")
    root = parent / "new-runtime"
    assert calls == ["transfer"] and root.is_dir() and root.stat().st_mode & 0o077 == 0
    assert (root / "python-archive/20260211").is_dir()
    with pytest.raises(FileExistsError):
        preparation._acquire(config, root)
    assert calls == ["transfer"]


def test_retained_private_tree_budget_is_explicit_postphase_guard(proposal):
    _, _, parent = proposal
    (parent / "large").write_bytes(b"x" * 64)
    with pytest.raises(ValueError, match="private tree budget"):
        preparation._private_tree_bytes(parent, 16)


def test_install_claim_precedes_archive_validation(proposal, monkeypatch):
    config, _, parent = proposal
    root = parent / "new-runtime"
    preparation._root_layout(root)
    with preparation.NativeTree(root, private=True) as tree:
        for name, raw in zip(("original.lock", "host-additions.lock", "runtime.lock"),
                             preparation._lock_bytes(), strict=True):
            tree.write_exclusive(name, raw)
        tree.write_exclusive("archive-result.json", preparation._canonical({
            "status": "OBSERVED_NOT_ADMITTED",
            "selection_sha256": hashlib.sha256(preparation._canonical(config)).hexdigest(),
            "runtime_lock_sha256": config["runtime_lock_sha256"],
        }))

    def fail(*_args):
        raise ValueError("archive bytes missing")

    monkeypatch.setattr(preparation, "_archives", fail)
    with pytest.raises(ValueError, match="archive bytes missing"):
        preparation._install(config, root, preparation._config(config)[1])
    assert json.loads((root / "install-claim.json").read_bytes())["status"] == "CONSUMED_UNCERTAIN"
    with pytest.raises(FileExistsError):
        preparation._install(config, root, preparation._config(config)[1])


def test_mutated_retained_lock_rejects_before_install_claim(proposal):
    config, _, parent = proposal
    root = parent / "new-runtime"
    preparation._root_layout(root)
    with preparation.NativeTree(root, private=True) as tree:
        for name, raw in zip(("original.lock", "host-additions.lock", "runtime.lock"),
                             preparation._lock_bytes(), strict=True):
            tree.write_exclusive(name, raw)
    (root / "runtime.lock").write_bytes(b"different")
    with pytest.raises(ValueError, match="retained lock bytes"):
        preparation._install(config, root, preparation._config(config)[1])
    assert not (root / "install-claim.json").exists()


def test_owned_command_flood_and_deadline_are_bounded(proposal):
    config, _, parent = proposal
    with pytest.raises(BoundedCommandError):
        preparation._command([sys.executable, "-c", "import sys;sys.stdout.write('x'*1048576)"],
                             parent, config)
    short = {**config, "maximum_worker_seconds": 0.05}
    with pytest.raises(BoundedCommandError):
        preparation._command([sys.executable, "-c", "import time;time.sleep(2)"], parent, short)


def _completed_install(proposal, monkeypatch):
    config, _, parent = proposal
    root = parent / "new-runtime"
    preparation._root_layout(root)
    (root / "uv-python").mkdir(mode=0o700)
    (root / "venv").mkdir(mode=0o700)
    for parent_name in preparation._UV_LOCK_PARENTS:
        lock = root / parent_name / ".lock"
        lock.write_bytes(b"")
        lock.chmod(0o666)  # Pinned uv's observed output, despite umask 077.
    uv = root / "tools/uv"
    uv.write_bytes(b"selected uv bytes")
    config["uv_sha256"] = hashlib.sha256(uv.read_bytes()).hexdigest()
    selection = hashlib.sha256(preparation._canonical(config)).hexdigest()
    expanded = 500687472
    installed = {"python_version": "3.11.14", "distribution_count": 54,
                 "inventory_sha256": "a" * 64, "executable": str(root / "uv-python/bin/python3.11"),
                 "executable_sha256": "b" * 64, "stdlib": str(root / "uv-python/lib/python3.11")}
    result = {**installed, "schema": "synaptic-native-private-runtime-install/v1",
              "status": "OBSERVED_NOT_ADMITTED", "selection_sha256": selection,
              "wheel_unpacked_bytes": expanded}
    with preparation.NativeTree(root, private=True) as tree:
        for name, raw in zip(("original.lock", "host-additions.lock", "runtime.lock"),
                             preparation._lock_bytes(), strict=True):
            tree.write_exclusive(name, raw)
        tree.write_exclusive("archive-result.json", preparation._canonical({
            "schema": "synaptic-native-private-runtime-archives/v1",
            "status": "OBSERVED_NOT_ADMITTED", "selection_sha256": selection,
            "archive_bytes": 166269790, "wheel_count": 54,
            "wheel_unpacked_bytes": expanded,
            "runtime_lock_sha256": config["runtime_lock_sha256"],
        }))
        tree.write_exclusive("install-claim.json", preparation._canonical({
            "schema": "synaptic-native-runtime-install-claim/v1",
            "status": "CONSUMED_UNCERTAIN", "selection_sha256": selection,
        }))
        tree.write_exclusive("install-result.json", preparation._canonical(result))
    monkeypatch.setattr(preparation, "_archives", lambda *_: expanded)
    monkeypatch.setattr(preparation, "_installed", lambda *_: installed)
    return config, root, preparation._canonical(result)


def test_completed_install_seals_only_uv_locks_and_preserves_prior_result(proposal, monkeypatch):
    config, root, original = _completed_install(proposal, monkeypatch)
    result = preparation._repair_uv_locks(config, root, preparation._config({
        **config, "uv_sha256": preparation._UV_SHA})[1])
    assert result["sealed_members"] == ["uv-python/.lock", "venv/.lock"]
    assert (root / "install-result.json").read_bytes() == original
    for parent in preparation._UV_LOCK_PARENTS:
        lock = root / parent / ".lock"
        assert lock.stat().st_mode & 0o777 == 0o600 and lock.stat().st_size == 0
    assert json.loads((root / "uv-lock-repair-claim.json").read_bytes())["status"] == "CONSUMED_UNCERTAIN"
    preparation._check_uv_repair_record(root, hashlib.sha256(preparation._canonical(config)).hexdigest())
    with pytest.raises(ValueError, match="slot is consumed"):
        preparation._repair_uv_locks(config, root, {})


@pytest.mark.parametrize("damage", ["symlink", "content", "hardlink", "mode"])
def test_repair_rejects_unexpected_uv_lock_without_mutation(proposal, monkeypatch, damage):
    config, root, original = _completed_install(proposal, monkeypatch)
    lock = root / "uv-python/.lock"
    if damage == "symlink":
        lock.unlink()
        lock.symlink_to(root / "venv/.lock")
    elif damage == "content":
        lock.write_bytes(b"changed")
    elif damage == "hardlink":
        os.link(lock, root / "uv-python/other")
    else:
        lock.chmod(0o644)
    with pytest.raises((ValueError, OSError)):
        preparation._repair_uv_locks(config, root, {})
    assert not (root / "uv-lock-repair-claim.json").exists()
    assert (root / "venv/.lock").stat().st_mode & 0o777 == 0o666
    assert (root / "install-result.json").read_bytes() == original


def test_repair_claim_collision_prevents_chmod(proposal, monkeypatch):
    config, root, _ = _completed_install(proposal, monkeypatch)
    (root / "uv-lock-repair-claim.json").write_bytes(b"prior claim")
    with pytest.raises(ValueError, match="slot is consumed"):
        preparation._repair_uv_locks(config, root, {})
    assert all((root / parent / ".lock").stat().st_mode & 0o777 == 0o666
               for parent in preparation._UV_LOCK_PARENTS)
    assert not (root / "uv-lock-repair-result.json").exists()


@pytest.mark.parametrize("drift", ["uv", "archive", "install_result", "selection", "inventory"])
def test_repair_rejects_prior_material_drift_before_claim(proposal, monkeypatch, drift):
    config, root, _ = _completed_install(proposal, monkeypatch)
    if drift == "uv":
        (root / "tools/uv").write_bytes(b"different uv")
    elif drift == "archive":
        (root / "archive-result.json").write_bytes(b"{}")
    elif drift == "install_result":
        (root / "install-result.json").write_bytes(b"{}")
    elif drift == "selection":
        config["maximum_worker_seconds"] -= 1
    else:
        monkeypatch.setattr(preparation, "_installed", lambda *_: {
            "python_version": "3.11.14", "distribution_count": 54,
            "inventory_sha256": "c" * 64, "executable": str(root / "uv-python/bin/python3.11"),
            "executable_sha256": "b" * 64, "stdlib": str(root / "uv-python/lib/python3.11"),
        })
    with pytest.raises(ValueError):
        preparation._repair_uv_locks(config, root, {})
    assert not (root / "uv-lock-repair-claim.json").exists()
    assert all((root / parent / ".lock").stat().st_mode & 0o777 == 0o666
               for parent in preparation._UV_LOCK_PARENTS)


def test_interrupted_repair_preserves_claim_and_partial_modes(proposal, monkeypatch):
    config, root, original = _completed_install(proposal, monkeypatch)
    real_fchmod = os.fchmod
    calls = 0

    def interrupt(fd, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        real_fchmod(fd, mode)

    monkeypatch.setattr(preparation.os, "fchmod", interrupt)
    with pytest.raises(KeyboardInterrupt):
        preparation._repair_uv_locks(config, root, {})
    assert (root / "uv-lock-repair-claim.json").exists()
    assert not (root / "uv-lock-repair-result.json").exists()
    assert (root / "uv-python/.lock").stat().st_mode & 0o777 == 0o600
    assert (root / "venv/.lock").stat().st_mode & 0o777 == 0o666
    assert (root / "install-result.json").read_bytes() == original
    with pytest.raises(ValueError, match="repair is uncertain"):
        preparation._check_uv_repair_record(root, hashlib.sha256(preparation._canonical(config)).hexdigest())
    with pytest.raises(ValueError, match="slot is consumed"):
        preparation._repair_uv_locks(config, root, {})


def test_fresh_install_seals_uv_lock_modes_before_result(proposal, monkeypatch):
    config, _, parent = proposal
    root = parent / "new-runtime"
    preparation._root_layout(root)
    uv = root / "tools/uv"
    uv.write_bytes(b"selected uv bytes")
    config["uv_sha256"] = hashlib.sha256(uv.read_bytes()).hexdigest()
    with preparation.NativeTree(root, private=True) as tree:
        for name, raw in zip(("original.lock", "host-additions.lock", "runtime.lock"),
                             preparation._lock_bytes(), strict=True):
            tree.write_exclusive(name, raw)
        tree.write_exclusive("archive-result.json", preparation._canonical({
            "status": "OBSERVED_NOT_ADMITTED",
            "selection_sha256": hashlib.sha256(preparation._canonical(config)).hexdigest(),
            "runtime_lock_sha256": config["runtime_lock_sha256"],
        }))
    monkeypatch.setattr(preparation, "_archives", lambda *_: 500687472)
    monkeypatch.setattr(preparation, "_UV_BANNER", b"selected uv banner")

    def command(argv, *_):
        if argv[-1] == "--version":
            return b"selected uv banner\n"
        if "install" in argv:
            managed = root / "uv-python" / config["python_install_key"] / "bin/python3.11"
            managed.parent.mkdir(parents=True, mode=0o700)
            managed.write_bytes(b"python")
            (root / "uv-python/.lock").write_bytes(b"")
            (root / "uv-python/.lock").chmod(0o666)
        elif "venv" in argv:
            (root / "venv").mkdir(mode=0o700)
            (root / "venv/.lock").write_bytes(b"")
            (root / "venv/.lock").chmod(0o666)
        return b""

    monkeypatch.setattr(preparation, "_command", command)
    monkeypatch.setattr(preparation, "_installed", lambda *_: {
        "python_version": "3.11.14", "distribution_count": 54,
        "inventory_sha256": "a" * 64, "executable": str(root / "uv-python/bin/python3.11"),
        "executable_sha256": "b" * 64, "stdlib": str(root / "uv-python/lib/python3.11"),
    })
    result = preparation._install(config, root, {})
    assert result["status"] == "OBSERVED_NOT_ADMITTED"
    assert all((root / parent_name / ".lock").stat().st_mode & 0o777 == 0o600
               for parent_name in preparation._UV_LOCK_PARENTS)
    assert (root / "install-result.json").exists()


def test_repair_rejects_lock_path_replacement_after_fchmod(proposal, monkeypatch):
    config, root, original = _completed_install(proposal, monkeypatch)
    real_fchmod = os.fchmod

    def replace_path(fd, mode):
        real_fchmod(fd, mode)
        lock = root / "uv-python/.lock"
        lock.rename(root / "uv-python/.lock.replaced")
        lock.write_bytes(b"")

    monkeypatch.setattr(preparation.os, "fchmod", replace_path)
    with pytest.raises(ValueError, match="sealed uv lock differs"):
        preparation._repair_uv_locks(config, root, {})
    assert (root / "uv-lock-repair-claim.json").exists()
    assert not (root / "uv-lock-repair-result.json").exists()
    assert (root / "install-result.json").read_bytes() == original


def test_repair_cli_interrupt_has_closed_diagnostic(proposal, monkeypatch, capsys):
    _, path, _ = proposal
    monkeypatch.setattr(preparation, "_host_target", lambda: None)

    def interrupt(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(preparation, "_repair_uv_locks", interrupt)
    assert preparation.main(["seal-uv-locks", str(path)]) == 2
    assert capsys.readouterr().out == '{"diagnostic_code":"native_runtime_preparation_refused"}\n'


def test_installed_custody_rejects_escaping_link(proposal):
    _, _, parent = proposal
    root = parent / "new-runtime"
    root.mkdir(mode=0o700)
    (root / "outside").symlink_to(parent)
    with pytest.raises(ValueError, match="link escapes"):
        preparation._installed_tree_custody(root, 1024)
