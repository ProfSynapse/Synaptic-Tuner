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
