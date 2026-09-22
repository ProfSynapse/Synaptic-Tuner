from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

import tuner.runtime.packaged_training_worker as worker
import tuner.runtime.packaged_worker_closure as closure


@pytest.fixture
def installed(tmp_path, monkeypatch):
    retained = tmp_path / "retained"
    retained.mkdir()
    packages = tmp_path / "site-packages"
    packages.mkdir()
    wheels, distributions = [], {}
    for name in ("synaptic-tuner", "bootstrap"):
        filename = name.replace("-", "_") + "-1.0.0-py3-none-any.whl"
        module = name.replace("-", "_") + ".py"
        raw = b"# reviewed package fixture\n"
        (packages / module).write_bytes(raw)
        with ZipFile(retained / filename, "w") as archive:
            archive.writestr(module, raw)
        digest = hashlib.sha256((retained / filename).read_bytes()).hexdigest()
        wheels.append({"filename": filename, "distribution": name, "version": "1.0.0", "sha256": digest})
        directory = packages / (name.replace("-", "_") + ".dist-info")
        directory.mkdir()
        direct = directory / "direct_url.json"
        direct.write_text(json.dumps({"url": "file:///opt/synaptic-runtime/" + filename, "archive_info": {"hashes": {"sha256": digest}}}))
        distributions[name] = SimpleNamespace(version="1.0.0", metadata={"Name": name}, requires=[], files=[direct.relative_to(packages)], locate_file=lambda path: packages / path)
    expected = {"wheel": wheels[0], "bootstrap": [wheels[1]],
                "python": {"implementation": "cpython", "version": "3.12.3", "executable": "/opt/python/bin/python3.12", "executable_digest": "a" * 64},
                "capabilities": {"compatibility": {}, "contracts": {}}}
    (retained / "build-inputs.json").write_bytes((json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode())
    read = closure.stable_read
    def local_read(path, maximum=1024 * 1024):
        if str(path).replace("\\", "/").startswith("/opt/synaptic-runtime/"):
            path = retained / path.name
        return read(path, maximum)
    monkeypatch.setattr(worker, "stable_read", local_read)
    monkeypatch.setattr(worker.importlib.metadata, "distribution", lambda name: distributions[name])
    monkeypatch.setattr(worker.importlib.metadata, "distributions", lambda: list(distributions.values()))
    return expected, retained, packages, distributions


def test_inspector_runs_real_installed_closure_verification(installed):
    measured = worker.inspect_installed_runtime(installed[0])
    assert measured["closure"]["verified_digest"] == closure.load_packaged_worker_closure().digest
    assert measured["package"]["digest"] == installed[0]["wheel"]["sha256"]


@pytest.mark.parametrize("mutation", ["missing-provenance", "wrong-provenance", "member", "wheel", "capability", "bootstrap-dependency"])
def test_installed_provenance_and_members_are_authenticated(installed, mutation):
    expected, retained, packages, distributions = installed
    if mutation == "missing-provenance": distributions["synaptic-tuner"].files = []
    elif mutation == "wrong-provenance": (packages / "synaptic_tuner.dist-info" / "direct_url.json").write_text('{}')
    elif mutation == "member": (packages / "synaptic_tuner.py").write_bytes(b"# substituted\n")
    elif mutation == "wheel": (retained / expected["wheel"]["filename"]).write_bytes(b"substituted")
    elif mutation == "capability": (retained / "build-inputs.json").write_bytes(b"{}")
    else: distributions["bootstrap"].requires = ["unreviewed>=1"]
    with pytest.raises(ValueError): worker.inspect_installed_runtime(expected)


def test_closure_verifies_members_not_manifest_hash_only(tmp_path, monkeypatch):
    source = Path(closure.__file__).parent
    manifest = tmp_path / "manifests"
    manifest.mkdir()
    (manifest / "packaged-training-worker-v1.json").write_bytes((source / "manifests" / "packaged-training-worker-v1.json").read_bytes())
    for name in closure.PACKAGED_WORKER_MEMBERS:
        (tmp_path / name).write_bytes((source / name).read_bytes())
    monkeypatch.setattr(closure.importlib.resources, "files", lambda _: tmp_path)
    assert closure.load_packaged_worker_closure().members
    (tmp_path / "releases.py").write_bytes(b"changed")
    with pytest.raises(closure.PackagedWorkerClosureError): closure.load_packaged_worker_closure()


def test_resource_reader_is_bounded():
    class Oversized:
        def open(self, _mode):
            from io import BytesIO
            return BytesIO(b"x" * 9)
    with pytest.raises(ValueError): closure._resource_bytes(Oversized(), 8)
