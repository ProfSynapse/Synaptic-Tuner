from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
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


def test_bootstrap_can_use_matching_dependency_from_pinned_base(installed):
    expected, _retained, _packages, distributions = installed
    distributions["bootstrap"].requires = ["base-dependency>=2,<3"]
    distributions["base-dependency"] = SimpleNamespace(
        version="2.1.0", metadata={"Name": "base-dependency"}, requires=[], files=[],
    )
    measured = worker.inspect_installed_runtime(expected)
    assert {item["name"] for item in measured["installed_distributions"]["inventory"]} == {
        "synaptic-tuner", "bootstrap", "base-dependency",
    }
    distributions["base-dependency"].version = "3.0.0"
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected)
    assert rejected.value.stage == "BOOTSTRAP_DEPENDENCIES"


@pytest.mark.parametrize("mutation,stage", [
    ("missing-provenance", "PROVENANCE"), ("wrong-provenance", "PROVENANCE"),
    ("member", "MEMBERS"), ("wheel", "WHEEL_BYTES"),
    ("capability", "INPUTS"), ("bootstrap-dependency", "BOOTSTRAP_DEPENDENCIES"),
])
def test_installed_provenance_and_members_are_authenticated(installed, mutation, stage):
    expected, retained, packages, distributions = installed
    if mutation == "missing-provenance": distributions["synaptic-tuner"].files = []
    elif mutation == "wrong-provenance": (packages / "synaptic_tuner.dist-info" / "direct_url.json").write_text('{}')
    elif mutation == "member": (packages / "synaptic_tuner.py").write_bytes(b"# substituted\n")
    elif mutation == "wheel": (retained / expected["wheel"]["filename"]).write_bytes(b"substituted")
    elif mutation == "capability": (retained / "build-inputs.json").write_bytes(b"{}")
    else: distributions["bootstrap"].requires = ["unreviewed>=1"]
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected)
    assert rejected.value.stage == stage
    assert str(rejected.value) == "PACKAGED_INSTALLED_RUNTIME_INSPECTION_REJECTED"
    assert rejected.value.__cause__ is None


@pytest.mark.parametrize("kind,stage", [
    ("distribution", "DISTRIBUTION"), ("closure", "CLOSURE"),
    ("enumeration", "INVENTORY_ENUMERATION"), ("empty", "INVENTORY_BOUNDS"),
    ("duplicate", "INVENTORY_DUPLICATE"),
])
def test_installed_inspection_substages_are_closed(installed, monkeypatch, kind, stage):
    expected, _retained, _packages, distributions = installed
    if kind == "distribution":
        distributions["synaptic-tuner"].version = "0.0.0"
    elif kind == "closure":
        def broken_closure():
            raise RuntimeError("PRIVATE_SENTINEL")
        monkeypatch.setattr(worker, "load_packaged_worker_closure", broken_closure)
    elif kind == "enumeration":
        def broken_enumeration():
            raise RuntimeError("PRIVATE_SENTINEL")
        monkeypatch.setattr(worker.importlib.metadata, "distributions", broken_enumeration)
    elif kind == "empty":
        monkeypatch.setattr(worker.importlib.metadata, "distributions", lambda: [])
    else:
        monkeypatch.setattr(worker.importlib.metadata, "distributions", lambda: [
            distributions["synaptic-tuner"], distributions["synaptic-tuner"]])
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected)
    assert rejected.value.stage == stage
    assert str(rejected.value) == "PACKAGED_INSTALLED_RUNTIME_INSPECTION_REJECTED"
    assert "PRIVATE_SENTINEL" not in str(rejected.value)


def _path_distribution(directory, name="shared-alias", version="1.0.0"):
    directory.mkdir(parents=True)
    (directory / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return importlib.metadata.PathDistribution(directory)


def test_repeated_same_physical_metadata_preserves_inventory(installed, monkeypatch, tmp_path):
    expected, _retained, _packages, distributions = installed
    original = worker.inspect_installed_runtime(expected)["installed_distributions"]
    directory = tmp_path / "shared-alias-1.0.0.dist-info"
    first = _path_distribution(directory)
    second = importlib.metadata.PathDistribution(directory)
    monkeypatch.setattr(worker.importlib.metadata, "distributions",
                        lambda: [*distributions.values(), first, second])
    measured = worker.inspect_installed_runtime(expected)["installed_distributions"]
    assert measured["count"] == original["count"] + 1
    assert measured["inventory"].count({"name": "shared-alias", "version": "1.0.0"}) == 1
    monkeypatch.setattr(worker.importlib.metadata, "distributions",
                        lambda: [*distributions.values(), first])
    assert worker.inspect_installed_runtime(expected)["installed_distributions"] == measured


@pytest.mark.parametrize("variant,stage", [
    ("distinct", "INVENTORY_DUPLICATE"),
    ("changed", "INVENTORY_DUPLICATE"),
    ("replaced", "INVENTORY_DUPLICATE"),
    ("raw-bound", "INVENTORY_BOUNDS"),
])
def test_physical_metadata_dedup_fails_closed(installed, monkeypatch, tmp_path, variant, stage):
    expected, _retained, _packages, distributions = installed
    directory = tmp_path / "shared-alias-1.0.0.dist-info"
    first = _path_distribution(directory)
    if variant == "distinct":
        second = _path_distribution(tmp_path / "other" / "shared-alias-1.0.0.dist-info")
        found = lambda: [*distributions.values(), first, second]
    elif variant == "raw-bound":
        found = lambda: itertools.chain(distributions.values(), itertools.repeat(first, 4095))
    else:
        def found():
            yield from distributions.values()
            yield first
            if variant == "changed":
                (directory / "METADATA").write_text(
                    "Metadata-Version: 2.1\nName: shared-alias\nVersion: 2.0.0\n")
            else:
                directory.rename(tmp_path / "original-metadata")
                _path_distribution(directory)
            yield importlib.metadata.PathDistribution(directory)
    monkeypatch.setattr(worker.importlib.metadata, "distributions", found)
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected)
    assert rejected.value.stage == stage
    assert str(rejected.value) == "PACKAGED_INSTALLED_RUNTIME_INSPECTION_REJECTED"


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
