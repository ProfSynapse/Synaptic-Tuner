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


class _Metadata(dict):
    def get_all(self, key, default=None):
        value = self.get(key)
        return default if value is None else list(value)


def _base_with_http_extra(distributions, *, aiohttp_version="3.14.3"):
    distributions["fsspec"] = SimpleNamespace(
        version="2025.9.0",
        metadata=_Metadata({"Name": "fsspec", "Provides-Extra": ["http", "s3"]}),
        requires=[
            "aiohttp!=4.0.0a0,!=4.0.0a1; extra == \"http\"",
            "s3fs; extra == \"s3\"",
        ],
        files=[],
    )
    distributions["aiohttp"] = SimpleNamespace(
        version=aiohttp_version, metadata=_Metadata({"Name": "aiohttp"}), requires=[], files=[],
    )


def test_bootstrap_dependency_extra_is_admitted_when_its_requirements_hold(installed):
    # datasets>=4.7 (needed by trl>=1.0) requires fsspec[http]; aiohttp is in the base.
    expected, _retained, _packages, distributions = installed
    distributions["bootstrap"].requires = ["fsspec[http]<=2026.2.0,>=2023.1.0"]
    _base_with_http_extra(distributions)
    measured = worker.inspect_installed_runtime(expected)
    assert {"fsspec", "aiohttp"} <= {
        item["name"] for item in measured["installed_distributions"]["inventory"]
    }


@pytest.mark.parametrize("mutation", [
    "extra-requirement-unsatisfied", "extra-undeclared", "extra-requirement-missing",
    "extra-on-bootstrap-wheel", "extra-requirement-has-extras",
])
def test_bootstrap_dependency_extra_fails_closed(installed, mutation):
    expected, _retained, _packages, distributions = installed
    distributions["bootstrap"].requires = ["fsspec[http]<=2026.2.0,>=2023.1.0"]
    _base_with_http_extra(distributions, aiohttp_version=(
        "4.0.0a1" if mutation == "extra-requirement-unsatisfied" else "3.14.3"))
    if mutation == "extra-undeclared":
        distributions["bootstrap"].requires = ["fsspec[gcs]<=2026.2.0,>=2023.1.0"]
    elif mutation == "extra-requirement-missing":
        del distributions["aiohttp"]
    elif mutation == "extra-on-bootstrap-wheel":
        distributions["bootstrap"].requires = ["synaptic-tuner[extra]>=1"]
    elif mutation == "extra-requirement-has-extras":
        distributions["fsspec"].requires = ["aiohttp[speedups]; extra == \"http\""]
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
    ("duplicate", "INVENTORY_DUPLICATE_IDENTITY_UNPROVEN_MAIN_UNKNOWN"),
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
    ("distinct", "INVENTORY_DUPLICATE_DISTINCT_PHYSICAL_OTHER_UNKNOWN"),
    ("changed", "INVENTORY_DUPLICATE_VERSION_MISMATCH_OTHER_UNKNOWN"),
    ("replaced", "INVENTORY_DUPLICATE_DISTINCT_PHYSICAL_OTHER_UNKNOWN"),
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


def test_reviewed_package_roots_require_pinned_venv_paths(monkeypatch, tmp_path):
    prefix = tmp_path / "venv"
    package_root = prefix / "lib" / "python3.12" / "site-packages"
    package_root.mkdir(parents=True)
    expected = {"python": {
        "executable": str(prefix / "bin" / "python3"),
        "purelib": str(package_root), "platlib": str(package_root),
    }}
    monkeypatch.setattr(worker.sys, "prefix", str(prefix))
    monkeypatch.setattr(worker.sysconfig, "get_paths", lambda **_kwargs: {
        "purelib": str(package_root), "platlib": str(package_root)})
    roots = worker._reviewed_package_roots(expected)
    assert roots is not None and worker._roots_stable(roots)
    assert roots[0][1] == package_root
    expected["python"]["purelib"] = str(tmp_path / "unreviewed")
    assert worker._reviewed_package_roots(expected) is None


@pytest.mark.parametrize("package", ("MAIN", "BOOTSTRAP", "OTHER"))
@pytest.mark.parametrize("location", ("BOTH_IN", "CROSS_ROOT", "BOTH_OUT", "UNKNOWN"))
def test_distinct_metadata_pair_reports_closed_role_and_location(
    installed, monkeypatch, tmp_path, package, location,
):
    expected, _retained, _packages, _distributions = installed
    root = tmp_path / "reviewed-site"
    root.mkdir()
    outside = tmp_path / "outside-site"
    outside.mkdir()
    first_parent = root if location in ("BOTH_IN", "CROSS_ROOT") else outside
    second_parent = root if location == "BOTH_IN" else outside
    name = (expected["wheel"]["distribution"] if package == "MAIN" else
            expected["bootstrap"][0]["distribution"] if package == "BOOTSTRAP" else
            "unreviewed-package")
    first = _path_distribution(first_parent / "first.dist-info", name=name)
    second = _path_distribution(second_parent / "second.dist-info", name=name)
    info = root.stat()
    roots = ((root, root, (info.st_dev, info.st_ino)),)
    monkeypatch.setattr(worker, "_reviewed_package_roots",
                        lambda _expected: None if location == "UNKNOWN" else roots)
    monkeypatch.setattr(worker.importlib.metadata, "distributions", lambda: [first, second])
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected)
    assert rejected.value.stage == (
        "INVENTORY_DUPLICATE_DISTINCT_PHYSICAL_" + package + "_" + location)


def test_same_physical_metadata_changed_name_is_closed(installed, monkeypatch, tmp_path):
    expected, _retained, _packages, _distributions = installed
    root = tmp_path / "reviewed-site"
    root.mkdir()
    directory = root / "shared.dist-info"
    first = _path_distribution(directory, name=expected["wheel"]["distribution"])
    info = root.stat()
    monkeypatch.setattr(worker, "_reviewed_package_roots",
                        lambda _expected: ((root, root, (info.st_dev, info.st_ino)),))
    def found():
        yield first
        (directory / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: unrelated-package\nVersion: 1.0.0\n")
        yield importlib.metadata.PathDistribution(directory)
    monkeypatch.setattr(worker.importlib.metadata, "distributions", found)
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected)
    assert rejected.value.stage == (
        "INVENTORY_DUPLICATE_PHYSICAL_METADATA_MISMATCH_MAIN_BOTH_IN")


@pytest.fixture
def reviewed_root_inventory(installed, monkeypatch, tmp_path):
    expected, retained, _packages, _distributions = installed
    prefix = tmp_path / "venv"
    site = prefix / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    expected["python"].update({"executable": str(prefix / "bin" / "python3"),
                               "purelib": str(site), "platlib": str(site)})
    (retained / "build-inputs.json").write_bytes(
        (json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode())
    monkeypatch.setattr(worker.sys, "prefix", str(prefix))
    monkeypatch.setattr(worker.sysconfig, "get_paths", lambda **_kwargs: {
        "purelib": str(site), "platlib": str(site)})
    release = SimpleNamespace(
        python_implementation=expected["python"]["implementation"],
        python_version=expected["python"]["version"],
        python_executable=expected["python"]["executable"],
        python_executable_digest=expected["python"]["executable_digest"],
    )
    return expected, retained, site, release


def test_reviewed_root_inventory_matches_only_proved_metadata(reviewed_root_inventory):
    _expected, _retained, site, release = reviewed_root_inventory
    _path_distribution(site / "alpha-1.0.0.dist-info", name="alpha")
    _path_distribution(site / "beta-2.0.0.dist-info", name="beta", version="2.0.0")
    inventory = [{"name": "alpha", "version": "1.0.0"},
                 {"name": "beta", "version": "2.0.0"}]
    assert worker.inspect_reviewed_root_inventory(release) == {
        "digest": hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest(),
        "count": 2,
    }


def test_parent_scope_selects_only_authenticated_roots(reviewed_root_inventory, monkeypatch):
    expected, _retained, site, _release = reviewed_root_inventory
    _path_distribution(site / "alpha-1.0.0.dist-info", name="alpha")
    outside = _path_distribution(site.parent / "outside.dist-info", name="alpha", version="9.0.0")
    distinct = _path_distribution(site.parent / "other.dist-info", name="alpha", version="8.0.0")
    monkeypatch.setattr(worker.importlib.metadata, "distributions", lambda: [outside, distinct])
    measured = worker.inspect_installed_runtime(expected, inventory_scope="reviewed_roots")
    assert measured["installed_distributions"]["inventory"] == [{"name": "alpha", "version": "1.0.0"}]
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected)
    assert rejected.value.stage.startswith("INVENTORY_DUPLICATE_")


def test_parent_scope_rejects_unproved_or_mutated_root(reviewed_root_inventory, monkeypatch, tmp_path):
    expected, retained, site, _release = reviewed_root_inventory
    directory = site / "alpha-1.0.0.dist-info"
    _path_distribution(directory, name="alpha")
    original = worker.inspect_installed_runtime(expected, inventory_scope="reviewed_roots")
    (directory / "METADATA").write_text("Metadata-Version: 2.1\nName: alpha\nVersion: 2.0.0\n")
    changed = worker.inspect_installed_runtime(expected, inventory_scope="reviewed_roots")
    assert changed["installed_distributions"]["digest"] != original["installed_distributions"]["digest"]
    outside = _path_distribution(tmp_path / "outside" / "beta.dist-info", name="beta")
    monkeypatch.setattr(worker.importlib.metadata.MetadataPathFinder, "find_distributions",
                        lambda _context: [importlib.metadata.PathDistribution(directory), outside])
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected, inventory_scope="reviewed_roots")
    assert rejected.value.stage == "INVENTORY_ROOT_UNPROVEN"
    assert str(rejected.value) == "PACKAGED_INSTALLED_RUNTIME_INSPECTION_REJECTED"
    (retained / "build-inputs.json").write_bytes(b"{}")
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(expected, inventory_scope="reviewed_roots")
    assert rejected.value.stage == "INPUTS"


def test_inventory_scope_is_closed(installed):
    with pytest.raises(worker.PackagedInstalledRuntimeInspectionError) as rejected:
        worker.inspect_installed_runtime(installed[0], inventory_scope="unreviewed")
    assert rejected.value.stage == "INPUTS"


def test_reviewed_root_inventory_rejects_unbound_or_unproved_input(reviewed_root_inventory, monkeypatch, tmp_path):
    _expected, retained, site, release = reviewed_root_inventory
    first = _path_distribution(site / "alpha-1.0.0.dist-info", name="alpha")
    assert worker.inspect_reviewed_root_inventory(release) is not None
    unbound = SimpleNamespace(**vars(release))
    unbound.python_executable = str(tmp_path / "other-python")
    assert worker.inspect_reviewed_root_inventory(unbound) is None
    outside = _path_distribution(tmp_path / "outside" / "beta-1.0.0.dist-info", name="beta")
    monkeypatch.setattr(worker.importlib.metadata.MetadataPathFinder, "find_distributions",
                        lambda _context: [first, outside])
    assert worker.inspect_reviewed_root_inventory(release) is None
    (retained / "build-inputs.json").write_bytes(b"{}")
    assert worker.inspect_reviewed_root_inventory(release) is None


def test_reviewed_root_inventory_rejects_ambiguous_metadata(reviewed_root_inventory, monkeypatch):
    _expected, _retained, site, release = reviewed_root_inventory
    first = _path_distribution(site / "alpha-1.0.0.dist-info", name="alpha")
    def changed_name(_context):
        yield first
        (site / "alpha-1.0.0.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: beta\nVersion: 1.0.0\n")
        yield importlib.metadata.PathDistribution(site / "alpha-1.0.0.dist-info")
    monkeypatch.setattr(worker.importlib.metadata.MetadataPathFinder, "find_distributions", changed_name)
    assert worker.inspect_reviewed_root_inventory(release) is None
    monkeypatch.setattr(worker.importlib.metadata.MetadataPathFinder, "find_distributions",
                        lambda _context: itertools.repeat(first, 4097))
    assert worker.inspect_reviewed_root_inventory(release) is None


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
