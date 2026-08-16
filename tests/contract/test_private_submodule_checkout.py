"""Contracts for locked private and nested submodule source identities."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pytest
import yaml

import tuner.cloud.checkout as checkout_module
from tuner.cloud.checkout import CheckoutPolicy, checkout_source_lock
from tuner.project.errors import (
    ManifestValidationError,
    RepositoryUrlError,
    SourceLockError,
)
from tuner.project.manifest import load_project_manifest
from tuner.project.secrets import SecretRef
from tuner.project.source_bundle import (
    GitSource,
    RepositoryLocation,
    SourceLock,
    inspect_git_source,
    resolve_relative_repository_url,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "host-project"


def test_recursive_checkout_reconstructs_locked_private_submodules(
    tmp_path: Path,
) -> None:
    fixture = _recursive_checkout_fixture(tmp_path)
    before = _source_snapshot(fixture.source_roots)

    result = checkout_source_lock(
        fixture.lock,
        tmp_path / "checkout",
        policy=_checkout_policy(),
        clone_url_resolver=fixture.resolve_clone,
        provider_secret=lambda name: (
            "private-engine-contract-secret"
            if name == fixture.credential.name
            else None
        ),
    )

    engine_root = result.project_root / fixture.engine_path
    plugin_root = engine_root / fixture.plugin_path
    assert result.engine_root == engine_root.resolve()
    assert _git("rev-parse", "HEAD", cwd=result.project_root) == fixture.host_commit
    assert _git("rev-parse", "HEAD", cwd=engine_root) == fixture.engine_commit
    assert _git("rev-parse", "HEAD", cwd=plugin_root) == fixture.plugin_commit
    assert _git("ls-tree", "HEAD", fixture.engine_path, cwd=result.project_root).split()[2] == (
        fixture.lock.engine_source.commit
    )
    assert _git("ls-tree", "HEAD", fixture.plugin_path, cwd=engine_root).split()[2] == (
        fixture.plugin_commit
    )
    assert fixture.resolve_calls == ["host", "engine", "private-plugin"]
    assert _source_snapshot(fixture.source_roots) == before


def test_scoped_credentials_are_removed_after_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fixture = _recursive_checkout_fixture(tmp_path)
    before = _source_snapshot(fixture.source_roots)
    secret = "credential-value-that-must-never-be-logged"
    helper_directories: list[Path] = []
    command_surfaces: list[tuple[str, ...]] = []
    process_environments: list[Mapping[str, str]] = []
    original_run_git = checkout_module._run_git

    def make_helper(prefix: str) -> str:
        assert prefix == "synaptic-git-credential-"
        path = tmp_path / f"ephemeral-helper-{len(helper_directories)}"
        path.mkdir()
        helper_directories.append(path)
        return str(path)

    def record_git(arguments: list[str], **kwargs: object) -> str:
        command_surfaces.append(tuple(arguments))
        environment = kwargs.get("env")
        if isinstance(environment, Mapping):
            process_environments.append(environment)
        return original_run_git(arguments, **kwargs)

    monkeypatch.setattr(checkout_module.tempfile, "mkdtemp", make_helper)
    monkeypatch.setattr(checkout_module, "_run_git", record_git)

    success = checkout_source_lock(
        fixture.lock,
        tmp_path / "success",
        policy=_checkout_policy(),
        clone_url_resolver=fixture.resolve_clone,
        provider_secret=lambda name: secret if name == fixture.credential.name else None,
    )
    assert helper_directories
    assert all(not path.exists() for path in helper_directories)
    assert any(
        argument.startswith("credential.https://example.test.helper=")
        for arguments in command_surfaces
        for argument in arguments
    )
    assert all("SYNAPTIC_GIT_SECRET" not in environment for environment in process_environments)
    assert secret not in success.source_lock.to_json()
    assert all(secret not in " ".join(arguments) for arguments in command_surfaces)
    assert secret not in caplog.text

    def missing_plugin(location: RepositoryLocation) -> str:
        if Path(location.path).stem == "private-plugin":
            return str(tmp_path / "missing-private-plugin.git")
        return fixture.resolve_clone(location)

    with pytest.raises(SourceLockError) as captured:
        checkout_source_lock(
            fixture.lock,
            tmp_path / "failure",
            policy=_checkout_policy(),
            clone_url_resolver=missing_plugin,
            provider_secret=lambda name: secret if name == fixture.credential.name else None,
        )

    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert all(not path.exists() for path in helper_directories)
    assert all("SYNAPTIC_GIT_SECRET" not in environment for environment in process_environments)
    assert all(secret not in " ".join(arguments) for arguments in command_surfaces)
    assert secret not in caplog.text
    assert _source_snapshot(fixture.source_roots) == before


@pytest.mark.parametrize(
    ("nested_url", "maximum_depth", "message"),
    [
        ("ext::sh -c exploit", 2, "Rejected .gitmodules"),
        ("https://evil.test/research/private-plugin.git", 2, "Rejected .gitmodules"),
        ("./private-plugin.git", 1, "maximum depth"),
    ],
)
def test_rejects_unexpected_nested_submodule_before_fetch(
    tmp_path: Path,
    nested_url: str,
    maximum_depth: int,
    message: str,
) -> None:
    fixture = _recursive_checkout_fixture(tmp_path, nested_url=nested_url)
    before = _source_snapshot(fixture.source_roots)
    attempted: list[str] = []

    def reject_child_fetch(location: RepositoryLocation) -> str:
        name = Path(location.path).stem
        attempted.append(name)
        if name == "private-plugin":
            pytest.fail("nested .gitmodules policy must reject before child fetch")
        return fixture.remotes[name]

    with pytest.raises(SourceLockError, match=message):
        checkout_source_lock(
            fixture.lock,
            tmp_path / "rejected",
            policy=_checkout_policy(maximum_depth=maximum_depth),
            clone_url_resolver=reject_child_fetch,
            provider_secret=lambda _name: "test-only-secret",
        )

    assert attempted == ["host", "engine"]
    assert _source_snapshot(fixture.source_roots) == before


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip()


def _repository(path: Path, remote: str, filename: str) -> str:
    path.mkdir(parents=True)
    _git("init", "--quiet", cwd=path)
    (path / filename).write_text("contract source\n", encoding="utf-8")
    _git("add", ".", cwd=path)
    _git(
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "initial",
        cwd=path,
    )
    _git("remote", "add", "origin", remote, cwd=path)
    return _git("rev-parse", "HEAD", cwd=path)


@dataclass
class _RecursiveCheckoutFixture:
    lock: SourceLock
    credential: SecretRef
    host_commit: str
    engine_commit: str
    plugin_commit: str
    engine_path: str
    plugin_path: str
    remotes: dict[str, str]
    source_roots: tuple[Path, ...]
    resolve_calls: list[str]
    resolve_clone: Callable[[RepositoryLocation], str]


def _publish_bare(source: Path, destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--bare", str(source), str(destination)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _source_snapshot(
    roots: tuple[Path, ...],
) -> tuple[tuple[str, tuple[tuple[str, bytes], ...]], ...]:
    snapshot: list[tuple[str, tuple[tuple[str, bytes], ...]]] = []
    for root in roots:
        tracked = tuple(
            (relative, (root / relative).read_bytes())
            for relative in _git("ls-files", cwd=root).splitlines()
            if (root / relative).is_file()
        )
        snapshot.append((_git("rev-parse", "HEAD", cwd=root), tracked))
    return tuple(snapshot)


def _checkout_policy(*, maximum_depth: int = 2) -> CheckoutPolicy:
    return CheckoutPolicy(
        allowed_hosts=frozenset({"example.test"}),
        allowed_schemes=frozenset({"https"}),
        nested_submodules=True,
        max_submodule_depth=maximum_depth,
    )


def _recursive_checkout_fixture(
    tmp_path: Path,
    *,
    nested_url: str = "./private-plugin.git",
) -> _RecursiveCheckoutFixture:
    plugin = tmp_path / "private plugin source"
    plugin_commit = _repository(
        plugin,
        "https://example.test/research/private-plugin.git",
        "plugin.txt",
    )
    plugin_bare = tmp_path / "private-plugin.git"
    _publish_bare(plugin, plugin_bare)

    engine = tmp_path / "private engine source"
    _repository(
        engine,
        "https://example.test/research/engine.git",
        "engine.txt",
    )
    plugin_path = "plugins/private plugin"
    _git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--name",
        "private-plugin",
        str(plugin_bare),
        plugin_path,
        cwd=engine,
    )
    _git(
        "config",
        "--file",
        str(engine / ".gitmodules"),
        "submodule.private-plugin.url",
        nested_url,
        cwd=engine,
    )
    _git("add", ".", cwd=engine)
    _git(
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "lock private plugin",
        cwd=engine,
    )
    engine_commit = _git("rev-parse", "HEAD", cwd=engine)
    engine_bare = tmp_path / "engine.git"
    _publish_bare(engine, engine_bare)

    host = tmp_path / "host source"
    _repository(
        host,
        "https://example.test/research/host.git",
        "host.txt",
    )
    engine_path = "dependencies/nonstandard engine location"
    _git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--name",
        "engine",
        str(engine_bare),
        engine_path,
        cwd=host,
    )
    _git(
        "config",
        "--file",
        str(host / ".gitmodules"),
        "submodule.engine.url",
        "https://example.test/research/engine.git",
        cwd=host,
    )
    _git("add", ".", cwd=host)
    _git(
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "lock private engine",
        cwd=host,
    )
    host_commit = _git("rev-parse", "HEAD", cwd=host)
    host_bare = tmp_path / "host.git"
    _publish_bare(host, host_bare)

    credential = SecretRef("provider_secret", "PRIVATE_GIT_TOKEN")
    project_source = GitSource(
        location=RepositoryLocation.parse("https://example.test/research/host.git"),
        commit=host_commit,
        pushed=True,
    )
    engine_source = GitSource(
        location=RepositoryLocation.parse(
            "https://example.test/research/engine.git",
            credential=credential,
        ),
        commit=engine_commit,
        pushed=True,
        submodule_path=engine_path,
        gitlink_commit=engine_commit,
    )
    lock = SourceLock(
        run_id="contract-private-recursive",
        mode="superproject",
        project_source=project_source,
        engine_source=engine_source,
        project={"id": "contract-host-project"},
        configuration={"manifest": "synaptic.yaml"},
    )
    remotes = {
        "host": str(host_bare),
        "engine": str(engine_bare),
        "private-plugin": str(plugin_bare),
    }
    resolve_calls: list[str] = []

    def resolve_clone(location: RepositoryLocation) -> str:
        name = Path(location.path).stem
        resolve_calls.append(name)
        return remotes[name]

    return _RecursiveCheckoutFixture(
        lock=lock,
        credential=credential,
        host_commit=host_commit,
        engine_commit=engine_commit,
        plugin_commit=plugin_commit,
        engine_path=engine_path,
        plugin_path=plugin_path,
        remotes=remotes,
        source_roots=(host, engine, plugin),
        resolve_calls=resolve_calls,
        resolve_clone=resolve_clone,
    )


def test_private_engine_lock_records_exact_gitlink_and_only_secret_reference(
    tmp_path: Path,
) -> None:
    host_repo = tmp_path / "host superproject"
    engine_repo = tmp_path / "private nested engine"
    host_commit = _repository(
        host_repo, "https://example.test/research/host.git", "host.txt"
    )
    engine_commit = _repository(
        engine_repo, "https://example.test/research/engine.git", "engine.txt"
    )
    credential = SecretRef(provider="provider_secret", name="PRIVATE_GIT_TOKEN")

    host_source = inspect_git_source(host_repo)
    engine_source = inspect_git_source(
        engine_repo,
        submodule_path="dependencies/nonstandard engine location",
        gitlink_commit=engine_commit,
        credential=credential,
    )
    lock = SourceLock(
        run_id="contract-private-recursive",
        mode="superproject",
        project_source=host_source,
        engine_source=engine_source,
        project={"id": "contract-host-project", "commit": host_commit},
        configuration={"manifest": "synaptic.yaml"},
    )

    serialized = lock.to_dict()
    encoded = lock.to_json()
    assert serialized["sources"]["engine"]["commit"] == engine_commit
    assert serialized["sources"]["engine"]["gitlink_commit"] == engine_commit
    assert serialized["sources"]["engine"]["submodule_path"] == (
        "dependencies/nonstandard engine location"
    )
    assert serialized["sources"]["engine"]["credential"] == credential.to_dict()
    assert "token-value" not in encoded
    assert json.loads(encoded) == serialized


def test_relative_nested_submodule_urls_stay_on_the_approved_parent_host() -> None:
    manifest = load_project_manifest(FIXTURE_ROOT / "synaptic.yaml")
    policy = manifest.policies
    parent = RepositoryLocation.parse(
        "https://example.test/research/host.git",
        allowed_hosts=set(policy["repository_hosts"]),
        allowed_schemes=set(policy["repository_schemes"]),
    )

    engine = resolve_relative_repository_url("./private-engine.git", parent)
    nested_plugin = resolve_relative_repository_url("./private-plugin.git", engine)

    for location in (engine, nested_plugin):
        approved = RepositoryLocation.parse(
            location.canonical_url,
            allowed_hosts=set(policy["repository_hosts"]),
            allowed_schemes=set(policy["repository_schemes"]),
        )
        assert approved.host == parent.host
    assert policy["nested_submodules"] is True
    assert policy["max_submodule_depth"] == 2


@pytest.mark.parametrize(
    ("maximum_depth", "valid"),
    [(0, True), (16, True), (-1, False), (17, False)],
)
def test_nested_submodule_depth_policy_enforces_schema_boundaries(
    tmp_path: Path, maximum_depth: int, valid: bool
) -> None:
    document = yaml.safe_load((FIXTURE_ROOT / "synaptic.yaml").read_text(encoding="utf-8"))
    document["policies"]["max_submodule_depth"] = maximum_depth
    manifest_path = tmp_path / f"depth-{maximum_depth}.yaml"
    manifest_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    if valid:
        assert (
            load_project_manifest(manifest_path).policies["max_submodule_depth"]
            == maximum_depth
        )
    else:
        with pytest.raises(ManifestValidationError):
            load_project_manifest(manifest_path)


@pytest.mark.parametrize(
    "nested_url",
    [
        "https://evil.test/research/plugin.git",
        "ssh://git@evil.test/research/plugin.git",
        "file:///tmp/plugin.git",
    ],
)
def test_nested_submodule_policy_rejects_host_or_transport_changes(
    nested_url: str,
) -> None:
    manifest = load_project_manifest(FIXTURE_ROOT / "synaptic.yaml")

    with pytest.raises(RepositoryUrlError):
        RepositoryLocation.parse(
            nested_url,
            allowed_hosts=set(manifest.policies["repository_hosts"]),
            allowed_schemes=set(manifest.policies["repository_schemes"]),
        )


def test_locked_engine_cannot_drift_from_the_host_gitlink() -> None:
    project = GitSource(
        location=RepositoryLocation.parse("https://example.test/research/host.git"),
        commit="1" * 40,
    )
    engine = GitSource(
        location=RepositoryLocation.parse("https://example.test/research/engine.git"),
        commit="2" * 40,
        submodule_path="vendor/engine",
        gitlink_commit="3" * 40,
    )

    with pytest.raises(SourceLockError, match="host gitlink"):
        SourceLock(
            run_id="drifted-engine",
            mode="superproject",
            project_source=project,
            engine_source=engine,
            project={},
            configuration={},
        )


def test_submodule_path_cannot_escape_the_host_checkout() -> None:
    with pytest.raises(SourceLockError, match="contained relative path"):
        GitSource(
            location=RepositoryLocation.parse(
                "https://example.test/research/engine.git"
            ),
            commit="2" * 40,
            submodule_path="../engine",
            gitlink_commit="2" * 40,
        )
