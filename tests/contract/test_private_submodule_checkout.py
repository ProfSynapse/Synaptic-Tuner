"""Contracts for locked private and nested submodule source identities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

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


@pytest.mark.skip(
    reason=(
        "Node I activation gate: provider-neutral checkout must reconstruct the "
        "locked private engine and its approved recursive submodules"
    )
)
def test_deferred_node_i_recursive_checkout_reconstructs_locked_private_submodules() -> None:
    """Activate with Node I's real checkout API and temporary private remotes."""
    pytest.fail("Node I must replace this deferred gate with recursive checkout assertions")


@pytest.mark.skip(
    reason=(
        "Node I activation gate: ephemeral host-scoped credential helpers must "
        "exist and clean up after both successful and failed checkout"
    )
)
def test_deferred_node_i_scoped_credentials_are_removed_after_success_and_failure() -> None:
    """Activate with Node I's credential-helper lifecycle implementation."""
    pytest.fail("Node I must replace this deferred gate with cleanup assertions")


@pytest.mark.skip(
    reason=(
        "Node I activation gate: recursive .gitmodules preflight must reject an "
        "unexpected nested submodule before any fetch"
    )
)
def test_deferred_node_i_rejects_unexpected_nested_submodule_before_fetch() -> None:
    """Activate with Node I's pre-fetch recursive submodule policy API."""
    pytest.fail("Node I must replace this deferred gate with pre-fetch rejection assertions")


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
