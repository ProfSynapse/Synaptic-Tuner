import json
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tuner.project.errors import RepositoryUrlError, SourceLockError
from tuner.project.secrets import SecretRef
from tuner.project.source_bundle import (
    GitSource,
    RepositoryLocation,
    SourceLock,
    inspect_git_source,
    resolve_relative_repository_url,
)


COMMIT = "a" * 40
PROJECT_COMMIT = "b" * 40


def source(url: str, commit: str, **kwargs: object) -> GitSource:
    return GitSource(location=RepositoryLocation.parse(url), commit=commit, **kwargs)


@pytest.mark.parametrize(
    "url",
    [
        "https://token@github.com/org/repo.git",
        "https://github.com/org/repo.git?token=secret",
        "https://github.com/org/repo.git#secret",
        "ssh://user:password@github.com/org/repo.git",
        "file:///tmp/repo",
        "ext::sh -c token-value",
        "git@github.com:org/repo.git;touch-pwned",
    ],
)
def test_repository_urls_reject_credential_bearing_or_local_transports(url: str) -> None:
    with pytest.raises(RepositoryUrlError) as error:
        RepositoryLocation.parse(url)
    assert "secret" not in str(error.value).lower()


def test_repository_url_canonicalization_is_deterministic() -> None:
    https = RepositoryLocation.parse("HTTPS://GitHub.COM:443/org/repo.git")
    ssh = RepositoryLocation.parse("git@GitHub.COM:org/repo.git")
    assert https.canonical_url == "https://github.com/org/repo.git"
    assert ssh.canonical_url == "ssh://git@github.com/org/repo.git"
    assert resolve_relative_repository_url("../engine.git", https).canonical_url == (
        "https://github.com/engine.git"
    )


def test_superproject_lock_requires_engine_commit_to_match_gitlink() -> None:
    with pytest.raises(SourceLockError, match="gitlink"):
        SourceLock(
            run_id="run-1",
            mode="superproject",
            project_source=source("https://github.com/org/host.git", PROJECT_COMMIT),
            engine_source=source(
                "https://github.com/org/engine.git",
                COMMIT,
                submodule_path="vendor/engine",
                gitlink_commit="c" * 40,
            ),
            project={},
            configuration={},
        )


def test_source_lock_round_trip_validates_schema_without_credentials(tmp_path: Path) -> None:
    credential = SecretRef("env", "GITHUB_TOKEN")
    engine_location = RepositoryLocation.parse(
        "https://github.com/org/engine.git", credential=credential
    )
    lock = SourceLock(
        run_id="run-1",
        mode="superproject",
        project_source=source("https://github.com/org/host.git", PROJECT_COMMIT),
        engine_source=GitSource(
            location=engine_location,
            commit=COMMIT,
            submodule_path="vendor/engine",
            gitlink_commit=COMMIT,
        ),
        project={
            "manifest_uri": "project://synaptic.yaml",
            "manifest_sha256": "d" * 64,
            "engine_requires": ">=1,<2",
        },
        configuration={
            "resolved_uri": "tracking://runs/run-1/resolved-config.json",
            "resolved_sha256": "e" * 64,
            "documents": [],
        },
    )
    payload = lock.to_dict()
    serialized = lock.to_json()
    assert "GITHUB_TOKEN" in serialized
    assert "token@" not in serialized.lower()
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "schemas" / "synaptic-source-lock-v1.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(payload)
    assert SourceLock.from_dict(payload).to_dict() == payload


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_git_source_dirtiness_includes_untracked_but_ignores_synaptic(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "remote", "add", "origin", "https://github.com/org/repo.git")
    (repository / ".gitignore").write_text(".synaptic/\n", encoding="utf-8")
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "tracked.txt")
    _git(
        repository,
        "-c",
        "user.name=Synaptic Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )

    untracked = repository / "untracked.txt"
    untracked.write_text("source\n", encoding="utf-8")
    assert inspect_git_source(repository).dirty is True

    untracked.unlink()
    runtime = repository / ".synaptic" / "artifacts" / "result.json"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("{}\n", encoding="utf-8")
    assert inspect_git_source(repository).dirty is False
