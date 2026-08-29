import json
import hashlib
import os
import subprocess
from dataclasses import replace
from types import MappingProxyType
from collections.abc import Mapping
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tuner.project.errors import RepositoryUrlError, SourceLockError
from tuner.project.secrets import SecretRef
from tuner.project.source_bundle import (
    GitSource,
    RepositoryLocation,
    SourceLock,
    SourceLockBindingV1,
    inspect_git_source,
    resolve_relative_repository_url,
)
from tuner.project import source_bundle as source_bundle_module


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


def _complete_lock() -> SourceLock:
    return SourceLock.from_dict({
        "schema_version": "synaptic-source-lock/v1",
        "run_id": "run-binding",
        "created_at": "2026-08-25T12:00:00Z",
        "mode": "superproject",
        "sources": {
            "project": {
                "url": "https://github.com/org/host.git", "commit": PROJECT_COMMIT,
                "dirty": False, "pushed": True,
            },
            "engine": {
                "url": "https://github.com/org/engine.git", "commit": COMMIT,
                "dirty": False, "pushed": True, "submodule_path": "vendor/engine",
                "gitlink_commit": COMMIT,
            },
        },
        "project": {"manifest_sha256": "1" * 64},
        "configuration": {"training_input_digest": "2" * 64},
        "plugins": [{"name": "plugin-a"}],
        "inputs": [{"name": "dataset-a"}],
        "runtime": {"python": "3.12.7"},
        "outputs": {"artifact": "final-model"},
    })


def test_source_lock_binding_is_exact_canonical_and_round_trips() -> None:
    lock = _complete_lock()
    expected = hashlib.sha256(
        b"synaptic-source-lock-binding/v1\0" + lock.canonical_bytes
    ).hexdigest()
    assert lock.canonical_bytes == lock.to_json().encode("utf-8")
    assert lock.binding == SourceLockBindingV1(
        "synaptic-source-lock-binding/v1", "synaptic-source-lock/v1", expected
    )
    assert SourceLockBindingV1.from_dict(lock.binding.to_dict()) == lock.binding
    with pytest.raises(SourceLockError):
        SourceLockBindingV1.from_dict({**lock.binding.to_dict(), "extra": True})


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "run-other"),
        ("created_at", "2026-08-25T12:00:01Z"),
        ("project", {"manifest_sha256": "3" * 64}),
        ("configuration", {"training_input_digest": "3" * 64}),
        ("plugins", ({"name": "plugin-b"},)),
        ("inputs", ({"name": "dataset-b"},)),
        ("runtime", {"python": "3.13.0"}),
        ("outputs", {"artifact": "checkpoint"}),
    ],
)
def test_every_source_lock_section_changes_binding(field: str, replacement: object) -> None:
    lock = _complete_lock()
    assert replace(lock, **{field: replacement}).binding != lock.binding


def test_source_lock_source_identities_change_binding() -> None:
    lock = _complete_lock()
    changed_project = replace(
        lock,
        project_source=replace(lock.project_source, commit="c" * 40),
    )
    changed_engine = replace(
        lock,
        engine_source=replace(
            lock.engine_source, commit="d" * 40, gitlink_commit="d" * 40
        ),
    )
    assert changed_project.binding != lock.binding
    assert changed_engine.binding != lock.binding


def test_source_lock_canonical_json_is_exact_finite_utf8() -> None:
    lock = replace(_complete_lock(), configuration={"label": "café"})
    assert b"caf\xc3\xa9" in lock.canonical_bytes
    assert b"\\u00e9" not in lock.canonical_bytes
    assert lock.canonical_bytes == lock.to_json().encode("utf-8")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "\ud800", object()])
def test_source_lock_canonicalization_totalizes_invalid_json_without_context(bad: object) -> None:
    lock = replace(_complete_lock(), configuration={"member": bad})
    with pytest.raises(SourceLockError, match="^Source lock cannot be canonicalized$") as caught:
        _ = lock.canonical_bytes
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


class _ExplodingMapping(Mapping):
    def __getitem__(self, key):
        raise RuntimeError("private-value")

    def __iter__(self):
        raise RuntimeError("private-value")

    def __len__(self):
        raise RuntimeError("private-value")


def test_source_lock_canonicalization_totalizes_pre_serialization_callbacks() -> None:
    lock = replace(_complete_lock(), configuration=_ExplodingMapping())
    with pytest.raises(SourceLockError, match="^Source lock cannot be canonicalized$") as caught:
        _ = lock.binding
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "private-value" not in str(caught.value)


def test_new_binding_and_source_lock_parsers_require_exact_builtins() -> None:
    lock = _complete_lock()
    with pytest.raises(SourceLockError):
        SourceLockBindingV1.from_dict(MappingProxyType(lock.binding.to_dict()))
    with pytest.raises(SourceLockError):
        SourceLock.from_dict(MappingProxyType(lock.to_dict()))
    class Text(str):
        pass
    class Array(list):
        pass
    text_value=lock.to_dict();text_value["configuration"]={"name":Text("value")}
    with pytest.raises(SourceLockError):
        SourceLock.from_dict(text_value)
    array_value=lock.to_dict();array_value["plugins"]=Array()
    with pytest.raises(SourceLockError):
        SourceLock.from_dict(array_value)


class _HostileFieldName(str):
    armed = False
    calls = 0

    def __hash__(self):
        if type(self).armed:
            type(self).calls += 1
            raise RuntimeError("private-field-name")
        return str.__hash__(self)

    def __eq__(self, other):
        if type(self).armed:
            type(self).calls += 1
            raise RuntimeError("private-field-name")
        return str.__eq__(self, other)


def _replace_key_with_hostile_field(value: dict[str, object], name: str) -> None:
    original = value.pop(name)
    value[_HostileFieldName(name)] = original


@pytest.mark.parametrize("family", ["binding", "git-source", "lock", "sources", "section"])
def test_source_parser_field_inventory_rejects_hostile_string_subclass_without_callbacks(
    family: str,
) -> None:
    lock = _complete_lock()
    if family == "binding":
        value = lock.binding.to_dict()
        _replace_key_with_hostile_field(value, "schema_version")
        parse = lambda: SourceLockBindingV1.from_dict(value)
    elif family == "git-source":
        value = lock.project_source.to_dict()
        _replace_key_with_hostile_field(value, "url")
        parse = lambda: GitSource.from_dict(value)
    else:
        value = lock.to_dict()
        if family == "lock":
            _replace_key_with_hostile_field(value, "schema_version")
        elif family == "sources":
            _replace_key_with_hostile_field(value["sources"], "project")
        else:
            _replace_key_with_hostile_field(value["configuration"], "training_input_digest")
        parse = lambda: SourceLock.from_dict(value)
    _HostileFieldName.calls = 0
    _HostileFieldName.armed = True
    try:
        with pytest.raises(SourceLockError) as caught:
            parse()
    finally:
        _HostileFieldName.armed = False
    assert _HostileFieldName.calls == 0
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "private-field-name" not in str(caught.value)


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repository: Path, name: str, content: str) -> str:
    (repository / name).write_text(content, encoding="utf-8")
    _git(repository, "add", name)
    _git(
        repository,
        "-c",
        "user.name=Synaptic Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        name,
    )
    return _git_output(repository, "rev-parse", "HEAD")


def _bare_fixture(tmp_path: Path) -> tuple[Path, Path]:
    bare = tmp_path / "origin.git"
    repository = tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(bare))
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "branch", "-M", "main")
    _git(repository, "remote", "add", "origin", "https://example.invalid/org/repo.git")
    _git(repository, "config", "branch.main.remote", "origin")
    _git(repository, "config", "branch.main.merge", "refs/heads/main")
    return repository, bare


def _push_to_bare(repository: Path, bare: Path) -> None:
    _git(repository, "push", str(bare), "HEAD:refs/heads/main")


def _bare_remote_probe(bare: Path):
    def probe(location: RepositoryLocation, ref: str) -> str | None:
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", str(bare), ref],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        fields = result.stdout.strip().split()
        return fields[0].lower() if len(fields) == 2 and fields[1] == ref else None

    return probe


def test_git_source_dirtiness_includes_untracked_but_ignores_synaptic(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "--local", "core.autocrlf", "false")
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


def test_pushed_requires_exact_head_at_current_origin_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, bare = _bare_fixture(tmp_path)
    head = _commit(repository, "tracked.txt", "one\n")
    _push_to_bare(repository, bare)
    monkeypatch.setattr(source_bundle_module, "_remote_ref_sha", _bare_remote_probe(bare))

    inspected = inspect_git_source(repository)
    assert inspected.commit == head
    assert inspected.branch == "main"
    assert inspected.pushed is True


def test_local_commit_ahead_of_origin_is_not_pushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, bare = _bare_fixture(tmp_path)
    _commit(repository, "first.txt", "one\n")
    _push_to_bare(repository, bare)
    _commit(repository, "second.txt", "two\n")
    monkeypatch.setattr(source_bundle_module, "_remote_ref_sha", _bare_remote_probe(bare))
    assert inspect_git_source(repository).pushed is False


def test_local_commit_behind_or_diverged_from_origin_is_not_pushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, bare = _bare_fixture(tmp_path)
    first = _commit(repository, "first.txt", "one\n")
    _commit(repository, "second.txt", "two\n")
    _push_to_bare(repository, bare)
    monkeypatch.setattr(source_bundle_module, "_remote_ref_sha", _bare_remote_probe(bare))

    _git(repository, "reset", "--hard", first)
    assert inspect_git_source(repository).pushed is False

    _commit(repository, "diverged.txt", "local\n")
    assert inspect_git_source(repository).pushed is False


def test_changed_origin_ignores_stale_remote_tracking_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, bare = _bare_fixture(tmp_path)
    head = _commit(repository, "tracked.txt", "one\n")
    _push_to_bare(repository, bare)
    _git(repository, "update-ref", "refs/remotes/origin/main", head)
    _git(repository, "remote", "set-url", "origin", "https://other.invalid/new/repo.git")
    assert "origin/main" in _git_output(repository, "branch", "-r", "--contains", head)

    contacted: list[str] = []

    def absent(location: RepositoryLocation, ref: str) -> None:
        contacted.append(location.canonical_url)
        return None

    monkeypatch.setattr(source_bundle_module, "_remote_ref_sha", absent)
    inspected = inspect_git_source(repository)
    assert inspected.location.canonical_url == "https://other.invalid/new/repo.git"
    assert inspected.pushed is False
    assert contacted == ["https://other.invalid/new/repo.git"]


def test_changed_origin_is_true_only_if_new_origin_advertises_exact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _ = _bare_fixture(tmp_path)
    second_bare = tmp_path / "second.git"
    _git(tmp_path, "init", "--bare", str(second_bare))
    head = _commit(repository, "tracked.txt", "one\n")
    _git(repository, "push", str(second_bare), "HEAD:refs/heads/main")
    _git(repository, "remote", "set-url", "origin", "https://other.invalid/new/repo.git")
    monkeypatch.setattr(
        source_bundle_module, "_remote_ref_sha", _bare_remote_probe(second_bare)
    )
    inspected = inspect_git_source(repository)
    assert inspected.commit == head
    assert inspected.pushed is True


@pytest.mark.parametrize("state", ["no-upstream", "other-remote", "detached"])
def test_missing_origin_upstream_or_attached_branch_fails_closed_without_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    repository, _ = _bare_fixture(tmp_path)
    _commit(repository, "tracked.txt", "one\n")
    if state == "no-upstream":
        _git(repository, "config", "--unset", "branch.main.remote")
    elif state == "other-remote":
        _git(repository, "remote", "add", "other", "https://other.invalid/org/repo.git")
        _git(repository, "config", "branch.main.remote", "other")
    else:
        _git(repository, "checkout", "--detach")

    def unexpected(*args: object) -> None:
        raise AssertionError("remote probe must not run")

    monkeypatch.setattr(source_bundle_module, "_remote_ref_sha", unexpected)
    assert inspect_git_source(repository).pushed is False


def test_unsafe_origin_is_rejected_before_remote_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _ = _bare_fixture(tmp_path)
    _commit(repository, "tracked.txt", "one\n")
    _git(repository, "remote", "set-url", "origin", "https://token@example.invalid/org/repo.git")

    def unexpected(*args: object) -> None:
        raise AssertionError("unsafe origin must never be contacted")

    with pytest.raises(RepositoryUrlError):
        inspect_git_source(repository, remote_proof=unexpected)


def test_missing_origin_fails_closed_without_remote_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "work"
    repository.mkdir()
    _git(repository, "init")
    _commit(repository, "tracked.txt", "one\n")

    def unexpected(*args: object) -> None:
        raise AssertionError("missing origin must not be contacted")

    monkeypatch.setattr(source_bundle_module, "_remote_ref_sha", unexpected)
    with pytest.raises(SourceLockError):
        inspect_git_source(repository)


def test_remote_probe_uses_exact_ref_scrubbed_environment_and_no_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    location = RepositoryLocation.parse("ssh://git@example.invalid/org/repo.git")
    commit = "a" * 40
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"{commit}\trefs/heads/feature/nested\n",
            stderr="credential-bearing failure text must stay internal",
        )

    monkeypatch.setattr(source_bundle_module.subprocess, "run", fake_run)
    assert source_bundle_module._remote_ref_sha(location, "refs/heads/feature/nested") == commit
    assert seen["argv"] == [
        "git",
        "ls-remote",
        "--exit-code",
        "ssh://git@example.invalid/org/repo.git",
        "refs/heads/feature/nested",
    ]
    assert seen["timeout"] == source_bundle_module._REMOTE_PROOF_TIMEOUT_SECONDS
    assert seen["capture_output"] is True
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GCM_INTERACTIVE"] == "Never"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert f"-F {os.devnull}" in env["GIT_SSH_COMMAND"]
    assert "ProxyCommand=none" in env["GIT_SSH_COMMAND"]
    assert "IdentityAgent=none" in env["GIT_SSH_COMMAND"]
    assert "IdentitiesOnly=yes" in env["GIT_SSH_COMMAND"]
    assert "IdentityFile=none" in env["GIT_SSH_COMMAND"]
    assert "PreferredAuthentications=none" in env["GIT_SSH_COMMAND"]
    for forbidden in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_PROXY_COMMAND",
        "GIT_CONFIG_COUNT",
        "HOME",
        "XDG_CONFIG_HOME",
        "SSH_AUTH_SOCK",
    ):
        assert forbidden not in env


@pytest.mark.parametrize(
    "ref",
    [
        "refs/tags/main",
        "refs/heads/../main",
        "refs/heads/bad//name",
        "refs/heads/bad@{name",
        "refs/heads/main.lock",
    ],
)
def test_remote_probe_rejects_nonbranch_or_malformed_ref_without_contact(
    monkeypatch: pytest.MonkeyPatch, ref: str
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid ref must not be contacted")

    monkeypatch.setattr(source_bundle_module.subprocess, "run", unexpected)
    location = RepositoryLocation.parse("https://example.invalid/org/repo.git")
    assert source_bundle_module._remote_ref_sha(location, ref) is None


@pytest.mark.parametrize(
    "returncode,stdout",
    [
        (128, ""),
        (0, "malformed"),
        (0, f"{'a' * 40}\trefs/heads/other\n"),
        (0, f"{'a' * 40}\trefs/heads/main\n{'b' * 40}\trefs/heads/main\n"),
    ],
)
def test_remote_probe_failure_or_ambiguous_output_returns_none_without_leak(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str
) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="token=private")

    monkeypatch.setattr(source_bundle_module.subprocess, "run", fake_run)
    location = RepositoryLocation.parse("https://example.invalid/org/repo.git")
    assert source_bundle_module._remote_ref_sha(location, "refs/heads/main") is None


def test_remote_probe_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(argv: list[str], **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(argv, 8, stderr="authorization: private")

    monkeypatch.setattr(source_bundle_module.subprocess, "run", timeout)
    location = RepositoryLocation.parse("https://example.invalid/org/repo.git")
    assert source_bundle_module._remote_ref_sha(location, "refs/heads/main") is None


def test_local_inspection_scrubs_ambient_git_controls_and_reads_literal_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, bare = _bare_fixture(tmp_path)
    head = _commit(repository, "tracked.txt", "intended\n")
    _push_to_bare(repository, bare)

    hostile = tmp_path / "hostile"
    hostile.mkdir()
    _git(hostile, "init")
    hostile_head = _commit(hostile, "hostile.txt", "wrong\n")
    assert hostile_head != head

    rewrite_target = (tmp_path / "rewrite-target.git").as_uri()
    _git(
        repository,
        "config",
        "--local",
        f"url.{rewrite_target}.insteadOf",
        "https://example.invalid/",
    )
    global_config = tmp_path / "hostile-global.gitconfig"
    global_config.write_text(
        '[url "ext::hostile-helper"]\n\tinsteadOf = https://example.invalid/\n',
        encoding="utf-8",
    )
    fake_exec = tmp_path / "fake-git-exec"
    fake_exec.mkdir()

    monkeypatch.setenv("GIT_DIR", str(hostile / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(hostile))
    monkeypatch.setenv("GIT_EXEC_PATH", str(fake_exec))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "remote.origin.url")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "ext::hostile-helper")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("GIT_ASKPASS", "hostile-askpass")
    monkeypatch.setenv("GIT_SSH_COMMAND", "hostile-ssh")

    seen: list[tuple[str, str, str]] = []

    def trusted(location: RepositoryLocation, ref: str, expected: str) -> str:
        seen.append((location.canonical_url, ref, expected))
        return expected

    inspected = inspect_git_source(repository, remote_proof=trusted)
    assert inspected.commit == head
    assert inspected.location.canonical_url == "https://example.invalid/org/repo.git"
    assert inspected.pushed is True
    assert seen == [
        ("https://example.invalid/org/repo.git", "refs/heads/main", head)
    ]


def test_each_local_git_call_receives_minimal_scrubbed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, stdout="value\n", stderr="")

    monkeypatch.setattr(source_bundle_module.subprocess, "run", fake_run)
    assert source_bundle_module._git(tmp_path, "rev-parse", "HEAD") == "value"
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_CONFIG_SYSTEM"] == os.devnull
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    for forbidden in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_EXEC_PATH",
        "GIT_CONFIG_COUNT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "SSH_AUTH_SOCK",
        "HOME",
        "XDG_CONFIG_HOME",
    ):
        assert forbidden not in env


def test_credential_reference_requires_authenticated_remote_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _ = _bare_fixture(tmp_path)
    _commit(repository, "tracked.txt", "one\n")
    credential = SecretRef("env", "PRIVATE_GIT_TOKEN")

    def unexpected(*args: object) -> None:
        raise AssertionError("credential-bearing source must not use public proof")

    monkeypatch.setattr(source_bundle_module, "_remote_ref_sha", unexpected)
    inspected = inspect_git_source(repository, credential=credential)
    assert inspected.location.credential == credential
    assert inspected.pushed is False


def test_authenticated_callback_boolean_is_true_only_after_exact_verification(
    tmp_path: Path,
) -> None:
    repository, bare = _bare_fixture(tmp_path)
    _commit(repository, "tracked.txt", "one\n")
    _push_to_bare(repository, bare)
    credential = SecretRef("env", "PRIVATE_GIT_TOKEN")
    bare_probe = _bare_remote_probe(bare)
    calls: list[tuple[RepositoryLocation, str, str]] = []

    def authenticated(location: RepositoryLocation, ref: str, expected: str) -> bool:
        calls.append((location, ref, expected))
        return bare_probe(location, ref) == expected

    exact = inspect_git_source(
        repository,
        credential=credential,
        remote_proof=authenticated,
    )
    assert exact.pushed is True
    assert calls[-1][0].credential == credential
    assert calls[-1][1:] == ("refs/heads/main", exact.commit)

    _commit(repository, "ahead.txt", "ahead\n")
    ahead = inspect_git_source(
        repository,
        credential=credential,
        remote_proof=authenticated,
    )
    assert ahead.pushed is False


@pytest.mark.parametrize(
    "proof_result,expected",
    [("a" * 40, True), ("b" * 40, False), (True, True), (False, False), (None, False), ("bad", False)],
)
def test_remote_proof_result_is_fail_closed(
    tmp_path: Path, proof_result: str | bool | None, expected: bool
) -> None:
    repository, _ = _bare_fixture(tmp_path)
    head = _commit(repository, "tracked.txt", "one\n")
    if isinstance(proof_result, str) and proof_result == "a" * 40:
        proof_result = head

    def proof(location: RepositoryLocation, ref: str, expected_commit: str):
        return proof_result

    assert inspect_git_source(repository, remote_proof=proof).pushed is expected


def test_remote_proof_exception_returns_false_without_exposing_message(tmp_path: Path) -> None:
    repository, _ = _bare_fixture(tmp_path)
    _commit(repository, "tracked.txt", "one\n")

    def proof(location: RepositoryLocation, ref: str, expected_commit: str) -> bool:
        raise RuntimeError("token=must-not-escape")

    assert inspect_git_source(repository, remote_proof=proof).pushed is False
