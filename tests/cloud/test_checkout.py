from __future__ import annotations

import base64
import copy
import os
import subprocess
from types import SimpleNamespace
from argparse import Namespace
from pathlib import Path
from urllib.parse import quote, quote_plus

import pytest

from tuner.cloud import bootstrap_core
from tuner.cloud.checkout import (
    CheckoutPolicy,
    SSHCheckoutPolicy,
    _authenticated_remote_proof,
    _credential_scope,
    _committed_engine_identity,
    _git_environment,
    build_source_lock,
    checkout_source_lock,
    ssh_checkout_policy_from_environment,
    standalone_credential_from_environment,
    validate_source_lock_for_cloud,
)
from tuner.project.errors import SourceLockError
from tuner.project.context import ProjectContext
from tuner.project.secrets import SecretRef
from tuner.project.source_bundle import GitSource, RepositoryLocation, SourceLock


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True, capture_output=True, text=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def _repository(root: Path, name: str, *, filename: str = "source.txt") -> tuple[Path, Path, str]:
    work = root / name
    bare = root / f"{name}.git"
    work.mkdir()
    _git(work, "init", "-b", "main")
    _git(work, "config", "user.name", "Synaptic Test")
    _git(work, "config", "user.email", "test@example.invalid")
    (work / filename).write_text(f"{name}\n", encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "initial")
    subprocess.run(["git", "clone", "--bare", str(work), str(bare)], check=True, capture_output=True)
    return work, bare, _git(work, "rev-parse", "HEAD")


def _location(name: str, credential: SecretRef | None = None) -> RepositoryLocation:
    return RepositoryLocation.parse(f"https://git.example.test/team/{name}.git", credential=credential)


def _source(name: str, commit: str, **kwargs) -> GitSource:
    return GitSource(location=_location(name, kwargs.pop("credential", None)), commit=commit, pushed=True, **kwargs)


def _standalone_lock(commit: str) -> SourceLock:
    source = _source("standalone", commit)
    return SourceLock("run-standalone", "standalone", source, source, {}, {})


def _superproject_fixture(tmp_path: Path, *, unsafe_extra: bool = False):
    engine, engine_bare, engine_commit = _repository(tmp_path, "engine", filename="engine.txt")
    host, _unused_host_bare, _ = _repository(tmp_path, "host-seed", filename="host.txt")
    _git(host, "-c", "protocol.file.allow=always", "submodule", "add", str(engine_bare), "vendor/engine")
    modules = host / ".gitmodules"
    _git(host, "config", "--file", str(modules), "submodule.vendor/engine.url", _location("engine").canonical_url)
    text = modules.read_text(encoding="utf-8")
    if unsafe_extra:
        text += '\n[submodule "unexpected"]\n\tpath = vendor/unexpected\n\turl = ext::sh -c exploit\n'
    modules.write_text(text, encoding="utf-8")
    _git(host, "add", ".")
    _git(host, "commit", "-m", "lock engine")
    host_bare = tmp_path / "host.git"
    subprocess.run(["git", "clone", "--bare", str(host), str(host_bare)], check=True, capture_output=True)
    host_commit = _git(host, "rev-parse", "HEAD")
    return host_bare, host_commit, engine_bare, engine_commit


def _lock(host_commit: str, engine_commit: str, *, mode: str, credential: SecretRef | None = None) -> SourceLock:
    return SourceLock(
        run_id=f"run-{mode}",
        mode=mode,
        project_source=_source("host", host_commit),
        engine_source=_source(
            "engine",
            engine_commit,
            submodule_path="vendor/engine",
            gitlink_commit=engine_commit,
            credential=credential,
        ),
        project={},
        configuration={},
    )


def _policy(**kwargs) -> CheckoutPolicy:
    return CheckoutPolicy(
        allowed_hosts=frozenset({"git.example.test"}),
        nested_submodules=kwargs.get("nested_submodules", True),
        max_submodule_depth=kwargs.get("max_submodule_depth", 2),
    )


def test_local_checkout_delegates_to_canonical_bootstrap_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = _source("standalone", "a" * 40)
    lock = SourceLock("parity", "standalone", source, source, {}, {})
    observed: dict[str, object] = {}

    def fake_core(source_lock, destination, **kwargs):
        observed.update(source_lock=source_lock, destination=destination, **kwargs)
        return {
            "project_root": str(tmp_path / "checkout/project"),
            "engine_root": str(tmp_path / "checkout/project"),
        }

    monkeypatch.setattr(bootstrap_core, "reconstruct_source_lock", fake_core)
    result = checkout_source_lock(
        lock, tmp_path / "checkout",
        policy=CheckoutPolicy(frozenset({"git.example.test"}), frozenset({"https"})),
    )
    assert observed["source_lock"] == lock.to_dict()
    assert observed["policy"] == {
        "allowed_hosts": ["git.example.test"],
        "allowed_schemes": ["https"],
        "nested_submodules": False,
        "max_submodule_depth": 0,
    }
    assert result.project_root == result.engine_root == tmp_path / "checkout/project"


def test_bootstrap_wire_documents_require_exact_shapes_and_boolean_types() -> None:
    document = _standalone_lock("a" * 40).to_dict()
    document["project"] = {
        "manifest_uri": "project://synaptic.yaml",
        "manifest_sha256": "d" * 64,
        "engine_requires": ">=1,<2",
    }
    document["configuration"] = {
        "resolved_uri": "tracking://runs/parity/resolved-config.json",
        "resolved_sha256": "e" * 64,
        "documents": [],
    }
    policy = {
        "allowed_hosts": ["git.example.test"],
        "allowed_schemes": ["https"],
        "nested_submodules": False,
        "max_submodule_depth": 0,
    }
    bootstrap_core.normalize_source_lock(document, bootstrap_core.normalize_policy(policy))

    malformed_documents = []
    missing = copy.deepcopy(document)
    missing.pop("outputs")
    malformed_documents.append(missing)
    extra = copy.deepcopy(document)
    extra["unexpected"] = True
    malformed_documents.append(extra)
    truncated = {"schema_version": "synaptic-source-lock/v1", "mode": "standalone"}
    malformed_documents.append(truncated)
    source_extra = copy.deepcopy(document)
    source_extra["sources"]["project"]["unexpected"] = True
    malformed_documents.append(source_extra)
    for malformed in malformed_documents:
        with pytest.raises(bootstrap_core.BootstrapError, match="canonical wire shape"):
            bootstrap_core.normalize_source_lock(malformed, bootstrap_core.normalize_policy(policy))

    incomplete_metadata = copy.deepcopy(document)
    incomplete_metadata["project"].pop("manifest_sha256")
    with pytest.raises(bootstrap_core.BootstrapError, match="project.manifest_sha256"):
        bootstrap_core.normalize_source_lock(
            incomplete_metadata, bootstrap_core.normalize_policy(policy),
        )

    for field, value in (("pushed", "false"), ("dirty", 0)):
        malformed = copy.deepcopy(document)
        malformed["sources"]["project"][field] = value
        with pytest.raises(bootstrap_core.BootstrapError, match="must be a boolean"):
            bootstrap_core.normalize_source_lock(malformed, bootstrap_core.normalize_policy(policy))

    for field, value in (("nested_submodules", "false"), ("max_submodule_depth", True)):
        malformed_policy = dict(policy)
        malformed_policy[field] = value
        with pytest.raises(bootstrap_core.BootstrapError):
            bootstrap_core.normalize_policy(malformed_policy)

    extra_policy = dict(policy, unexpected=True)
    with pytest.raises(bootstrap_core.BootstrapError, match="canonical wire shape"):
        bootstrap_core.normalize_policy(extra_policy)


@pytest.mark.parametrize(
    "arguments",
    [
        ["rev-parse", "HEAD"],
        ["ls-tree", "HEAD", "--", "vendor/engine"],
        ["show", "HEAD:.gitmodules"],
    ],
)
def test_every_runtime_git_object_command_disables_replacement_refs(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, str]] = []

    def fake_run(_command, **kwargs):
        observed.append(dict(kwargs["env"]))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(bootstrap_core.subprocess, "run", fake_run)
    assert bootstrap_core.run_git(
        arguments, env={"GIT_NO_REPLACE_OBJECTS": "0", "PATH": os.environ.get("PATH", "")},
    ) == "ok"
    assert observed == [{"GIT_NO_REPLACE_OBJECTS": "1", "PATH": os.environ.get("PATH", "")}]


def test_runtime_git_environment_canonicalizes_mixed_case_replacement_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, str]] = []

    def fake_run(_command, **kwargs):
        observed.append(dict(kwargs["env"]))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(bootstrap_core.subprocess, "run", fake_run)
    supplied = {
        "git_no_replace_objects": "0",
        "Git_No_Replace_Objects": "false",
        "GIT_NO_REPLACE_OBJECTS": "disabled",
        "CaseSensitiveUnrelated": "preserved",
    }
    assert bootstrap_core.run_git(["rev-parse", "HEAD"], env=supplied) == "ok"

    child_environment = observed[0]
    guard_keys = [
        key for key in child_environment
        if key.upper() == "GIT_NO_REPLACE_OBJECTS"
    ]
    assert guard_keys == ["GIT_NO_REPLACE_OBJECTS"]
    assert child_environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert child_environment["CaseSensitiveUnrelated"] == "preserved"


@pytest.mark.parametrize(
    ("source_bytes", "policy_bytes", "message"),
    [
        (
            b'{"schema_version":"synaptic-source-lock/v1","schema_version":"drift"}',
            b"{}",
            "duplicate object keys",
        ),
        (b'{"project":{"nested":1,"nested":2}}', b"{}", "duplicate object keys"),
        (b"{}", b'{"allowed_hosts":[],"allowed_hosts":[]}', "duplicate object keys"),
        (b"\xff", b"{}", "input JSON is invalid"),
        (b'{"broken":', b"{}", "input JSON is invalid"),
    ],
)
def test_remote_json_rejects_duplicate_keys_and_decode_errors_before_reconstruction(
    source_bytes: bytes, policy_bytes: bytes, message: str, tmp_path: Path,
) -> None:
    destination = tmp_path / "must-not-exist"
    with pytest.raises(bootstrap_core.BootstrapError, match=message) as error:
        bootstrap_core.reconstruct_source_lock_json(
            source_bytes, policy_bytes, destination,
        )
    assert not destination.exists()
    assert "schema_version" not in str(error.value)
    assert "allowed_hosts" not in str(error.value)


def test_submodule_paths_reject_casefold_collisions_before_child_fetch(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".gitmodules").write_text("fixture\n", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(arguments, **_kwargs):
        calls.append(list(arguments))
        if "--get-regexp" in arguments:
            return "submodule.one.path Vendor/Plugin\nsubmodule.two.path vendor/plugin"
        if arguments[-1] == "submodule.one.url":
            return "https://git.example.test/team/plugin.git"
        if "ls-tree" in arguments:
            return f"160000 commit {'b' * 40}\tVendor/Plugin"
        raise AssertionError(f"unexpected Git call: {arguments}")

    policy = bootstrap_core.normalize_policy(
        {
            "allowed_hosts": ["git.example.test"],
            "allowed_schemes": ["https"],
            "nested_submodules": True,
            "max_submodule_depth": 2,
        }
    )
    parent = bootstrap_core.canonicalize_repository_url(
        "https://git.example.test/team/host.git"
    )
    with pytest.raises(bootstrap_core.BootstrapError, match="unique and contained"):
        bootstrap_core._read_submodules(
            repository, parent, policy=policy, depth=0, command_runner=runner,
        )
    assert not any("clone" in call for call in calls)
    assert sum("submodule.one.url" in call for call in calls) == 1
    assert not any("submodule.two.url" in call for call in calls)


def test_encoded_secret_forms_are_redacted_from_git_errors() -> None:
    secret = "token value+/with?symbols"
    raw = secret.encode("utf-8")
    variants = (
        secret,
        quote(secret, safe=""),
        quote_plus(secret, safe=""),
        base64.b64encode(raw).decode("ascii"),
        base64.b64encode(b"x-access-token:" + raw).decode("ascii").rstrip("="),
    )
    rendered = bootstrap_core.redact(" | ".join(variants) + " | useful-context", (secret,))
    assert "useful-context" in rendered
    assert all(variant not in rendered for variant in variants)


def test_checkout_destination_rejects_real_junction_before_runner_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tuner.cloud.checkout as checkout_module

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    _directory_link(linked_parent, real_parent)
    calls: list[list[str]] = []

    def runner(arguments, **_kwargs):
        calls.append(list(arguments))
        raise AssertionError("Git runner must not execute through a linked destination")

    monkeypatch.setattr(checkout_module, "_run_git", runner)
    with pytest.raises(SourceLockError, match="links or reparse"):
        checkout_source_lock(
            _standalone_lock("a" * 40),
            linked_parent / "checkout",
            policy=CheckoutPolicy(frozenset({"git.example.test"}), frozenset({"https"})),
        )
    assert not calls
    assert not (real_parent / "checkout").exists()


def test_standalone_checkout_uses_exact_commit(tmp_path: Path) -> None:
    _work, bare, commit = _repository(tmp_path, "standalone")
    result = checkout_source_lock(
        _standalone_lock(commit),
        tmp_path / "checkout",
        policy=_policy(),
        clone_url_resolver=lambda _location: str(bare),
    )
    assert result.project_root == result.engine_root
    assert _git(result.project_root, "rev-parse", "HEAD") == commit


@pytest.mark.parametrize("mode", ["superproject", "dual_clone"])
def test_host_modes_verify_gitlink_and_reconstruct_exact_engine(tmp_path: Path, mode: str) -> None:
    host_bare, host_commit, engine_bare, engine_commit = _superproject_fixture(tmp_path)
    remotes = {"host": host_bare, "engine": engine_bare}
    result = checkout_source_lock(
        _lock(host_commit, engine_commit, mode=mode),
        tmp_path / "checkout",
        policy=_policy(),
        clone_url_resolver=lambda location: str(remotes[Path(location.path).stem]),
    )
    assert _git(result.project_root, "rev-parse", "HEAD") == host_commit
    assert _git(result.engine_root, "rev-parse", "HEAD") == engine_commit
    if mode == "superproject":
        assert result.engine_root == (result.project_root / "vendor" / "engine").resolve()
    else:
        assert result.engine_root.parent == (tmp_path / "checkout").resolve()


def test_checkout_rejects_actual_gitlink_mismatch(tmp_path: Path) -> None:
    host_bare, host_commit, engine_bare, engine_commit = _superproject_fixture(tmp_path)
    engine_work = tmp_path / "engine"
    (engine_work / "second.txt").write_text("second\n", encoding="utf-8")
    _git(engine_work, "add", ".")
    _git(engine_work, "commit", "-m", "second")
    second_commit = _git(engine_work, "rev-parse", "HEAD")
    _git(engine_work, "push", str(engine_bare), "main")
    lock = _lock(host_commit, second_commit, mode="superproject")
    with pytest.raises(SourceLockError, match="gitlink"):
        checkout_source_lock(
            lock,
            tmp_path / "checkout",
            policy=_policy(),
            clone_url_resolver=lambda location: str(host_bare if "host" in location.path else engine_bare),
        )


def test_dual_clone_rejects_engine_identity_not_named_by_host(tmp_path: Path) -> None:
    host_bare, host_commit, engine_bare, engine_commit = _superproject_fixture(tmp_path)
    engine = GitSource(
        location=_location("different-engine"),
        commit=engine_commit,
        pushed=True,
        submodule_path="vendor/engine",
        gitlink_commit=engine_commit,
    )
    lock = SourceLock(
        "run-dual-identity",
        "dual_clone",
        _source("host", host_commit),
        engine,
        {},
        {},
    )
    with pytest.raises(SourceLockError, match="URL"):
        checkout_source_lock(
            lock,
            tmp_path / "checkout",
            policy=_policy(),
            clone_url_resolver=lambda location: str(host_bare if "host" in location.path else engine_bare),
        )


@pytest.mark.parametrize(("dirty", "pushed", "message"), [(True, True, "clean"), (False, False, "pushed")])
def test_cloud_preflight_rejects_dirty_or_unpushed_sources(dirty: bool, pushed: bool, message: str) -> None:
    source = GitSource(location=_location("standalone"), commit="a" * 40, dirty=dirty, pushed=pushed)
    lock = SourceLock("run", "standalone", source, source, {}, {})
    with pytest.raises(SourceLockError, match=message):
        validate_source_lock_for_cloud(lock)


def test_all_gitmodules_urls_are_rejected_before_first_submodule_fetch(tmp_path: Path) -> None:
    host_bare, host_commit, engine_bare, engine_commit = _superproject_fixture(tmp_path, unsafe_extra=True)
    resolved: list[str] = []

    def resolver(location: RepositoryLocation) -> str:
        resolved.append(location.path)
        return str(host_bare if "host" in location.path else engine_bare)

    with pytest.raises(SourceLockError, match="Rejected .gitmodules"):
        checkout_source_lock(
            _lock(host_commit, engine_commit, mode="superproject"),
            tmp_path / "checkout",
            policy=_policy(),
            clone_url_resolver=resolver,
        )
    assert resolved == ["/team/host.git"]


def test_secret_is_resolved_at_execution_and_helper_removed_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_bare, host_commit, engine_bare, engine_commit = _superproject_fixture(tmp_path)
    reference = SecretRef("provider_secret", "PRIVATE_GIT_TOKEN")
    created: list[Path] = []
    counter = 0

    def make_helper(prefix: str) -> str:
        nonlocal counter
        counter += 1
        path = tmp_path / f"helper-{counter}"
        path.mkdir()
        created.append(path)
        return str(path)

    monkeypatch.setattr("tuner.cloud.checkout.tempfile.mkdtemp", make_helper)
    lock = _lock(host_commit, engine_commit, mode="superproject", credential=reference)
    resolver = lambda location: str(host_bare if "host" in location.path else engine_bare)
    checkout_source_lock(
        lock,
        tmp_path / "success",
        policy=_policy(),
        clone_url_resolver=resolver,
        provider_secret=lambda name: "memory-only-value" if name == reference.name else None,
    )
    assert created and all(not path.exists() for path in created)

    with pytest.raises(SourceLockError) as error:
        checkout_source_lock(
            lock,
            tmp_path / "failure",
            policy=_policy(),
            clone_url_resolver=lambda location: str(host_bare if "host" in location.path else tmp_path / "missing.git"),
            provider_secret=lambda _name: "memory-only-value",
        )
    assert "memory-only-value" not in str(error.value)
    assert all(not path.exists() for path in created)


def test_git_environment_is_allowlisted_and_scrubs_execution_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hostile = {
        "GIT_TEMPLATE_DIR": str(tmp_path / "template"),
        "GIT_EXEC_PATH": str(tmp_path / "exec"),
        "GIT_SSH": str(tmp_path / "ssh"),
        "GIT_SSH_COMMAND": "malicious-ssh --steal",
        "GIT_PROXY_COMMAND": "malicious-proxy",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "url.evil.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://git.example.test/",
        "GIT_ASKPASS": str(tmp_path / "askpass"),
        "SSH_ASKPASS": str(tmp_path / "askpass"),
        "HTTP_PROXY": "http://evil.invalid",
        "HTTPS_PROXY": "http://evil.invalid",
        "CUSTOM_INJECTION": "malicious",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    environment = _git_environment()
    assert not set(hostile).intersection(environment)
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"]
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_ALLOW_PROTOCOL"] == "https:ssh"
    assert any(key.upper() == "PATH" for key in environment)


def test_private_https_remote_proof_exposes_secret_only_to_exact_ls_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuner.cloud.checkout as checkout_module

    reference = SecretRef("env", "PRIVATE_GIT_TOKEN")
    location = _location("private", reference)
    expected = "a" * 40
    observed: dict[str, object] = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = list(arguments)
        observed["environment"] = dict(kwargs["env"])
        return f"{expected}\trefs/heads/main"

    monkeypatch.setattr(checkout_module, "_run_git", fake_run)
    proof = _authenticated_remote_proof(
        environment={"PRIVATE_GIT_TOKEN": "memory-only-value", "GIT_EXEC_PATH": "evil"},
        provider_secret=None,
        credential_helper=None,
    )
    assert proof(location, "refs/heads/main", expected) == expected
    assert "ls-remote" in observed["arguments"]
    environment = observed["environment"]
    assert environment["SYNAPTIC_GIT_SECRET"] == "memory-only-value"
    assert "GIT_EXEC_PATH" not in environment


def test_controlled_ssh_scope_ignores_ambient_identity_and_proxy_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ssh = tmp_path / "fake-ssh"
    known_hosts = tmp_path / "known_hosts"
    ssh.write_text("fake", encoding="utf-8")
    known_hosts.write_text("git.example.test ssh-ed25519 AAAA\n", encoding="utf-8")
    policy = SSHCheckoutPolicy(ssh.resolve(), "explicit-agent", known_hosts.resolve())
    location = RepositoryLocation.parse("ssh://git@git.example.test/team/repo.git")
    monkeypatch.setenv("SSH_AUTH_SOCK", "ambient-agent")
    monkeypatch.setenv("GIT_SSH_COMMAND", "ambient-ssh --steal")
    with _credential_scope(
        location,
        environment=None,
        provider_secret=None,
        credential_helper=None,
        ssh_policy=policy,
    ) as (_config, environment, _secrets, _helper):
        command = environment["GIT_SSH_COMMAND"]
        assert environment["SSH_AUTH_SOCK"] == "explicit-agent"
        assert "ambient" not in command
        for required in (
            "StrictHostKeyChecking=yes",
            "UserKnownHostsFile=",
            "IdentityFile=none",
            "ProxyCommand=none",
            "ProxyJump=none",
            "ForwardAgent=no",
            "PermitLocalCommand=no",
        ):
            assert required in command


def test_ssh_scope_fails_closed_without_controlled_policy() -> None:
    location = RepositoryLocation.parse("ssh://git@git.example.test/team/repo.git")
    with pytest.raises(SourceLockError, match="explicit controlled agent"):
        with _credential_scope(
            location,
            environment=None,
            provider_secret=None,
            credential_helper=None,
        ):
            pass


def test_hostile_ambient_git_template_hook_is_not_installed_or_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _work, bare, commit = _repository(tmp_path, "standalone")
    template_hooks = tmp_path / "hostile-template" / "hooks"
    template_hooks.mkdir(parents=True)
    marker = tmp_path / "ambient-hook-ran"
    hook = template_hooks / "post-checkout"
    hook.write_text(
        "#!/bin/sh\nprintf pwned > " + str(marker).replace("\\", "/") + "\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(template_hooks.parent))
    monkeypatch.setenv("GIT_EXEC_PATH", str(tmp_path / "missing-exec"))
    monkeypatch.setenv("GIT_SSH", str(tmp_path / "malicious-ssh"))
    result = checkout_source_lock(
        _standalone_lock(commit),
        tmp_path / "checkout",
        policy=_policy(),
        clone_url_resolver=lambda _location: str(bare),
    )
    assert _git(result.project_root, "rev-parse", "HEAD") == commit
    assert not marker.exists()
    assert not (result.project_root / ".git" / "hooks" / "post-checkout").exists()


def test_secret_environment_is_removed_before_checkout_and_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuner.cloud.checkout as checkout_module

    host_bare, host_commit, engine_bare, engine_commit = _superproject_fixture(tmp_path)
    reference = SecretRef("provider_secret", "PRIVATE_GIT_TOKEN")
    lock = _lock(host_commit, engine_commit, mode="superproject", credential=reference)
    observed: list[tuple[list[str], dict[str, str] | None]] = []
    original = checkout_module._run_git

    def recording_run(arguments, **kwargs):
        environment = kwargs.get("env")
        observed.append((list(arguments), dict(environment) if environment is not None else None))
        return original(arguments, **kwargs)

    monkeypatch.setattr(checkout_module, "_run_git", recording_run)
    checkout_source_lock(
        lock,
        tmp_path / "checkout",
        policy=_policy(),
        clone_url_resolver=lambda location: str(
            host_bare if "host" in location.path else engine_bare
        ),
        provider_secret=lambda _name: "memory-only-value",
    )
    secret_calls = [arguments for arguments, env in observed if env and env.get("SYNAPTIC_GIT_SECRET")]
    assert len(secret_calls) == 1
    assert "clone" in secret_calls[0]
    for arguments, env in observed:
        if "checkout" in arguments or "rev-parse" in arguments:
            assert not env or "SYNAPTIC_GIT_SECRET" not in env


def test_requested_source_mode_must_match_standalone_topology(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    context = ProjectContext.standalone(engine_root=engine)
    with pytest.raises(SourceLockError, match="does not match standalone topology"):
        build_source_lock(context, run_id="mode-mismatch", mode="superproject")


def test_requested_source_mode_must_match_discovered_host_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuner.cloud.checkout as checkout_module

    project = tmp_path / "host"
    engine = project / "vendor" / "engine"
    engine.mkdir(parents=True)
    manifest_path = project / "synaptic.yaml"
    manifest_path.write_text("fixture\n", encoding="utf-8")
    context = ProjectContext.host(
        engine_root=engine,
        project_root=project,
        manifest_path=manifest_path,
    )
    project_source = _source("host", "b" * 40)
    engine_source = _source(
        "engine",
        "a" * 40,
        submodule_path="vendor/engine",
        gitlink_commit="a" * 40,
    )

    class Manifest:
        data = {
            "project": {},
            "engine": {"path": "project://vendor/engine"},
        }
        policies = {
            "repository_hosts": ["git.example.test"],
            "repository_schemes": ["https"],
            "max_submodule_depth": 1,
        }
        project_id = "host"
        engine_requires = ">=1,<2"

    monkeypatch.setattr(checkout_module, "load_project_manifest", lambda _path: Manifest())
    monkeypatch.setattr(
        checkout_module,
        "inspect_git_source",
        lambda repository, **_kwargs: project_source if repository == project else engine_source,
    )
    monkeypatch.setattr(
        checkout_module,
        "_committed_engine_identity",
        lambda *_args, **_kwargs: (engine_source.location, engine_source.commit),
    )
    with pytest.raises(SourceLockError, match="does not match discovered 'superproject'"):
        build_source_lock(context, run_id="host-mode-mismatch", mode="dual_clone")


def test_cloud_train_forwards_requested_source_mode_before_provider_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tuner.handlers.cloud_train_handler as handler_module

    handler = handler_module.CloudTrainHandler(
        args=Namespace(source_mode="dual_clone", run_id="requested-mode", json=False)
    )
    source = _source("standalone", "a" * 40)
    lock = SourceLock("requested-mode", "standalone", source, source, {}, {})
    observed: dict[str, object] = {}

    def fake_build(context, *, run_id, mode, **_kwargs):
        observed.update(context=context, run_id=run_id, mode=mode, **_kwargs)
        return lock

    monkeypatch.setattr(handler_module, "build_source_lock", fake_build)
    monkeypatch.setattr(
        handler_module,
        "checkout_policy_from_context",
        lambda _context, **_kwargs: _policy(),
    )
    monkeypatch.setattr(handler_module, "build_runtime_layout", lambda _context: object())
    handler._prepare_source_contract()
    assert observed["run_id"] == "requested-mode"
    assert observed["mode"] == "dual_clone"
    assert "standalone_credential" in observed
    assert "ssh_policy" in observed


def test_committed_gitmodules_engine_identity_is_read_before_checkout(tmp_path: Path) -> None:
    host_bare, _host_commit, _engine_bare, engine_commit = _superproject_fixture(tmp_path)
    checkout = tmp_path / "host-checkout"
    subprocess.run(["git", "clone", str(host_bare), str(checkout)], check=True, capture_output=True)
    location, gitlink = _committed_engine_identity(
        checkout,
        _location("host"),
        engine_path="vendor/engine",
        policy=_policy(),
    )
    assert location.canonical_url == _location("engine").canonical_url
    assert gitlink == engine_commit


def test_committed_gitmodules_rejects_credential_url_without_echo(tmp_path: Path) -> None:
    host_bare, _host_commit, _engine_bare, _engine_commit = _superproject_fixture(tmp_path)
    checkout = tmp_path / "host-checkout"
    subprocess.run(["git", "clone", str(host_bare), str(checkout)], check=True, capture_output=True)
    _git(checkout, "config", "user.name", "Synaptic Test")
    _git(checkout, "config", "user.email", "test@example.invalid")
    _git(
        checkout,
        "config",
        "--file",
        str(checkout / ".gitmodules"),
        "submodule.vendor/engine.url",
        "https://token-value@git.example.test/team/engine.git",
    )
    _git(checkout, "add", ".gitmodules")
    _git(checkout, "commit", "-m", "unsafe identity")
    with pytest.raises(SourceLockError) as error:
        _committed_engine_identity(
            checkout,
            _location("host"),
            engine_path="vendor/engine",
            policy=_policy(),
        )
    assert "token-value" not in str(error.value)


def test_committed_gitmodules_rejects_case_variant_duplicate_options(tmp_path: Path) -> None:
    host_bare, _host_commit, _engine_bare, _engine_commit = _superproject_fixture(tmp_path)
    checkout = tmp_path / "host-checkout"
    subprocess.run(["git", "clone", str(host_bare), str(checkout)], check=True, capture_output=True)
    _git(checkout, "config", "user.name", "Synaptic Test")
    _git(checkout, "config", "user.email", "test@example.invalid")
    modules = checkout / ".gitmodules"
    modules.write_text(
        modules.read_text(encoding="utf-8")
        + "\tURL = https://git.example.test/team/different.git\n",
        encoding="utf-8",
    )
    _git(checkout, "add", ".gitmodules")
    _git(checkout, "commit", "-m", "duplicate option")
    with pytest.raises(SourceLockError, match="duplicate option"):
        _committed_engine_identity(
            checkout,
            _location("host"),
            engine_path="vendor/engine",
            policy=_policy(),
        )


def test_private_standalone_source_uses_opaque_env_reference_and_cleans_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuner.cloud.checkout as checkout_module

    engine = tmp_path / "engine"
    engine.mkdir()
    context = ProjectContext.standalone(engine_root=engine)
    expected = "b" * 40
    process_environments: list[dict[str, str]] = []

    def fake_run(arguments, **kwargs):
        process_environments.append(kwargs["env"])
        assert kwargs["env"]["SYNAPTIC_GIT_SECRET"] == "fake-token-value"
        return f"{expected}\trefs/heads/main"

    def fake_inspect(_repository, *, credential, remote_proof, **_kwargs):
        assert credential == SecretRef("env", "FAKE_PRIVATE_TOKEN")
        location = _location("private-standalone", credential)
        pushed = remote_proof(location, "refs/heads/main", expected) == expected
        return GitSource(location=location, commit=expected, branch="main", pushed=pushed)

    monkeypatch.setattr(checkout_module, "_run_git", fake_run)
    monkeypatch.setattr(checkout_module, "inspect_git_source", fake_inspect)
    environment = {
        "SYNAPTIC_GIT_CREDENTIAL_PROVIDER": "env",
        "SYNAPTIC_GIT_CREDENTIAL_NAME": "FAKE_PRIVATE_TOKEN",
        "FAKE_PRIVATE_TOKEN": "fake-token-value",
    }
    reference = standalone_credential_from_environment(environment)
    lock = build_source_lock(
        context,
        run_id="private-standalone",
        environment=environment,
        standalone_credential=reference,
    )
    assert lock.project_source.location.credential == reference
    assert process_environments and all(
        "SYNAPTIC_GIT_SECRET" not in process_environment
        for process_environment in process_environments
    )
    assert "fake-token-value" not in lock.to_json()


@pytest.mark.parametrize(
    "environment",
    [
        {"SYNAPTIC_GIT_CREDENTIAL_PROVIDER": "env"},
        {"SYNAPTIC_GIT_CREDENTIAL_NAME": "TOKEN_NAME"},
        {
            "SYNAPTIC_GIT_CREDENTIAL_PROVIDER": "literal-token",
            "SYNAPTIC_GIT_CREDENTIAL_NAME": "TOKEN_NAME",
        },
    ],
)
def test_standalone_credential_declaration_fails_closed(environment) -> None:
    with pytest.raises(SourceLockError, match="credential reference"):
        standalone_credential_from_environment(environment)


def test_ssh_policy_environment_is_all_or_none_and_exact(tmp_path: Path) -> None:
    ssh = (tmp_path / "ssh").resolve()
    known_hosts = (tmp_path / "known_hosts").resolve()
    ssh.write_text("fake", encoding="utf-8")
    known_hosts.write_text("host ssh-ed25519 AAAA\n", encoding="utf-8")
    environment = {
        "SYNAPTIC_GIT_SSH_EXECUTABLE": str(ssh),
        "SYNAPTIC_GIT_SSH_AGENT_SOCKET": "explicit-agent-id",
        "SYNAPTIC_GIT_SSH_KNOWN_HOSTS": str(known_hosts),
    }
    policy = ssh_checkout_policy_from_environment(environment)
    assert policy == SSHCheckoutPolicy(ssh, "explicit-agent-id", known_hosts)
    with pytest.raises(SourceLockError, match="requires executable, agent, and known_hosts"):
        ssh_checkout_policy_from_environment(
            {"SYNAPTIC_GIT_SSH_EXECUTABLE": str(ssh)}
        )


def test_ssh_exact_remote_proof_uses_controlled_agent_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuner.cloud.checkout as checkout_module

    ssh = (tmp_path / "fake-ssh").resolve()
    known_hosts = (tmp_path / "known_hosts").resolve()
    ssh.write_text("fake", encoding="utf-8")
    known_hosts.write_text("git.example.test ssh-ed25519 AAAA\n", encoding="utf-8")
    policy = SSHCheckoutPolicy(ssh, "explicit-agent", known_hosts)
    expected = "c" * 40
    observed: dict[str, object] = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed["environment"] = dict(kwargs["env"])
        return f"{expected}\trefs/heads/main"

    monkeypatch.setattr(checkout_module, "_run_git", fake_run)
    proof = _authenticated_remote_proof(
        environment={"SSH_AUTH_SOCK": "ambient-agent", "GIT_SSH_COMMAND": "evil"},
        provider_secret=None,
        credential_helper=None,
        ssh_policy=policy,
    )
    location = RepositoryLocation.parse("ssh://git@git.example.test/team/repo.git")
    assert proof(location, "refs/heads/main", expected) == expected
    environment = observed["environment"]
    command = environment["GIT_SSH_COMMAND"]
    assert environment["SSH_AUTH_SOCK"] == "explicit-agent"
    assert "IdentityFile=none" in command and "IdentitiesOnly=no" in command
    assert "StrictHostKeyChecking=yes" in command and "UserKnownHostsFile=" in command
    assert "ambient-agent" not in command and "evil" not in command
