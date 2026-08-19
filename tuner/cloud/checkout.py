"""Provider-neutral, exact-commit checkout for cloud execution."""

from __future__ import annotations

import hashlib
import configparser
import os
import re
import tempfile  # compatibility seam; bootstrap_core uses the same stdlib module object
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

from tuner.cloud import bootstrap_core
from tuner.project.errors import RepositoryUrlError, SecretReferenceError, SourceLockError
from tuner.project.context import ProjectContext
from tuner.project.manifest import load_project_manifest
from tuner.project.secrets import SecretResolver, SecretRef, resolve_secret
from tuner.project.source_bundle import (
    GitSource,
    RepositoryLocation,
    SourceLock,
    resolve_relative_repository_url,
    inspect_git_source,
)


CloneUrlResolver = Callable[[RepositoryLocation], str]


@dataclass(frozen=True)
class SSHCheckoutPolicy:
    """Explicit external-agent SSH boundary with controlled host verification."""

    ssh_executable: Path
    agent_socket: str
    known_hosts: Path

    def __post_init__(self) -> None:
        if not self.ssh_executable.is_absolute() or not self.known_hosts.is_absolute():
            raise SourceLockError("Controlled SSH paths must be absolute")
        executable = self.ssh_executable.resolve()
        known_hosts = self.known_hosts.resolve()
        if not executable.is_file() or self.ssh_executable.is_symlink():
            raise SourceLockError("Controlled SSH executable must be an absolute regular file")
        if not known_hosts.is_file() or self.known_hosts.is_symlink():
            raise SourceLockError("Controlled SSH requires an explicit regular known_hosts file")
        if not self.agent_socket.strip():
            raise SourceLockError("Controlled SSH requires an explicit agent socket")


@dataclass(frozen=True)
class CheckoutPolicy:
    """Fail-closed repository and recursion policy."""

    allowed_hosts: frozenset[str]
    allowed_schemes: frozenset[str] = frozenset({"https", "ssh"})
    nested_submodules: bool = False
    max_submodule_depth: int = 0
    ssh: SSHCheckoutPolicy | None = None

    def __post_init__(self) -> None:
        if not self.allowed_hosts:
            raise SourceLockError("Cloud checkout requires at least one allowed repository host")
        if not self.allowed_schemes or not self.allowed_schemes <= {"https", "ssh"}:
            raise SourceLockError("Cloud checkout permits only approved https and ssh schemes")
        if not 0 <= self.max_submodule_depth <= 16:
            raise SourceLockError("Submodule depth must be between 0 and 16")

    def validate(self, location: RepositoryLocation) -> RepositoryLocation:
        return RepositoryLocation.parse(
            location.canonical_url,
            credential=location.credential,
            allowed_hosts={host.lower() for host in self.allowed_hosts},
            allowed_schemes=set(self.allowed_schemes),
        )


@dataclass(frozen=True)
class CheckoutResult:
    """Verified local reconstruction of a source lock."""

    source_lock: SourceLock
    project_root: Path
    engine_root: Path


def standalone_credential_from_environment(
    environment: Mapping[str, str] | None,
) -> SecretRef | None:
    """Parse an optional standalone SecretRef declaration, never its value."""

    values = environment or {}
    provider = values.get("SYNAPTIC_GIT_CREDENTIAL_PROVIDER")
    name = values.get("SYNAPTIC_GIT_CREDENTIAL_NAME")
    if provider is None and name is None:
        return None
    if not provider or not provider.strip() or not name or not name.strip():
        raise SourceLockError("Standalone Git credential reference must declare provider and name")
    try:
        return SecretRef.from_dict({"provider": provider.strip(), "name": name.strip()})
    except SecretReferenceError:
        raise SourceLockError("Standalone Git credential reference is invalid") from None


def ssh_checkout_policy_from_environment(
    environment: Mapping[str, str] | None,
) -> SSHCheckoutPolicy | None:
    """Parse an all-or-none controlled SSH declaration from opaque paths/IDs."""

    values = environment or {}
    names = (
        "SYNAPTIC_GIT_SSH_EXECUTABLE",
        "SYNAPTIC_GIT_SSH_AGENT_SOCKET",
        "SYNAPTIC_GIT_SSH_KNOWN_HOSTS",
    )
    declared = [values.get(name) for name in names]
    if all(value is None for value in declared):
        return None
    if any(value is None or not value.strip() for value in declared):
        raise SourceLockError("Controlled SSH policy requires executable, agent, and known_hosts")
    return SSHCheckoutPolicy(
        ssh_executable=Path(declared[0].strip()),
        agent_socket=declared[1].strip(),
        known_hosts=Path(declared[2].strip()),
    )


def checkout_policy_from_context(
    context: ProjectContext,
    *,
    ssh_policy: SSHCheckoutPolicy | None = None,
    source_lock: SourceLock | None = None,
) -> CheckoutPolicy:
    """Load repository policy from a host manifest or safe standalone defaults."""

    if context.manifest_path and context.manifest_path.is_file():
        policies = load_project_manifest(context.manifest_path).policies
        hosts = frozenset(str(host).lower() for host in policies.get("repository_hosts", ()))
        schemes = frozenset(str(scheme).lower() for scheme in policies.get("repository_schemes", ()))
        return CheckoutPolicy(
            allowed_hosts=hosts,
            allowed_schemes=schemes or frozenset({"https", "ssh"}),
            nested_submodules=bool(policies.get("nested_submodules", False)),
            max_submodule_depth=int(policies.get("max_submodule_depth", 0)),
            ssh=ssh_policy,
        )
    source = source_lock.engine_source if source_lock is not None else inspect_git_source(
        context.engine_root,
        remote_proof=_authenticated_remote_proof(
            environment=None,
            provider_secret=None,
            credential_helper=None,
            ssh_policy=ssh_policy,
        ),
    )
    return CheckoutPolicy(
        allowed_hosts=frozenset({source.location.host}),
        ssh=ssh_policy,
    )


def _manifest_credential(value: object) -> SecretRef | None:
    return SecretRef.from_dict(value) if isinstance(value, Mapping) else None


def build_source_lock(
    context: ProjectContext,
    *,
    run_id: str,
    mode: str | None = None,
    environment: Mapping[str, str] | None = None,
    provider_secret: SecretResolver | None = None,
    credential_helper: SecretResolver | None = None,
    standalone_credential: SecretRef | None = None,
    ssh_policy: SSHCheckoutPolicy | None = None,
) -> SourceLock:
    """Inspect the active context and create its one canonical cloud source lock."""

    if context.mode == "standalone":
        if mode is not None and mode != "standalone":
            raise SourceLockError(
                f"Requested source mode {mode!r} does not match standalone topology"
            )
        source = inspect_git_source(
            context.engine_root,
            credential=standalone_credential,
            remote_proof=_authenticated_remote_proof(
                environment=environment,
                provider_secret=provider_secret,
                credential_helper=credential_helper,
                ssh_policy=ssh_policy,
            ),
        )
        lock = SourceLock(
            run_id=run_id,
            mode="standalone",
            project_source=source,
            engine_source=source,
            project={"id": context.engine_root.name},
            configuration={},
        )
        validate_source_lock_for_cloud(lock)
        return lock

    if not context.manifest_path or not context.manifest_path.is_file():
        raise SourceLockError("Host cloud execution requires a validated synaptic.yaml")
    manifest = load_project_manifest(context.manifest_path)
    engine_document = manifest.data.get("engine", {})
    project_document = manifest.data.get("project", {})
    engine_path_value = engine_document.get("path") if isinstance(engine_document, Mapping) else None
    if isinstance(engine_path_value, str) and engine_path_value.startswith("project://"):
        submodule_path = engine_path_value[len("project://") :].replace("\\", "/")
    else:
        try:
            submodule_path = context.engine_root.resolve().relative_to(context.project_root.resolve()).as_posix()
        except ValueError as exc:
            raise SourceLockError("Host manifest must declare engine.path for a separate engine checkout") from exc
    project_credential = _manifest_credential(
        project_document.get("credential") if isinstance(project_document, Mapping) else None
    )
    engine_credential = _manifest_credential(
        engine_document.get("credential") if isinstance(engine_document, Mapping) else None
    )
    project_source = inspect_git_source(
        context.project_root,
        credential=project_credential,
        remote_proof=_authenticated_remote_proof(
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
            ssh_policy=ssh_policy,
        ),
    )
    policies = manifest.policies
    policy = CheckoutPolicy(
        allowed_hosts=frozenset(str(host).lower() for host in policies.get("repository_hosts", ())),
        allowed_schemes=frozenset(
            str(scheme).lower() for scheme in policies.get("repository_schemes", ("https", "ssh"))
        ),
        nested_submodules=bool(policies.get("nested_submodules", False)),
        max_submodule_depth=int(policies.get("max_submodule_depth", 0)),
        ssh=ssh_policy,
    )
    committed_location, committed_gitlink = _committed_engine_identity(
        context.project_root,
        project_source.location,
        engine_path=submodule_path,
        policy=policy,
    )
    gitlink = committed_gitlink
    engine_source = inspect_git_source(
        context.engine_root,
        submodule_path=submodule_path,
        gitlink_commit=gitlink,
        credential=engine_credential,
        remote_proof=_authenticated_remote_proof(
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
            ssh_policy=ssh_policy,
        ),
    )
    discovered_mode = (
        "superproject"
        if context.engine_root.resolve() == (context.project_root / submodule_path).resolve()
        else "dual_clone"
    )
    if mode is not None and mode != discovered_mode:
        raise SourceLockError(
            f"Requested source mode {mode!r} does not match discovered {discovered_mode!r} topology"
        )
    if committed_location.canonical_url != engine_source.location.canonical_url:
        raise SourceLockError("Committed engine submodule URL does not match the engine source identity")
    if committed_gitlink.lower() != engine_source.commit.lower():
        raise SourceLockError("Committed engine gitlink does not match the engine source commit")
    lock = SourceLock(
        run_id=run_id,
        mode=discovered_mode,
        project_source=project_source,
        engine_source=engine_source,
        project={
            "id": manifest.project_id,
            "manifest_sha256": hashlib.sha256(context.manifest_path.read_bytes()).hexdigest(),
            "engine_requires": manifest.engine_requires,
        },
        configuration={},
    )
    validate_source_lock_for_cloud(lock)
    return lock


def _committed_engine_identity(
    repository: Path,
    parent_location: RepositoryLocation,
    *,
    engine_path: str,
    policy: CheckoutPolicy,
) -> tuple[RepositoryLocation, str]:
    """Read the committed root .gitmodules without config expansion or includes."""

    document = _run_git(["show", "HEAD:.gitmodules"], cwd=repository)
    parser = configparser.RawConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    try:
        parser.read_string(document)
    except configparser.Error as exc:
        raise SourceLockError("Committed .gitmodules is malformed") from exc
    if parser.defaults():
        raise SourceLockError("Committed .gitmodules cannot contain default options")
    matches: list[RepositoryLocation] = []
    seen_paths: set[str] = set()
    for section in parser.sections():
        if not re.fullmatch(r'submodule "[^"\r\n]+"', section):
            raise SourceLockError("Committed .gitmodules contains an unsupported section")
        values: dict[str, str] = {}
        for key, value in parser.items(section, raw=True):
            normalized_key = key.lower()
            if normalized_key in values:
                raise SourceLockError("Committed .gitmodules contains duplicate option names")
            values[normalized_key] = value
        raw_path = values.get("path", "").strip().replace("\\", "/")
        raw_url = values.get("url", "").strip()
        if not raw_path or raw_path.startswith("/") or ".." in raw_path.split("/"):
            raise SourceLockError("Committed .gitmodules contains an unsafe submodule path")
        if raw_path in seen_paths:
            raise SourceLockError("Committed .gitmodules contains duplicate submodule paths")
        seen_paths.add(raw_path)
        try:
            location = (
                resolve_relative_repository_url(raw_url, parent_location)
                if "://" not in raw_url and not re.match(r"^(?:[^@]+@)?[^:]+:.+$", raw_url)
                else RepositoryLocation.parse(raw_url)
            )
            location = policy.validate(location)
        except RepositoryUrlError as exc:
            raise SourceLockError("Committed .gitmodules contains a rejected repository URL") from exc
        if raw_path == engine_path:
            matches.append(location)
    if len(matches) != 1:
        raise SourceLockError("Committed .gitmodules must declare the engine path exactly once")
    tree_line = _run_git(["ls-tree", "HEAD", "--", engine_path], cwd=repository)
    fields = tree_line.split()
    if len(fields) < 3 or fields[0] != "160000":
        raise SourceLockError("Committed engine path is not a gitlink")
    return matches[0], fields[2]


def _redact(text: object, secrets: tuple[str, ...] = ()) -> str:
    return bootstrap_core.redact(text, secrets)


def _git_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    return bootstrap_core.git_environment(overrides)


def _run_git(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    secrets: tuple[str, ...] = (),
) -> str:
    try:
        return bootstrap_core.run_git(arguments, cwd=cwd, env=env, secrets=secrets)
    except bootstrap_core.BootstrapError as exc:
        raise SourceLockError(str(exc)) from exc


def _policy_mapping(policy: CheckoutPolicy) -> dict[str, object]:
    result: dict[str, object] = {
        "allowed_hosts": sorted(host.lower() for host in policy.allowed_hosts),
        "allowed_schemes": sorted(policy.allowed_schemes),
        "nested_submodules": policy.nested_submodules,
        "max_submodule_depth": policy.max_submodule_depth,
    }
    if policy.ssh is not None:
        result["ssh"] = {
            "executable": str(policy.ssh.ssh_executable.resolve()),
            "agent_socket": policy.ssh.agent_socket,
            "known_hosts": str(policy.ssh.known_hosts.resolve()),
        }
    return result


def _credential_resolver(
    *, environment: Mapping[str, str] | None, provider_secret: SecretResolver | None,
    credential_helper: SecretResolver | None,
) -> bootstrap_core.CredentialResolver:
    def resolve(reference: Mapping[str, object]) -> str:
        return resolve_secret(
            SecretRef.from_dict(reference), environment=environment or os.environ,
            provider_secret=provider_secret, credential_helper=credential_helper,
        )

    return resolve


@contextmanager
def _credential_scope(
    location: RepositoryLocation,
    *,
    environment: Mapping[str, str] | None,
    provider_secret: SecretResolver | None,
    credential_helper: SecretResolver | None,
    ssh_policy: SSHCheckoutPolicy | None = None,
) -> Iterator[tuple[list[str], dict[str, str], tuple[str, ...], Path | None]]:
    """Adapt typed local credential callbacks to the shared stdlib core."""

    location_mapping = location.to_dict()
    policy = CheckoutPolicy(allowed_hosts=frozenset({location.host}), ssh=ssh_policy)
    try:
        with bootstrap_core.credential_scope(
            location_mapping, policy=_policy_mapping(policy), environment=environment,
            credential_resolver=_credential_resolver(
                environment=environment, provider_secret=provider_secret,
                credential_helper=credential_helper,
            ),
        ) as (config, process_env, secrets):
            yield config, process_env, secrets, None
    except bootstrap_core.BootstrapError as exc:
        raise SourceLockError(str(exc)) from exc


def _authenticated_remote_proof(
    *,
    environment: Mapping[str, str] | None,
    provider_secret: SecretResolver | None,
    credential_helper: SecretResolver | None,
    ssh_policy: SSHCheckoutPolicy | None = None,
) -> Callable[[RepositoryLocation, str, str], str | None]:
    """Build A's exact-ref callback using scoped HTTPS authentication."""

    def prove(location: RepositoryLocation, exact_ref: str, expected_head: str) -> str | None:
        with _credential_scope(
            location,
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
            ssh_policy=ssh_policy,
        ) as (config, process_env, secrets, _helper_dir):
            advertised = _run_git(
                [*config, "ls-remote", "--exit-code", location.canonical_url, exact_ref],
                env=process_env,
                secrets=secrets,
            )
            process_env.pop("SYNAPTIC_GIT_SECRET", None)
            process_env.pop("SYNAPTIC_GIT_HOST", None)
        matches: list[str] = []
        for line in advertised.splitlines():
            fields = line.split()
            if len(fields) != 2 or fields[1] != exact_ref or not re.fullmatch(
                r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?", fields[0]
            ):
                return None
            matches.append(fields[0].lower())
        if len(matches) != 1 or matches[0] != expected_head.lower():
            return None
        return matches[0]

    return prove


def validate_source_lock_for_cloud(source_lock: SourceLock) -> None:
    """Reject mutable or unavailable source identities before paid work."""

    for label, source in (
        ("project", source_lock.project_source),
        ("engine", source_lock.engine_source),
    ):
        if source.dirty:
            raise SourceLockError(f"Cloud checkout requires a clean {label} worktree")
        if not source.pushed:
            raise SourceLockError(f"Cloud checkout requires the exact {label} commit to be pushed")
    if source_lock.mode in {"superproject", "dual_clone"}:
        if source_lock.engine_source.commit.lower() != (
            source_lock.engine_source.gitlink_commit or ""
        ).lower():
            raise SourceLockError("Locked engine commit does not match the host gitlink")


def checkout_source_lock(
    source_lock: SourceLock,
    destination: Path,
    *,
    policy: CheckoutPolicy,
    clone_url_resolver: CloneUrlResolver | None = None,
    environment: Mapping[str, str] | None = None,
    provider_secret: SecretResolver | None = None,
    credential_helper: SecretResolver | None = None,
) -> CheckoutResult:
    """Adapt typed local values to the one canonical reconstruction core."""

    validate_source_lock_for_cloud(source_lock)

    def resolve_clone(location: Mapping[str, object]) -> str:
        typed = RepositoryLocation.parse(
            str(location["url"]),
            credential=location.get("credential") if isinstance(location.get("credential"), Mapping) else None,
        )
        return clone_url_resolver(typed) if clone_url_resolver else typed.canonical_url

    try:
        result = bootstrap_core.reconstruct_source_lock(
            source_lock.to_dict(), Path(destination), policy=_policy_mapping(policy),
            clone_url_resolver=resolve_clone if clone_url_resolver else None,
            environment=environment,
            credential_resolver=_credential_resolver(
                environment=environment, provider_secret=provider_secret,
                credential_helper=credential_helper,
            ),
            command_runner=_run_git,
            allow_legacy_metadata=True,
        )
    except bootstrap_core.BootstrapError as exc:
        raise SourceLockError(str(exc)) from exc
    return CheckoutResult(
        source_lock=source_lock,
        project_root=Path(str(result["project_root"])),
        engine_root=Path(str(result["engine_root"])),
    )
