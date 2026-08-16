"""Provider-neutral, exact-commit checkout for cloud execution."""

from __future__ import annotations

import os
import hashlib
import configparser
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

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
_GITMODULE_KEY = re.compile(r"^submodule\.(?P<name>.+)\.path$")


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


@dataclass(frozen=True)
class _Submodule:
    name: str
    path: str
    location: RepositoryLocation
    commit: str


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
    rendered = str(text)
    for secret in secrets:
        if secret:
            rendered = rendered.replace(secret, "<redacted>")
    # Defense in depth for URL-like text returned by Git.
    rendered = re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", rendered)
    return rendered


def _git_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a minimal cross-platform environment with no Git injection knobs."""

    source = dict(os.environ)
    if overrides:
        source.update(overrides)
    allowed = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "CURL_CA_BUNDLE",
    }
    environment = {key: value for key, value in source.items() if key.upper() in allowed}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": "https:ssh",
        }
    )
    return environment


def _run_git(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    secrets: tuple[str, ...] = (),
) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else _git_environment(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceLockError(f"Git checkout operation failed: {_redact(exc, secrets)}") from exc
    if completed.returncode:
        detail = _redact(completed.stderr.strip() or completed.stdout.strip(), secrets)
        raise SourceLockError(f"Git checkout operation failed: {detail or 'unknown error'}")
    return completed.stdout.strip()


@contextmanager
def _credential_scope(
    location: RepositoryLocation,
    *,
    environment: Mapping[str, str] | None,
    provider_secret: SecretResolver | None,
    credential_helper: SecretResolver | None,
    ssh_policy: SSHCheckoutPolicy | None = None,
) -> Iterator[tuple[list[str], dict[str, str], tuple[str, ...], Path | None]]:
    """Resolve one SecretRef and expose it only to an ephemeral scoped helper."""

    helper_dir = Path(tempfile.mkdtemp(prefix="synaptic-git-credential-"))
    home_dir = helper_dir / "home"
    template_dir = helper_dir / "template"
    hooks_dir = helper_dir / "hooks"
    home_dir.mkdir()
    template_dir.mkdir()
    hooks_dir.mkdir()
    process_env = _git_environment(environment)
    process_env["HOME"] = str(home_dir)
    process_env["USERPROFILE"] = str(home_dir)
    config = [
        "-c",
        "credential.helper=",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.ext.allow=never",
        "-c",
        f"core.hooksPath={hooks_dir}",
        "-c",
        f"init.templateDir={template_dir}",
    ]
    reference = location.credential
    if reference is None:
        if location.scheme == "ssh":
            if ssh_policy is None:
                shutil.rmtree(helper_dir, ignore_errors=True)
                raise SourceLockError(
                    "SSH checkout requires an explicit controlled agent and known_hosts policy"
                )
            process_env["SSH_AUTH_SOCK"] = ssh_policy.agent_socket
            process_env["GIT_SSH_VARIANT"] = "ssh"
            ssh_arguments = [
                str(ssh_policy.ssh_executable.resolve()),
                "-F", os.devnull,
                "-oBatchMode=yes",
                "-oStrictHostKeyChecking=yes",
                f"-oUserKnownHostsFile={ssh_policy.known_hosts.resolve()}",
                "-oGlobalKnownHostsFile=none",
                "-oIdentityFile=none",
                "-oIdentitiesOnly=no",
                "-oProxyCommand=none",
                "-oProxyJump=none",
                "-oForwardAgent=no",
                "-oClearAllForwardings=yes",
                "-oPermitLocalCommand=no",
                "-oLocalCommand=none",
                "-oRequestTTY=no",
            ]
            process_env["GIT_SSH_COMMAND"] = " ".join(
                shlex.quote(argument) for argument in ssh_arguments
            )
        try:
            yield config, process_env, (), helper_dir
        finally:
            shutil.rmtree(helper_dir, ignore_errors=True)
        return

    if location.scheme != "https":
        shutil.rmtree(helper_dir, ignore_errors=True)
        raise SourceLockError(
            "SecretRef-backed checkout currently requires HTTPS; SSH must use an external agent"
        )

    try:
        value = resolve_secret(
            reference,
            environment=environment or os.environ,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
        )
    except Exception:
        shutil.rmtree(helper_dir, ignore_errors=True)
        raise
    helper = helper_dir / "credential_helper.py"
    helper.write_text(
        "import os, sys\n"
        "request = dict(line.rstrip('\\n').split('=', 1) for line in sys.stdin if '=' in line)\n"
        "if request.get('host', '').lower() != os.environ['SYNAPTIC_GIT_HOST'].lower():\n"
        "    raise SystemExit(1)\n"
        "print('username=x-access-token')\n"
        "print('password=' + os.environ['SYNAPTIC_GIT_SECRET'])\n",
        encoding="utf-8",
    )
    process_env["SYNAPTIC_GIT_HOST"] = location.host
    process_env["SYNAPTIC_GIT_SECRET"] = value
    helper_command = f"!\"{sys.executable}\" \"{helper}\""
    config.extend(
        [
            "-c",
            f"credential.https://{location.host}.helper={helper_command}",
            "-c",
            "credential.useHttpPath=true",
        ]
    )
    try:
        yield config, process_env, (value,), helper_dir
    finally:
        process_env.pop("SYNAPTIC_GIT_SECRET", None)
        process_env.pop("SYNAPTIC_GIT_HOST", None)
        shutil.rmtree(helper_dir, ignore_errors=True)


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


def _clone_exact(
    source: GitSource,
    destination: Path,
    *,
    policy: CheckoutPolicy,
    clone_url_resolver: CloneUrlResolver | None,
    environment: Mapping[str, str] | None,
    provider_secret: SecretResolver | None,
    credential_helper: SecretResolver | None,
) -> None:
    location = policy.validate(source.location)
    clone_url = clone_url_resolver(location) if clone_url_resolver else location.canonical_url
    with _credential_scope(
        location,
        environment=environment,
        provider_secret=provider_secret,
        credential_helper=credential_helper,
        ssh_policy=policy.ssh,
    ) as (config, process_env, secrets, _helper_dir):
        if clone_url_resolver:
            # Local fixture transport is an explicit test seam; locked identity
            # validation still applies to the declared canonical URL.
            fixture_config: list[str] = []
            for index in range(0, len(config), 2):
                if config[index + 1] != "protocol.file.allow=never":
                    fixture_config.extend(config[index : index + 2])
            config = [*fixture_config, "-c", "protocol.file.allow=always"]
            process_env["GIT_ALLOW_PROTOCOL"] = "https:ssh:file"
        _run_git(
            [
                *config,
                "clone",
                f"--template={next(item.split('=', 1)[1] for item in config if item.startswith('init.templateDir='))}",
                "--no-checkout",
                "--no-recurse-submodules",
                clone_url,
                str(destination),
            ],
            env=process_env,
            secrets=secrets,
        )
        # The resolved value is needed only by clone authentication. Checkout,
        # verification, and hooks receive the same scrubbed env without it.
        process_env.pop("SYNAPTIC_GIT_SECRET", None)
        process_env.pop("SYNAPTIC_GIT_HOST", None)
        _run_git([*config, "checkout", "--detach", source.commit], cwd=destination, env=process_env, secrets=secrets)
        actual = _run_git(["rev-parse", "HEAD"], cwd=destination, env=process_env, secrets=secrets)
    if actual.lower() != source.commit.lower():
        raise SourceLockError("Checkout HEAD does not match the exact locked commit")


def _read_submodules(
    repository: Path,
    parent_location: RepositoryLocation,
    *,
    policy: CheckoutPolicy,
    depth: int,
) -> list[_Submodule]:
    document = repository / ".gitmodules"
    if not document.is_file():
        return []
    paths = _run_git(["config", "--file", str(document), "--get-regexp", r"^submodule\..*\.path$"] , cwd=repository)
    if not paths:
        return []
    if depth > 0 and not policy.nested_submodules:
        raise SourceLockError("Nested submodules are disabled by project policy")
    if depth >= policy.max_submodule_depth:
        raise SourceLockError("Submodule graph exceeds the approved maximum depth")

    entries: list[_Submodule] = []
    seen_paths: set[str] = set()
    for line in paths.splitlines():
        key, separator, raw_path = line.partition(" ")
        match = _GITMODULE_KEY.fullmatch(key)
        if not separator or not match:
            raise SourceLockError("Malformed .gitmodules submodule path entry")
        path = raw_path.strip().replace("\\", "/")
        if not path or path.startswith("/") or ".." in path.split("/") or path in seen_paths:
            raise SourceLockError("Submodule path must be unique and contained")
        seen_paths.add(path)
        name = match.group("name")
        raw_url = _run_git(["config", "--file", str(document), "--get", f"submodule.{name}.url"], cwd=repository)
        try:
            location = (
                resolve_relative_repository_url(raw_url, parent_location)
                if "://" not in raw_url and not re.match(r"^(?:[^@]+@)?[^:]+:.+$", raw_url)
                else RepositoryLocation.parse(raw_url)
            )
            location = policy.validate(location)
        except RepositoryUrlError as exc:
            raise SourceLockError(f"Rejected .gitmodules repository URL: {_redact(exc)}") from exc
        tree_line = _run_git(["ls-tree", "HEAD", "--", path], cwd=repository)
        fields = tree_line.split()
        if len(fields) < 3 or fields[0] != "160000":
            raise SourceLockError("Submodule declaration does not match a committed gitlink")
        entries.append(_Submodule(name=name, path=path, location=location, commit=fields[2]))
    return entries


def _materialize_submodules(
    repository: Path,
    parent_location: RepositoryLocation,
    *,
    policy: CheckoutPolicy,
    depth: int,
    locked_engine: GitSource,
    clone_url_resolver: CloneUrlResolver | None,
    environment: Mapping[str, str] | None,
    provider_secret: SecretResolver | None,
    credential_helper: SecretResolver | None,
) -> None:
    # All declarations at this level are parsed and policy-validated before
    # the first child fetch, preventing partial traversal of a hostile graph.
    entries = _read_submodules(repository, parent_location, policy=policy, depth=depth)
    for entry in entries:
        credential: SecretRef | None = None
        if entry.path == locked_engine.submodule_path and depth == 0:
            if entry.commit.lower() != locked_engine.commit.lower():
                raise SourceLockError("Host gitlink does not match the locked engine commit")
            if entry.location.canonical_url != locked_engine.location.canonical_url:
                raise SourceLockError("Host engine submodule URL does not match the source lock")
            credential = locked_engine.location.credential
        elif entry.location.host == parent_location.host:
            credential = parent_location.credential
        child_location = RepositoryLocation.parse(entry.location.canonical_url, credential=credential)
        child = GitSource(location=child_location, commit=entry.commit, pushed=True)
        target = (repository / entry.path).resolve()
        try:
            target.relative_to(repository.resolve())
        except ValueError as exc:
            raise SourceLockError("Submodule destination escapes its parent checkout") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        _clone_exact(
            child,
            target,
            policy=policy,
            clone_url_resolver=clone_url_resolver,
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
        )
        _materialize_submodules(
            target,
            child_location,
            policy=policy,
            depth=depth + 1,
            locked_engine=locked_engine,
            clone_url_resolver=clone_url_resolver,
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
        )


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
    """Reconstruct and verify a locked standalone, superproject, or dual clone."""

    validate_source_lock_for_cloud(source_lock)
    project_location = policy.validate(source_lock.project_source.location)
    policy.validate(source_lock.engine_source.location)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SourceLockError("Checkout destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)

    if source_lock.mode == "standalone":
        project_root = destination / "project"
        _clone_exact(
            source_lock.project_source,
            project_root,
            policy=policy,
            clone_url_resolver=clone_url_resolver,
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
        )
        return CheckoutResult(source_lock, project_root, project_root)

    project_root = destination / "project"
    _clone_exact(
        source_lock.project_source,
        project_root,
        policy=policy,
        clone_url_resolver=clone_url_resolver,
        environment=environment,
        provider_secret=provider_secret,
        credential_helper=credential_helper,
    )
    engine_path = source_lock.engine_source.submodule_path or ""
    root_entries = _read_submodules(project_root, project_location, policy=policy, depth=0)
    matching = [entry for entry in root_entries if entry.path == engine_path]
    if len(matching) != 1 or matching[0].commit.lower() != source_lock.engine_source.commit.lower():
        raise SourceLockError("Locked host gitlink does not identify the locked engine commit")
    if matching[0].location.canonical_url != source_lock.engine_source.location.canonical_url:
        raise SourceLockError("Locked host submodule URL does not identify the locked engine source")

    if source_lock.mode == "superproject":
        _materialize_submodules(
            project_root,
            project_location,
            policy=policy,
            depth=0,
            locked_engine=source_lock.engine_source,
            clone_url_resolver=clone_url_resolver,
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
        )
        engine_root = (project_root / engine_path).resolve()
    else:
        engine_root = destination / "engine"
        _clone_exact(
            source_lock.engine_source,
            engine_root,
            policy=policy,
            clone_url_resolver=clone_url_resolver,
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
        )
        _materialize_submodules(
            engine_root,
            source_lock.engine_source.location,
            policy=policy,
            depth=1,
            locked_engine=source_lock.engine_source,
            clone_url_resolver=clone_url_resolver,
            environment=environment,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
        )

    actual_engine = _run_git(["rev-parse", "HEAD"], cwd=engine_root)
    if actual_engine.lower() != source_lock.engine_source.commit.lower():
        raise SourceLockError("Reconstructed engine does not match the source lock")
    return CheckoutResult(source_lock, project_root, engine_root)
