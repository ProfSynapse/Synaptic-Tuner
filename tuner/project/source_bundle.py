"""Canonical source-lock model shared by local and cloud execution."""

from __future__ import annotations

import json
import os
import posixpath
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

from .errors import RepositoryUrlError, SourceLockError
from .secrets import SecretRef, redact_secrets, reject_literal_secrets

SourceMode = Literal["standalone", "superproject", "dual_clone"]
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_SCP_RE = re.compile(
    r"^(?:(?P<user>[A-Za-z0-9._-]+)@)?"
    r"(?P<host>[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?):"
    r"(?P<path>[A-Za-z0-9._~/-]+)$"
)
_SAFE_REPO_PATH_RE = re.compile(r"^/[A-Za-z0-9._~/-]+$")
_REMOTE_BRANCH_REF_RE = re.compile(r"^refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]*$")
_REMOTE_PROOF_TIMEOUT_SECONDS = 3

RemoteProof = Callable[["RepositoryLocation", str, str], str | bool | None]


def _normalize_repo_path(value: str) -> str:
    path = "/" + value.replace("\\", "/").lstrip("/")
    if not _SAFE_REPO_PATH_RE.fullmatch(path):
        raise RepositoryUrlError("Repository URL path contains unsupported characters")
    normalized = posixpath.normpath(path)
    if normalized in {"/", "/."} or normalized.startswith("/../"):
        raise RepositoryUrlError("Repository URL must contain a valid repository path")
    return normalized


@dataclass(frozen=True)
class RepositoryLocation:
    canonical_url: str
    scheme: Literal["https", "ssh"]
    host: str
    path: str
    credential: SecretRef | None = None

    @classmethod
    def parse(
        cls,
        url: str,
        *,
        credential: SecretRef | Mapping[str, object] | None = None,
        allowed_hosts: set[str] | None = None,
        allowed_schemes: set[str] | None = None,
    ) -> "RepositoryLocation":
        location = canonicalize_repository_url(url)
        if allowed_hosts is not None and location.host not in {h.lower() for h in allowed_hosts}:
            raise RepositoryUrlError(f"Repository host is not allowed: {location.host}")
        if allowed_schemes is not None and location.scheme not in allowed_schemes:
            raise RepositoryUrlError(f"Repository scheme is not allowed: {location.scheme}")
        if credential is not None:
            reference = (
                credential
                if isinstance(credential, SecretRef)
                else SecretRef.from_dict(credential)
            )
            return cls(
                canonical_url=location.canonical_url,
                scheme=location.scheme,
                host=location.host,
                path=location.path,
                credential=reference,
            )
        return location

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "url": self.canonical_url,
            "scheme": self.scheme,
            "host": self.host,
            "path": self.path,
        }
        if self.credential:
            result["credential"] = self.credential.to_dict()
        return result


def canonicalize_repository_url(url: str) -> RepositoryLocation:
    if not isinstance(url, str) or not url.strip():
        raise RepositoryUrlError("Repository URL must be a non-empty string")
    candidate = url.strip()
    scp = _SCP_RE.fullmatch(candidate) if "://" not in candidate else None
    if scp:
        user = scp.group("user") or "git"
        if user != "git":
            raise RepositoryUrlError("SSH repository URLs may use only the 'git' username")
        host = scp.group("host").lower()
        path = _normalize_repo_path(scp.group("path"))
        return RepositoryLocation(
            canonical_url=f"ssh://git@{host}{path}", scheme="ssh", host=host, path=path
        )

    parsed = urlsplit(candidate)
    scheme = parsed.scheme.lower()
    if scheme not in {"https", "ssh"}:
        raise RepositoryUrlError("Repository URL scheme must be https or ssh")
    if parsed.query or parsed.fragment:
        raise RepositoryUrlError("Repository URLs cannot contain query strings or fragments")
    try:
        port = parsed.port
    except ValueError as exc:
        raise RepositoryUrlError("Repository URL has an invalid port") from exc
    if not parsed.hostname:
        raise RepositoryUrlError("Repository URL must contain a host")
    host = parsed.hostname.lower()
    path = _normalize_repo_path(parsed.path)
    if scheme == "https":
        if parsed.username is not None or parsed.password is not None:
            raise RepositoryUrlError("HTTPS repository URLs cannot contain userinfo")
        netloc = host if port in {None, 443} else f"{host}:{port}"
    else:
        if parsed.password is not None:
            raise RepositoryUrlError("SSH repository URLs cannot contain passwords")
        if parsed.username not in {None, "git"}:
            raise RepositoryUrlError("SSH repository URLs may use only the 'git' username")
        netloc = f"git@{host}"
        if port not in {None, 22}:
            netloc += f":{port}"
    canonical = urlunsplit((scheme, netloc, path, "", ""))
    return RepositoryLocation(canonical_url=canonical, scheme=scheme, host=host, path=path)


def resolve_relative_repository_url(
    url: str, parent: RepositoryLocation
) -> RepositoryLocation:
    """Resolve a relative .gitmodules URL against an approved parent identity."""

    if "://" in url or _SCP_RE.fullmatch(url):
        return RepositoryLocation.parse(url)
    if not url or url.startswith(("/", "\\")):
        raise RepositoryUrlError("Relative repository URL must be repository-relative")
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(parent.path), url))
    if joined.startswith("../"):
        raise RepositoryUrlError("Relative repository URL escapes the repository host root")
    if parent.scheme == "ssh":
        return RepositoryLocation.parse(f"ssh://git@{parent.host}/{joined.lstrip('/')}")
    return RepositoryLocation.parse(f"https://{parent.host}/{joined.lstrip('/')}")


@dataclass(frozen=True)
class GitSource:
    location: RepositoryLocation
    commit: str
    branch: str | None = None
    dirty: bool = False
    pushed: bool = False
    submodule_path: str | None = None
    gitlink_commit: str | None = None

    def __post_init__(self) -> None:
        if not _COMMIT_RE.fullmatch(self.commit):
            raise SourceLockError("Source commit must be a full 40- or 64-character hash")
        if self.gitlink_commit and not _COMMIT_RE.fullmatch(self.gitlink_commit):
            raise SourceLockError("Gitlink commit must be a full 40- or 64-character hash")
        if self.submodule_path:
            path = self.submodule_path.replace("\\", "/")
            if path.startswith("/") or ".." in path.split("/"):
                raise SourceLockError("Submodule path must be a contained relative path")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GitSource":
        raw_url = value.get("url")
        if not isinstance(raw_url, str):
            raise SourceLockError("Git source requires a repository URL")
        credential = value.get("credential")
        return cls(
            location=RepositoryLocation.parse(
                raw_url,
                credential=credential if isinstance(credential, Mapping) else None,
            ),
            commit=str(value.get("commit", "")),
            branch=str(value["branch"]) if value.get("branch") is not None else None,
            dirty=bool(value.get("dirty", False)),
            pushed=bool(value.get("pushed", False)),
            submodule_path=(
                str(value["submodule_path"])
                if value.get("submodule_path") is not None
                else None
            ),
            gitlink_commit=(
                str(value["gitlink_commit"])
                if value.get("gitlink_commit") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "url": self.location.canonical_url,
            "commit": self.commit.lower(),
            "dirty": self.dirty,
            "pushed": self.pushed,
        }
        if self.branch is not None:
            result["branch"] = self.branch
        if self.submodule_path is not None:
            result["submodule_path"] = self.submodule_path.replace("\\", "/")
        if self.gitlink_commit is not None:
            result["gitlink_commit"] = self.gitlink_commit.lower()
        if self.location.credential is not None:
            result["credential"] = self.location.credential.to_dict()
        return result


@dataclass(frozen=True)
class SourceLock:
    run_id: str
    mode: SourceMode
    project_source: GitSource
    engine_source: GitSource
    project: Mapping[str, Any]
    configuration: Mapping[str, Any]
    plugins: tuple[Mapping[str, Any], ...] = ()
    inputs: tuple[Mapping[str, Any], ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    schema_version: str = "synaptic-source-lock/v1"

    def __post_init__(self) -> None:
        if self.mode not in {"standalone", "superproject", "dual_clone"}:
            raise SourceLockError(f"Unsupported source-lock mode: {self.mode}")
        if not self.run_id:
            raise SourceLockError("Source lock requires a run_id")
        if self.mode in {"superproject", "dual_clone"}:
            if not self.engine_source.submodule_path or not self.engine_source.gitlink_commit:
                raise SourceLockError(
                    f"{self.mode} mode requires engine submodule_path and gitlink_commit"
                )
            if self.engine_source.gitlink_commit.lower() != self.engine_source.commit.lower():
                raise SourceLockError("Engine commit must equal the host gitlink commit")
        if self.mode == "standalone" and (
            self.project_source.commit.lower() != self.engine_source.commit.lower()
            or self.project_source.location.canonical_url
            != self.engine_source.location.canonical_url
        ):
            raise SourceLockError("Standalone mode requires identical project and engine sources")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SourceLock":
        if value.get("schema_version") != "synaptic-source-lock/v1":
            raise SourceLockError("Unsupported source-lock schema version")
        sources = value.get("sources")
        if not isinstance(sources, Mapping):
            raise SourceLockError("Source lock requires sources.project and sources.engine")
        project_source = sources.get("project")
        engine_source = sources.get("engine")
        if not isinstance(project_source, Mapping) or not isinstance(engine_source, Mapping):
            raise SourceLockError("Source lock requires project and engine source mappings")
        return cls(
            run_id=str(value.get("run_id", "")),
            created_at=str(value.get("created_at", "")),
            mode=str(value.get("mode", "")),  # type: ignore[arg-type]
            project_source=GitSource.from_dict(project_source),
            engine_source=GitSource.from_dict(engine_source),
            project=dict(value.get("project", {})) if isinstance(value.get("project"), Mapping) else {},
            configuration=(
                dict(value.get("configuration", {}))
                if isinstance(value.get("configuration"), Mapping)
                else {}
            ),
            plugins=tuple(value.get("plugins", ())) if isinstance(value.get("plugins"), list) else (),
            inputs=tuple(value.get("inputs", ())) if isinstance(value.get("inputs"), list) else (),
            runtime=dict(value.get("runtime", {})) if isinstance(value.get("runtime"), Mapping) else {},
            outputs=dict(value.get("outputs", {})) if isinstance(value.get("outputs"), Mapping) else {},
        )

    def to_dict(self) -> dict[str, object]:
        for section in (
            self.project,
            self.configuration,
            self.plugins,
            self.inputs,
            self.runtime,
            self.outputs,
        ):
            reject_literal_secrets(section)
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "mode": self.mode,
            "sources": {
                "project": self.project_source.to_dict(),
                "engine": self.engine_source.to_dict(),
            },
            "project": redact_secrets(dict(self.project)),
            "configuration": redact_secrets(dict(self.configuration)),
            "plugins": [redact_secrets(dict(item)) for item in self.plugins],
            "inputs": [redact_secrets(dict(item)) for item in self.inputs],
            "runtime": redact_secrets(dict(self.runtime)),
            "outputs": redact_secrets(dict(self.outputs)),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _minimal_git_environment() -> dict[str, str]:
    """Return an ambient-independent environment for read-only Git inspection."""

    inherited = {
        key: os.environ[key]
        for key in (
            "PATH",
            "SystemRoot",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "LANG",
            "LC_ALL",
        )
        if key in os.environ
    }
    inherited.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return inherited


def _git(repository: Path, *args: str, allow_failure: bool = False) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=not allow_failure,
            capture_output=True,
            text=True,
            timeout=20,
            env=_minimal_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceLockError(f"Could not inspect Git source at {repository}: {exc}") from exc
    return result.stdout.strip()


def _remote_probe_environment() -> dict[str, str]:
    """Build a minimal noninteractive environment for an untrusted remote probe."""

    inherited = _minimal_git_environment()
    inherited.update(
        {
            "GIT_SSH_COMMAND": (
                f"ssh -F {os.devnull} -oBatchMode=yes -oClearAllForwardings=yes "
                "-oProxyCommand=none -oProxyJump=none -oPermitLocalCommand=no "
                "-oIdentityAgent=none -oIdentitiesOnly=yes -oIdentityFile=none "
                "-oPreferredAuthentications=none -oCanonicalizeHostname=no"
            ),
            "GIT_SSH_VARIANT": "ssh",
        }
    )
    return inherited


def _valid_remote_branch_ref(ref: str) -> bool:
    return bool(
        _REMOTE_BRANCH_REF_RE.fullmatch(ref)
        and ".." not in ref
        and "//" not in ref
        and "@{" not in ref
        and not ref.endswith(("/", ".", ".lock"))
    )


def _remote_ref_sha(location: RepositoryLocation, ref: str) -> str | None:
    """Return the exact SHA advertised by a validated origin ref, if provable."""

    if location.credential is not None:
        return None
    if (
        not _valid_remote_branch_ref(ref)
    ):
        return None
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", location.canonical_url, ref],
            check=False,
            capture_output=True,
            text=True,
            timeout=_REMOTE_PROOF_TIMEOUT_SECONDS,
            env=_remote_probe_environment(),
            cwd=Path(location.canonical_url).anchor or Path.cwd().anchor,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None

    matches: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[1] != ref or not _COMMIT_RE.fullmatch(fields[0]):
            return None
        matches.append(fields[0].lower())
    if len(matches) != 1:
        return None
    return matches[0]


def inspect_git_source(
    repository: Path,
    *,
    submodule_path: str | None = None,
    gitlink_commit: str | None = None,
    credential: SecretRef | None = None,
    remote_proof: RemoteProof | None = None,
) -> GitSource:
    """Inspect one local source without persisting authentication material.

    ``remote_proof`` is a trusted authenticated boundary. It receives only a
    validated repository location, exact upstream ref, and expected HEAD. A
    boolean true is accepted as the caller's assertion that it verified that
    exact tuple; returning the advertised SHA is preferred.
    """

    commit = _git(repository, "rev-parse", "HEAD")
    url = _git(repository, "config", "--local", "--get", "remote.origin.url")
    # Reject unsafe or credential-bearing origins before any remote operation.
    location = RepositoryLocation.parse(url, credential=credential)
    branch = _git(repository, "branch", "--show-current", allow_failure=True) or None
    # Normal status includes untracked source files while respecting .gitignore,
    # so host-owned .synaptic runtime data does not make a source dirty.
    dirty = bool(_git(repository, "status", "--porcelain", "--untracked-files=normal"))
    pushed = False
    if branch:
        upstream_remote = _git(
            repository,
            "config",
            "--local",
            "--get",
            f"branch.{branch}.remote",
            allow_failure=True,
        )
        upstream_ref = _git(
            repository,
            "config",
            "--local",
            "--get",
            f"branch.{branch}.merge",
            allow_failure=True,
        )
        if upstream_remote == "origin" and _valid_remote_branch_ref(upstream_ref):
            if remote_proof is not None:
                try:
                    proof = remote_proof(location, upstream_ref, commit.lower())
                except Exception:
                    proof = None
                if isinstance(proof, bool):
                    pushed = proof
                elif isinstance(proof, str) and _COMMIT_RE.fullmatch(proof):
                    pushed = proof.lower() == commit.lower()
            elif location.credential is None:
                advertised = _remote_ref_sha(location, upstream_ref)
                pushed = advertised == commit.lower()
    return GitSource(
        location=location,
        commit=commit,
        branch=branch,
        dirty=dirty,
        pushed=pushed,
        submodule_path=submodule_path,
        gitlink_commit=gitlink_commit,
    )
