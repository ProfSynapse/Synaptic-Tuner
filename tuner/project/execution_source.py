"""Canonical execution-source identity crossing the engine runtime boundary."""

from __future__ import annotations

import hashlib
import json
import re
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

from .context import ProjectContext
from .source_bundle import GitSource, RepositoryLocation, SourceLock, SourceLockError


EXECUTION_SOURCE_SCHEMA = "synaptic-execution-source/v1"
SOURCE_EVIDENCE_SCHEMA = "synaptic-authenticated-source-evidence/v1"
RUNTIME_SCHEMA = "synaptic-modal-runtime/v1"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_ROOT_NAMES = ("engine", "project", "artifacts", "state", "tracking", "cache", "tmp")


def _canonical(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SourceLockError("execution source must contain only finite JSON values") from exc


def _text(value: object, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceLockError(f"{name} must be nonblank canonical text")
    if len(value.encode("utf-8")) > maximum or any(ord(char) < 0x20 for char in value):
        raise SourceLockError(f"{name} must be bounded text without controls")
    return value


def _safe_ref(value: object, name: str) -> str:
    result = _text(value, name, maximum=256)
    if _SAFE_REF_RE.fullmatch(result) is None:
        raise SourceLockError(f"{name} must be a safe reference")
    return result


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise SourceLockError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise SourceLockError(f"{name} must be an exact lowercase commit")
    return value


def _timestamp(value: object, name: str) -> str:
    result = _text(value, name)
    try:
        parsed = datetime.fromisoformat(result[:-1] + "+00:00" if result.endswith("Z") else result)
    except ValueError as exc:
        raise SourceLockError(f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SourceLockError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _submodule_path(value: object) -> str:
    path = _text(value, "engine_submodule_path")
    if path.startswith(("/", "\\")) or "\\" in path or any(
        part in {"", ".", ".."} for part in path.split("/")
    ):
        raise SourceLockError("engine_submodule_path must be canonical relative POSIX")
    return path


def _source_dict(source: GitSource, *, engine: bool) -> dict[str, object]:
    if source.location.credential is not None:
        raise SourceLockError("execution source cannot embed repository credentials")
    result: dict[str, object] = {
        "url": source.location.canonical_url,
        "commit": _commit(source.commit.lower(), "source commit"),
    }
    if engine:
        result.update(
            {
                "submodule_path": _submodule_path(source.submodule_path),
                "gitlink_commit": _commit(source.gitlink_commit, "gitlink_commit"),
            }
        )
    return result


@dataclass(frozen=True, slots=True)
class AuthenticatedSourceEvidenceV1:
    project_url: str
    project_commit: str
    engine_url: str
    engine_commit: str
    engine_submodule_path: str
    gitlink_commit: str
    issuer_ref: str
    evidence_ref: str
    audience_ref: str
    challenge_nonce: str
    verified_at: str
    expires_at: str
    key_ref: str
    tag_base64: str
    attestation_digest: str
    schema_version: str = SOURCE_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_EVIDENCE_SCHEMA:
            raise SourceLockError("unsupported authenticated source-evidence schema")
        object.__setattr__(self, "project_url", RepositoryLocation.parse(self.project_url).canonical_url)
        object.__setattr__(self, "engine_url", RepositoryLocation.parse(self.engine_url).canonical_url)
        for name in ("project_commit", "engine_commit", "gitlink_commit"):
            object.__setattr__(self, name, _commit(getattr(self, name), name))
        object.__setattr__(self, "engine_submodule_path", _submodule_path(self.engine_submodule_path))
        object.__setattr__(self, "issuer_ref", _safe_ref(self.issuer_ref, "issuer_ref"))
        object.__setattr__(self, "evidence_ref", _safe_ref(self.evidence_ref, "evidence_ref"))
        for name in ("audience_ref", "challenge_nonce", "key_ref"):
            object.__setattr__(self, name, _safe_ref(getattr(self, name), name))
        from tuner.execution.evidence import canonical_utc
        object.__setattr__(self, "verified_at", canonical_utc(self.verified_at, "verified_at"))
        object.__setattr__(self, "expires_at", canonical_utc(self.expires_at, "expires_at"))
        if not isinstance(self.tag_base64, str) or not self.tag_base64 or not self.tag_base64.isascii():
            raise SourceLockError("tag_base64 must be canonical Base64")
        try:
            tag = base64.b64decode(self.tag_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise SourceLockError("tag_base64 must be canonical Base64") from exc
        if not tag or base64.b64encode(tag).decode("ascii") != self.tag_base64:
            raise SourceLockError("tag_base64 must be canonical Base64")
        object.__setattr__(
            self, "attestation_digest", _digest(self.attestation_digest, "attestation_digest")
        )

    def binds(self, source_lock: SourceLock) -> bool:
        return (
            source_lock.mode == "superproject"
            and self.project_url == source_lock.project_source.location.canonical_url
            and self.project_commit == source_lock.project_source.commit.lower()
            and self.engine_url == source_lock.engine_source.location.canonical_url
            and self.engine_commit == source_lock.engine_source.commit.lower()
            and self.engine_submodule_path == source_lock.engine_source.submodule_path
            and self.gitlink_commit == source_lock.engine_source.gitlink_commit
            and self.engine_commit == self.gitlink_commit
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "project": {"url": self.project_url, "commit": self.project_commit},
            "engine": {
                "url": self.engine_url,
                "commit": self.engine_commit,
                "submodule_path": self.engine_submodule_path,
                "gitlink_commit": self.gitlink_commit,
            },
            "issuer_ref": self.issuer_ref,
            "evidence_ref": self.evidence_ref,
            "audience_ref": self.audience_ref,
            "challenge_nonce": self.challenge_nonce,
            "verified_at": self.verified_at,
            "expires_at": self.expires_at,
            "key_ref": self.key_ref,
            "tag_base64": self.tag_base64,
            "attestation_digest": self.attestation_digest,
        }

    @property
    def authenticated_payload(self) -> bytes:
        value = self.to_dict()
        value.pop("tag_base64")
        value.pop("attestation_digest")
        return _canonical(value)

    @property
    def tag(self) -> bytes:
        return base64.b64decode(self.tag_base64, validate=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AuthenticatedSourceEvidenceV1":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "project", "engine", "issuer_ref", "evidence_ref",
            "audience_ref", "challenge_nonce", "verified_at", "expires_at", "key_ref",
            "tag_base64", "attestation_digest",
        }:
            raise SourceLockError("authenticated source evidence has missing or unknown fields")
        project = value.get("project")
        engine = value.get("engine")
        if not isinstance(project, Mapping) or set(project) != {"url", "commit"}:
            raise SourceLockError("authenticated project evidence is malformed")
        if not isinstance(engine, Mapping) or set(engine) != {
            "url", "commit", "submodule_path", "gitlink_commit"
        }:
            raise SourceLockError("authenticated engine evidence is malformed")
        return cls(
            schema_version=value["schema_version"], project_url=project["url"],
            project_commit=project["commit"], engine_url=engine["url"],
            engine_commit=engine["commit"], engine_submodule_path=engine["submodule_path"],
            gitlink_commit=engine["gitlink_commit"], issuer_ref=value["issuer_ref"],
            evidence_ref=value["evidence_ref"], audience_ref=value["audience_ref"],
            challenge_nonce=value["challenge_nonce"], verified_at=value["verified_at"],
            expires_at=value["expires_at"], key_ref=value["key_ref"],
            tag_base64=value["tag_base64"],
            attestation_digest=value["attestation_digest"],
        )


@dataclass(frozen=True, slots=True)
class ExecutionSourceV1:
    run_id: str
    created_at: str
    project_source: GitSource
    engine_source: GitSource
    engine_submodule_path: str
    source_evidence: AuthenticatedSourceEvidenceV1
    deployment_member_sha256: str
    roots: Mapping[str, str]
    writable_capability_root: str
    python_implementation: str
    python_version: str
    python_executable: str
    python_executable_digest: str
    environment: Mapping[str, str]
    secret_requirements_digest: str
    provider_runtime_requirements_digest: str
    schema_version: str = EXECUTION_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_SOURCE_SCHEMA:
            raise SourceLockError("unsupported execution-source schema")
        object.__setattr__(self, "run_id", _safe_ref(self.run_id, "run_id"))
        if "/" in self.run_id:
            raise SourceLockError("run_id must be one canonical path segment")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if not isinstance(self.project_source, GitSource) or not isinstance(self.engine_source, GitSource):
            raise TypeError("execution sources must be GitSource values")
        if self.project_source.dirty or self.engine_source.dirty:
            raise SourceLockError("execution sources must be clean")
        path = _submodule_path(self.engine_submodule_path)
        object.__setattr__(self, "engine_submodule_path", path)
        if self.engine_source.submodule_path != path:
            raise SourceLockError("engine submodule path does not match the source")
        if self.engine_source.commit.lower() != str(self.engine_source.gitlink_commit).lower():
            raise SourceLockError("engine commit does not match the host gitlink")
        if not isinstance(self.source_evidence, AuthenticatedSourceEvidenceV1):
            raise TypeError("source_evidence must be AuthenticatedSourceEvidenceV1")
        provisional = SourceLock(
            run_id=self.run_id, created_at=self.created_at, mode="superproject",
            project_source=self.project_source, engine_source=self.engine_source,
            project={}, configuration={},
        )
        if not self.source_evidence.binds(provisional):
            raise SourceLockError("authenticated evidence does not bind the execution sources")
        object.__setattr__(
            self, "deployment_member_sha256",
            _digest(self.deployment_member_sha256, "deployment_member_sha256"),
        )
        roots = dict(self.roots)
        if set(roots) != set(_ROOT_NAMES):
            raise SourceLockError("execution source requires the exact seven runtime roots")
        for name, value in roots.items():
            value = _text(value, f"runtime root {name}")
            if not (value.startswith("/") or Path(value).is_absolute()) or value.endswith(("/", "\\")):
                raise SourceLockError("runtime roots must be canonical absolute paths")
            roots[name] = value
        if len(set(roots.values())) != len(roots):
            raise SourceLockError("runtime roots must not alias")
        capability = _text(
            self.writable_capability_root, "writable capability root"
        )
        if not (
            capability.startswith("/") or Path(capability).is_absolute()
        ) or capability.endswith(("/", "\\")):
            raise SourceLockError(
                "writable capability root must be a canonical absolute path"
            )
        capability_path = Path(capability)
        if any(
            capability_path not in Path(roots[name]).parents
            for name in ("artifacts", "state", "tracking", "cache", "tmp")
        ):
            raise SourceLockError(
                "writable runtime roots must descend from their capability root"
            )
        if any(
            capability_path == Path(roots[name])
            or capability_path in Path(roots[name]).parents
            or Path(roots[name]) in capability_path.parents
            for name in ("engine", "project")
        ):
            raise SourceLockError(
                "writable capability root must be disjoint from sources"
            )
        object.__setattr__(self, "writable_capability_root", capability)
        for writable in ("artifacts", "state", "tracking", "cache", "tmp"):
            if any(
                Path(roots[writable]) == Path(roots[source])
                or Path(roots[writable]) in Path(roots[source]).parents
                or Path(roots[source]) in Path(roots[writable]).parents
                for source in ("engine", "project")
            ):
                raise SourceLockError("writable runtime roots must be disjoint from sources")
        object.__setattr__(self, "roots", MappingProxyType(roots))
        if self.python_implementation != "cpython":
            raise SourceLockError("execution runtime requires CPython")
        if re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", self.python_version) is None:
            raise SourceLockError("python_version must be exact major.minor.micro")
        executable = _text(self.python_executable, "python_executable")
        if not (executable.startswith("/") or Path(executable).is_absolute()):
            raise SourceLockError("python_executable must be an exact absolute provider path")
        object.__setattr__(self, "python_executable", executable)
        object.__setattr__(
            self, "python_executable_digest",
            _digest(self.python_executable_digest, "python_executable_digest"),
        )
        environment = dict(self.environment)
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            or any(ord(char) < 0x20 for char in key + value)
            for key, value in environment.items()
        ):
            raise SourceLockError("runtime environment must be a closed text map")
        required_environment = {
            "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
            "PYTHONPATH": roots["engine"], "SYNAPTIC_ENGINE_ROOT": roots["engine"],
            "SYNAPTIC_PROJECT_ROOT": roots["project"],
            "SYNAPTIC_ARTIFACT_ROOT": roots["artifacts"],
            "SYNAPTIC_STATE_ROOT": roots["state"],
            "SYNAPTIC_TRACKING_ROOT": roots["tracking"],
            "SYNAPTIC_CACHE_ROOT": roots["cache"], "SYNAPTIC_TMP_ROOT": roots["tmp"],
            "HF_HOME": roots["cache"] + "/huggingface",
            "TRANSFORMERS_CACHE": roots["cache"] + "/transformers",
            "WANDB_DISABLED": "true",
        }
        if any(environment.get(key) != value for key, value in required_environment.items()):
            raise SourceLockError("runtime environment does not bind the exact roots and isolation")
        object.__setattr__(self, "environment", MappingProxyType(environment))
        object.__setattr__(
            self, "secret_requirements_digest",
            _digest(self.secret_requirements_digest, "secret_requirements_digest"),
        )
        object.__setattr__(
            self, "provider_runtime_requirements_digest",
            _digest(self.provider_runtime_requirements_digest, "provider_runtime_requirements_digest"),
        )
        _canonical(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "topology": {
                "provenance_mode": "superproject", "execution_mode": "dual_clone",
                "engine_submodule_path": self.engine_submodule_path,
            },
            "sources": {
                "project": _source_dict(self.project_source, engine=False),
                "engine": _source_dict(self.engine_source, engine=True),
            },
            "source_evidence": self.source_evidence.to_dict(),
            "deployment_member_sha256": self.deployment_member_sha256,
            "runtime": {
                "schema_version": RUNTIME_SCHEMA,
                "roots": dict(self.roots),
                "capability_roots": {
                    "writable": self.writable_capability_root,
                },
                "interpreter": {
                    "implementation": self.python_implementation,
                    "version": self.python_version,
                    "executable": self.python_executable,
                    "executable_digest": self.python_executable_digest,
                },
                "environment": {
                    "clear_inherited": True,
                    "variables": dict(self.environment),
                    "secret_requirements_digest": self.secret_requirements_digest,
                    "provider_runtime_requirements_digest": self.provider_runtime_requirements_digest,
                },
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExecutionSourceV1":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "run_id", "created_at", "topology", "sources",
            "source_evidence", "deployment_member_sha256", "runtime",
        }:
            raise SourceLockError("execution source has missing or unknown fields")
        topology = value.get("topology")
        sources = value.get("sources")
        runtime = value.get("runtime")
        if not isinstance(topology, Mapping) or set(topology) != {
            "provenance_mode", "execution_mode", "engine_submodule_path"
        } or topology.get("provenance_mode") != "superproject" or topology.get("execution_mode") != "dual_clone":
            raise SourceLockError("execution source topology is malformed")
        if not isinstance(sources, Mapping) or set(sources) != {"project", "engine"}:
            raise SourceLockError("execution sources are malformed")
        project = sources.get("project")
        engine = sources.get("engine")
        if not isinstance(project, Mapping) or set(project) != {"url", "commit"}:
            raise SourceLockError("project execution source is malformed")
        if not isinstance(engine, Mapping) or set(engine) != {
            "url", "commit", "submodule_path", "gitlink_commit"
        }:
            raise SourceLockError("engine execution source is malformed")
        if not isinstance(runtime, Mapping) or set(runtime) != {
            "schema_version", "roots", "capability_roots", "interpreter", "environment"
        } or runtime.get("schema_version") != RUNTIME_SCHEMA:
            raise SourceLockError("execution runtime is malformed")
        roots = runtime.get("roots")
        capability_roots = runtime.get("capability_roots")
        interpreter = runtime.get("interpreter")
        environment = runtime.get("environment")
        if (
            not isinstance(roots, Mapping)
            or not isinstance(capability_roots, Mapping)
            or set(capability_roots) != {"writable"}
            or not isinstance(interpreter, Mapping)
            or set(interpreter) != {
            "implementation", "version", "executable", "executable_digest"
            }
        ):
            raise SourceLockError("execution interpreter is malformed")
        if not isinstance(environment, Mapping) or set(environment) != {
            "clear_inherited", "variables", "secret_requirements_digest",
            "provider_runtime_requirements_digest",
        } or environment.get("clear_inherited") is not True or not isinstance(environment.get("variables"), Mapping):
            raise SourceLockError("execution environment is malformed")
        return cls(
            schema_version=value["schema_version"], run_id=value["run_id"],
            created_at=value["created_at"],
            project_source=GitSource.from_dict({**project, "dirty": False, "pushed": True}),
            engine_source=GitSource.from_dict({**engine, "dirty": False, "pushed": True}),
            engine_submodule_path=topology["engine_submodule_path"],
            source_evidence=AuthenticatedSourceEvidenceV1.from_dict(value["source_evidence"]),
            deployment_member_sha256=value["deployment_member_sha256"], roots=roots,
            writable_capability_root=capability_roots["writable"],
            python_implementation=interpreter["implementation"],
            python_version=interpreter["version"], python_executable=interpreter["executable"],
            python_executable_digest=interpreter["executable_digest"],
            environment=environment["variables"],
            secret_requirements_digest=environment["secret_requirements_digest"],
            provider_runtime_requirements_digest=environment["provider_runtime_requirements_digest"],
        )

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(b"synaptic-execution-source/v1\0" + self.canonical_bytes).hexdigest()


@runtime_checkable
class LocalSourceInspectionPort(Protocol):
    def inspect(self, *, context: ProjectContext) -> SourceLock: ...


@runtime_checkable
class PushedSourceVerificationPort(Protocol):
    def verify(self, source_lock: SourceLock) -> AuthenticatedSourceEvidenceV1: ...


__all__ = [
    "AuthenticatedSourceEvidenceV1", "EXECUTION_SOURCE_SCHEMA", "ExecutionSourceV1",
    "LocalSourceInspectionPort", "PushedSourceVerificationPort", "RUNTIME_SCHEMA",
    "SOURCE_EVIDENCE_SCHEMA",
]
