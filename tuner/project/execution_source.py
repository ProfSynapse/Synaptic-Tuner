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
from .source_bundle import (
    GitSource,
    RepositoryLocation,
    SourceLock,
    SourceLockBindingV1,
    SourceLockError,
)


EXECUTION_SOURCE_SCHEMA = "synaptic-execution-source/v1"
SOURCE_EVIDENCE_SCHEMA = "synaptic-authenticated-source-evidence/v1"
RUNTIME_SCHEMA = "synaptic-training-runtime/v1"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_ROOT_NAMES = ("engine", "project", "artifacts", "state", "tracking", "cache", "tmp")
_TRAINING_PROVENANCE_KEYS = (
    "training_input_digest", "training_contract_identity_digest",
    "training_source_sha256", "training_ingress_digest", "provider_policy_digest",
)
_DICT_COPY = dict.copy
_DICT_KEYS = dict.keys


def _exact_object_fields(
    value: object, *, required: tuple[str, ...] | None, failure: str
) -> dict[str, object]:
    """Snapshot and validate one closed exact-built-in JSON object."""

    snapshot: dict[str, object] | None = None
    valid = False
    try:
        if type(value) is dict:
            candidate = _DICT_COPY(value)
            keys = tuple(_DICT_KEYS(candidate))
            if all(type(key) is str for key in keys):
                valid = required is None or (
                    len(keys) == len(required) and all(key in keys for key in required)
                )
                if valid:
                    snapshot = candidate
    except BaseException:
        snapshot = None
        valid = False
    if not valid or snapshot is None:
        raise SourceLockError(failure) from None
    return snapshot


def _canonical(value: Mapping[str, object]) -> bytes:
    encoded: bytes | None = None
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except BaseException:
        pass
    if encoded is None:
        raise SourceLockError("execution source must contain only finite JSON values") from None
    return encoded


def _text(value: object, name: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value or value != value.strip():
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
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise SourceLockError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object, name: str) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
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


def _source_tuple_matches(
    evidence: "AuthenticatedSourceEvidenceV1",
    project_source: GitSource,
    engine_source: GitSource,
    engine_submodule_path: str | None,
) -> bool:
    return (
        type(evidence) is AuthenticatedSourceEvidenceV1
        and type(project_source) is GitSource
        and type(engine_source) is GitSource
        and evidence.project_url == project_source.location.canonical_url
        and evidence.project_commit == project_source.commit.lower()
        and evidence.engine_url == engine_source.location.canonical_url
        and evidence.engine_commit == engine_source.commit.lower()
        and evidence.engine_submodule_path == engine_submodule_path
        and evidence.gitlink_commit == engine_source.gitlink_commit
        and evidence.engine_commit == evidence.gitlink_commit
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedSourceEvidenceV1:
    project_url: str
    project_commit: str
    engine_url: str
    engine_commit: str
    engine_submodule_path: str
    gitlink_commit: str
    source_lock_binding: SourceLockBindingV1
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
        if type(self.schema_version) is not str or self.schema_version != SOURCE_EVIDENCE_SCHEMA:
            raise SourceLockError("unsupported authenticated source-evidence schema")
        if type(self.project_url) is not str or type(self.engine_url) is not str:
            raise TypeError("source evidence URLs must be exact strings")
        object.__setattr__(self, "project_url", RepositoryLocation.parse(self.project_url).canonical_url)
        object.__setattr__(self, "engine_url", RepositoryLocation.parse(self.engine_url).canonical_url)
        for name in ("project_commit", "engine_commit", "gitlink_commit"):
            object.__setattr__(self, name, _commit(getattr(self, name), name))
        object.__setattr__(self, "engine_submodule_path", _submodule_path(self.engine_submodule_path))
        if type(self.source_lock_binding) is not SourceLockBindingV1:
            raise TypeError("source_lock_binding must be exact SourceLockBindingV1")
        object.__setattr__(self, "issuer_ref", _safe_ref(self.issuer_ref, "issuer_ref"))
        object.__setattr__(self, "evidence_ref", _safe_ref(self.evidence_ref, "evidence_ref"))
        for name in ("audience_ref", "challenge_nonce", "key_ref"):
            object.__setattr__(self, name, _safe_ref(getattr(self, name), name))
        from tuner.execution.evidence import canonical_utc
        object.__setattr__(self, "verified_at", canonical_utc(self.verified_at, "verified_at"))
        object.__setattr__(self, "expires_at", canonical_utc(self.expires_at, "expires_at"))
        if type(self.tag_base64) is not str or not self.tag_base64 or not self.tag_base64.isascii():
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
            type(source_lock) is SourceLock
            and self.binds_sources(
                source_lock.project_source,
                source_lock.engine_source,
                source_lock.engine_source.submodule_path,
            )
            and self.source_lock_binding == source_lock.binding
        )

    def binds_sources(
        self,
        project_source: GitSource,
        engine_source: GitSource,
        engine_submodule_path: str | None,
    ) -> bool:
        return _source_tuple_matches(
            self, project_source, engine_source, engine_submodule_path
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
            "source_lock_binding": self.source_lock_binding.to_dict(),
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
        value = _exact_object_fields(
            value,
            required=(
                "schema_version", "project", "engine", "issuer_ref", "evidence_ref",
                "audience_ref", "challenge_nonce", "verified_at", "expires_at", "key_ref",
                "tag_base64", "attestation_digest", "source_lock_binding",
            ),
            failure="authenticated source evidence has missing or unknown fields",
        )
        project = _exact_object_fields(
            value.get("project"), required=("url", "commit"),
            failure="authenticated project evidence is malformed",
        )
        engine = _exact_object_fields(
            value.get("engine"),
            required=("url", "commit", "submodule_path", "gitlink_commit"),
            failure="authenticated engine evidence is malformed",
        )
        binding = value.get("source_lock_binding")
        if type(binding) is not dict:
            raise SourceLockError("authenticated source-lock binding is malformed")
        return cls(
            schema_version=value["schema_version"], project_url=project["url"],
            project_commit=project["commit"], engine_url=engine["url"],
            engine_commit=engine["commit"], engine_submodule_path=engine["submodule_path"],
            gitlink_commit=engine["gitlink_commit"],
            source_lock_binding=SourceLockBindingV1.from_dict(binding),
            issuer_ref=value["issuer_ref"],
            evidence_ref=value["evidence_ref"], audience_ref=value["audience_ref"],
            challenge_nonce=value["challenge_nonce"], verified_at=value["verified_at"],
            expires_at=value["expires_at"], key_ref=value["key_ref"],
            tag_base64=value["tag_base64"],
            attestation_digest=value["attestation_digest"],
        )


@dataclass(frozen=True, slots=True)
class _SealedSourceEvidenceSnapshotV1:
    evidence: AuthenticatedSourceEvidenceV1
    document_bytes: bytes
    authenticated_payload: bytes
    tag: bytes
    attestation_digest: str
    replay_identity: tuple[str, str, str, str, str, str]


def _capture_source_evidence_snapshot(
    evidence: AuthenticatedSourceEvidenceV1,
) -> _SealedSourceEvidenceSnapshotV1:
    if type(evidence) is not AuthenticatedSourceEvidenceV1:
        raise TypeError("exact AuthenticatedSourceEvidenceV1 is required")
    try:
        document = evidence.to_dict()
        if type(document) is not dict:
            raise TypeError
        reconstructed = AuthenticatedSourceEvidenceV1.from_dict(document)
        document_bytes = _canonical(reconstructed.to_dict())
        payload = reconstructed.authenticated_payload
        tag = reconstructed.tag
    except BaseException:
        raise SourceLockError("authenticated source evidence is malformed") from None
    return _SealedSourceEvidenceSnapshotV1(
        reconstructed,
        document_bytes,
        payload,
        tag,
        reconstructed.attestation_digest,
        (
            reconstructed.issuer_ref,
            reconstructed.evidence_ref,
            reconstructed.challenge_nonce,
            reconstructed.audience_ref,
            reconstructed.attestation_digest,
            reconstructed.expires_at,
        ),
    )


def _require_source_evidence_snapshot(
    evidence: AuthenticatedSourceEvidenceV1,
    baseline: _SealedSourceEvidenceSnapshotV1,
) -> None:
    current = _capture_source_evidence_snapshot(evidence)
    if current != baseline:
        raise SourceLockError("authenticated source evidence changed during processing")


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
        if type(self.schema_version) is not str or self.schema_version != EXECUTION_SOURCE_SCHEMA:
            raise SourceLockError("unsupported execution-source schema")
        object.__setattr__(self, "run_id", _safe_ref(self.run_id, "run_id"))
        if "/" in self.run_id:
            raise SourceLockError("run_id must be one canonical path segment")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if type(self.project_source) is not GitSource or type(self.engine_source) is not GitSource:
            raise TypeError("execution sources must be GitSource values")
        if self.project_source.dirty or self.engine_source.dirty:
            raise SourceLockError("execution sources must be clean")
        path = _submodule_path(self.engine_submodule_path)
        object.__setattr__(self, "engine_submodule_path", path)
        if self.engine_source.submodule_path != path:
            raise SourceLockError("engine submodule path does not match the source")
        if self.engine_source.commit.lower() != str(self.engine_source.gitlink_commit).lower():
            raise SourceLockError("engine commit does not match the host gitlink")
        if type(self.source_evidence) is not AuthenticatedSourceEvidenceV1:
            raise TypeError("source_evidence must be exact AuthenticatedSourceEvidenceV1")
        evidence_snapshot = _capture_source_evidence_snapshot(self.source_evidence)
        canonical_evidence = evidence_snapshot.evidence
        project_snapshot = GitSource.from_dict(
            {**_source_dict(self.project_source, engine=False), "dirty": False, "pushed": False}
        )
        engine_snapshot = GitSource.from_dict(
            {**_source_dict(self.engine_source, engine=True), "dirty": False, "pushed": False}
        )
        if not _source_tuple_matches(
            canonical_evidence, project_snapshot, engine_snapshot, self.engine_submodule_path
        ):
            raise SourceLockError("authenticated evidence does not bind the execution sources")
        if not _source_tuple_matches(
            canonical_evidence,
            self.project_source, self.engine_source, self.engine_submodule_path
        ):
            raise SourceLockError("authenticated evidence does not bind the execution sources")
        object.__setattr__(self, "source_evidence", canonical_evidence)
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
        if type(self.python_implementation) is not str or self.python_implementation != "cpython":
            raise SourceLockError("execution runtime requires CPython")
        if (
            type(self.python_version) is not str
            or re.fullmatch(
                r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)",
                self.python_version,
            ) is None
        ):
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
            type(key) is not str or not key or type(value) is not str
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
        value = _exact_object_fields(
            value,
            required=(
                "schema_version", "run_id", "created_at", "topology", "sources",
                "source_evidence", "deployment_member_sha256", "runtime",
            ),
            failure="execution source has missing or unknown fields",
        )
        topology = _exact_object_fields(
            value.get("topology"),
            required=("provenance_mode", "execution_mode", "engine_submodule_path"),
            failure="execution source topology is malformed",
        )
        sources = _exact_object_fields(
            value.get("sources"), required=("project", "engine"),
            failure="execution sources are malformed",
        )
        runtime = _exact_object_fields(
            value.get("runtime"),
            required=("schema_version", "roots", "capability_roots", "interpreter", "environment"),
            failure="execution runtime is malformed",
        )
        if (
            type(topology.get("provenance_mode")) is not str
            or topology.get("provenance_mode") != "superproject"
            or type(topology.get("execution_mode")) is not str
            or topology.get("execution_mode") != "dual_clone"
        ):
            raise SourceLockError("execution source topology is malformed")
        project = _exact_object_fields(
            sources.get("project"), required=("url", "commit"),
            failure="project execution source is malformed",
        )
        engine = _exact_object_fields(
            sources.get("engine"),
            required=("url", "commit", "submodule_path", "gitlink_commit"),
            failure="engine execution source is malformed",
        )
        if type(runtime.get("schema_version")) is not str or runtime.get("schema_version") != RUNTIME_SCHEMA:
            raise SourceLockError("execution runtime is malformed")
        roots = _exact_object_fields(
            runtime.get("roots"), required=_ROOT_NAMES,
            failure="execution interpreter is malformed",
        )
        capability_roots = _exact_object_fields(
            runtime.get("capability_roots"), required=("writable",),
            failure="execution interpreter is malformed",
        )
        interpreter = _exact_object_fields(
            runtime.get("interpreter"),
            required=("implementation", "version", "executable", "executable_digest"),
            failure="execution interpreter is malformed",
        )
        environment = _exact_object_fields(
            runtime.get("environment"),
            required=(
                "clear_inherited", "variables", "secret_requirements_digest",
                "provider_runtime_requirements_digest",
            ),
            failure="execution environment is malformed",
        )
        variables = environment.get("variables")
        if environment.get("clear_inherited") is not True or type(variables) is not dict:
            raise SourceLockError("execution environment is malformed")
        variables = _exact_object_fields(
            variables, required=None,
            failure="execution environment is malformed",
        )
        if type(value.get("source_evidence")) is not dict:
            raise SourceLockError("execution source evidence is malformed")
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
            environment=variables,
            secret_requirements_digest=environment["secret_requirements_digest"],
            provider_runtime_requirements_digest=environment["provider_runtime_requirements_digest"],
        )

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(b"synaptic-execution-source/v1\0" + self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceLockProvenanceViewV1:
    """Validated provider-neutral training provenance bound to one complete lock."""

    binding: SourceLockBindingV1
    projection: tuple[tuple[str, str], ...]


def validate_source_lock_provenance_v1(
    source: ExecutionSourceV1,
    source_lock: SourceLock,
    expected: dict[str, str],
) -> SourceLockProvenanceViewV1:
    if type(source) is not ExecutionSourceV1 or type(source_lock) is not SourceLock:
        raise SourceLockError("exact execution source and source lock are required")
    expected = _exact_object_fields(
        expected, required=_TRAINING_PROVENANCE_KEYS,
        failure="training provenance projection is malformed",
    )
    projection = tuple((key, expected[key]) for key in _TRAINING_PROVENANCE_KEYS)
    if any(type(value) is not str or _DIGEST_RE.fullmatch(value) is None for _, value in projection):
        raise SourceLockError("training provenance projection is malformed")
    configuration = dict(source_lock.configuration)
    if tuple(configuration) != _TRAINING_PROVENANCE_KEYS or configuration != dict(projection):
        raise SourceLockError("source lock training provenance differs")
    if (
        source.source_evidence.source_lock_binding != source_lock.binding
        or not source.source_evidence.binds(source_lock)
    ):
        raise SourceLockError("execution source does not bind the complete source lock")
    return SourceLockProvenanceViewV1(source_lock.binding, projection)


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
