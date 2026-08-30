"""Stable config-first training facade and immutable public contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .context import ProjectContext

from .execution import (
    ArtifactRef,
    ArtifactState,
    AuthorizationRequirement,
    ExecutionError,
    ExecutionGrant,
    RunRef,
    RunState,
    RunStatus,
)
from .sources import ExecutionSourceV1


def _required(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _positive(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PINNED_IMAGE_PATTERN = re.compile(r"^\S+@sha256:(?P<digest>[0-9a-f]{64})$")


def _canonical_document(value: Mapping[str, object]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("document must be a mapping")
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("document must contain only JSON values") from exc
    if not isinstance(json.loads(encoded), dict):  # pragma: no cover - mapping invariant
        raise ValueError("document must encode a JSON object")
    return encoded


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    """Immutable canonical JSON object used instead of mutable untyped config."""

    canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str):
            raise TypeError("canonical_json must be a string")
        try:
            value = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical_json must contain valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("canonical_json must encode a JSON object")
        object.__setattr__(self, "canonical_json", _canonical_document(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CanonicalDocument":
        return cls(_canonical_document(value))

    def to_dict(self) -> dict[str, object]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise TypeError("canonical document must decode to an object")
        return value


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    image: str
    dependency_lock_digest: str
    python_version: str

    def __post_init__(self) -> None:
        image = _required(self.image, "image")
        match = _PINNED_IMAGE_PATTERN.fullmatch(image)
        if match is None:
            raise ValueError("image must be pinned to an exact sha256 digest")
        dependency_digest = _required(
            self.dependency_lock_digest, "dependency_lock_digest"
        )
        if _SHA256_PATTERN.fullmatch(dependency_digest) is None:
            raise ValueError("dependency_lock_digest must be a lowercase SHA-256 digest")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "dependency_lock_digest", dependency_digest)
        object.__setattr__(
            self, "python_version", _required(self.python_version, "python_version")
        )


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    accelerator: str
    accelerator_count: int = 1
    timeout_seconds: int = 3600

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "accelerator", _required(self.accelerator, "accelerator")
        )
        _positive(self.accelerator_count, "accelerator_count")
        _positive(self.timeout_seconds, "timeout_seconds")


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    required_kinds: tuple[str, ...] = ("training_lineage", "final_model")
    retain_checkpoints: bool = True

    def __post_init__(self) -> None:
        kinds = tuple(_required(item, "required_kind") for item in self.required_kinds)
        if not kinds or len(kinds) != len(set(kinds)):
            raise ValueError("required_kinds must be non-empty and unique")
        if not isinstance(self.retain_checkpoints, bool):
            raise TypeError("retain_checkpoints must be a boolean")
        object.__setattr__(self, "required_kinds", kinds)


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    document: CanonicalDocument

    def __post_init__(self) -> None:
        if not isinstance(self.document, CanonicalDocument):
            raise TypeError("document must be a CanonicalDocument")


@dataclass(frozen=True, slots=True)
class ResolvedTrainingRequest:
    request: TrainingRequest
    execution_source: ExecutionSourceV1
    execution_context: CanonicalDocument
    resolved_config: CanonicalDocument
    workload: CanonicalDocument
    runtime: RuntimeSpec
    resources: ResourceSpec
    artifact_policy: ArtifactPolicy = ArtifactPolicy()

    def __post_init__(self) -> None:
        expected = (
            (self.request, TrainingRequest, "request"),
            (self.execution_source, ExecutionSourceV1, "execution_source"),
            (self.execution_context, CanonicalDocument, "execution_context"),
            (self.resolved_config, CanonicalDocument, "resolved_config"),
            (self.workload, CanonicalDocument, "workload"),
            (self.runtime, RuntimeSpec, "runtime"),
            (self.resources, ResourceSpec, "resources"),
            (self.artifact_policy, ArtifactPolicy, "artifact_policy"),
        )
        for value, kind, name in expected:
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be {kind.__name__}")


class TrainingResolutionError(ValueError):
    """Stable failure raised when a host cannot resolve an exact request."""


@dataclass(frozen=True, slots=True)
class ResolvedTrainingComponents:
    """Exact host resolver output before deterministic workload compilation."""

    execution_source: ExecutionSourceV1
    execution_context: CanonicalDocument
    resolved_config: CanonicalDocument
    runtime: RuntimeSpec
    resources: ResourceSpec
    artifact_policy: ArtifactPolicy = ArtifactPolicy()

    def __post_init__(self) -> None:
        checks = (
            (self.execution_source, ExecutionSourceV1, "execution_source"),
            (self.execution_context, CanonicalDocument, "execution_context"),
            (self.resolved_config, CanonicalDocument, "resolved_config"),
            (self.runtime, RuntimeSpec, "runtime"),
            (self.resources, ResourceSpec, "resources"),
            (self.artifact_policy, ArtifactPolicy, "artifact_policy"),
        )
        for value, expected, name in checks:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")


@runtime_checkable
class TrainingRequestResolver(Protocol):
    """Host seam for config, source, model, and dataset resolution."""

    def resolve(
        self,
        request: TrainingRequest,
        *,
        context: "ProjectContext",
    ) -> ResolvedTrainingComponents: ...


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    execution_source: ExecutionSourceV1
    execution_context: CanonicalDocument
    resolved_config: CanonicalDocument
    workload: CanonicalDocument
    runtime: RuntimeSpec
    resources: ResourceSpec
    artifact_policy: ArtifactPolicy

    def __post_init__(self) -> None:
        expected = (
            (self.execution_source, ExecutionSourceV1, "execution_source"),
            (self.execution_context, CanonicalDocument, "execution_context"),
            (self.resolved_config, CanonicalDocument, "resolved_config"),
            (self.workload, CanonicalDocument, "workload"),
            (self.runtime, RuntimeSpec, "runtime"),
            (self.resources, ResourceSpec, "resources"),
            (self.artifact_policy, ArtifactPolicy, "artifact_policy"),
        )
        for value, kind, name in expected:
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be {kind.__name__}")

    @property
    def fingerprint(self) -> str:
        payload = {
            "artifact_policy": {
                "required_kinds": list(self.artifact_policy.required_kinds),
                "retain_checkpoints": self.artifact_policy.retain_checkpoints,
            },
            "resources": {
                "accelerator": self.resources.accelerator,
                "accelerator_count": self.resources.accelerator_count,
                "timeout_seconds": self.resources.timeout_seconds,
            },
            "resolved_config": self.resolved_config.to_dict(),
            "runtime": {
                "dependency_lock_digest": self.runtime.dependency_lock_digest,
                "image": self.runtime.image,
                "python_version": self.runtime.python_version,
            },
            "execution_source": self.execution_source.to_dict(),
            "execution_context": self.execution_context.to_dict(),
            "workload": self.workload.to_dict(),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(b"synaptic-training-plan/v1\0" + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    plan_fingerprint: str
    ready: bool
    checked_at: str
    expires_at: str
    authorization: tuple[AuthorizationRequirement, ...] = ()
    errors: tuple[ExecutionError, ...] = ()

    def __post_init__(self) -> None:
        fingerprint = _required(self.plan_fingerprint, "plan_fingerprint")
        if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("plan_fingerprint must be a lowercase SHA-256 digest")
        if not isinstance(self.ready, bool):
            raise TypeError("ready must be a boolean")
        checked_at = _required(self.checked_at, "checked_at")
        expires_at = _required(self.expires_at, "expires_at")
        from datetime import datetime
        try:
            checked = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
            expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("preflight times must be ISO 8601 timestamps") from exc
        if checked.tzinfo is None or expires.tzinfo is None or expires <= checked:
            raise ValueError("preflight expiry must be after its checked time")
        object.__setattr__(self, "checked_at", checked_at)
        object.__setattr__(self, "expires_at", expires_at)
        authorization = tuple(self.authorization)
        errors = tuple(self.errors)
        if any(not isinstance(item, AuthorizationRequirement) for item in authorization):
            raise TypeError("authorization must contain AuthorizationRequirement values")
        if any(not isinstance(item, ExecutionError) for item in errors):
            raise TypeError("errors must contain ExecutionError values")
        if self.ready and errors:
            raise ValueError("ready preflight cannot contain errors")
        if not self.ready and not errors:
            raise ValueError("failed preflight requires an error")
        object.__setattr__(self, "authorization", authorization)
        object.__setattr__(self, "errors", errors)

    def binds(self, plan: TrainingPlan) -> bool:
        return self.plan_fingerprint == plan.fingerprint


@dataclass(frozen=True, slots=True)
class TrainingSubmission:
    run: RunRef
    plan_fingerprint: str
    submitted_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        fingerprint = _required(self.plan_fingerprint, "plan_fingerprint")
        if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("plan_fingerprint must be a lowercase SHA-256 digest")
        object.__setattr__(self, "submitted_at", _required(self.submitted_at, "submitted_at"))


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    submission: TrainingSubmission
    status: RunStatus
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.submission, TrainingSubmission):
            raise TypeError("submission must be a TrainingSubmission")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if self.status.run != self.submission.run:
            raise ValueError("status must refer to the submitted run")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise TypeError("artifacts must contain ArtifactRef values")
        if any(item.run != self.submission.run for item in artifacts):
            raise ValueError("artifacts must refer to the submitted run")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def success(self) -> bool:
        return (
            self.status.state is RunState.SUCCEEDED
            and bool(self.artifacts)
            and all(item.state is ArtifactState.VERIFIED for item in self.artifacts)
        )


class TrainingOperations(Protocol):
    def load(self, document: CanonicalDocument) -> TrainingRequest: ...

    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest: ...

    def plan(self, resolved: ResolvedTrainingRequest) -> TrainingPlan: ...

    def preflight(self, plan: TrainingPlan) -> TrainingPreflight: ...

    def start(
        self,
        plan: TrainingPlan,
        preflight: TrainingPreflight,
        grant: ExecutionGrant,
    ) -> TrainingSubmission: ...

    def outcome(self, submission: TrainingSubmission) -> TrainingOutcome: ...

    def reverify(self, submission: TrainingSubmission) -> TrainingOutcome: ...



class TrainingAPI:
    """Import-light facade over a host-selected training implementation."""

    __slots__ = ("_operations",)

    def __init__(self, operations: TrainingOperations) -> None:
        self._operations = operations

    def load(self, document: CanonicalDocument) -> TrainingRequest:
        return self._operations.load(document)

    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest:
        return self._operations.resolve(request)

    def plan(self, resolved: ResolvedTrainingRequest) -> TrainingPlan:
        return self._operations.plan(resolved)

    def preflight(self, plan: TrainingPlan) -> TrainingPreflight:
        return self._operations.preflight(plan)

    def start(
        self,
        plan: TrainingPlan,
        preflight: TrainingPreflight,
        grant: ExecutionGrant,
    ) -> TrainingSubmission:
        if not preflight.binds(plan):
            raise ValueError("preflight does not bind the exact training plan")
        if not preflight.ready:
            raise ValueError("training plan did not pass preflight")
        return self._operations.start(plan, preflight, grant)

    def outcome(self, submission: TrainingSubmission) -> TrainingOutcome:
        return self._operations.outcome(submission)

    def reverify(self, submission: TrainingSubmission) -> TrainingOutcome:
        return self._operations.reverify(submission)

__all__ = [
    "ArtifactPolicy",
    "CanonicalDocument",
    "ResolvedTrainingComponents",
    "ResolvedTrainingRequest",
    "ResourceSpec",
    "RuntimeSpec",
    "TrainingAPI",
    "TrainingOperations",
    "TrainingOutcome",
    "TrainingPlan",
    "TrainingPreflight",
    "TrainingRequest",
    "TrainingRequestResolver",
    "TrainingResolutionError",
    "TrainingSubmission",
]
