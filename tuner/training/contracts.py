"""Immutable rich training compilation contracts owned by the engine."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, BinaryIO, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from synaptic_tuner.api.v1.training_input import TrainingInputV1
    from tuner.project.context import ProjectContext

from synaptic_tuner.api.v1._contract import contract_digest, PreparedTrainingInputIdentity
from tuner.project.execution_source import ExecutionSourceV1


GitExecutionSourceV1 = ExecutionSourceV1
# Host execution material is opaque to the legacy closure. The explicit host
# boundary exports the precise union; these constructors enforce exact types.
_ExecutionMaterialAnnotation = object


def __getattr__(name: str):
    if name in {"ExecutionMaterialV1", "compile_training_plan_for_execution_v1"}:
        from importlib import import_module
        return getattr(import_module("tuner.training.packaged_boundary"), name)
    raise AttributeError(name)


def is_packaged_execution_material(value: object) -> bool:
    # Do not import the packaged boundary while running the fixed Git closure.
    module = sys.modules.get("tuner.training.packaged_boundary")
    return module is not None and type(value) is vars(module).get("PackagedExecutionBindingV1")


def require_execution_material(value: object) -> None:
    if type(value) is not GitExecutionSourceV1 and not is_packaged_execution_material(value):
        raise TypeError("execution_source must be exact ExecutionSourceV1 or PackagedExecutionBindingV1")


def execution_material_digest(value: _ExecutionMaterialAnnotation) -> str:
    require_execution_material(value)
    return value.fingerprint if type(value) is GitExecutionSourceV1 else value.binding_digest


def execution_material_run(value: _ExecutionMaterialAnnotation) -> str:
    require_execution_material(value)
    return value.run_id if type(value) is GitExecutionSourceV1 else value.run_ref


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
_ACCELERATOR_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9._-]*$")


class RetainedTrainingInputStreamLease:
    """One-use verified stream retained only for one staging upload attempt."""

    __slots__ = ("identity", "_stream", "_state", "_lock")

    def __init__(self, identity: PreparedTrainingInputIdentity, stream: BinaryIO) -> None:
        if type(identity) is not PreparedTrainingInputIdentity:
            raise TypeError("exact prepared training input identity required")
        if not callable(getattr(stream, "read", None)) or not callable(
            getattr(stream, "close", None)
        ):
            raise TypeError("prepared training input lease requires a binary stream")
        self.identity = identity
        self._stream = stream
        self._state = "available"
        self._lock = Lock()

    def take_stream(self) -> BinaryIO:
        with self._lock:
            if self._state != "available":
                raise ValueError("prepared training input lease was already consumed")
            self._state = "transferred"
            return self._stream

    def close(self) -> None:
        with self._lock:
            if self._state == "closed":
                return
            self._state = "closed"
            self._stream.close()

    def __copy__(self):
        raise TypeError("prepared training input leases are not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("prepared training input leases are not copyable")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("prepared training input leases are not serializable")


@runtime_checkable
class VerifiedTrainingInputSource(Protocol):
    """Internal source port; public requests carry only the prepared identity."""

    @property
    def identity(self) -> PreparedTrainingInputIdentity: ...

    def open_lease(self) -> RetainedTrainingInputStreamLease: ...


def _canonical_document(value: Mapping[str, object]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("document must be a mapping")
    _bound_json(value)
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("document must contain only JSON values") from exc
    if len(encoded.encode("utf-8")) > 512 * 1024:
        raise ValueError("document exceeds its byte bound")
    if not isinstance(json.loads(encoded), dict):  # pragma: no cover - mapping invariant
        raise ValueError("document must encode a JSON object")
    return encoded


def _bound_json(value: object) -> None:
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 16384 or depth > 32:
            raise ValueError("document exceeds its structural bound")
        if type(item) is dict:
            if len(item) > 4096 or any(type(key) is not str for key in item):
                raise ValueError("document requires bounded string keys")
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) in (list, tuple):
            if len(item) > 4096:
                raise ValueError("document exceeds its container bound")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > 512 * 1024:
                raise ValueError("document exceeds its text bound")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("document requires finite numbers")
        elif item is not None and type(item) not in (int, bool):
            raise ValueError("document must contain only JSON values")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("document contains duplicate keys")
        result[key] = value
    return result


def bounded_json_object(payload: str) -> dict[str, object]:
    if type(payload) is not str or len(payload) > 512 * 1024:
        raise ValueError("document exceeds its byte bound")
    try:
        if len(payload.encode("utf-8")) > 512 * 1024:
            raise ValueError("document exceeds its byte bound")
        value = json.loads(payload, object_pairs_hook=_json_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("document must contain bounded valid JSON") from None
    if type(value) is not dict:
        raise ValueError("canonical_json must encode a JSON object")
    _bound_json(value)
    return value


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    """Immutable canonical JSON object used instead of mutable untyped config."""

    canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str):
            raise TypeError("canonical_json must be a string")
        value = bounded_json_object(self.canonical_json)
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
class AcceleratorDeviceRequestV1:
    """Canonical provider-neutral selection of concrete accelerator devices."""

    kind: str
    device_indices: tuple[int, ...]
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not str:
            raise TypeError("kind must be an exact string")
        if _ACCELERATOR_TOKEN_PATTERN.fullmatch(self.kind) is None:
            raise ValueError("kind must be a canonical lowercase token")
        if type(self.device_indices) is not tuple:
            raise TypeError("device_indices must be an exact tuple")
        if any(type(value) is not int for value in self.device_indices):
            raise TypeError("device_indices must contain exact integers")
        if any(value < 0 for value in self.device_indices):
            raise ValueError("device_indices must be nonnegative")
        if (
            self.device_indices != tuple(sorted(self.device_indices))
            or len(self.device_indices) != len(set(self.device_indices))
        ):
            raise ValueError("device_indices must be unique canonical ascending")
        if type(self.capabilities) is not tuple:
            raise TypeError("capabilities must be an exact tuple")
        if any(type(value) is not str for value in self.capabilities):
            raise TypeError("capabilities must contain exact strings")
        if any(
            _ACCELERATOR_TOKEN_PATTERN.fullmatch(value) is None
            for value in self.capabilities
        ):
            raise ValueError("capabilities must contain canonical lowercase tokens")
        if (
            self.capabilities != tuple(sorted(self.capabilities))
            or len(self.capabilities) != len(set(self.capabilities))
        ):
            raise ValueError("capabilities must be unique canonical ascending")
        if self.kind == "cpu":
            if self.device_indices or self.capabilities:
                raise ValueError("cpu requests cannot select devices or capabilities")
        elif not self.device_indices or not self.capabilities:
            raise ValueError(
                "accelerator requests require device indices and capabilities"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "device_indices": list(self.device_indices),
            "capabilities": list(self.capabilities),
        }

    @property
    def accelerator_device_request_digest(self) -> str:
        return contract_digest(
            "synaptic-accelerator-device-request/v1", self.to_dict()
        )


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
    execution_source: _ExecutionMaterialAnnotation
    execution_context: CanonicalDocument
    resolved_config: CanonicalDocument
    workload: CanonicalDocument
    runtime: RuntimeSpec
    resources: ResourceSpec
    artifact_policy: ArtifactPolicy = ArtifactPolicy()

    def __post_init__(self) -> None:
        require_execution_material(self.execution_source)
        expected = (
            (self.request, TrainingRequest, "request"),
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

    execution_source: _ExecutionMaterialAnnotation
    execution_context: CanonicalDocument
    resolved_config: CanonicalDocument
    runtime: RuntimeSpec
    resources: ResourceSpec
    artifact_policy: ArtifactPolicy = ArtifactPolicy()

    def __post_init__(self) -> None:
        require_execution_material(self.execution_source)
        checks = (
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


class _StaticTrainingRequestResolverAdapter:
    __slots__ = ("_resolve",)

    def __init__(self, resolve) -> None:
        self._resolve = resolve

    def resolve(
        self,
        request: TrainingRequest,
        *,
        context: "ProjectContext",
    ) -> ResolvedTrainingComponents:
        return self._resolve(request, context=context)


def _safe_training_request_resolver(
    resolver: object,
) -> _StaticTrainingRequestResolverAdapter:
    from inspect import getattr_static
    from types import FunctionType, MethodType

    missing = object()
    member = getattr_static(type(resolver), "resolve", missing)
    if type(member) is FunctionType:
        bound_resolve = MethodType(member, resolver)
    elif type(member) is staticmethod and type(member.__func__) is FunctionType:
        bound_resolve = member.__func__
    elif type(member) is classmethod and type(member.__func__) is FunctionType:
        bound_resolve = MethodType(member.__func__, type(resolver))
    else:
        raise TypeError("resolver must implement TrainingRequestResolver")
    return _StaticTrainingRequestResolverAdapter(bound_resolve)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    execution_source: _ExecutionMaterialAnnotation
    execution_context: CanonicalDocument
    resolved_config: CanonicalDocument
    workload: CanonicalDocument
    runtime: RuntimeSpec
    resources: ResourceSpec
    artifact_policy: ArtifactPolicy

    def __post_init__(self) -> None:
        require_execution_material(self.execution_source)
        expected = (
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


def compile_training_plan_v1(
    *,
    training_input: "TrainingInputV1",
    context: "ProjectContext",
    resolver: TrainingRequestResolver,
) -> TrainingPlan:
    """Compile one canonical input through the provider-neutral planning core."""

    from synaptic_tuner.api.v1.training_input import TrainingInputV1
    from tuner.project.context import ProjectContext

    if type(training_input) is not TrainingInputV1:
        raise TypeError("training_input must be exact TrainingInputV1")
    if type(context) is not ProjectContext:
        raise TypeError("context must be exact ProjectContext")
    safe_resolver = _safe_training_request_resolver(resolver)

    from tuner.training import TrainingService, default_recipe_registry

    service = TrainingService(
        context=context,
        resolver=safe_resolver,
        recipes=default_recipe_registry(),
    )
    document = CanonicalDocument(training_input.canonical_json())
    request = service.load(document)
    resolved = service.resolve(request)
    plan = service.plan(resolved)
    if type(plan) is not TrainingPlan:
        raise TypeError("planning must return exact TrainingPlan")
    fingerprint = plan.fingerprint
    if type(fingerprint) is not str or _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError("training plan fingerprint is invalid")
    return plan




__all__ = [
    "AcceleratorDeviceRequestV1", "ArtifactPolicy", "CanonicalDocument",
    "ResolvedTrainingComponents", "ResolvedTrainingRequest", "ResourceSpec",
    "RuntimeSpec", "TrainingPlan", "TrainingRequest", "TrainingRequestResolver",
    "TrainingResolutionError", "compile_training_plan_v1",
    "GitExecutionSourceV1", "ExecutionMaterialV1", "compile_training_plan_for_execution_v1",
]
