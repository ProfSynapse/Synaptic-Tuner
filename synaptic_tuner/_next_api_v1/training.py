"""Frozen provider-neutral model-training contracts.

This standard-library-only module separates pure validation, disclosed read-only
resolution/preflight, deterministic planning, and authorized execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from .execution import (
    AccessContext, ArtifactRef, AuthorizationRequirement, ExecutionGrant,
    RunRef, RunState, RunStatus,
)

TRAINING_SCHEMA_VERSION = "synaptic.training/v1"
TRAINING_KIND = "training"


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _optional(value: str | None, name: str) -> str | None:
    return None if value is None else _required(value, name)


def _positive(value: int | None, name: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _finite(value: float | None, name: str, *, allow_zero: bool = True) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be a finite non-negative number")


def _unique(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    result = tuple(_required(item, name) for item in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _require_type(value: object, expected: type, name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__}")


def _require_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


_REF_PATTERN = re.compile(r"^(?:secret|credential)://[A-Za-z0-9_-]+(?:/[A-Za-z0-9._-]+)*$")
_PUBLICATION_PATTERN = re.compile(r"^publication://[A-Za-z0-9_-]+(?:/[A-Za-z0-9._-]+)*$")
_HEX_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_IMAGE_PATTERN = re.compile(r"^[^@\s]+@(?P<digest>sha256:[0-9a-f]{64})$")


def _refs(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    refs = _unique(values, name)
    if any(_REF_PATTERN.fullmatch(value) is None for value in refs):
        raise ValueError(f"{name} must contain only opaque secret references")
    return refs


def _canonical_digest(value: str, name: str, *, image: bool = False) -> str:
    value = _required(value, name)
    pattern = _IMAGE_DIGEST_PATTERN if image else _HEX_DIGEST_PATTERN
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical SHA-256 digest")
    return value

def _immutable_revision(value: str, name: str) -> str:
    value = _required(value, name)
    if _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an immutable 40-character lowercase commit revision")
    return value

class TrainingMethod(str, Enum):
    SFT = "sft"
    KTO = "kto"
    GRPO = "grpo"
    EMBEDDING = "embedding"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    identifier: str
    revision: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier", _required(self.identifier, "model.identifier"))
        object.__setattr__(self, "revision", _optional(self.revision, "model.revision"))
        object.__setattr__(
            self, "tokenizer_revision",
            _optional(self.tokenizer_revision, "model.tokenizer_revision"),
        )
        _require_bool(self.trust_remote_code, "model.trust_remote_code")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code is not permitted")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    identifier: str
    revision: str | None = None
    split: str = "train"
    file: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier", _required(self.identifier, "dataset.identifier"))
        object.__setattr__(self, "revision", _optional(self.revision, "dataset.revision"))
        object.__setattr__(self, "split", _required(self.split, "dataset.split"))
        object.__setattr__(self, "file", _optional(self.file, "dataset.file"))


@dataclass(frozen=True, slots=True)
class TrainingParameters:
    num_train_epochs: float | None = None
    max_steps: int | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.0002
    max_seq_length: int = 2048
    seed: int = 3407
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    packing: bool = False
    beta: float | None = None
    num_generations: int | None = None
    temperature: float | None = None
    margin: float | None = None

    def __post_init__(self) -> None:
        _finite(self.num_train_epochs, "num_train_epochs", allow_zero=False)
        _positive(self.max_steps, "max_steps")
        _positive(self.per_device_train_batch_size, "per_device_train_batch_size")
        _positive(self.gradient_accumulation_steps, "gradient_accumulation_steps")
        _finite(self.learning_rate, "learning_rate", allow_zero=False)
        _positive(self.max_seq_length, "max_seq_length")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        _finite(self.warmup_ratio, "warmup_ratio")
        if self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must not exceed one")
        _finite(self.weight_decay, "weight_decay")
        _finite(self.beta, "beta")
        _positive(self.num_generations, "num_generations")
        _finite(self.temperature, "temperature")
        _finite(self.margin, "margin")
        if self.num_train_epochs is None and self.max_steps is None:
            raise ValueError("one of num_train_epochs or max_steps is required")


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    mode: str
    rank: int | None = None
    alpha: int | None = None
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ()
    use_rslora: bool = False
    use_dora: bool = False
    init: str | None = None

    def __post_init__(self) -> None:
        mode = _required(self.mode, "adapter.mode").lower()
        if mode not in {"none", "full", "lora", "qlora", "frozen_head"}:
            raise ValueError("unsupported adapter mode")
        object.__setattr__(self, "mode", mode)
        if mode in {"lora", "qlora"}:
            _positive(self.rank, "adapter.rank")
            _positive(self.alpha, "adapter.alpha")
        elif self.rank is not None or self.alpha is not None:
            raise ValueError("rank and alpha apply only to lora or qlora")
        _finite(self.dropout, "adapter.dropout")
        if self.dropout > 1:
            raise ValueError("adapter.dropout must not exceed one")
        object.__setattr__(
            self, "target_modules", _unique(tuple(self.target_modules), "target_modules")
        )
        _require_bool(self.use_rslora, "adapter.use_rslora")
        _require_bool(self.use_dora, "adapter.use_dora")
        if mode not in {"lora", "qlora"} and (
            self.dropout != 0 or self.target_modules
            or self.use_rslora or self.use_dora or self.init
        ):
            raise ValueError("LoRA options require lora or qlora mode")
        object.__setattr__(self, "init", _optional(self.init, "adapter.init"))


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    provider: str
    accelerator: str | None = None
    accelerator_count: int = 1
    runtime_image: str | None = None
    runtime_image_digest: str | None = None
    dependency_lock_digest: str | None = None
    timeout_seconds: int = 3600
    maximum_cost_minor_units: int | None = None
    currency: str | None = None
    allowed_secret_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _required(self.provider, "execution.provider").lower())
        object.__setattr__(self, "accelerator", _optional(self.accelerator, "accelerator"))
        _positive(self.accelerator_count, "accelerator_count")
        object.__setattr__(self, "runtime_image", _optional(self.runtime_image, "runtime_image"))
        object.__setattr__(
            self, "runtime_image_digest",
            None if self.runtime_image_digest is None else _canonical_digest(
                self.runtime_image_digest, "runtime_image_digest", image=True
            ),
        )
        object.__setattr__(
            self, "dependency_lock_digest",
            None if self.dependency_lock_digest is None else _canonical_digest(
                self.dependency_lock_digest, "dependency_lock_digest"
            ),
        )
        _positive(self.timeout_seconds, "timeout_seconds")
        if self.maximum_cost_minor_units is not None and (
            not isinstance(self.maximum_cost_minor_units, int)
            or isinstance(self.maximum_cost_minor_units, bool)
            or self.maximum_cost_minor_units < 0
        ):
            raise ValueError("maximum_cost_minor_units must be non-negative")
        currency = _optional(self.currency, "currency")
        if (self.maximum_cost_minor_units is None) != (currency is None):
            raise ValueError("maximum cost and currency must be supplied together")
        if currency is not None:
            currency = currency.upper()
            if len(currency) != 3 or not currency.isalpha():
                raise ValueError("currency must be a three-letter code")
        object.__setattr__(self, "currency", currency)
        object.__setattr__(
            self, "allowed_secret_refs",
            _refs(tuple(self.allowed_secret_refs), "allowed_secret_refs"),
        )


def _validate_runtime_image_binding(execution: ExecutionSpec) -> None:
    image = execution.runtime_image
    digest = execution.runtime_image_digest
    if image is None or digest is None:
        raise ValueError("resolved execution requires runtime image and digest")
    match = _RUNTIME_IMAGE_PATTERN.fullmatch(image)
    if match is None or match.group("digest") != digest:
        raise ValueError("runtime image must be pinned to its exact canonical sha256 digest")


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    retain_checkpoints: bool = True
    publish_final_model: bool = False
    publication_ref: str | None = None
    required_artifacts: tuple[str, ...] = ("training_lineage", "final_model")

    def __post_init__(self) -> None:
        _require_bool(self.retain_checkpoints, "retain_checkpoints")
        _require_bool(self.publish_final_model, "publish_final_model")
        publication_ref = _optional(self.publication_ref, "publication_ref")
        if publication_ref is not None and _PUBLICATION_PATTERN.fullmatch(publication_ref) is None:
            raise ValueError("publication_ref must be an opaque publication reference")
        if self.publish_final_model != (publication_ref is not None):
            raise ValueError("publication_ref is required exactly when publishing")
        object.__setattr__(self, "publication_ref", publication_ref)
        object.__setattr__(
            self, "required_artifacts",
            _unique(tuple(self.required_artifacts), "required_artifacts"),
        )


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    method: TrainingMethod
    model: ModelSpec
    dataset: DatasetSpec
    parameters: TrainingParameters
    adapter: AdapterSpec
    execution: ExecutionSpec
    artifacts: ArtifactPolicy = ArtifactPolicy()
    schema_version: str = TRAINING_SCHEMA_VERSION
    kind: str = TRAINING_KIND

    def __post_init__(self) -> None:
        _require_type(self.method, TrainingMethod, "method")
        _require_type(self.model, ModelSpec, "model")
        _require_type(self.dataset, DatasetSpec, "dataset")
        _require_type(self.parameters, TrainingParameters, "parameters")
        _require_type(self.adapter, AdapterSpec, "adapter")
        _require_type(self.execution, ExecutionSpec, "execution")
        _require_type(self.artifacts, ArtifactPolicy, "artifacts")
        if self.schema_version != TRAINING_SCHEMA_VERSION or self.kind != TRAINING_KIND:
            raise ValueError("unsupported training document kind or schema version")

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(self)


@dataclass(frozen=True, slots=True)
class ResolvedTrainingRequest:
    request: TrainingRequest
    model: ModelSpec
    dataset: DatasetSpec
    source_digest: str
    resolved_at: str

    def __post_init__(self) -> None:
        _require_type(self.request, TrainingRequest, "request")
        _require_type(self.model, ModelSpec, "model")
        _require_type(self.dataset, DatasetSpec, "dataset")
        if self.model.identifier != self.request.model.identifier:
            raise ValueError("resolved model identifier must match request")
        if self.dataset.identifier != self.request.dataset.identifier:
            raise ValueError("resolved dataset identifier must match request")
        if self.model.revision is None or self.model.tokenizer_revision is None:
            raise ValueError("resolved model and tokenizer revisions are required")
        if self.dataset.revision is None:
            raise ValueError("resolved dataset revision is required")
        _immutable_revision(self.model.revision, "model.revision")
        _immutable_revision(self.model.tokenizer_revision, "model.tokenizer_revision")
        _immutable_revision(self.dataset.revision, "dataset.revision")
        execution = self.request.execution
        if (
            execution.runtime_image is None
            or execution.runtime_image_digest is None
            or execution.dependency_lock_digest is None
        ):
            raise ValueError("resolved execution requires runtime image and dependency lock")
        _validate_runtime_image_binding(execution)
        object.__setattr__(self, "source_digest", _canonical_digest(self.source_digest, "source_digest"))
        object.__setattr__(self, "resolved_at", _required(self.resolved_at, "resolved_at"))


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    provider: str
    ready: bool
    checked_at: str
    quote_ref: str | None = None
    authorization: tuple[AuthorizationRequirement, ...] = ()
    refusal_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _required(self.provider, "provider").lower())
        object.__setattr__(self, "checked_at", _required(self.checked_at, "checked_at"))
        object.__setattr__(self, "quote_ref", _optional(self.quote_ref, "quote_ref"))
        authorization = tuple(self.authorization)
        if any(not isinstance(item, AuthorizationRequirement) for item in authorization):
            raise TypeError("authorization must contain AuthorizationRequirement values")
        object.__setattr__(self, "authorization", authorization)
        object.__setattr__(
            self, "refusal_codes", _unique(tuple(self.refusal_codes), "refusal_codes")
        )
        if self.ready and self.refusal_codes:
            raise ValueError("ready preflight cannot contain refusal codes")
        if not self.ready and not self.refusal_codes:
            raise ValueError("refused preflight requires a refusal code")


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    method: TrainingMethod
    model: ModelSpec
    dataset: DatasetSpec
    parameters: TrainingParameters
    adapter: AdapterSpec
    execution: ExecutionSpec
    artifacts: ArtifactPolicy
    source_digest: str
    workload_digest: str
    artifact_slot_ref: str
    quote_ref: str | None
    authorization: tuple[AuthorizationRequirement, ...]

    def __post_init__(self) -> None:
        _require_type(self.method, TrainingMethod, "method")
        _require_type(self.model, ModelSpec, "model")
        _require_type(self.dataset, DatasetSpec, "dataset")
        _require_type(self.parameters, TrainingParameters, "parameters")
        _require_type(self.adapter, AdapterSpec, "adapter")
        _require_type(self.execution, ExecutionSpec, "execution")
        _require_type(self.artifacts, ArtifactPolicy, "artifacts")
        if self.model.revision is None or self.model.tokenizer_revision is None:
            raise ValueError("plan requires immutable model and tokenizer revisions")
        if self.dataset.revision is None:
            raise ValueError("plan requires immutable dataset revision")
        _immutable_revision(self.model.revision, "model.revision")
        _immutable_revision(self.model.tokenizer_revision, "model.tokenizer_revision")
        _immutable_revision(self.dataset.revision, "dataset.revision")
        if (
            self.execution.runtime_image is None
            or self.execution.runtime_image_digest is None
            or self.execution.dependency_lock_digest is None
        ):
            raise ValueError("plan requires resolved execution runtime image and dependency lock")
        _validate_runtime_image_binding(self.execution)
        object.__setattr__(self, "source_digest", _canonical_digest(self.source_digest, "source_digest"))
        object.__setattr__(
            self, "workload_digest", _canonical_digest(self.workload_digest, "workload_digest")
        )
        object.__setattr__(
            self, "artifact_slot_ref", _required(self.artifact_slot_ref, "artifact_slot_ref")
        )
        object.__setattr__(self, "quote_ref", _optional(self.quote_ref, "quote_ref"))
        authorization = tuple(self.authorization)
        if any(not isinstance(item, AuthorizationRequirement) for item in authorization):
            raise TypeError("authorization must contain AuthorizationRequirement values")
        object.__setattr__(self, "authorization", authorization)

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(self)

    @property
    def fingerprint(self) -> str:
        payload = b"synaptic.training-plan/v1\0" + _canonical_json(self).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingSubmission:
    run: RunRef
    plan_fingerprint: str
    submitted_at: str

    def __post_init__(self) -> None:
        _require_type(self.run, RunRef, "run")
        object.__setattr__(
            self, "plan_fingerprint", _required(self.plan_fingerprint, "plan_fingerprint")
        )
        object.__setattr__(self, "submitted_at", _required(self.submitted_at, "submitted_at"))


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    submission: TrainingSubmission
    status: RunStatus
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _require_type(self.submission, TrainingSubmission, "submission")
        _require_type(self.status, RunStatus, "status")
        if self.status.run != self.submission.run:
            raise ValueError("outcome status must refer to submitted run")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise TypeError("artifacts must contain ArtifactRef values")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def success(self) -> bool:
        return self.status.state is RunState.SUCCEEDED and bool(self.artifacts) and all(
            item.verification.value == "verified" for item in self.artifacts
        )


@runtime_checkable
class TrainingAPI(Protocol):
    """Pure/read-only/effectful boundaries for provider-neutral training."""

    def load(self, document: Mapping[str, Any]) -> TrainingRequest: ...
    def validate(self, request: TrainingRequest) -> TrainingRequest: ...
    def resolve_revisions(
        self, access: AccessContext, request: TrainingRequest
    ) -> ResolvedTrainingRequest: ...
    def preflight(
        self, access: AccessContext, resolved: ResolvedTrainingRequest
    ) -> TrainingPreflight: ...
    def plan(
        self, resolved: ResolvedTrainingRequest, preflight: TrainingPreflight
    ) -> TrainingPlan: ...
    def start(
        self, access: AccessContext, plan: TrainingPlan, grant: ExecutionGrant
    ) -> TrainingSubmission: ...
    def outcome(
        self, access: AccessContext, submission: TrainingSubmission
    ) -> TrainingOutcome: ...


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _canonical_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("canonical mappings require string keys")
            result[key] = _canonical_value(value[key])
        return result
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical documents cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )


__all__ = [
    "AdapterSpec", "ArtifactPolicy", "DatasetSpec", "ExecutionSpec", "ModelSpec",
    "ResolvedTrainingRequest", "TrainingAPI", "TrainingMethod", "TrainingOutcome",
    "TrainingParameters", "TrainingPlan", "TrainingPreflight", "TrainingRequest",
    "TrainingSubmission",
]
