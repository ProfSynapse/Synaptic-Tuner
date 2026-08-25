"""Private, provider-neutral execution SPI for the next API candidate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Protocol, runtime_checkable

from ._workloads import CanonicalWorkload, LiveVerified, require_live_verified


_SAFE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value) > 256 or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{name} must be bounded text without controls")
    return value


def _safe_ref(value: str, name: str) -> str:
    value = _text(value, name)
    if _SAFE_REF_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded safe reference")
    return value


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical SHA-256 digest")
    return value


class EffectKind(str, Enum):
    SUBMIT = "submit"
    CANCEL = "cancel"


class LookupResult(str, Enum):
    FOUND = "found"
    DEFINITIVELY_ABSENT = "definitively_absent"
    INDETERMINATE = "indeterminate"
    COLLISION = "collision"


class ProviderRunState(str, Enum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    provider: str
    account_ref: str
    namespace_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _text(self.provider, "provider").lower())
        object.__setattr__(self, "account_ref", _text(self.account_ref, "account_ref"))
        object.__setattr__(self, "namespace_ref", _text(self.namespace_ref, "namespace_ref"))


@dataclass(frozen=True, slots=True)
class EffectIdentity:
    effect_id: str
    effect_key: str
    kind: EffectKind
    scope: ExecutionScope

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", _text(self.effect_id, "effect_id"))
        object.__setattr__(self, "effect_key", _text(self.effect_key, "effect_key"))
        if not isinstance(self.kind, EffectKind):
            raise TypeError("kind must be EffectKind")
        if not isinstance(self.scope, ExecutionScope):
            raise TypeError("scope must be ExecutionScope")


@dataclass(frozen=True, slots=True)
class ProviderJobRef:
    provider_job_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_job_id", _safe_ref(self.provider_job_id, "provider_job_id"))


_STAGING_SEAL = object()


class _ProviderStagingCapability:
    __slots__ = ("_seal",)

    def __init__(self, seal: object = None) -> None:
        if seal is not _STAGING_SEAL:
            raise TypeError("provider staging capabilities are production-composition values")
        self._seal = seal


class VerifiedWorkload:
    """Redacted transport; canonical bytes require a staging capability."""

    __slots__ = ("_workload",)

    def __init__(self, workload: CanonicalWorkload, provenance: LiveVerified) -> None:
        require_live_verified(provenance)
        if not isinstance(workload, CanonicalWorkload):
            raise TypeError("workload must be CanonicalWorkload")
        self._workload = workload

    @property
    def digest(self) -> str:
        return self._workload.digest

    def bytes_for_staging(self, capability: object) -> bytes:
        if not isinstance(capability, _ProviderStagingCapability) or capability._seal is not _STAGING_SEAL:
            raise TypeError("canonical workload bytes require provider staging authority")
        return self._workload.canonical_bytes

    def __repr__(self) -> str:
        return f"VerifiedWorkload(digest={self.digest!r}, payload=<redacted>)"

@dataclass(frozen=True, slots=True)
class SubmitRequest:
    identity: EffectIdentity
    plan_fingerprint: str
    source_digest: str
    workload: VerifiedWorkload
    artifact_slot_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EffectIdentity):
            raise TypeError("identity must be EffectIdentity")
        for name in ("plan_fingerprint", "source_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.workload, VerifiedWorkload):
            raise TypeError("workload must be VerifiedWorkload")
        object.__setattr__(self, "artifact_slot_ref", _safe_ref(self.artifact_slot_ref, "artifact_slot_ref"))


@dataclass(frozen=True, slots=True)
class CancelRequest:
    identity: EffectIdentity
    job: ProviderJobRef


@dataclass(frozen=True, slots=True)
class EffectReceipt:
    identity: EffectIdentity
    job: ProviderJobRef
    receipt_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EffectIdentity):
            raise TypeError("identity must be EffectIdentity")
        if not isinstance(self.job, ProviderJobRef):
            raise TypeError("job must be ProviderJobRef")
        object.__setattr__(self, "receipt_digest", _digest(self.receipt_digest, "receipt_digest"))


@dataclass(frozen=True, slots=True)
class EffectObservation:
    result: LookupResult
    identity: EffectIdentity
    job: ProviderJobRef | None = None
    receipt_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, LookupResult):
            raise TypeError("result must be LookupResult")
        if not isinstance(self.identity, EffectIdentity):
            raise TypeError("identity must be EffectIdentity")
        if self.result is LookupResult.FOUND and self.job is None:
            raise ValueError("FOUND observation requires a provider job")
        if self.result is not LookupResult.FOUND and self.job is not None:
            raise ValueError("only FOUND observations may include a provider job")
        if self.result is LookupResult.FOUND and self.receipt_digest is None:
            raise ValueError("FOUND observation requires a receipt digest")
        if self.result is not LookupResult.FOUND and self.receipt_digest is not None:
            raise ValueError("only FOUND observations may include a receipt digest")
        if self.receipt_digest is not None:
            object.__setattr__(self, "receipt_digest", _digest(self.receipt_digest, "receipt_digest"))


@dataclass(frozen=True, slots=True)
class RunObservation:
    job: ProviderJobRef
    state: ProviderRunState
    observation_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.job, ProviderJobRef):
            raise TypeError("job must be ProviderJobRef")
        if not isinstance(self.state, ProviderRunState):
            raise TypeError("state must be ProviderRunState")
        object.__setattr__(self, "observation_digest", _digest(self.observation_digest, "observation_digest"))


class ProviderAuth:
    """Opaque, deliberately non-serializable provider authentication handle."""

    __slots__ = ("_scope", "_value")

    def __init__(self, scope: ExecutionScope, value: object) -> None:
        if not isinstance(scope, ExecutionScope):
            raise TypeError("scope must be ExecutionScope")
        self._scope = scope
        self._value = value

    @property
    def scope(self) -> ExecutionScope:
        return self._scope

    def __repr__(self) -> str:
        return "ProviderAuth(<redacted>)"

    def unwrap_for_provider(self) -> object:
        return self._value

    def __reduce__(self) -> object:
        raise TypeError("ProviderAuth cannot be serialized")


class ProviderEffectError(RuntimeError):
    """Base class whose message is never persisted or returned to callers."""


class DefinitiveNoEffect(ProviderEffectError):
    pass


class EffectOutcomeUnknown(ProviderEffectError):
    pass


class AuthenticationUnavailable(ProviderEffectError):
    pass


class ProviderUnavailable(ProviderEffectError):
    pass


class ProtocolViolation(ProviderEffectError):
    pass


@runtime_checkable
class ExecutionProvider(Protocol):
    """Every mutating method must be idempotently keyed by ``identity``."""

    @property
    def scope(self) -> ExecutionScope: ...

    def submit(self, auth: ProviderAuth, request: SubmitRequest) -> EffectReceipt: ...

    def lookup_submission(
        self, auth: ProviderAuth, identity: EffectIdentity
    ) -> EffectObservation: ...

    def observe(self, auth: ProviderAuth, job: ProviderJobRef) -> RunObservation: ...

    def cancel(self, auth: ProviderAuth, request: CancelRequest) -> EffectReceipt: ...

    def lookup_cancellation(
        self, auth: ProviderAuth, identity: EffectIdentity
    ) -> EffectObservation: ...


__all__ = [
    "AuthenticationUnavailable", "CancelRequest", "DefinitiveNoEffect",
    "EffectIdentity", "EffectKind", "EffectObservation", "EffectOutcomeUnknown",
    "EffectReceipt", "ExecutionProvider", "ExecutionScope", "LookupResult",
    "ProtocolViolation", "ProviderAuth", "ProviderJobRef", "ProviderRunState",
    "ProviderUnavailable", "RunObservation", "SubmitRequest", "VerifiedWorkload",
]
