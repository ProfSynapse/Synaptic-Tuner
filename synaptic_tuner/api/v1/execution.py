"""Provider-neutral execution identities and immutable lifecycle values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _required(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _optional(value: str | None, field_name: str) -> str | None:
    return None if value is None else _required(value, field_name)


class RunState(str, Enum):
    """Closed lifecycle states understood by the public API."""

    PLANNED = "planned"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    SUBMISSION_AMBIGUOUS = "submission_ambiguous"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    CANCEL_AMBIGUOUS = "cancel_ambiguous"
    RECONCILE_REQUIRED = "reconcile_required"
    RECONCILING = "reconciling"


class ErrorCode(str, Enum):
    """Closed, machine-actionable public failure categories."""

    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    NOT_AUTHORIZED = "not_authorized"
    PREFLIGHT_FAILED = "preflight_failed"
    SUBMISSION_FAILED = "submission_failed"
    EXECUTION_FAILED = "execution_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RECONCILE_REQUIRED = "reconcile_required"
    VERIFICATION_FAILED = "verification_failed"
    INTERNAL = "internal"


class ArtifactState(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RunRef:
    """Stable engine-owned identity; never a raw provider job identifier."""

    run_id: str
    project_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _required(self.run_id, "run_id"))
        object.__setattr__(
            self, "project_ref", _required(self.project_ref, "project_ref")
        )


@dataclass(frozen=True, slots=True)
class ExecutionError:
    code: ErrorCode
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, ErrorCode):
            raise TypeError("code must be an ErrorCode")
        object.__setattr__(self, "message", _required(self.message, "message"))
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")


@dataclass(frozen=True, slots=True)
class AuthorizationRequirement:
    operation: str
    paid_effect: bool
    maximum_cost_minor_units: int | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _required(self.operation, "operation"))
        if not isinstance(self.paid_effect, bool):
            raise TypeError("paid_effect must be a boolean")
        amount = self.maximum_cost_minor_units
        if amount is not None and (
            not isinstance(amount, int) or isinstance(amount, bool) or amount < 0
        ):
            raise ValueError("maximum_cost_minor_units must be a non-negative integer")
        currency = _optional(self.currency, "currency")
        if (amount is None) != (currency is None):
            raise ValueError("maximum cost and currency must be supplied together")
        if currency is not None:
            currency = currency.upper()
            if len(currency) != 3 or not currency.isalpha():
                raise ValueError("currency must be a three-letter code")
        object.__setattr__(self, "currency", currency)


@dataclass(frozen=True, slots=True)
class ExecutionGrant:
    """Opaque reference to host-held execution authority."""

    grant_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "grant_ref", _required(self.grant_ref, "grant_ref"))


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    run: RunRef
    kind: str
    state: ArtifactState = ArtifactState.PENDING

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        if not isinstance(self.state, ArtifactState):
            raise TypeError("state must be an ArtifactState")
        object.__setattr__(
            self, "artifact_id", _required(self.artifact_id, "artifact_id")
        )
        object.__setattr__(self, "kind", _required(self.kind, "kind"))


@dataclass(frozen=True, slots=True)
class RunStatus:
    run: RunRef
    state: RunState
    updated_at: str
    error: ExecutionError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        if not isinstance(self.state, RunState):
            raise TypeError("state must be a RunState")
        object.__setattr__(self, "updated_at", _required(self.updated_at, "updated_at"))
        if self.error is not None and not isinstance(self.error, ExecutionError):
            raise TypeError("error must be an ExecutionError")


__all__ = [
    "ArtifactRef",
    "ArtifactState",
    "AuthorizationRequirement",
    "ErrorCode",
    "ExecutionError",
    "ExecutionGrant",
    "RunRef",
    "RunState",
    "RunStatus",
]
