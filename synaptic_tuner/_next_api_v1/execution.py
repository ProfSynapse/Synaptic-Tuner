"""Provider-neutral execution and lifecycle contracts.

These values carry Synaptic-owned identities only. Provider identifiers,
credentials, and secret values are deliberately absent from the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Protocol, runtime_checkable


def _required(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _optional(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _required(value, field_name)


def _require_bool(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")


def _require_instance(value: object, expected: type, field_name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{field_name} must be {expected.__name__}")


def _timestamp(value: str, field_name: str) -> str:
    text = _required(value, field_name)
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serializable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("contract values cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


class SerializableContract:
    """JSON-safe representation shared by immutable public values."""

    def to_dict(self) -> dict[str, Any]:
        value = _serializable(self)
        if not isinstance(value, dict):  # pragma: no cover - defensive invariant
            raise TypeError("contract serialization must produce an object")
        return value


@dataclass(frozen=True, slots=True)
class AccessContext(SerializableContract):
    """Authenticated principal and project references, never credentials."""

    principal_ref: str
    project_ref: str
    authentication_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "principal_ref", _required(self.principal_ref, "principal_ref")
        )
        object.__setattr__(
            self, "project_ref", _required(self.project_ref, "project_ref")
        )
        object.__setattr__(
            self,
            "authentication_ref",
            _required(self.authentication_ref, "authentication_ref"),
        )


@dataclass(frozen=True, slots=True)
class RunRef(SerializableContract):
    """Stable Synaptic-owned run identity; never a raw provider job ID."""

    run_id: str
    project_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _required(self.run_id, "run_id"))
        object.__setattr__(
            self, "project_ref", _required(self.project_ref, "project_ref")
        )


class RunState(str, Enum):
    PLANNED = "planned"
    AUTHORIZED = "authorized"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    NOT_SUBMITTED = "not_submitted"
    SUBMISSION_AMBIGUOUS = "submission_ambiguous"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    CANCEL_FAILED = "cancel_failed"
    CANCEL_AMBIGUOUS = "cancel_ambiguous"
    RECONCILE_REQUIRED = "reconcile_required"
    RECONCILING = "reconciling"


class ArtifactVerificationState(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArtifactRef(SerializableContract):
    artifact_id: str
    run: RunRef
    kind: str
    verification: ArtifactVerificationState = ArtifactVerificationState.PENDING

    def __post_init__(self) -> None:
        _require_instance(self.run, RunRef, "run")
        _require_instance(
            self.verification, ArtifactVerificationState, "verification"
        )
        object.__setattr__(
            self, "artifact_id", _required(self.artifact_id, "artifact_id")
        )
        object.__setattr__(self, "kind", _required(self.kind, "kind"))


@dataclass(frozen=True, slots=True)
class AuthorizationRequirement(SerializableContract):
    """Disclosed authority requirement; this is not execution authority."""

    operation: str
    paid_effect: bool
    maximum_cost_minor_units: int | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", _required(self.operation, "operation")
        )
        _require_bool(self.paid_effect, "paid_effect")
        if self.maximum_cost_minor_units is not None and (
            not isinstance(self.maximum_cost_minor_units, int)
            or isinstance(self.maximum_cost_minor_units, bool)
        ):
            raise TypeError("maximum_cost_minor_units must be an integer")
        if self.maximum_cost_minor_units is not None and self.maximum_cost_minor_units < 0:
            raise ValueError("maximum_cost_minor_units must be non-negative")
        currency = _optional(self.currency, "currency")
        if self.maximum_cost_minor_units is None and currency is not None:
            raise ValueError("currency requires maximum_cost_minor_units")
        if self.maximum_cost_minor_units is not None and currency is None:
            raise ValueError("maximum_cost_minor_units requires currency")
        if currency is not None:
            currency = currency.upper()
            if len(currency) != 3 or not currency.isalpha():
                raise ValueError("currency must be a three-letter code")
        object.__setattr__(self, "currency", currency)


@dataclass(frozen=True, slots=True)
class ExecutionGrant(SerializableContract):
    """Opaque handle to authority held and verified by the execution service."""

    grant_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "grant_ref", _required(self.grant_ref, "grant_ref")
        )


@dataclass(frozen=True, slots=True)
class ProviderDescriptor(SerializableContract):
    provider: str
    display_name: str
    supported_methods: tuple[str, ...] = ()
    available: bool = False
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        _require_bool(self.available, "available")
        object.__setattr__(
            self, "provider", _required(self.provider, "provider").lower()
        )
        object.__setattr__(
            self, "display_name", _required(self.display_name, "display_name")
        )
        if isinstance(self.supported_methods, (str, bytes)):
            raise TypeError("supported_methods must be a sequence of strings")
        try:
            methods = tuple(
                _required(method, "supported_method").lower()
                for method in self.supported_methods
            )
        except TypeError as exc:
            raise TypeError(
                "supported_methods must be a sequence of strings"
            ) from exc
        if len(methods) != len(set(methods)):
            raise ValueError("supported_methods must not contain duplicates")
        object.__setattr__(self, "supported_methods", methods)
        reason = _optional(self.unavailable_reason, "unavailable_reason")
        if self.available and reason is not None:
            raise ValueError(
                "an available provider cannot have an unavailable_reason"
            )
        if not self.available and reason is None:
            raise ValueError(
                "an unavailable provider requires unavailable_reason"
            )
        object.__setattr__(self, "unavailable_reason", reason)


@dataclass(frozen=True, slots=True)
class RunStatus(SerializableContract):
    run: RunRef
    state: RunState
    artifact_state: ArtifactVerificationState
    updated_at: str
    message_code: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.run, RunRef, "run")
        _require_instance(self.state, RunState, "state")
        _require_instance(
            self.artifact_state, ArtifactVerificationState, "artifact_state"
        )
        object.__setattr__(
            self, "updated_at", _timestamp(self.updated_at, "updated_at")
        )
        object.__setattr__(
            self, "message_code", _optional(self.message_code, "message_code")
        )


@dataclass(frozen=True, slots=True)
class LogCursor(SerializableContract):
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _required(self.value, "cursor"))


@dataclass(frozen=True, slots=True)
class LogEntry(SerializableContract):
    sequence: int
    timestamp: str
    level: str
    event: str
    message: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")
        object.__setattr__(
            self, "timestamp", _timestamp(self.timestamp, "timestamp")
        )
        object.__setattr__(
            self, "level", _required(self.level, "level").lower()
        )
        object.__setattr__(self, "event", _required(self.event, "event"))
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")


@dataclass(frozen=True, slots=True)
class LogPage(SerializableContract):
    run: RunRef
    entries: tuple[LogEntry, ...]
    next_cursor: LogCursor | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        _require_instance(self.run, RunRef, "run")
        if isinstance(self.entries, (str, bytes)):
            raise TypeError("entries must be a sequence of LogEntry values")
        try:
            entries = tuple(self.entries)
        except TypeError as exc:
            raise TypeError(
                "entries must be a sequence of LogEntry values"
            ) from exc
        if any(not isinstance(entry, LogEntry) for entry in entries):
            raise TypeError("entries must contain only LogEntry values")
        object.__setattr__(self, "entries", entries)
        if self.next_cursor is not None:
            _require_instance(self.next_cursor, LogCursor, "next_cursor")
        _require_bool(self.truncated, "truncated")


@dataclass(frozen=True, slots=True)
class CancelResult(SerializableContract):
    run: RunRef
    state: RunState
    accepted: bool
    message_code: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.run, RunRef, "run")
        _require_instance(self.state, RunState, "state")
        _require_bool(self.accepted, "accepted")
        if self.state not in {
            RunState.CANCEL_REQUESTED,
            RunState.CANCELLING,
            RunState.CANCELLED,
            RunState.CANCEL_FAILED,
            RunState.CANCEL_AMBIGUOUS,
        }:
            raise ValueError("cancel result must use a cancellation state")
        object.__setattr__(
            self, "message_code", _optional(self.message_code, "message_code")
        )


@runtime_checkable
class JobLifecycle(Protocol):
    """Internal shape shared by lifecycle-facing services."""

    def status(self, access: AccessContext, run: RunRef) -> RunStatus: ...


__all__ = [
    "AccessContext",
    "ArtifactRef",
    "ArtifactVerificationState",
    "AuthorizationRequirement",
    "CancelResult",
    "ExecutionGrant",
    "LogCursor",
    "LogEntry",
    "LogPage",
    "ProviderDescriptor",
    "RunRef",
    "RunState",
    "RunStatus",
]
