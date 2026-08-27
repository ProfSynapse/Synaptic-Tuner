"""Run observation and provider-staging verification facade."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol

from ._contract import exact_fields, required_text
from ._timestamps import require_rfc3339
from .results import TrainingRunRef, TrainingRunState, VerifiedArtifact


def _artifact_object(artifacts: tuple[VerifiedArtifact, ...]) -> dict[str, object]:
    return {
        artifact.role: {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes}
        for artifact in sorted(artifacts, key=lambda item: item.role)
    }


def _parse_artifact_object(value: object) -> tuple[VerifiedArtifact, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("artifacts must be a role-keyed object")
    artifacts = []
    for role in sorted(value):
        required_text(role, "artifact_role")
        descriptor = value[role]
        if not isinstance(descriptor, Mapping):
            raise TypeError("artifact descriptors must be objects")
        exact_fields(descriptor, frozenset({"sha256", "size_bytes"}), "artifact_descriptor")
        artifacts.append(VerifiedArtifact(role, descriptor["sha256"], descriptor["size_bytes"]))  # type: ignore[arg-type]
    return tuple(artifacts)


def _exact_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def _ascii_cursor(value: str | None, name: str = "cursor") -> str | None:
    if value is None:
        return None
    value = required_text(value, name)
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be ASCII") from None
    if len(encoded) > 256:
        raise ValueError(f"{name} exceeds 256 bytes")
    return value


@dataclass(frozen=True, slots=True)
class RunOutcome:
    schema_version: str
    run: TrainingRunRef
    state: TrainingRunState
    artifacts: tuple[VerifiedArtifact, ...] = ()
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-run-outcome/v1":
            raise ValueError("unsupported run outcome schema version")
        if not isinstance(self.run, TrainingRunRef) or not isinstance(self.state, TrainingRunState):
            raise TypeError("run/state have invalid types")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, VerifiedArtifact) for item in artifacts):
            raise TypeError("artifacts must contain VerifiedArtifact values")
        roles = tuple(item.role for item in artifacts)
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        if self.diagnostic_code is not None:
            object.__setattr__(self, "diagnostic_code", required_text(self.diagnostic_code, "diagnostic_code"))
        object.__setattr__(self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.role)))

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "run": self.run.to_dict(), "state": self.state.value, "artifacts": _artifact_object(self.artifacts), "diagnostic_code": self.diagnostic_code}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RunOutcome":
        exact_fields(value, frozenset({"schema_version", "run", "state", "artifacts", "diagnostic_code"}), "run_outcome")
        if not isinstance(value["run"], Mapping):
            raise TypeError("run must be an object")
        return cls(value["schema_version"], TrainingRunRef.from_dict(value["run"]), TrainingRunState(value["state"]), _parse_artifact_object(value["artifacts"]), value["diagnostic_code"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RunListRequest:
    project_ref: str
    cursor: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", required_text(self.project_ref, "project_ref"))
        object.__setattr__(self, "cursor", _ascii_cursor(self.cursor))
        _exact_integer(self.limit, "limit", minimum=1, maximum=100)


@dataclass(frozen=True, slots=True)
class RunPage:
    request: RunListRequest
    outcomes: tuple[RunOutcome, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not RunListRequest:
            raise TypeError("request must be exact RunListRequest")
        if type(self.outcomes) is not tuple or any(type(item) is not RunOutcome for item in self.outcomes):
            raise TypeError("outcomes must be an exact tuple of RunOutcome")
        if len(self.outcomes) > self.request.limit:
            raise ValueError("outcomes exceed requested limit")
        if any(item.run.project_ref != self.request.project_ref for item in self.outcomes):
            raise ValueError("outcome project does not match list request")
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be an exact boolean")
        object.__setattr__(self, "next_cursor", _ascii_cursor(self.next_cursor, "next_cursor"))
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.outcomes:
            raise ValueError("a truncated page must contain an outcome")


class RunLogLevel(str, Enum):
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class RunLogEntry:
    sequence: int
    timestamp: str
    level: RunLogLevel
    event: str
    message: str
    size_bytes: int

    def __post_init__(self) -> None:
        _exact_integer(self.sequence, "sequence", minimum=0, maximum=2**63 - 1)
        object.__setattr__(
            self, "timestamp", require_rfc3339(self.timestamp, "timestamp")
        )
        if type(self.level) is not RunLogLevel:
            raise TypeError("level must be exact RunLogLevel")
        object.__setattr__(self, "event", required_text(self.event, "event"))
        object.__setattr__(self, "message", required_text(self.message, "message"))
        size = len(self.message.encode("utf-8"))
        if size > 4096:
            raise ValueError("message exceeds 4096 UTF-8 bytes")
        _exact_integer(self.size_bytes, "size_bytes", minimum=1, maximum=4096)
        if self.size_bytes != size:
            raise ValueError("size_bytes does not match message UTF-8 bytes")

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "event": self.event,
            "message": self.message,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RunLogEntry":
        exact_fields(
            value,
            frozenset({"sequence", "timestamp", "level", "event", "message", "size_bytes"}),
            "run_log_entry",
        )
        return cls(
            sequence=value["sequence"],  # type: ignore[arg-type]
            timestamp=value["timestamp"],  # type: ignore[arg-type]
            level=RunLogLevel(value["level"]),
            event=value["event"],  # type: ignore[arg-type]
            message=value["message"],  # type: ignore[arg-type]
            size_bytes=value["size_bytes"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RunLogsRequest:
    run: TrainingRunRef
    cursor: str | None = None
    limit: int = 200
    maximum_bytes: int = 262144

    def __post_init__(self) -> None:
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        object.__setattr__(self, "cursor", _ascii_cursor(self.cursor))
        _exact_integer(self.limit, "limit", minimum=1, maximum=200)
        _exact_integer(self.maximum_bytes, "maximum_bytes", minimum=4096, maximum=262144)


@dataclass(frozen=True, slots=True)
class RunLogPage:
    request: RunLogsRequest
    entries: tuple[RunLogEntry, ...]
    total_bytes: int
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not RunLogsRequest:
            raise TypeError("request must be exact RunLogsRequest")
        if type(self.entries) is not tuple or any(type(item) is not RunLogEntry for item in self.entries):
            raise TypeError("entries must be an exact tuple of RunLogEntry")
        if len(self.entries) > self.request.limit:
            raise ValueError("entries exceed requested limit")
        sequences = tuple(item.sequence for item in self.entries)
        if any(left >= right for left, right in zip(sequences, sequences[1:])):
            raise ValueError("log sequences must be unique and strictly increasing")
        total = sum(item.size_bytes for item in self.entries)
        _exact_integer(self.total_bytes, "total_bytes", minimum=0, maximum=262144)
        if self.total_bytes != total or self.total_bytes > self.request.maximum_bytes:
            raise ValueError("total_bytes does not match bounded log entries")
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be an exact boolean")
        object.__setattr__(self, "next_cursor", _ascii_cursor(self.next_cursor, "next_cursor"))
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.entries:
            raise ValueError("a truncated log page must contain an entry")


@dataclass(frozen=True, slots=True)
class RunVerification:
    run: TrainingRunRef
    verified: bool
    checked_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.run, TrainingRunRef) or not isinstance(self.verified, bool):
            raise TypeError("run/verified have invalid types")
        object.__setattr__(self, "checked_at", required_text(self.checked_at, "checked_at"))


@dataclass(frozen=True, slots=True)
class RunArtifactRequest:
    run: TrainingRunRef
    role: str
    maximum_bytes: int

    def __post_init__(self) -> None:
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        object.__setattr__(self, "role", required_text(self.role, "role"))
        _exact_integer(self.maximum_bytes, "maximum_bytes", minimum=1, maximum=2**63 - 1)


class RunArtifactStream(Protocol):
    @property
    def run(self) -> TrainingRunRef: ...
    @property
    def artifact(self) -> VerifiedArtifact: ...
    @property
    def maximum_bytes(self) -> int: ...
    def iter_bytes(self) -> Iterator[bytes]: ...


class RunOperationCode(str, Enum):
    RUN_MISSING = "run_missing"
    CURSOR_INVALID = "cursor_invalid"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    READ_INELIGIBLE = "read_ineligible"
    PROVIDER_READ_INVALID = "provider_read_invalid"
    LOG_BOUNDS_INVALID = "log_bounds_invalid"
    CANCEL_INELIGIBLE = "cancel_ineligible"
    ARTIFACTS_UNVERIFIED = "artifacts_unverified"
    ARTIFACT_ROLE_MISSING = "artifact_role_missing"
    ARTIFACT_LIMIT_EXCEEDED = "artifact_limit_exceeded"
    ARTIFACT_CONTENT_INVALID = "artifact_content_invalid"
    STATE_CONFLICT = "state_conflict"
    INTEGRITY_ERROR = "integrity_error"


class RunOperationError(ValueError):
    def __init__(self, code: RunOperationCode) -> None:
        if type(code) is not RunOperationCode:
            raise TypeError("code must be exact RunOperationCode")
        self.code = code
        super().__init__(code.value)


class RunsOperations(Protocol):
    def list(self, request: RunListRequest) -> RunPage: ...
    def show(self, run: TrainingRunRef) -> RunOutcome: ...
    def outcome(self, run: TrainingRunRef) -> RunOutcome: ...
    def logs(self, request: RunLogsRequest) -> RunLogPage: ...
    def cancel(self, run: TrainingRunRef, reason: str) -> RunOutcome: ...
    def reconcile(self, run: TrainingRunRef) -> RunOutcome: ...
    def verify(self, run: TrainingRunRef) -> RunVerification: ...
    def reverify(self, run: TrainingRunRef) -> RunVerification: ...
    def artifacts(self, request: RunArtifactRequest) -> RunArtifactStream: ...


class RunsAPI:
    __slots__ = ("_operations",)
    def __init__(self, operations: RunsOperations) -> None: self._operations = operations
    def list(self, request: RunListRequest) -> RunPage: return self._operations.list(request)
    def show(self, run: TrainingRunRef) -> RunOutcome: return self._operations.show(run)
    def outcome(self, run: TrainingRunRef) -> RunOutcome: return self._operations.outcome(run)
    def logs(self, request: RunLogsRequest) -> RunLogPage: return self._operations.logs(request)
    def cancel(self, run: TrainingRunRef, reason: str) -> RunOutcome: return self._operations.cancel(run, reason)
    def reconcile(self, run: TrainingRunRef) -> RunOutcome: return self._operations.reconcile(run)
    def verify(self, run: TrainingRunRef) -> RunVerification: return self._operations.verify(run)
    def reverify(self, run: TrainingRunRef) -> RunVerification: return self._operations.reverify(run)
    def artifacts(self, request: RunArtifactRequest) -> RunArtifactStream: return self._operations.artifacts(request)


__all__ = [
    "RunArtifactRequest", "RunArtifactStream", "RunListRequest", "RunLogEntry",
    "RunLogLevel", "RunLogPage", "RunLogsRequest", "RunOperationCode",
    "RunOperationError", "RunOutcome", "RunPage", "RunVerification", "RunsAPI",
    "RunsOperations",
]
