"""Run observation and provider-staging verification facade."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ._contract import required_text
from ._timestamps import require_rfc3339
from .results import TrainingRunRef, TrainingRunState, VerifiedArtifact


def _artifact_object(artifacts: tuple[VerifiedArtifact, ...]) -> dict[str, object]:
    return {
        artifact.role: {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes}
        for artifact in sorted(artifacts, key=lambda item: item.role)
    }


def _exact_object(
    value: object, expected: frozenset[str], name: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact object")
    keys = tuple(dict.keys(value))
    if any(type(key) is not str for key in keys):
        raise TypeError(f"{name} field names must be exact strings")
    actual = frozenset(keys)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details = []
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        raise ValueError(f"{name} has invalid fields ({'; '.join(details)})")
    return {key: dict.__getitem__(value, key) for key in keys}


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return required_text(value, name)


def _parse_artifact_object(value: object) -> tuple[VerifiedArtifact, ...]:
    if type(value) is not dict:
        raise TypeError("artifacts must be a role-keyed object")
    artifacts = []
    roles = tuple(dict.keys(value))
    if any(type(role) is not str for role in roles):
        raise TypeError("artifact roles must be exact strings")
    for role in sorted(roles):
        role = _text(role, "artifact_role")
        descriptor = _exact_object(
            dict.__getitem__(value, role),
            frozenset({"sha256", "size_bytes"}),
            "artifact_descriptor",
        )
        artifacts.append(VerifiedArtifact(
            role,
            _text(descriptor["sha256"], "sha256"),
            descriptor["size_bytes"],  # type: ignore[arg-type]
        ))
    return tuple(artifacts)


def _exact_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def _ascii_cursor(value: str | None, name: str = "cursor") -> str | None:
    if value is None:
        return None
    value = _text(value, name)
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
        if type(self.schema_version) is not str or self.schema_version != "synaptic-run-outcome/v1":
            raise ValueError("unsupported run outcome schema version")
        if type(self.run) is not TrainingRunRef or type(self.state) is not TrainingRunState:
            raise TypeError("run/state have invalid types")
        object.__setattr__(self, "run", TrainingRunRef.from_dict(self.run.to_dict()))
        if type(self.artifacts) is not tuple or any(type(item) is not VerifiedArtifact for item in self.artifacts):
            raise TypeError("artifacts must be an exact tuple of VerifiedArtifact values")
        artifacts = tuple(VerifiedArtifact.from_dict(item.to_dict()) for item in self.artifacts)
        roles = tuple(item.role for item in artifacts)
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        if self.diagnostic_code is not None:
            object.__setattr__(self, "diagnostic_code", _text(self.diagnostic_code, "diagnostic_code"))
        object.__setattr__(self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.role)))

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "run": self.run.to_dict(), "state": self.state.value, "artifacts": _artifact_object(self.artifacts), "diagnostic_code": self.diagnostic_code}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RunOutcome":
        value = _exact_object(value, frozenset({"schema_version", "run", "state", "artifacts", "diagnostic_code"}), "run_outcome")
        run = _exact_object(value["run"], frozenset({"run_id", "project_ref"}), "run")
        state = value["state"]
        if type(state) is not str:
            raise TypeError("state must be an exact string")
        diagnostic = value["diagnostic_code"]
        if diagnostic is not None and type(diagnostic) is not str:
            raise TypeError("diagnostic_code must be an exact string or null")
        return cls(
            _text(value["schema_version"], "schema_version"),
            TrainingRunRef(_text(run["run_id"], "run_id"), _text(run["project_ref"], "project_ref")),
            TrainingRunState(state),
            _parse_artifact_object(value["artifacts"]),
            diagnostic,
        )


@dataclass(frozen=True, slots=True)
class RunListRequest:
    project_ref: str
    cursor: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
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
        object.__setattr__(self, "event", _text(self.event, "event"))
        object.__setattr__(self, "message", _text(self.message, "message"))
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
    def from_dict(cls, value: dict[str, object]) -> "RunLogEntry":
        value = _exact_object(
            value,
            frozenset({"sequence", "timestamp", "level", "event", "message", "size_bytes"}),
            "run_log_entry",
        )
        level = value["level"]
        if type(level) is not str:
            raise TypeError("level must be an exact string")
        return cls(
            sequence=value["sequence"],  # type: ignore[arg-type]
            timestamp=_text(value["timestamp"], "timestamp"),
            level=RunLogLevel(level),
            event=_text(value["event"], "event"),
            message=_text(value["message"], "message"),
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
        object.__setattr__(self, "run", TrainingRunRef.from_dict(self.run.to_dict()))
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
        if type(self.run) is not TrainingRunRef or type(self.verified) is not bool:
            raise TypeError("run/verified have invalid types")
        object.__setattr__(self, "run", TrainingRunRef.from_dict(self.run.to_dict()))
        object.__setattr__(self, "checked_at", require_rfc3339(_text(self.checked_at, "checked_at"), "checked_at"))


@dataclass(frozen=True, slots=True)
class RunArtifactRequest:
    run: TrainingRunRef
    role: str
    maximum_bytes: int

    def __post_init__(self) -> None:
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        object.__setattr__(self, "run", TrainingRunRef.from_dict(self.run.to_dict()))
        object.__setattr__(self, "role", _text(self.role, "role"))
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
    def __init__(self, operations: RunsOperations) -> None:
        self._operations = operations

    @staticmethod
    def _run(value: TrainingRunRef) -> TrainingRunRef:
        if type(value) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        return TrainingRunRef.from_dict(value.to_dict())

    @staticmethod
    def _list_request(value: RunListRequest) -> RunListRequest:
        if type(value) is not RunListRequest:
            raise TypeError("request must be exact RunListRequest")
        return RunListRequest(value.project_ref, value.cursor, value.limit)

    @classmethod
    def _logs_request(cls, value: RunLogsRequest) -> RunLogsRequest:
        if type(value) is not RunLogsRequest:
            raise TypeError("request must be exact RunLogsRequest")
        return RunLogsRequest(cls._run(value.run), value.cursor, value.limit, value.maximum_bytes)

    @classmethod
    def _artifact_request(cls, value: RunArtifactRequest) -> RunArtifactRequest:
        if type(value) is not RunArtifactRequest:
            raise TypeError("request must be exact RunArtifactRequest")
        return RunArtifactRequest(cls._run(value.run), value.role, value.maximum_bytes)

    @staticmethod
    def _matches(current: object, baseline: object, rebuild) -> bool:
        try:
            return rebuild(current) == baseline
        except BaseException:
            return False

    @staticmethod
    def _changed() -> None:
        raise ValueError("run operation input changed during callback") from None

    @classmethod
    def _unchanged(cls, current: object, baseline: object, rebuild) -> None:
        if not cls._matches(current, baseline, rebuild):
            cls._changed()

    @classmethod
    def _call(cls, callback, original, baseline, presentation, rebuild, *extra):
        mutation_after_failure = False
        try:
            result = callback(presentation, *extra)
        except BaseException:
            mutation_after_failure = (
                not cls._matches(original, baseline, rebuild)
                or not cls._matches(presentation, baseline, rebuild)
            )
            if not mutation_after_failure:
                raise
        if mutation_after_failure:
            cls._changed()
        cls._unchanged(original, baseline, rebuild)
        cls._unchanged(presentation, baseline, rebuild)
        return result

    def list(self, request: RunListRequest) -> RunPage:
        baseline = self._list_request(request)
        presented = self._list_request(baseline)
        result = self._call(self._operations.list, request, baseline, presented, self._list_request)
        if type(result) is not RunPage:
            raise TypeError("run list result must be exact RunPage")
        rebuilt = RunPage(
            RunListRequest(result.request.project_ref, result.request.cursor, result.request.limit),
            tuple(RunOutcome.from_dict(item.to_dict()) for item in result.outcomes),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("run list result does not bind the request")
        self._unchanged(request, baseline, self._list_request)
        self._unchanged(presented, baseline, self._list_request)
        return rebuilt

    @staticmethod
    def _outcome(value: object, run: TrainingRunRef) -> RunOutcome:
        if type(value) is not RunOutcome:
            raise TypeError("run result must be exact RunOutcome")
        rebuilt = RunOutcome.from_dict(value.to_dict())
        if rebuilt.run != run:
            raise ValueError("run result does not bind the request")
        return rebuilt

    def show(self, run: TrainingRunRef) -> RunOutcome:
        baseline = self._run(run)
        presented = self._run(baseline)
        result = self._call(self._operations.show, run, baseline, presented, self._run)
        rebuilt = self._outcome(result, baseline)
        self._unchanged(run, baseline, self._run); self._unchanged(presented, baseline, self._run)
        return rebuilt

    def outcome(self, run: TrainingRunRef) -> RunOutcome:
        baseline = self._run(run)
        presented = self._run(baseline)
        rebuilt = self._outcome(self._call(self._operations.outcome, run, baseline, presented, self._run), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    def logs(self, request: RunLogsRequest) -> RunLogPage:
        baseline = self._logs_request(request)
        presented = self._logs_request(baseline)
        result = self._call(self._operations.logs, request, baseline, presented, self._logs_request)
        if type(result) is not RunLogPage:
            raise TypeError("run log result must be exact RunLogPage")
        rebuilt = RunLogPage(
            RunLogsRequest(
                self._run(result.request.run), result.request.cursor,
                result.request.limit, result.request.maximum_bytes,
            ),
            tuple(RunLogEntry.from_dict(item.to_dict()) for item in result.entries),
            result.total_bytes,
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("run log result does not bind the request")
        self._unchanged(request, baseline, self._logs_request)
        self._unchanged(presented, baseline, self._logs_request)
        return rebuilt

    def cancel(self, run: TrainingRunRef, reason: str) -> RunOutcome:
        baseline = self._run(run)
        presented = self._run(baseline)
        reason = _text(reason, "reason")
        rebuilt = self._outcome(self._call(self._operations.cancel, run, baseline, presented, self._run, reason), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    def reconcile(self, run: TrainingRunRef) -> RunOutcome:
        baseline = self._run(run)
        presented = self._run(baseline)
        rebuilt = self._outcome(self._call(self._operations.reconcile, run, baseline, presented, self._run), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    @staticmethod
    def _verification(value: object, run: TrainingRunRef) -> RunVerification:
        if type(value) is not RunVerification:
            raise TypeError("verification result must be exact RunVerification")
        rebuilt = RunVerification(
            TrainingRunRef.from_dict(value.run.to_dict()),
            value.verified,
            value.checked_at,
        )
        if rebuilt.run != run:
            raise ValueError("verification result does not bind the request")
        return rebuilt

    def verify(self, run: TrainingRunRef) -> RunVerification:
        baseline = self._run(run)
        presented = self._run(baseline)
        rebuilt = self._verification(self._call(self._operations.verify, run, baseline, presented, self._run), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    def reverify(self, run: TrainingRunRef) -> RunVerification:
        baseline = self._run(run)
        presented = self._run(baseline)
        rebuilt = self._verification(self._call(self._operations.reverify, run, baseline, presented, self._run), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    def artifacts(self, request: RunArtifactRequest) -> RunArtifactStream:
        baseline = self._artifact_request(request)
        presented = self._artifact_request(baseline)
        stream = self._call(self._operations.artifacts, request, baseline, presented, self._artifact_request)
        try:
            if (self._run(stream.run) != baseline.run or type(stream.artifact) is not VerifiedArtifact
                    or stream.artifact.role != baseline.role
                    or type(stream.maximum_bytes) is not int
                    or stream.maximum_bytes != baseline.maximum_bytes
                    or not callable(stream.iter_bytes)):
                raise ValueError("artifact stream does not bind the request")
        except Exception:
            raise ValueError("artifact stream is invalid") from None
        self._unchanged(request, baseline, self._artifact_request)
        self._unchanged(presented, baseline, self._artifact_request)
        return stream


__all__ = [
    "RunArtifactRequest", "RunArtifactStream", "RunListRequest", "RunLogEntry",
    "RunLogLevel", "RunLogPage", "RunLogsRequest", "RunOperationCode",
    "RunOperationError", "RunOutcome", "RunPage", "RunVerification", "RunsAPI",
    "RunsOperations",
]
