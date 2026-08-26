"""Stable run-observation and lifecycle-control facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .execution import ArtifactRef, ArtifactState, ExecutionError, RunRef, RunState, RunStatus


def _required(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _limit(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000")


@dataclass(frozen=True, slots=True)
class RunListRequest:
    project_ref: str
    states: tuple[RunState, ...] = ()
    cursor: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_ref", _required(self.project_ref, "project_ref")
        )
        states = tuple(self.states)
        if any(not isinstance(item, RunState) for item in states):
            raise TypeError("states must contain RunState values")
        if len(states) != len(set(states)):
            raise ValueError("states must not contain duplicates")
        if self.cursor is not None:
            object.__setattr__(self, "cursor", _required(self.cursor, "cursor"))
        _limit(self.limit)
        object.__setattr__(self, "states", states)


@dataclass(frozen=True, slots=True)
class RunPage:
    runs: tuple[RunStatus, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        if any(not isinstance(item, RunStatus) for item in runs):
            raise TypeError("runs must contain RunStatus values")
        if self.next_cursor is not None:
            object.__setattr__(
                self, "next_cursor", _required(self.next_cursor, "next_cursor")
            )
        object.__setattr__(self, "runs", runs)


@dataclass(frozen=True, slots=True)
class RunLogsRequest:
    run: RunRef
    cursor: str | None = None
    limit: int = 200

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        if self.cursor is not None:
            object.__setattr__(self, "cursor", _required(self.cursor, "cursor"))
        _limit(self.limit)


@dataclass(frozen=True, slots=True)
class LogEntry:
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
        for name in ("timestamp", "level", "event", "message"):
            object.__setattr__(self, name, _required(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class LogPage:
    run: RunRef
    entries: tuple[LogEntry, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        entries = tuple(self.entries)
        if any(not isinstance(item, LogEntry) for item in entries):
            raise TypeError("entries must contain LogEntry values")
        if self.next_cursor is not None:
            object.__setattr__(
                self, "next_cursor", _required(self.next_cursor, "next_cursor")
            )
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a boolean")
        object.__setattr__(self, "entries", entries)


@dataclass(frozen=True, slots=True)
class RunCancelRequest:
    run: RunRef
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        object.__setattr__(self, "reason", _required(self.reason, "reason"))


@dataclass(frozen=True, slots=True)
class CancelResult:
    run: RunRef
    state: RunState
    accepted: bool
    error: ExecutionError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        if self.state not in {
            RunState.CANCEL_REQUESTED,
            RunState.CANCELLING,
            RunState.CANCELLED,
            RunState.CANCEL_AMBIGUOUS,
        }:
            raise ValueError("cancel result requires a cancellation state")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean")
        if self.error is not None and not isinstance(self.error, ExecutionError):
            raise TypeError("error must be an ExecutionError")


@dataclass(frozen=True, slots=True)
class ReconcileRequest:
    run: RunRef

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")


@dataclass(frozen=True, slots=True)
class VerifyRequest:
    run: RunRef

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")


@dataclass(frozen=True, slots=True)
class RunVerification:
    run: RunRef
    state: ArtifactState
    checked_at: str
    errors: tuple[ExecutionError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        if not isinstance(self.state, ArtifactState):
            raise TypeError("state must be an ArtifactState")
        object.__setattr__(self, "checked_at", _required(self.checked_at, "checked_at"))
        errors = tuple(self.errors)
        if any(not isinstance(item, ExecutionError) for item in errors):
            raise TypeError("errors must contain ExecutionError values")
        if self.state is ArtifactState.VERIFIED and errors:
            raise ValueError("verified result cannot contain errors")
        if self.state is ArtifactState.FAILED and not errors:
            raise ValueError("failed verification requires an error")
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class ArtifactsRequest:
    run: RunRef

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")


@dataclass(frozen=True, slots=True)
class ArtifactPage:
    run: RunRef
    artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise TypeError("artifacts must contain ArtifactRef values")
        if any(item.run != self.run for item in artifacts):
            raise ValueError("artifacts must refer to the requested run")
        object.__setattr__(self, "artifacts", artifacts)


class RunsOperations(Protocol):
    def list(self, request: RunListRequest) -> RunPage: ...

    def show(self, run: RunRef) -> RunStatus: ...

    def logs(self, request: RunLogsRequest) -> LogPage: ...

    def cancel(self, request: RunCancelRequest) -> CancelResult: ...

    def reconcile(self, request: ReconcileRequest) -> RunStatus: ...

    def verify(self, request: VerifyRequest) -> RunVerification: ...

    def artifacts(self, request: ArtifactsRequest) -> ArtifactPage: ...


class RunsAPI:
    """Import-light facade over a host-selected lifecycle repository."""

    __slots__ = ("_operations",)

    def __init__(self, operations: RunsOperations) -> None:
        self._operations = operations

    def list(self, request: RunListRequest) -> RunPage:
        return self._operations.list(request)

    def show(self, run: RunRef) -> RunStatus:
        status = self._operations.show(run)
        self._same_run(run, status.run)
        return status

    def logs(self, request: RunLogsRequest) -> LogPage:
        page = self._operations.logs(request)
        self._same_run(request.run, page.run)
        return page

    def cancel(self, request: RunCancelRequest) -> CancelResult:
        result = self._operations.cancel(request)
        self._same_run(request.run, result.run)
        return result

    def reconcile(self, request: ReconcileRequest) -> RunStatus:
        status = self._operations.reconcile(request)
        self._same_run(request.run, status.run)
        return status

    def verify(self, request: VerifyRequest) -> RunVerification:
        result = self._operations.verify(request)
        self._same_run(request.run, result.run)
        return result

    def artifacts(self, request: ArtifactsRequest) -> ArtifactPage:
        page = self._operations.artifacts(request)
        self._same_run(request.run, page.run)
        return page

    @staticmethod
    def _same_run(expected: RunRef, actual: RunRef) -> None:
        if actual != expected:
            raise ValueError("lifecycle result does not refer to the requested run")


__all__ = [
    "ArtifactPage",
    "ArtifactsRequest",
    "CancelResult",
    "LogEntry",
    "LogPage",
    "ReconcileRequest",
    "RunCancelRequest",
    "RunListRequest",
    "RunLogsRequest",
    "RunPage",
    "RunVerification",
    "RunsAPI",
    "RunsOperations",
    "VerifyRequest",
]
