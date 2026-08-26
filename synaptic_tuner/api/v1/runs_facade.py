"""Run observation and provider-staging verification facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from ._contract import exact_fields, required_text
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
class RunPage:
    outcomes: tuple[RunOutcome, ...]


@dataclass(frozen=True, slots=True)
class RunLogPage:
    run: TrainingRunRef
    entries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, TrainingRunRef):
            raise TypeError("run must be TrainingRunRef")
        object.__setattr__(self, "entries", tuple(required_text(v, "log_entry") for v in self.entries))


@dataclass(frozen=True, slots=True)
class RunVerification:
    run: TrainingRunRef
    verified: bool
    checked_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.run, TrainingRunRef) or not isinstance(self.verified, bool):
            raise TypeError("run/verified have invalid types")
        object.__setattr__(self, "checked_at", required_text(self.checked_at, "checked_at"))


class RunsOperations(Protocol):
    def list(self, project_ref: str) -> RunPage: ...
    def show(self, run: TrainingRunRef) -> RunOutcome: ...
    def outcome(self, run: TrainingRunRef) -> RunOutcome: ...
    def logs(self, run: TrainingRunRef) -> RunLogPage: ...
    def cancel(self, run: TrainingRunRef, reason: str) -> RunOutcome: ...
    def reconcile(self, run: TrainingRunRef) -> RunOutcome: ...
    def verify(self, run: TrainingRunRef) -> RunVerification: ...
    def reverify(self, run: TrainingRunRef) -> RunVerification: ...
    def artifacts(self, run: TrainingRunRef) -> tuple[VerifiedArtifact, ...]: ...


class RunsAPI:
    __slots__ = ("_operations",)
    def __init__(self, operations: RunsOperations) -> None: self._operations = operations
    def list(self, project_ref: str) -> RunPage: return self._operations.list(project_ref)
    def show(self, run: TrainingRunRef) -> RunOutcome: return self._operations.show(run)
    def outcome(self, run: TrainingRunRef) -> RunOutcome: return self._operations.outcome(run)
    def logs(self, run: TrainingRunRef) -> RunLogPage: return self._operations.logs(run)
    def cancel(self, run: TrainingRunRef, reason: str) -> RunOutcome: return self._operations.cancel(run, reason)
    def reconcile(self, run: TrainingRunRef) -> RunOutcome: return self._operations.reconcile(run)
    def verify(self, run: TrainingRunRef) -> RunVerification: return self._operations.verify(run)
    def reverify(self, run: TrainingRunRef) -> RunVerification: return self._operations.reverify(run)
    def artifacts(self, run: TrainingRunRef) -> tuple[VerifiedArtifact, ...]: return self._operations.artifacts(run)


__all__ = ["RunLogPage", "RunOutcome", "RunPage", "RunVerification", "RunsAPI", "RunsOperations"]
