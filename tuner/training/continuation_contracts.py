"""Provider-neutral, structural contracts for future SFT continuation.

These checks do not authenticate a run or artifact. The host must obtain the
parent through RunsAPI verification and stream-integrity checks before using
one of these requests to stage a new, separately authorized training run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re

from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunsAPI
from synaptic_tuner.api.v1.training_input import TrainingContinuationInputV1


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_FULL_STATE_MEMBERS = frozenset({"model", "optimizer", "scheduler", "rng", "data_cursor"})


def _digest(value: str, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _steps(value: int, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum or value > 2**63 - 1:
        raise ValueError(f"{name} is invalid")
    return value


def _revision(value: str, name: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{name} must be an immutable 40- or 64-hex revision")
    return value


class ContinuationMode(str, Enum):
    FULL_STATE_RESUME = "full_state_resume"
    ADAPTER_WARM_START = "adapter_warm_start"


class SchedulePolicy(str, Enum):
    PRESERVE = "preserve"
    EXPLICIT_EXTENSION = "explicit_extension"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class ParentTrainingStateV1:
    """Claimed parent metadata; authenticity is established elsewhere."""

    run: TrainingRunRef
    artifact: VerifiedArtifact
    dataset_digest: str
    training_invariants_digest: str
    base_model_revision: str
    adapter_layout_digest: str
    runtime_digest: str
    completed_steps: int
    planned_total_steps: int
    state_member_digests: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.run) is not TrainingRunRef or type(self.artifact) is not VerifiedArtifact:
            raise TypeError("parent run and artifact must use exact public contracts")
        if self.artifact.role not in {"trainer_state", "final_model"}:
            raise ValueError("unsupported parent artifact role")
        for name in ("dataset_digest", "training_invariants_digest",
                     "adapter_layout_digest", "runtime_digest"):
            _digest(getattr(self, name), name)
        _revision(self.base_model_revision, "base_model_revision")
        _steps(self.completed_steps, "completed_steps")
        _steps(self.planned_total_steps, "planned_total_steps")
        if self.completed_steps > self.planned_total_steps:
            raise ValueError("completed steps exceed parent plan")
        if type(self.state_member_digests) is not tuple:
            raise TypeError("state member digests must be an exact tuple")
        names: list[str] = []
        for member in self.state_member_digests:
            if type(member) is not tuple or len(member) != 2 or type(member[0]) is not str:
                raise TypeError("state member entry is invalid")
            names.append(member[0])
            _digest(member[1], member[0])
        if names != sorted(set(names)) or set(names) - _FULL_STATE_MEMBERS:
            raise ValueError("state members must be sorted, unique and recognized")
        if self.artifact.role == "trainer_state" and set(names) != _FULL_STATE_MEMBERS:
            raise ValueError("trainer state requires model, optimizer, scheduler, RNG and cursor")
        if self.artifact.role == "final_model" and names:
            raise ValueError("final model is not a full trainer state")


@dataclass(frozen=True, slots=True)
class ContinuationRequestV1:
    """Structural continuation intent, not provider or replay authority."""

    mode: ContinuationMode
    parent: ParentTrainingStateV1
    child_run: TrainingRunRef
    child_dataset_digest: str
    child_training_invariants_digest: str
    child_base_model_revision: str
    child_adapter_layout_digest: str
    child_runtime_digest: str
    child_total_steps: int
    schedule_policy: SchedulePolicy
    optimizer_reset: bool
    schedule_transition_digest: str | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not ContinuationMode or type(self.parent) is not ParentTrainingStateV1:
            raise TypeError("continuation mode or parent is invalid")
        if type(self.child_run) is not TrainingRunRef:
            raise TypeError("child run must use the exact public contract")
        if self.child_run == self.parent.run:
            raise ValueError("continuation requires a distinct child run")
        if self.child_run.project_ref != self.parent.run.project_ref:
            raise ValueError("cross-project continuation requires separate policy")
        for name in ("child_dataset_digest", "child_training_invariants_digest",
                     "child_adapter_layout_digest", "child_runtime_digest"):
            _digest(getattr(self, name), name)
        _revision(self.child_base_model_revision, "child_base_model_revision")
        _steps(self.child_total_steps, "child_total_steps")
        if type(self.schedule_policy) is not SchedulePolicy or type(self.optimizer_reset) is not bool:
            raise TypeError("schedule policy and optimizer reset must be explicit")
        if self.schedule_transition_digest is not None:
            _digest(self.schedule_transition_digest, "schedule_transition_digest")
        if self.child_base_model_revision != self.parent.base_model_revision:
            raise ValueError("base model revision changed")
        if self.child_adapter_layout_digest != self.parent.adapter_layout_digest:
            raise ValueError("adapter layout changed")

        if self.mode is ContinuationMode.FULL_STATE_RESUME:
            if self.parent.artifact.role != "trainer_state" or self.optimizer_reset:
                raise ValueError("full-state resume requires trainer state and retained optimizer")
            if (self.child_dataset_digest != self.parent.dataset_digest
                    or self.child_training_invariants_digest != self.parent.training_invariants_digest
                    or self.child_runtime_digest != self.parent.runtime_digest):
                raise ValueError("full-state resume changed dataset or training identity")
            if self.child_total_steps <= self.parent.completed_steps:
                raise ValueError("resume has no remaining steps")
            if self.schedule_policy is SchedulePolicy.PRESERVE:
                if (self.child_total_steps != self.parent.planned_total_steps
                        or self.schedule_transition_digest is not None):
                    raise ValueError("preserved schedule must retain the planned horizon")
            elif self.schedule_policy is SchedulePolicy.EXPLICIT_EXTENSION:
                if (self.child_total_steps <= self.parent.planned_total_steps
                        or self.schedule_transition_digest is None):
                    raise ValueError("extended schedule needs a longer horizon and transition")
            else:
                raise ValueError("full-state resume cannot reset scheduler")
        else:
            if self.parent.artifact.role != "final_model" or not self.optimizer_reset:
                raise ValueError("adapter warm-start requires final adapter and optimizer reset")
            if (self.schedule_policy is not SchedulePolicy.RESET
                    or self.schedule_transition_digest is not None):
                raise ValueError("adapter warm-start must start a fresh schedule")


@dataclass(frozen=True, slots=True)
class ContinuationSourceReceiptV1:
    """Evidence of one complete public stream read, not child staging authority."""

    run: TrainingRunRef
    artifact: VerifiedArtifact
    checked_at: str


def verify_continuation_source(
    runs: RunsAPI, intent: TrainingContinuationInputV1, *, project_ref: str,
    maximum_bytes: int,
) -> ContinuationSourceReceiptV1:
    """Read a verified parent artifact through the existing integrity port.

    The future stager must authenticate and transfer the artifact again in the
    exact child attempt; this receipt never grants execution or replay.
    """
    if type(runs) is not RunsAPI or type(intent) is not TrainingContinuationInputV1:
        raise TypeError("exact RunsAPI and continuation intent required")
    if type(project_ref) is not str or project_ref != intent.parent_project_ref:
        raise ValueError("continuation parent differs from the child project")
    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 2**63 - 1:
        raise ValueError("continuation read bound is invalid")
    parent_run = TrainingRunRef(intent.parent_run_id, intent.parent_project_ref)
    artifact = VerifiedArtifact(intent.artifact_role, intent.artifact_sha256,
                                intent.artifact_size_bytes)
    if artifact.size_bytes > maximum_bytes:
        raise ValueError("continuation artifact exceeds its read bound")
    verification = runs.verify(parent_run)
    if not verification.verified:
        raise ValueError("parent run is not verified")
    stream = runs.artifacts(RunArtifactRequest(
        parent_run, artifact.role, maximum_bytes,
    ))
    if stream.artifact != artifact:
        raise ValueError("parent artifact descriptor changed")
    digest = hashlib.sha256()
    size = 0
    for chunk in stream.iter_bytes():
        if type(chunk) is not bytes or not chunk:
            raise ValueError("parent artifact stream is invalid")
        size += len(chunk)
        if size > artifact.size_bytes or size > maximum_bytes:
            raise ValueError("parent artifact stream exceeds its bound")
        digest.update(chunk)
    if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
        raise ValueError("parent artifact stream differs from its descriptor")
    return ContinuationSourceReceiptV1(parent_run, artifact,
                                       verification.checked_at)


__all__ = [
    "ContinuationMode", "ContinuationRequestV1", "ContinuationSourceReceiptV1",
    "ParentTrainingStateV1", "SchedulePolicy", "verify_continuation_source",
]
