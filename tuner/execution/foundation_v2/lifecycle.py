"""Effect-specific lifecycle ambiguity and legal outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .identities import EffectKind


class LifecyclePhaseV2(str, Enum):
    PLANNED = "planned"
    STAGING = "staging"
    STAGED = "staged"
    SUBMITTING = "submitting"
    SUBMISSION_AMBIGUOUS = "submission_ambiguous"
    QUEUED = "queued"
    RUNNING = "running"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLING = "cancelling"
    CANCEL_AMBIGUOUS = "cancel_ambiguous"
    CANCELLED = "cancelled"
    RECONCILE_REQUIRED = "reconcile_required"
    RECONCILING = "reconciling"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True, slots=True)
class LifecycleStateV2:
    phase: LifecyclePhaseV2
    ambiguous_effect_kind: EffectKind | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, LifecyclePhaseV2):
            raise TypeError("phase must be LifecyclePhaseV2")
        if self.phase in {
            LifecyclePhaseV2.SUBMISSION_AMBIGUOUS,
            LifecyclePhaseV2.CANCEL_AMBIGUOUS,
            LifecyclePhaseV2.RECONCILE_REQUIRED,
            LifecyclePhaseV2.RECONCILING,
        }:
            if not isinstance(self.ambiguous_effect_kind, EffectKind):
                raise ValueError("ambiguous lifecycle phases require an effect kind")
        elif self.ambiguous_effect_kind is not None:
            raise ValueError("non-ambiguous lifecycle phases cannot retain an effect kind")
        if self.phase is LifecyclePhaseV2.SUBMISSION_AMBIGUOUS and self.ambiguous_effect_kind is not EffectKind.SUBMIT:
            raise ValueError("submission ambiguity requires submit effect kind")
        if self.phase is LifecyclePhaseV2.CANCEL_AMBIGUOUS and self.ambiguous_effect_kind is not EffectKind.CANCEL:
            raise ValueError("cancel ambiguity requires cancel effect kind")


_ALLOWED = {
    LifecyclePhaseV2.PLANNED: {LifecyclePhaseV2.STAGING, LifecyclePhaseV2.FAILED},
    LifecyclePhaseV2.STAGING: {LifecyclePhaseV2.STAGED, LifecyclePhaseV2.RECONCILE_REQUIRED, LifecyclePhaseV2.FAILED},
    LifecyclePhaseV2.STAGED: {LifecyclePhaseV2.SUBMITTING, LifecyclePhaseV2.FAILED},
    LifecyclePhaseV2.SUBMITTING: {LifecyclePhaseV2.QUEUED, LifecyclePhaseV2.SUBMISSION_AMBIGUOUS, LifecyclePhaseV2.FAILED},
    LifecyclePhaseV2.SUBMISSION_AMBIGUOUS: {LifecyclePhaseV2.RECONCILE_REQUIRED},
    LifecyclePhaseV2.QUEUED: {LifecyclePhaseV2.RUNNING, LifecyclePhaseV2.CANCEL_REQUESTED, LifecyclePhaseV2.FAILED},
    LifecyclePhaseV2.RUNNING: {LifecyclePhaseV2.VERIFYING, LifecyclePhaseV2.CANCEL_REQUESTED, LifecyclePhaseV2.FAILED},
    LifecyclePhaseV2.VERIFYING: {LifecyclePhaseV2.SUCCEEDED, LifecyclePhaseV2.FAILED},
    LifecyclePhaseV2.CANCEL_REQUESTED: {LifecyclePhaseV2.CANCELLING},
    LifecyclePhaseV2.CANCELLING: {LifecyclePhaseV2.CANCELLED, LifecyclePhaseV2.CANCEL_AMBIGUOUS, LifecyclePhaseV2.FAILED},
    LifecyclePhaseV2.CANCEL_AMBIGUOUS: {LifecyclePhaseV2.RECONCILE_REQUIRED},
    LifecyclePhaseV2.RECONCILE_REQUIRED: {LifecyclePhaseV2.RECONCILING},
    LifecyclePhaseV2.RECONCILING: {
        LifecyclePhaseV2.STAGED, LifecyclePhaseV2.QUEUED,
        LifecyclePhaseV2.CANCELLED, LifecyclePhaseV2.RECONCILE_REQUIRED,
        LifecyclePhaseV2.CONTRADICTED, LifecyclePhaseV2.FAILED,
    },
    LifecyclePhaseV2.SUCCEEDED: set(), LifecyclePhaseV2.FAILED: set(),
    LifecyclePhaseV2.CANCELLED: set(), LifecyclePhaseV2.CONTRADICTED: set(),
}


def transition(current: LifecycleStateV2, target: LifecycleStateV2) -> LifecycleStateV2:
    if target.phase not in _ALLOWED[current.phase]:
        raise ValueError("illegal lifecycle transition")
    expected_ambiguity = {
        LifecyclePhaseV2.STAGING: EffectKind.STAGE,
        LifecyclePhaseV2.SUBMISSION_AMBIGUOUS: EffectKind.SUBMIT,
        LifecyclePhaseV2.CANCEL_AMBIGUOUS: EffectKind.CANCEL,
    }.get(current.phase)
    if target.phase is LifecyclePhaseV2.RECONCILE_REQUIRED and expected_ambiguity is not None and target.ambiguous_effect_kind is not expected_ambiguity:
        raise ValueError("ambiguity kind does not match the source phase")
    if current.ambiguous_effect_kind is not None and target.phase in {
        LifecyclePhaseV2.RECONCILE_REQUIRED, LifecyclePhaseV2.RECONCILING,
    } and target.ambiguous_effect_kind is not current.ambiguous_effect_kind:
        raise ValueError("reconciliation effect kind cannot change")
    if current.phase is LifecyclePhaseV2.RECONCILING:
        allowed_by_kind = {
            EffectKind.STAGE: {LifecyclePhaseV2.STAGED, LifecyclePhaseV2.RECONCILE_REQUIRED, LifecyclePhaseV2.CONTRADICTED, LifecyclePhaseV2.FAILED},
            EffectKind.SUBMIT: {LifecyclePhaseV2.QUEUED, LifecyclePhaseV2.RECONCILE_REQUIRED, LifecyclePhaseV2.CONTRADICTED, LifecyclePhaseV2.FAILED},
            EffectKind.CANCEL: {LifecyclePhaseV2.CANCELLED, LifecyclePhaseV2.RECONCILE_REQUIRED, LifecyclePhaseV2.CONTRADICTED, LifecyclePhaseV2.FAILED},
        }[current.ambiguous_effect_kind]
        if target.phase not in allowed_by_kind:
            raise ValueError("outcome is illegal for the ambiguous effect kind")
    return target
