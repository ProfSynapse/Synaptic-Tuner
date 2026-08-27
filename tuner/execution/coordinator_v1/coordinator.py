"""Non-authoritative coordinator transition and durable-slot descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from tuner.execution.foundation_v2.canonical import digest_text, exact_integer, safe_ref
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.repository import EffectRecordV2

from .model import AuthenticatedFoundationRecordAssessmentV1, EffectIntentV1


class CoordinatorTransitionKindV1(str, Enum):
    BEGIN_PREPARATION = "begin_preparation"
    RECORD_STAGE_INTENT = "record_stage_intent"
    APPLY_STAGE_EFFECT = "apply_stage_effect"
    RECORD_SUBMIT_INTENT = "record_submit_intent"
    APPLY_SUBMIT_EFFECT = "apply_submit_effect"


@dataclass(frozen=True, slots=True)
class BeginPreparationTransitionV1:
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.BEGIN_PREPARATION

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.BEGIN_PREPARATION:
            raise ValueError("invalid begin-preparation transition")


@dataclass(frozen=True, slots=True)
class RecordStageIntentTransitionV1:
    preparation: CanonicalPreparationV2
    intent: EffectIntentV1
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.RECORD_STAGE_INTENT

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.RECORD_STAGE_INTENT:
            raise ValueError("invalid stage-intent transition")
        if type(self.preparation) is not CanonicalPreparationV2:
            raise TypeError("exact preparation required")
        if type(self.intent) is not EffectIntentV1:
            raise TypeError("exact effect intent required")


@dataclass(frozen=True, slots=True)
class ApplyStageEffectTransitionV1:
    record: EffectRecordV2
    assessment: AuthenticatedFoundationRecordAssessmentV1
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.APPLY_STAGE_EFFECT

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.APPLY_STAGE_EFFECT:
            raise ValueError("invalid stage-effect transition")
        if type(self.record) is not EffectRecordV2:
            raise TypeError("exact Foundation record required")
        if type(self.assessment) is not AuthenticatedFoundationRecordAssessmentV1:
            raise TypeError("exact Foundation assessment required")


@dataclass(frozen=True, slots=True)
class RecordSubmitIntentTransitionV1:
    intent: EffectIntentV1
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.RECORD_SUBMIT_INTENT

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.RECORD_SUBMIT_INTENT:
            raise ValueError("invalid submit-intent transition")
        if type(self.intent) is not EffectIntentV1:
            raise TypeError("exact effect intent required")


@dataclass(frozen=True, slots=True)
class ApplySubmitEffectTransitionV1:
    record: EffectRecordV2
    assessment: AuthenticatedFoundationRecordAssessmentV1
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.APPLY_SUBMIT_EFFECT

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.APPLY_SUBMIT_EFFECT:
            raise ValueError("invalid submit-effect transition")
        if type(self.record) is not EffectRecordV2:
            raise TypeError("exact Foundation record required")
        if type(self.assessment) is not AuthenticatedFoundationRecordAssessmentV1:
            raise TypeError("exact Foundation assessment required")


CoordinatorTransitionV1: TypeAlias = (
    BeginPreparationTransitionV1
    | RecordStageIntentTransitionV1
    | ApplyStageEffectTransitionV1
    | RecordSubmitIntentTransitionV1
    | ApplySubmitEffectTransitionV1
)


@dataclass(frozen=True, slots=True)
class ExecutionGrantSlotV1:
    effect_id: str
    command_digest: str
    command_bytes_digest: str

    def __post_init__(self) -> None:
        safe_ref(self.effect_id, "effect_id")
        digest_text(self.command_digest, "command_digest")
        digest_text(self.command_bytes_digest, "command_bytes_digest")


@dataclass(frozen=True, slots=True)
class ReconciliationGrantSlotV1:
    effect_id: str
    command_digest: str
    command_bytes_digest: str
    generation: int
    ownership_epoch: int
    prior_claim_digest: str | None
    predecessor_grant_digest: str | None

    def __post_init__(self) -> None:
        safe_ref(self.effect_id, "effect_id")
        digest_text(self.command_digest, "command_digest")
        digest_text(self.command_bytes_digest, "command_bytes_digest")
        exact_integer(self.generation, "generation", minimum=1)
        exact_integer(self.ownership_epoch, "ownership_epoch", minimum=1)
        if (self.prior_claim_digest is None) != (self.predecessor_grant_digest is None):
            raise ValueError("reconciliation lineage must be paired")
        if self.prior_claim_digest is not None:
            digest_text(self.prior_claim_digest, "prior_claim_digest")
            digest_text(self.predecessor_grant_digest, "predecessor_grant_digest")


__all__ = [
    "ApplyStageEffectTransitionV1",
    "ApplySubmitEffectTransitionV1",
    "BeginPreparationTransitionV1",
    "CoordinatorTransitionKindV1",
    "CoordinatorTransitionV1",
    "ExecutionGrantSlotV1",
    "ReconciliationGrantSlotV1",
    "RecordStageIntentTransitionV1",
    "RecordSubmitIntentTransitionV1",
]
