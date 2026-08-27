"""Typed side-effect boundaries consumed by later coordinator composition."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1,
    ResolvedTrainingRequest,
    TrainingPlan,
)
from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.training_facade import TrainingPreflight, TrainingRequest
from tuner.execution.foundation_v2.authority import (
    AuthenticatedGrantV2,
    AuthenticatedReconciliationGrantV1,
)
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.repository import EffectRecordV2, ReconciliationOwnershipV2
from tuner.execution.foundation_v2.receipts import QuiescenceProofV2
from tuner.execution.foundation_v2.receipts import AuthenticatedInvalidEvidenceV2, AuthenticatedReceiptV2

from .model import (
    ArtifactManifestV1,
    AuthenticatedFoundationRecordAssessmentV1,
    AuthenticatedArtifactVerificationReceiptV1,
    AuthenticatedProviderRunObservationV1,
    ProviderRunReadRequestV1,
    ProviderExecutionBindingV1,
    WorkflowRecordV1,
)
from .coordinator import (
    CoordinatorTransitionV1,
    ExecutionGrantSlotV1,
    ReconciliationGrantSlotV1,
)
from .foundation import QuiescenceRecoveryRequestV1


class RequestLoaderPortV1(Protocol):
    def load(self, canonical_json: str) -> TrainingRequest: ...


class RequestResolutionPortV1(Protocol):
    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest: ...


class PlanningPortV1(Protocol):
    def describe(self, provider: ProviderRef) -> ProviderDescriptor: ...
    def context(
        self, resolved: ResolvedTrainingRequest, provider: ProviderRef
    ) -> ProviderPlanContextV1: ...
    def preflight(self, plan: TrainingPlan) -> TrainingPreflight: ...


class PlanningStorePortV1(Protocol):
    def put_plan_if_absent(self, plan: TrainingPlan) -> bool: ...
    def get_plan(self, plan_fingerprint: str) -> TrainingPlan | None: ...
    def put_context_if_absent(self, context: ProviderPlanContextV1) -> bool: ...
    def get_context(self, context_digest: str) -> ProviderPlanContextV1 | None: ...


class WorkflowStorePortV1(Protocol):
    def create(self, record: WorkflowRecordV1) -> bool: ...
    def get(self, run: TrainingRunRef) -> WorkflowRecordV1 | None: ...
    def get_by_plan(
        self, project_ref: str, plan_fingerprint: str
    ) -> WorkflowRecordV1 | None: ...
    def list(self, project_ref: str) -> tuple[WorkflowRecordV1, ...]: ...
    def compare_and_swap(
        self,
        expected: WorkflowRecordV1,
        replacement: WorkflowRecordV1,
        *,
        transition: CoordinatorTransitionV1,
    ) -> bool: ...


class PreparationStorePortV1(Protocol):
    def put_if_absent(self, preparation: CanonicalPreparationV2) -> bool: ...
    def get(self, preparation_digest: str) -> CanonicalPreparationV2 | None: ...


class ExecutionGrantStorePortV1(Protocol):
    def put_if_absent(
        self, slot: ExecutionGrantSlotV1, grant: AuthenticatedGrantV2,
        command_bytes: bytes,
    ) -> bool: ...
    def get(
        self, slot: ExecutionGrantSlotV1, command_bytes: bytes
    ) -> AuthenticatedGrantV2 | None: ...


class ReconciliationGrantStorePortV1(Protocol):
    def put_if_absent(
        self,
        slot: ReconciliationGrantSlotV1,
        grant: AuthenticatedReconciliationGrantV1,
        command_bytes: bytes,
        record: EffectRecordV2,
    ) -> bool: ...
    def get(
        self, slot: ReconciliationGrantSlotV1,
        *, command_bytes: bytes, record: EffectRecordV2,
    ) -> AuthenticatedReconciliationGrantV1 | None: ...


class ProviderBindingResolverPortV1(Protocol):
    def resolve(
        self, provider: ProviderRef, context: ProviderPlanContextV1
    ) -> ProviderExecutionBindingV1: ...


class PreparationMaterializerPortV1(Protocol):
    def prepare(
        self,
        plan: TrainingPlan,
        run: TrainingRunRef,
        binding: ProviderExecutionBindingV1,
    ) -> CanonicalPreparationV2: ...

    def payload(
        self, preparation: CanonicalPreparationV2, kind: EffectKind
    ) -> CanonicalProviderPayloadV1: ...


class EffectFoundationPortV1(Protocol):
    def get(self, effect_id: str) -> EffectRecordV2 | None: ...
    def execute(
        self, command_bytes: bytes, grant: AuthenticatedGrantV2, *, now_epoch: int
    ) -> EffectRecordV2: ...
    def reconcile(
        self,
        command_bytes: bytes,
        grant: AuthenticatedReconciliationGrantV1,
        *,
        now_epoch: int,
        continuation: ReconciliationOwnershipV2 | None = None,
    ) -> EffectRecordV2: ...
    def recover_orphan(self, effect_id: str, *, now_epoch: int) -> EffectRecordV2: ...


class TrustedQuiescenceEvidencePortV1(Protocol):
    def obtain(
        self, request: QuiescenceRecoveryRequestV1, *, now_epoch: int
    ) -> QuiescenceProofV2: ...


class AuthorizationPortV1(Protocol):
    def commit_preflight(self, plan: TrainingPlan, preflight: TrainingPreflight) -> str: ...
    def issue_effect_grant(
        self,
        command_bytes: bytes,
        *,
        preflight_digest: str,
        now_epoch: int,
    ) -> AuthenticatedGrantV2: ...
    def issue_reconciliation_grant(
        self,
        record: EffectRecordV2,
        binding: ProviderExecutionBindingV1,
        *,
        slot: ReconciliationGrantSlotV1,
        now_epoch: int,
    ) -> AuthenticatedReconciliationGrantV1: ...


class ProviderRunReaderPortV1(Protocol):
    """Trusted reader boundary.

    Implementations must authenticate and validate the request's complete Foundation
    record and assessment before performing any provider or external I/O. Raw provider
    references and compact bound-run values are never authorization.
    """
    def observe(self, request: ProviderRunReadRequestV1) -> AuthenticatedProviderRunObservationV1: ...
    def logs(self, request: ProviderRunReadRequestV1, *, cursor: str | None) -> tuple[str, ...]: ...
    def artifacts(self, request: ProviderRunReadRequestV1) -> ArtifactManifestV1: ...
    def iter_artifact_bytes(
        self, request: ProviderRunReadRequestV1, manifest: ArtifactManifestV1,
        role: str, *, maximum_bytes: int
    ) -> Iterator[bytes]: ...


class FoundationEvidenceAuthenticatorPortV1(Protocol):
    def authenticate_grant(
        self, grant: AuthenticatedGrantV2, command_bytes: bytes
    ) -> bool: ...
    def authenticate_receipt(self, receipt: AuthenticatedReceiptV2) -> bool: ...
    def authenticate_invalid_evidence(self, evidence: AuthenticatedInvalidEvidenceV2) -> bool: ...

class FoundationRecordAssessmentPortV1(Protocol):
    def assess(self, record: EffectRecordV2) -> AuthenticatedFoundationRecordAssessmentV1: ...
    def authenticate(self, assessment: AuthenticatedFoundationRecordAssessmentV1) -> bool: ...

class ProviderObservationAuthenticatorPortV1(Protocol):
    def authenticate(self, observation: AuthenticatedProviderRunObservationV1) -> bool: ...


class ArtifactVerifierPortV1(Protocol):
    def verify(self, workflow: WorkflowRecordV1, manifest: ArtifactManifestV1) -> AuthenticatedArtifactVerificationReceiptV1: ...
    def replay(self, workflow: WorkflowRecordV1, manifest: ArtifactManifestV1, prior_receipt: AuthenticatedArtifactVerificationReceiptV1) -> AuthenticatedArtifactVerificationReceiptV1: ...
    def authenticate(self, receipt: AuthenticatedArtifactVerificationReceiptV1) -> bool: ...


class CoordinatorClockPortV1(Protocol):
    def now_epoch(self) -> int: ...
    def now_iso(self) -> str: ...


class RunIdentityPortV1(Protocol):
    def for_plan(self, plan: TrainingPlan) -> TrainingRunRef: ...


__all__ = [
    "ArtifactVerifierPortV1",
    "AuthorizationPortV1",
    "CoordinatorClockPortV1",
    "EffectFoundationPortV1",
    "ExecutionGrantStorePortV1",
    "FoundationEvidenceAuthenticatorPortV1",
    "FoundationRecordAssessmentPortV1",
    "PlanningPortV1",
    "PlanningStorePortV1",
    "PreparationMaterializerPortV1",
    "PreparationStorePortV1",
    "ProviderBindingResolverPortV1",
    "ProviderRunReaderPortV1",
    "ReconciliationGrantStorePortV1",
    "ProviderObservationAuthenticatorPortV1",
    "RequestLoaderPortV1",
    "RequestResolutionPortV1",
    "RunIdentityPortV1",
    "TrustedQuiescenceEvidencePortV1",
    "WorkflowStorePortV1",
]
