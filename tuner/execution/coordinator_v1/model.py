"""Derivation-owned provider-neutral records for the training coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunLogEntry
from synaptic_tuner.api.v1._timestamps import require_rfc3339
from tuner.execution.foundation_v2.authority import AuthenticatedGrantV2
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, digest_text, domain_digest, exact_integer,
    parse_canonical_object, safe_ref,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.executors import ExecutorDescriptorV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.receipts import AuthenticatedReceiptV2
from tuner.execution.foundation_v2.repository import DispatchState, EffectRecordV2, EffectState
from tuner.execution.foundation_v2.references import (
    CancellationRefV1, ExecutionScopeV1, ProviderStageRefV1, ScopedProviderRunRefV1,
)


class ProviderRunPhaseV1(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProviderReadPurposeV1(str, Enum):
    OBSERVE = "observe"
    LOGS = "logs"
    ARTIFACTS = "artifacts"


class WorkflowPhaseV1(str, Enum):
    PLANNED = "planned"
    PREPARING = "preparing"
    STAGE_INTENT_RECORDED = "stage_intent_recorded"
    STAGED = "staged"
    STAGE_RECONCILE_REQUIRED = "stage_reconcile_required"
    SUBMIT_INTENT_RECORDED = "submit_intent_recorded"
    QUEUED = "queued"
    SUBMIT_RECONCILE_REQUIRED = "submit_reconcile_required"
    RUNNING = "running"
    CANCEL_INTENT_RECORDED = "cancel_intent_recorded"
    CANCEL_REQUESTED = "cancel_requested"
    CANCEL_RECONCILE_REQUIRED = "cancel_reconcile_required"
    SUCCEEDED_UNVERIFIED = "succeeded_unverified"
    VERIFICATION_FAILED = "verification_failed"
    VERIFIED = "verified"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True, slots=True)
class ProviderExecutionBindingV1:
    provider: ProviderRef
    provider_descriptor_digest: str
    profile_digest: str
    scope: ExecutionScopeV1
    executor_descriptor: ExecutorDescriptorV1
    reconciliation_adapter_digest: str
    resource_digest: str
    quote_digest: str
    secret_requirements_digest: str

    def __post_init__(self) -> None:
        if type(self.provider) is not ProviderRef:
            raise TypeError("provider must be an exact ProviderRef")
        if type(self.scope) is not ExecutionScopeV1:
            raise TypeError("scope must be an exact ExecutionScopeV1")
        if type(self.executor_descriptor) is not ExecutorDescriptorV1:
            raise TypeError("executor_descriptor must be exact")
        if self.executor_descriptor.provider_id != self.provider.provider_id:
            raise ValueError("executor descriptor provider does not match binding")
        for name in (
            "provider_descriptor_digest", "profile_digest",
            "reconciliation_adapter_digest", "resource_digest", "quote_digest",
            "secret_requirements_digest",
        ):
            digest_text(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider.to_dict(),
            "provider_descriptor_digest": self.provider_descriptor_digest,
            "profile_digest": self.profile_digest,
            "scope": self.scope.to_dict(),
            "executor_descriptor": self.executor_descriptor.to_dict(),
            "reconciliation_adapter_digest": self.reconciliation_adapter_digest,
            "resource_digest": self.resource_digest,
            "quote_digest": self.quote_digest,
            "secret_requirements_digest": self.secret_requirements_digest,
        }

    @property
    def binding_digest(self) -> str:
        return domain_digest(
            "synaptic-provider-execution-binding/v1", canonical_bytes(self.to_dict())
        )


class FoundationDispositionV1(str, Enum):
    FOUND = "found"
    DEFINITELY_ABSENT = "definitely_absent"
    INDETERMINATE = "indeterminate"
    CONTRADICTED = "contradicted"

class ReceiptFreshnessV1(str, Enum):
    FRESH = "fresh"
    STALE = "stale"

@dataclass(frozen=True, slots=True)
class ReceiptAssessmentV1:
    authenticated_receipt_digest: str
    receipt_admission_digest: str
    source_kind: str
    source_owner_ref: str
    source_generation: int
    source_ownership_epoch: int
    freshness: ReceiptFreshnessV1
    finality_verified: bool
    generated_invalid_codes: tuple[str, ...]
    def __post_init__(self):
        digest_text(self.authenticated_receipt_digest,"authenticated_receipt_digest");digest_text(self.receipt_admission_digest,"receipt_admission_digest");safe_ref(self.source_owner_ref,"source_owner_ref")
        exact_integer(self.source_generation,"source_generation",minimum=1);exact_integer(self.source_ownership_epoch,"source_ownership_epoch",minimum=1)
        if self.source_kind not in {"dispatch","reconciliation"} or type(self.freshness) is not ReceiptFreshnessV1 or type(self.finality_verified) is not bool or type(self.generated_invalid_codes) is not tuple:raise ValueError("receipt assessment fields invalid")
        expected=(() if self.freshness is ReceiptFreshnessV1.FRESH else ("stale_result",))
        if self.finality_verified and "finality_unproven" in self.generated_invalid_codes:raise ValueError("verified finality cannot be unproven")
        if tuple(x for x in self.generated_invalid_codes if x=="stale_result")!=expected:raise ValueError("freshness invalid-code mismatch")
    def to_dict(self):return {"authenticated_receipt_digest":self.authenticated_receipt_digest,"receipt_admission_digest":self.receipt_admission_digest,"source_kind":self.source_kind,"source_owner_ref":self.source_owner_ref,"source_generation":self.source_generation,"source_ownership_epoch":self.source_ownership_epoch,"freshness":self.freshness.value,"finality_verified":self.finality_verified,"generated_invalid_codes":list(self.generated_invalid_codes)}

@dataclass(frozen=True, slots=True)
class FoundationRecordAssessmentContentV1:
    schema_version: str
    effect_id: str
    command_digest: str
    command_bytes_digest: str
    foundation_record_digest: str
    record_evidence_digest: str
    authenticated_receipt_digests: tuple[str, ...]
    terminal_content_digests: tuple[str, ...]
    receipt_assessments: tuple[ReceiptAssessmentV1, ...]
    invalid_evidence_admission_digests: tuple[str, ...]
    invalid_codes: tuple[str, ...]
    assessor_ref: str
    assessor_version: str
    assessed_at: str
    def __post_init__(self):
        if self.schema_version!="synaptic-foundation-record-assessment-content/v1":raise ValueError("assessment schema invalid")
        safe_ref(self.effect_id,"effect_id");safe_ref(self.assessor_ref,"assessor_ref");safe_ref(self.assessor_version,"assessor_version");safe_ref(self.assessed_at,"assessed_at")
        for n in ("command_digest","command_bytes_digest","foundation_record_digest","record_evidence_digest"):digest_text(getattr(self,n),n)
        if any(type(x) is not ReceiptAssessmentV1 for x in self.receipt_assessments):raise TypeError("receipt assessments must be exact")
        if tuple(x.authenticated_receipt_digest for x in self.receipt_assessments)!=self.authenticated_receipt_digests:raise ValueError("assessment order differs from record receipts")
        if type(self.invalid_evidence_admission_digests) is not tuple:raise TypeError("invalid evidence admission history must be tuple")
        for value in self.invalid_evidence_admission_digests:digest_text(value,"invalid_evidence_admission_digest")
    def to_dict(self):return {"schema_version":self.schema_version,"effect_id":self.effect_id,"command_digest":self.command_digest,"command_bytes_digest":self.command_bytes_digest,"foundation_record_digest":self.foundation_record_digest,"record_evidence_digest":self.record_evidence_digest,"authenticated_receipt_digests":list(self.authenticated_receipt_digests),"terminal_content_digests":list(self.terminal_content_digests),"receipt_assessments":[x.to_dict() for x in self.receipt_assessments],"invalid_evidence_admission_digests":list(self.invalid_evidence_admission_digests),"invalid_codes":list(self.invalid_codes),"assessor_ref":self.assessor_ref,"assessor_version":self.assessor_version,"assessed_at":self.assessed_at}
    @property
    def content_digest(self):return domain_digest("synaptic-foundation-record-assessment-content/v1",canonical_bytes(self.to_dict()))

@dataclass(frozen=True, slots=True)
class AuthenticatedFoundationRecordAssessmentV1:
    content: FoundationRecordAssessmentContentV1
    authority_ref: str
    key_ref: str
    tag: str
    def __post_init__(self):
        if type(self.content) is not FoundationRecordAssessmentContentV1:raise TypeError("exact assessment content required")
        safe_ref(self.authority_ref,"authority_ref");safe_ref(self.key_ref,"key_ref");digest_text(self.tag,"tag")
    @property
    def canonical_bytes(self):return canonical_bytes({"schema_version":"synaptic-authenticated-foundation-record-assessment/v1","content":{**self.content.to_dict(),"content_digest":self.content.content_digest},"authority_ref":self.authority_ref,"key_ref":self.key_ref,"tag":self.tag})
    @property
    def authenticated_assessment_digest(self):return domain_digest("synaptic-authenticated-foundation-record-assessment/v1",self.canonical_bytes)
    @classmethod
    def parse(cls,raw):
        doc=parse_canonical_object(raw,name="foundation assessment");expected={"schema_version","content","authority_ref","key_ref","tag"}
        if set(doc)!=expected or doc["schema_version"]!="synaptic-authenticated-foundation-record-assessment/v1" or not isinstance(doc["content"],dict):raise ValueError("assessment envelope invalid")
        content=dict(doc["content"]);claimed=content.pop("content_digest");assessments=[]
        for value in content["receipt_assessments"]:
            if not isinstance(value,dict) or set(value)!={"authenticated_receipt_digest","receipt_admission_digest","source_kind","source_owner_ref","source_generation","source_ownership_epoch","freshness","finality_verified","generated_invalid_codes"}:raise ValueError("receipt assessment invalid")
            assessments.append(ReceiptAssessmentV1(value["authenticated_receipt_digest"],value["receipt_admission_digest"],value["source_kind"],value["source_owner_ref"],value["source_generation"],value["source_ownership_epoch"],ReceiptFreshnessV1(value["freshness"]),value["finality_verified"],tuple(value["generated_invalid_codes"])))
        content["receipt_assessments"]=tuple(assessments);content["authenticated_receipt_digests"]=tuple(content["authenticated_receipt_digests"]);content["terminal_content_digests"]=tuple(content["terminal_content_digests"]);content["invalid_evidence_admission_digests"]=tuple(content["invalid_evidence_admission_digests"]);content["invalid_codes"]=tuple(content["invalid_codes"])
        owned_content=FoundationRecordAssessmentContentV1(**content)
        if claimed!=owned_content.content_digest:raise ValueError("assessment content digest mismatch")
        owned=cls(owned_content,doc["authority_ref"],doc["key_ref"],doc["tag"])
        if owned.canonical_bytes!=raw:raise ValueError("assessment bytes are not canonical")
        return owned


@dataclass(frozen=True, slots=True)
class FoundationEffectBindingV1:
    kind: EffectKind
    effect_id: str
    command_digest: str
    command_bytes_digest: str
    preparation_digest: str
    grant_digest: str
    foundation_record_digest: str
    canonical_snapshot_bytes: bytes
    snapshot_digest: str
    attempt_count: int
    dispatch_epoch: int
    dispatch_state: DispatchState
    effect_state: EffectState
    authenticated_receipt_digests: tuple[str, ...]
    terminal_content_digests: tuple[str, ...]
    invalid_codes: tuple[str, ...]
    reconciliation_claim_digests: tuple[str, ...]
    active_reconciliation_claim_digest: str | None
    canonical_assessment_bytes: bytes
    assessment_digest: str
    binding_digest: str

    def __post_init__(self):
        if type(self.kind) is not EffectKind or type(self.dispatch_state) is not DispatchState or type(self.effect_state) is not EffectState:
            raise TypeError("foundation binding enums must be exact")
        for name in ("command_digest","command_bytes_digest","preparation_digest","grant_digest","foundation_record_digest","snapshot_digest","assessment_digest","binding_digest"):
            digest_text(getattr(self,name),name)
        exact_integer(self.attempt_count,"attempt_count",minimum=1); exact_integer(self.dispatch_epoch,"dispatch_epoch",minimum=1)
        snapshot=parse_canonical_object(self.canonical_snapshot_bytes,name="foundation snapshot")
        if canonical_bytes(snapshot)!=self.canonical_snapshot_bytes or domain_digest("synaptic-foundation-effect-snapshot/v1",self.canonical_snapshot_bytes)!=self.snapshot_digest:
            raise ValueError("foundation snapshot identity invalid")
        parsed_assessment=AuthenticatedFoundationRecordAssessmentV1.parse(self.canonical_assessment_bytes)
        if parsed_assessment.authenticated_assessment_digest!=self.assessment_digest:raise ValueError("foundation assessment identity mismatch")
        document={"kind":self.kind.value,"effect_id":self.effect_id,"command_digest":self.command_digest,"command_bytes_digest":self.command_bytes_digest,"preparation_digest":self.preparation_digest,"grant_digest":self.grant_digest,"foundation_record_digest":self.foundation_record_digest,"snapshot_digest":self.snapshot_digest,"assessment_digest":self.assessment_digest}
        if self.binding_digest!=domain_digest("synaptic-foundation-binding/v1",canonical_bytes(document)): raise ValueError("foundation binding digest mismatch")
        for values,name in ((self.authenticated_receipt_digests,"receipt"),(self.terminal_content_digests,"terminal"),(self.invalid_codes,"invalid"),(self.reconciliation_claim_digests,"claim")):
            if type(values) is not tuple: raise TypeError(f"{name} history must be tuple")

def _fresh_found_reference(binding,digest,kind):
    try:index=binding.authenticated_receipt_digests.index(digest)
    except ValueError:return None
    assessment=AuthenticatedFoundationRecordAssessmentV1.parse(binding.canonical_assessment_bytes)
    if index>=len(assessment.content.receipt_assessments) or assessment.content.receipt_assessments[index].freshness is not ReceiptFreshnessV1.FRESH:return None
    snapshot=parse_canonical_object(binding.canonical_snapshot_bytes,name="foundation snapshot")
    receipts=snapshot.get("receipts")
    if type(receipts) is not list or index>=len(receipts):return None
    receipt=AuthenticatedReceiptV2.parse(canonical_bytes(receipts[index]))
    if receipt.authenticated_receipt_digest!=digest or receipt.content.disposition is not ObservationDisposition.FOUND:return None
    return {EffectKind.STAGE:receipt.content.stage_ref,EffectKind.SUBMIT:receipt.content.provider_run,EffectKind.CANCEL:receipt.content.cancellation}[kind]

@dataclass(frozen=True, slots=True)
class FoundationEffectOutcomeV1:
    binding_digest: str
    kind: EffectKind
    effect_id: str
    command_digest: str
    preparation_digest: str
    foundation_record_digest: str
    disposition: FoundationDispositionV1
    authenticated_receipt_digests: tuple[str, ...]
    receipt_content_digests: tuple[str, ...]
    observation_digests: tuple[str, ...]
    stage_reference: ProviderStageRefV1 | None
    run_reference: ScopedProviderRunRefV1 | None
    cancellation_reference: CancellationRefV1 | None
    finality_proof_digest: str | None
    outcome_digest: str

    def __post_init__(self):
        if type(self.kind) is not EffectKind or type(self.disposition) is not FoundationDispositionV1: raise TypeError("foundation outcome enums must be exact")
        for name in ("binding_digest","command_digest","preparation_digest","foundation_record_digest","outcome_digest"): digest_text(getattr(self,name),name)
        document={"binding_digest":self.binding_digest,"disposition":self.disposition.value,"receipts":list(self.authenticated_receipt_digests),"stage":None if self.stage_reference is None else self.stage_reference.to_dict(),"run":None if self.run_reference is None else self.run_reference.to_dict(),"cancel":None if self.cancellation_reference is None else {"run":self.cancellation_reference.run.to_dict(),"reason_digest":self.cancellation_reference.reason_digest},"finality":self.finality_proof_digest}
        if self.outcome_digest!=domain_digest("synaptic-foundation-outcome/v1",canonical_bytes(document)): raise ValueError("foundation outcome digest mismatch")

@dataclass(frozen=True, slots=True)
class EffectIntentV1:
    kind: EffectKind
    effect_id: str
    command_digest: str
    canonical_command_bytes: bytes
    foundation_bindings: tuple[FoundationEffectBindingV1, ...]
    foundation_outcomes: tuple[FoundationEffectOutcomeV1, ...]

    def __post_init__(self): self._validate()

    def _validate(self) -> None:
        if type(self.kind) is not EffectKind:
            raise TypeError("kind must be exact EffectKind")
        safe_ref(self.effect_id, "effect_id")
        digest_text(self.command_digest, "command_digest")
        if type(self.canonical_command_bytes) is not bytes:
            raise TypeError("canonical_command_bytes must be exact bytes")
        command = parse_exact_command(self.canonical_command_bytes)
        effect = command.operation.effect
        if (effect.kind, effect.effect_id, command.digest) != (
            self.kind, self.effect_id, self.command_digest,
        ):
            raise ValueError("effect intent does not bind exact command")
        if any(type(value) is not FoundationEffectBindingV1 for value in self.foundation_bindings):
            raise TypeError("foundation bindings must be exact")
        if any(type(value) is not FoundationEffectOutcomeV1 for value in self.foundation_outcomes):
            raise TypeError("foundation outcomes must be exact")
        if len(self.foundation_outcomes) > len(self.foundation_bindings):
            raise ValueError("foundation outcome requires binding")
        preparation_digest = command.preparation.preparation_digest
        for binding in self.foundation_bindings:
            if (binding.kind, binding.effect_id, binding.command_digest,
                    binding.preparation_digest) != (
                    self.kind, self.effect_id, self.command_digest,
                    preparation_digest,
            ):
                raise ValueError("foundation binding does not retain intent identity")
            snapshot = parse_canonical_object(
                binding.canonical_snapshot_bytes, name="foundation snapshot"
            )
            if (canonical_bytes(snapshot) != binding.canonical_snapshot_bytes
                    or domain_digest("synaptic-foundation-effect-snapshot/v1",
                                     binding.canonical_snapshot_bytes)
                    != binding.snapshot_digest):
                raise ValueError("foundation snapshot digest mismatch")
        binding_positions = {value.binding_digest: index
                             for index, value in enumerate(self.foundation_bindings)}
        positions = []
        for outcome in self.foundation_outcomes:
            if outcome.binding_digest not in binding_positions:
                raise ValueError("foundation outcome references unknown binding")
            positions.append(binding_positions[outcome.binding_digest])
            if (outcome.kind, outcome.effect_id, outcome.command_digest,
                    outcome.preparation_digest) != (
                    self.kind, self.effect_id, self.command_digest,
                    preparation_digest,
            ):
                raise ValueError("foundation outcome does not retain intent identity")
        if positions != sorted(positions) or len(positions) != len(set(positions)):
            raise ValueError("foundation outcomes are not an ordered binding history")

    @classmethod
    def from_command_bytes(cls, canonical_command_bytes: bytes) -> "EffectIntentV1":
        if type(canonical_command_bytes) is not bytes:
            raise TypeError("canonical_command_bytes must be exact bytes")
        command = parse_exact_command(canonical_command_bytes)
        effect = command.operation.effect
        return cls(effect.kind,effect.effect_id,command.digest,command.canonical_bytes,(),())

    @property
    def advanced(self) -> bool:
        return bool(self.foundation_bindings)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value, "effect_id": self.effect_id,
            "command_digest": self.command_digest,
            "foundation_binding_digests": [x.binding_digest for x in self.foundation_bindings],
            "foundation_outcome_digests": [x.outcome_digest for x in self.foundation_outcomes],
        }

    @property
    def intent_digest(self) -> str:
        return domain_digest("synaptic-effect-intent/v1", canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class BoundProviderStageRefV1:
    reference: ProviderStageRefV1
    effect_id: str
    command_digest: str
    command_bytes_digest: str
    preparation_digest: str
    foundation_binding_digest: str
    foundation_outcome_digest: str
    authenticated_receipt_digest: str
    binding_digest: str

    def __post_init__(self):
        if type(self.reference) is not ProviderStageRefV1: raise TypeError("exact stage reference required")
        for name in ("command_digest","command_bytes_digest","preparation_digest","foundation_binding_digest","foundation_outcome_digest","authenticated_receipt_digest","binding_digest"): digest_text(getattr(self,name),name)
        doc={"reference":self.reference.to_dict(),"effect_id":self.effect_id,"command_digest":self.command_digest,"command_bytes_digest":self.command_bytes_digest,"preparation_digest":self.preparation_digest,"foundation_binding_digest":self.foundation_binding_digest,"foundation_outcome_digest":self.foundation_outcome_digest,"authenticated_receipt_digest":self.authenticated_receipt_digest}
        if self.binding_digest!=domain_digest("synaptic-stage-evidence-binding/v1",canonical_bytes(doc)): raise ValueError("stage evidence binding digest mismatch")


@dataclass(frozen=True, slots=True)
class BoundProviderRunRefV1:
    reference: ScopedProviderRunRefV1
    effect_id: str
    command_digest: str
    command_bytes_digest: str
    preparation_digest: str
    foundation_binding_digest: str
    foundation_outcome_digest: str
    authenticated_receipt_digest: str
    binding_digest: str

    def __post_init__(self):
        if type(self.reference) is not ScopedProviderRunRefV1: raise TypeError("exact run reference required")
        for name in ("command_digest","command_bytes_digest","preparation_digest","foundation_binding_digest","foundation_outcome_digest","authenticated_receipt_digest","binding_digest"): digest_text(getattr(self,name),name)
        doc={"reference":self.reference.to_dict(),"effect_id":self.effect_id,"command_digest":self.command_digest,"command_bytes_digest":self.command_bytes_digest,"preparation_digest":self.preparation_digest,"foundation_binding_digest":self.foundation_binding_digest,"foundation_outcome_digest":self.foundation_outcome_digest,"authenticated_receipt_digest":self.authenticated_receipt_digest}
        if self.binding_digest!=domain_digest("synaptic-submit-evidence-binding/v1",canonical_bytes(doc)): raise ValueError("run evidence binding digest mismatch")


@dataclass(frozen=True, slots=True)
class BoundCancellationRefV1:
    reference: CancellationRefV1
    effect_id: str
    command_digest: str
    command_bytes_digest: str
    preparation_digest: str
    foundation_binding_digest: str
    foundation_outcome_digest: str
    authenticated_receipt_digest: str
    target_run_binding_digest: str
    binding_digest: str

    def __post_init__(self):
        if type(self.reference) is not CancellationRefV1: raise TypeError("exact cancellation reference required")
        for name in ("command_digest","command_bytes_digest","preparation_digest","foundation_binding_digest","foundation_outcome_digest","authenticated_receipt_digest","target_run_binding_digest","binding_digest"): digest_text(getattr(self,name),name)
        doc={"reference":{"run":self.reference.run.to_dict(),"reason_digest":self.reference.reason_digest},"effect_id":self.effect_id,"command_digest":self.command_digest,"command_bytes_digest":self.command_bytes_digest,"preparation_digest":self.preparation_digest,"foundation_binding_digest":self.foundation_binding_digest,"foundation_outcome_digest":self.foundation_outcome_digest,"authenticated_receipt_digest":self.authenticated_receipt_digest,"target_run_binding_digest":self.target_run_binding_digest}
        if self.binding_digest!=domain_digest("synaptic-cancel-evidence-binding/v1",canonical_bytes(doc)): raise ValueError("cancel evidence binding digest mismatch")


@dataclass(frozen=True, slots=True)
class ProviderRunReadRequestV1:
    purpose: ProviderReadPurposeV1
    source_workflow_record_digest: str
    source_revision: int
    run: TrainingRunRef
    provider_run: BoundProviderRunRefV1
    submit_command_bytes: bytes
    foundation_record: EffectRecordV2
    assessment: AuthenticatedFoundationRecordAssessmentV1
    foundation_binding: FoundationEffectBindingV1
    foundation_outcome: FoundationEffectOutcomeV1
    found_receipt_digest: str
    canonical_bytes: bytes
    request_digest: str
    def __post_init__(self):
        if type(self.purpose) is not ProviderReadPurposeV1:
            raise TypeError("exact provider read purpose required")
        digest_text(self.source_workflow_record_digest,"source_workflow_record_digest");exact_integer(self.source_revision,"source_revision")
        if type(self.run) is not TrainingRunRef or type(self.provider_run) is not BoundProviderRunRefV1 or type(self.foundation_record) is not EffectRecordV2 or type(self.assessment) is not AuthenticatedFoundationRecordAssessmentV1:raise TypeError("provider read request types invalid")
        doc={"schema_version":"synaptic-provider-run-read-request/v1","purpose":self.purpose.value,"source_workflow_record_digest":self.source_workflow_record_digest,"source_revision":self.source_revision,"run":self.run.to_dict(),"provider_run_binding_digest":self.provider_run.binding_digest,"submit_command_bytes_digest":domain_digest("synaptic-foundation-command-bytes/v1",self.submit_command_bytes),"foundation_record_digest":self.foundation_record.record_digest,"assessment_digest":self.assessment.authenticated_assessment_digest,"foundation_binding_digest":self.foundation_binding.binding_digest,"foundation_outcome_digest":self.foundation_outcome.outcome_digest,"found_receipt_digest":self.found_receipt_digest}
        expected=canonical_bytes(doc)
        if self.canonical_bytes!=expected or self.request_digest!=domain_digest("synaptic-provider-run-read-request/v1",expected):raise ValueError("provider read request identity mismatch")

@dataclass(frozen=True, slots=True)
class ProviderRunObservationContentV1:
    schema_version: str
    request_digest: str
    source_workflow_record_digest: str
    source_revision: int
    run: TrainingRunRef
    provider_run_binding_digest: str
    provider_id: str
    profile_ref: str
    account_ref: str
    namespace_ref: str
    provider_job_ref: str
    phase: ProviderRunPhaseV1
    canonical_evidence: bytes
    diagnostic_code: str | None
    observer_ref: str
    observer_version: str
    observed_at: str
    def __post_init__(self):
        if self.schema_version!="synaptic-provider-run-observation-content/v1" or type(self.phase) is not ProviderRunPhaseV1:raise ValueError("observation content invalid")
        for n in ("request_digest","source_workflow_record_digest","provider_run_binding_digest"):digest_text(getattr(self,n),n)
        for n in ("provider_id","profile_ref","account_ref","namespace_ref","provider_job_ref","observer_ref","observer_version","observed_at"):safe_ref(getattr(self,n),n)
        exact_integer(self.source_revision,"source_revision");e=parse_canonical_object(self.canonical_evidence,name="provider observation evidence")
        if canonical_bytes(e)!=self.canonical_evidence:raise ValueError("observation evidence not canonical")
        if self.diagnostic_code is not None:safe_ref(self.diagnostic_code,"diagnostic_code")
        if (self.phase is ProviderRunPhaseV1.FAILED)!=(self.diagnostic_code is not None):raise ValueError("observation diagnostic matrix invalid")
    @property
    def evidence_digest(self):return domain_digest("synaptic-provider-run-evidence/v1",self.canonical_evidence)
    def to_dict(self):return {"schema_version":self.schema_version,"request_digest":self.request_digest,"source_workflow_record_digest":self.source_workflow_record_digest,"source_revision":self.source_revision,"run":self.run.to_dict(),"provider_run_binding_digest":self.provider_run_binding_digest,"provider_id":self.provider_id,"profile_ref":self.profile_ref,"account_ref":self.account_ref,"namespace_ref":self.namespace_ref,"provider_job_ref":self.provider_job_ref,"phase":self.phase.value,"canonical_evidence":parse_canonical_object(self.canonical_evidence,name="provider observation evidence"),"evidence_digest":self.evidence_digest,"diagnostic_code":self.diagnostic_code,"observer_ref":self.observer_ref,"observer_version":self.observer_version,"observed_at":self.observed_at}
    @property
    def content_digest(self):return domain_digest("synaptic-provider-run-observation-content/v1",canonical_bytes(self.to_dict()))

@dataclass(frozen=True, slots=True)
class AuthenticatedProviderRunObservationV1:
    content: ProviderRunObservationContentV1
    authority_ref: str
    key_ref: str
    tag: str
    def __post_init__(self):
        if type(self.content) is not ProviderRunObservationContentV1:raise TypeError("exact observation content required")
        safe_ref(self.authority_ref,"authority_ref");safe_ref(self.key_ref,"key_ref");digest_text(self.tag,"tag")
    @property
    def canonical_bytes(self):return canonical_bytes({"schema_version":"synaptic-authenticated-provider-run-observation/v1","content":{**self.content.to_dict(),"content_digest":self.content.content_digest},"authority_ref":self.authority_ref,"key_ref":self.key_ref,"tag":self.tag})
    @property
    def authenticated_observation_digest(self):return domain_digest("synaptic-authenticated-provider-run-observation/v1",self.canonical_bytes)
    @classmethod
    def parse(cls,raw):
        doc=parse_canonical_object(raw,name="authenticated provider observation")
        if set(doc)!={"schema_version","content","authority_ref","key_ref","tag"} or doc["schema_version"]!="synaptic-authenticated-provider-run-observation/v1":raise ValueError("observation envelope invalid")
        x=dict(doc["content"]);claimed=x.pop("content_digest");evidence=x.pop("evidence_digest");x["run"]=TrainingRunRef.from_dict(x["run"]);x["phase"]=ProviderRunPhaseV1(x["phase"]);x["canonical_evidence"]=canonical_bytes(x["canonical_evidence"])
        content=ProviderRunObservationContentV1(**x)
        if evidence!=content.evidence_digest or claimed!=content.content_digest:raise ValueError("observation digest mismatch")
        owned=cls(content,doc["authority_ref"],doc["key_ref"],doc["tag"])
        if owned.canonical_bytes!=raw:raise ValueError("observation bytes not canonical")
        return owned


@dataclass(frozen=True, slots=True)
class ProviderLogQueryV1:
    after_sequence: int | None
    limit: int
    maximum_bytes: int

    def __post_init__(self) -> None:
        if self.after_sequence is not None:
            exact_integer(self.after_sequence, "after_sequence")
            if self.after_sequence > 2**63 - 1:
                raise ValueError("after_sequence exceeds the signed 64-bit bound")
        exact_integer(self.limit, "limit", minimum=1)
        exact_integer(self.maximum_bytes, "maximum_bytes", minimum=4096)
        if self.limit > 200:
            raise ValueError("log query limit exceeds 200")
        if self.maximum_bytes > 262144:
            raise ValueError("log query maximum_bytes exceeds 262144")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(
            {
                "schema_version": "synaptic-provider-log-query/v1",
                "after_sequence": self.after_sequence,
                "limit": self.limit,
                "maximum_bytes": self.maximum_bytes,
            }
        )

    @property
    def log_query_digest(self) -> str:
        return domain_digest("synaptic-provider-log-query/v1", self.canonical_bytes)


def _provider_log_canonical_bytes(value: dict[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("provider log value is not canonical JSON") from exc


def _parse_provider_log_object(raw: bytes, *, name: str, maximum_bytes: int) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        raise ValueError(f"{name} must be bounded nonempty bytes")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError(f"{name} contains a non-finite number")
            ),
            parse_float=lambda _: (_ for _ in ()).throw(
                ValueError(f"{name} contains a non-integer number")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical UTF-8 JSON") from exc
    if type(value) is not dict or _provider_log_canonical_bytes(value) != raw:
        raise ValueError(f"{name} must be a canonical JSON object")
    return value


@dataclass(frozen=True, slots=True)
class ProviderLogPageContentV1:
    schema_version: str
    read_request_digest: str
    log_query_digest: str
    source_workflow_record_digest: str
    source_revision: int
    run: TrainingRunRef
    provider_run_binding_digest: str
    provider_id: str
    profile_ref: str
    account_ref: str
    namespace_ref: str
    provider_job_ref: str
    after_sequence: int | None
    entries: tuple[RunLogEntry, ...]
    total_bytes: int
    truncated: bool
    canonical_evidence: bytes
    reader_ref: str
    reader_version: str
    read_at: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-provider-log-page-content/v1":
            raise ValueError("provider log page content schema invalid")
        for name in (
            "read_request_digest",
            "log_query_digest",
            "source_workflow_record_digest",
            "provider_run_binding_digest",
        ):
            digest_text(getattr(self, name), name)
        exact_integer(self.source_revision, "source_revision")
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        for name in (
            "provider_id", "profile_ref", "account_ref", "namespace_ref",
            "provider_job_ref", "reader_ref", "reader_version",
        ):
            safe_ref(getattr(self, name), name)
        require_rfc3339(self.read_at, "read_at")
        if self.after_sequence is not None:
            exact_integer(self.after_sequence, "after_sequence")
            if self.after_sequence > 2**63 - 1:
                raise ValueError("after_sequence exceeds the signed 64-bit bound")
        if type(self.entries) is not tuple or any(type(item) is not RunLogEntry for item in self.entries):
            raise TypeError("entries must be an exact tuple of RunLogEntry")
        if len(self.entries) > 200:
            raise ValueError("provider log page exceeds 200 entries")
        sequences = tuple(item.sequence for item in self.entries)
        if any(left >= right for left, right in zip(sequences, sequences[1:])):
            raise ValueError("provider log sequences must be unique and strictly increasing")
        if self.after_sequence is not None and any(
            sequence <= self.after_sequence for sequence in sequences
        ):
            raise ValueError("provider log sequence does not advance cursor")
        exact_integer(self.total_bytes, "total_bytes")
        if self.total_bytes != sum(item.size_bytes for item in self.entries) or self.total_bytes > 262144:
            raise ValueError("provider log total_bytes mismatch")
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be an exact boolean")
        if type(self.canonical_evidence) is not bytes or len(self.canonical_evidence) > 65536:
            raise ValueError("provider log canonical evidence exceeds 65536 bytes")
        _parse_provider_log_object(
            self.canonical_evidence,
            name="provider log evidence",
            maximum_bytes=65536,
        )

    @property
    def evidence_digest(self) -> str:
        return domain_digest("synaptic-provider-log-evidence/v1", self.canonical_evidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "read_request_digest": self.read_request_digest,
            "log_query_digest": self.log_query_digest,
            "source_workflow_record_digest": self.source_workflow_record_digest,
            "source_revision": self.source_revision,
            "run": self.run.to_dict(),
            "provider_run_binding_digest": self.provider_run_binding_digest,
            "provider_id": self.provider_id,
            "profile_ref": self.profile_ref,
            "account_ref": self.account_ref,
            "namespace_ref": self.namespace_ref,
            "provider_job_ref": self.provider_job_ref,
            "after_sequence": self.after_sequence,
            "entries": [entry.to_dict() for entry in self.entries],
            "total_bytes": self.total_bytes,
            "truncated": self.truncated,
            "canonical_evidence": _parse_provider_log_object(
                self.canonical_evidence,
                name="provider log evidence",
                maximum_bytes=65536,
            ),
            "evidence_digest": self.evidence_digest,
            "reader_ref": self.reader_ref,
            "reader_version": self.reader_version,
            "read_at": self.read_at,
        }

    @property
    def content_digest(self) -> str:
        return domain_digest(
            "synaptic-provider-log-page-content/v1",
            _provider_log_canonical_bytes(self.to_dict()),
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedProviderLogPageV1:
    content: ProviderLogPageContentV1
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.content) is not ProviderLogPageContentV1:
            raise TypeError("content must be exact ProviderLogPageContentV1")
        safe_ref(self.authority_ref, "authority_ref")
        safe_ref(self.key_ref, "key_ref")
        digest_text(self.tag, "tag")

    @property
    def canonical_bytes(self) -> bytes:
        return _provider_log_canonical_bytes(
            {
                "schema_version": "synaptic-authenticated-provider-log-page/v1",
                "content": {
                    **self.content.to_dict(),
                    "content_digest": self.content.content_digest,
                },
                "authority_ref": self.authority_ref,
                "key_ref": self.key_ref,
                "tag": self.tag,
            }
        )

    @property
    def authenticated_log_page_digest(self) -> str:
        return domain_digest(
            "synaptic-authenticated-provider-log-page/v1", self.canonical_bytes
        )

    @classmethod
    def parse(cls, raw: bytes) -> "AuthenticatedProviderLogPageV1":
        document = _parse_provider_log_object(
            raw, name="authenticated provider log page", maximum_bytes=1048576
        )
        if (
            set(document) != {"schema_version", "content", "authority_ref", "key_ref", "tag"}
            or document["schema_version"] != "synaptic-authenticated-provider-log-page/v1"
            or type(document["content"]) is not dict
        ):
            raise ValueError("provider log page envelope invalid")
        values = dict(document["content"])
        claimed_content_digest = values.pop("content_digest")
        claimed_evidence_digest = values.pop("evidence_digest")
        values["run"] = TrainingRunRef.from_dict(values["run"])
        values["entries"] = tuple(RunLogEntry.from_dict(item) for item in values["entries"])
        values["canonical_evidence"] = _provider_log_canonical_bytes(
            values["canonical_evidence"]
        )
        content = ProviderLogPageContentV1(**values)
        if (
            claimed_evidence_digest != content.evidence_digest
            or claimed_content_digest != content.content_digest
        ):
            raise ValueError("provider log page digest mismatch")
        owned = cls(
            content,
            document["authority_ref"],
            document["key_ref"],
            document["tag"],
        )
        if owned.canonical_bytes != raw:
            raise ValueError("provider log page bytes are not canonical")
        return owned


def _validated_artifacts(values: tuple[VerifiedArtifact, ...]):
    if type(values) is not tuple:
        raise TypeError("artifacts must be an exact tuple")
    artifacts = values
    if any(type(item) is not VerifiedArtifact for item in artifacts):
        raise TypeError("artifacts must contain exact VerifiedArtifact values")
    roles = tuple(item.role for item in artifacts)
    if len(roles) != len(set(roles)):
        raise ValueError("artifact roles must be unique")
    if roles != tuple(sorted(roles)):
        raise ValueError("artifact roles must already be ascending")
    return artifacts


@dataclass(frozen=True, slots=True)
class ArtifactManifestV1:
    run: TrainingRunRef
    provider_run: ScopedProviderRunRefV1
    artifacts: tuple[VerifiedArtifact, ...]
    artifact_source_digest: str
    canonical_evidence: bytes
    manifest_digest: str

    def __post_init__(self) -> None:
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        if type(self.provider_run) is not ScopedProviderRunRefV1:
            raise TypeError("provider_run must be exact ScopedProviderRunRefV1")
        object.__setattr__(self, "artifacts", _validated_artifacts(self.artifacts))
        if not self.artifacts:
            raise ValueError("artifact manifest must not be empty")
        digest_text(self.artifact_source_digest, "artifact_source_digest")
        digest_text(self.manifest_digest, "manifest_digest")
        if type(self.canonical_evidence) is not bytes:
            raise TypeError("canonical_evidence must be exact bytes")
        parse_canonical_object(self.canonical_evidence, name="artifact evidence")
        if self.manifest_digest != self.expected_manifest_digest:
            raise ValueError("manifest digest does not bind manifest")

    @classmethod
    def build(cls, *, run, provider_run, artifacts, artifact_source_digest,
              canonical_evidence):
        ordered = _validated_artifacts(artifacts)
        document = cls._document(
            run, provider_run, ordered, artifact_source_digest, canonical_evidence
        )
        return cls(
            run, provider_run, ordered, artifact_source_digest, canonical_evidence,
            domain_digest("synaptic-artifact-manifest/v1", canonical_bytes(document)),
        )

    @staticmethod
    def _document(run, provider_run, artifacts, artifact_source_digest,
                  canonical_evidence):
        return {
            "schema_version": "synaptic-artifact-manifest/v1",
            "run": run.to_dict(), "provider_run": provider_run.to_dict(),
            "artifacts": [item.to_dict() for item in artifacts],
            "artifact_source_digest": artifact_source_digest,
            "evidence_digest": domain_digest(
                "synaptic-artifact-manifest-evidence/v1", canonical_evidence
            ),
        }

    @property
    def expected_manifest_digest(self) -> str:
        return domain_digest(
            "synaptic-artifact-manifest/v1",
            canonical_bytes(self._document(
                self.run, self.provider_run, self.artifacts,
                self.artifact_source_digest, self.canonical_evidence,
            )),
        )


class VerificationVerdictV1(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ArtifactVerificationContentV1:
    schema_version: str
    source_workflow_record_digest: str
    source_revision: int
    run: TrainingRunRef
    provider_run_binding_digest: str
    manifest_digest: str
    artifact_source_digest: str
    manifest_artifacts: tuple[VerifiedArtifact, ...]
    verified_artifacts: tuple[VerifiedArtifact, ...]
    verdict: VerificationVerdictV1
    diagnostic_code: str | None
    verifier_ref: str
    verifier_version: str
    canonical_evidence: bytes
    checked_at: str

    def __post_init__(self): self._validate()

    def _validate(self):
        if self.schema_version != "synaptic-artifact-verification-content/v1":
            raise ValueError("unsupported artifact verification content")
        exact_integer(self.source_revision, "source_revision")
        if type(self.run) is not TrainingRunRef or type(self.verdict) is not VerificationVerdictV1:
            raise TypeError("verification content types invalid")
        for name in ("source_workflow_record_digest", "provider_run_binding_digest", "manifest_digest", "artifact_source_digest"):
            digest_text(getattr(self, name), name)
        for name in ("verifier_ref", "verifier_version", "checked_at"):
            safe_ref(getattr(self, name), name)
        object.__setattr__(self, "manifest_artifacts", _validated_artifacts(self.manifest_artifacts))
        object.__setattr__(self, "verified_artifacts", _validated_artifacts(self.verified_artifacts))
        if type(self.canonical_evidence) is not bytes or len(self.canonical_evidence) > 1_048_576:
            raise ValueError("verification evidence must be bounded exact bytes")
        evidence = parse_canonical_object(self.canonical_evidence, name="verification evidence")
        if canonical_bytes(evidence) != self.canonical_evidence:
            raise ValueError("verification evidence is not exact canonical bytes")
        if self.verdict is VerificationVerdictV1.VERIFIED:
            if self.verified_artifacts != self.manifest_artifacts or self.diagnostic_code is not None:
                raise ValueError("verified receipt artifacts/diagnostic invalid")
        else:
            if self.verified_artifacts or self.diagnostic_code is None:
                raise ValueError("rejected receipt requires diagnostic and no artifacts")
            safe_ref(self.diagnostic_code, "diagnostic_code")

    def to_dict(self):
        artifacts = lambda xs: [x.to_dict() for x in xs]
        return {
            "schema_version": self.schema_version,
            "source_workflow_record_digest": self.source_workflow_record_digest,
            "source_revision": self.source_revision, "run": self.run.to_dict(),
            "provider_run_binding_digest": self.provider_run_binding_digest,
            "manifest_digest": self.manifest_digest,
            "artifact_source_digest": self.artifact_source_digest,
            "manifest_artifacts": artifacts(self.manifest_artifacts),
            "verified_artifacts": artifacts(self.verified_artifacts),
            "verdict": self.verdict.value, "diagnostic_code": self.diagnostic_code,
            "verifier_ref": self.verifier_ref, "verifier_version": self.verifier_version,
            "canonical_evidence": parse_canonical_object(self.canonical_evidence, name="verification evidence"),
            "evidence_digest": self.evidence_digest, "checked_at": self.checked_at,
        }

    @property
    def evidence_digest(self):
        return domain_digest("synaptic-artifact-verification-evidence/v1", self.canonical_evidence)

    @property
    def content_digest(self):
        return domain_digest("synaptic-artifact-verification-content/v1", canonical_bytes(self.to_dict()))

@dataclass(frozen=True, slots=True)
class AuthenticatedArtifactVerificationReceiptV1:
    content: ArtifactVerificationContentV1
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self):
        if type(self.content) is not ArtifactVerificationContentV1: raise TypeError("exact verification content required")
        safe_ref(self.authority_ref,"authority_ref"); safe_ref(self.key_ref,"key_ref"); digest_text(self.tag,"tag")

    @property
    def canonical_bytes(self):
        content = {**self.content.to_dict(), "content_digest": self.content.content_digest}
        return canonical_bytes({"schema_version": "synaptic-authenticated-artifact-verification/v1", "content": content, "authority_ref": self.authority_ref, "key_ref": self.key_ref, "tag": self.tag})

    @classmethod
    def parse(cls, raw: bytes):
        doc = parse_canonical_object(raw, name="authenticated artifact verification")
        if set(doc) != {"schema_version", "content", "authority_ref", "key_ref", "tag"}:
            raise ValueError("authenticated artifact verification fields invalid")
        if doc["schema_version"] != "synaptic-authenticated-artifact-verification/v1":
            raise ValueError("unsupported artifact verification receipt")
        content_doc = doc["content"]
        expected_content_fields = (set(ArtifactVerificationContentV1.__dataclass_fields__)
                                   - {"canonical_evidence"}) | {"canonical_evidence", "evidence_digest", "content_digest"}
        if not isinstance(content_doc, dict) or set(content_doc) != expected_content_fields:
            raise ValueError("artifact verification content fields invalid")
        claimed = content_doc.pop("content_digest")
        def artifacts(value):
            if not isinstance(value, list): raise ValueError("artifact array malformed")
            result = []
            for item in value:
                if not isinstance(item, dict) or set(item) != {"role", "sha256", "size_bytes"}:
                    raise ValueError("artifact entry malformed")
                result.append(VerifiedArtifact.from_dict(item))
            return tuple(result)
        values = dict(content_doc)
        values["run"] = TrainingRunRef.from_dict(values["run"])
        values["manifest_artifacts"] = artifacts(values["manifest_artifacts"])
        values["verified_artifacts"] = artifacts(values["verified_artifacts"])
        values["verdict"] = VerificationVerdictV1(values["verdict"])
        claimed_evidence = values.pop("evidence_digest")
        values["canonical_evidence"] = canonical_bytes(values["canonical_evidence"])
        content = ArtifactVerificationContentV1(**values)
        if claimed_evidence != content.evidence_digest: raise ValueError("artifact verification evidence digest mismatch")
        if claimed != content.content_digest: raise ValueError("artifact verification content digest mismatch")
        owned = cls(content,doc["authority_ref"],doc["key_ref"],doc["tag"])
        if owned.canonical_bytes != raw: raise ValueError("verification receipt is not exact canonical bytes")
        return owned

    @property
    def authenticated_receipt_digest(self):
        return domain_digest("synaptic-authenticated-artifact-verification/v1", self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class WorkflowRecordV1:
    schema_version: str
    run: TrainingRunRef
    plan_fingerprint: str
    preflight_digest: str
    provider: ProviderRef
    provider_context_digest: str
    provider_descriptor_digest: str
    phase: WorkflowPhaseV1
    revision: int
    preparation_digest: str | None
    stage: EffectIntentV1 | None
    submit: EffectIntentV1 | None
    cancel: EffectIntentV1 | None
    provider_stage_ref: BoundProviderStageRefV1 | None
    provider_run_ref: BoundProviderRunRefV1 | None
    bound_cancellation: BoundCancellationRefV1 | None
    pre_cancel_phase: WorkflowPhaseV1 | None
    run_observation_digests: tuple[str, ...]
    artifact_manifest: ArtifactManifestV1 | None
    artifact_manifest_digest: str | None
    verified_artifacts: tuple[VerifiedArtifact, ...]
    verification_receipts: tuple[AuthenticatedArtifactVerificationReceiptV1, ...]
    verification_receipt_digests: tuple[str, ...]
    diagnostic_codes: tuple[str, ...]
    provider_run_observations: tuple[AuthenticatedProviderRunObservationV1, ...] = ()

    def __post_init__(self): self._validate()

    @classmethod
    def planned(cls, *, run: TrainingRunRef, plan: TrainingPlan,
                preflight_digest: str, context: ProviderPlanContextV1,
                provider: ProviderRef, descriptor: ProviderDescriptor):
        if type(run) is not TrainingRunRef or type(plan) is not TrainingPlan:
            raise TypeError("run and plan must be exact B1 contracts")
        if type(context) is not ProviderPlanContextV1:
            raise TypeError("context must be exact ProviderPlanContextV1")
        if type(provider) is not ProviderRef or type(descriptor) is not ProviderDescriptor:
            raise TypeError("provider and descriptor must be exact B1 contracts")
        digest_text(preflight_digest, "preflight_digest")
        expected = (
            plan.basis.project_ref, plan.basis.basis_digest,
            plan.provider_plan.context_digest, context.provider,
            context.descriptor_digest, descriptor.provider_id,
        )
        actual = (
            run.project_ref, context.basis_digest,
            context.provider_context_digest, provider,
            descriptor.descriptor_digest, provider.provider_id,
        )
        if actual != expected:
            raise ValueError("plan, context, provider, descriptor, and run do not bind")
        return cls(
            schema_version="synaptic-coordinator-workflow/v1", run=run,
            plan_fingerprint=plan.plan_fingerprint,
            preflight_digest=preflight_digest, provider=provider,
            provider_context_digest=context.provider_context_digest,
            provider_descriptor_digest=descriptor.descriptor_digest,
            phase=WorkflowPhaseV1.PLANNED, revision=0,
            preparation_digest=None, stage=None, submit=None, cancel=None,
            provider_stage_ref=None, provider_run_ref=None, pre_cancel_phase=None,
            bound_cancellation=None,
            run_observation_digests=(), artifact_manifest=None,
            artifact_manifest_digest=None, verified_artifacts=(),
            verification_receipts=(), verification_receipt_digests=(), diagnostic_codes=(),
            provider_run_observations=(),
        )

    def _validate(self) -> None:
        if self.schema_version != "synaptic-coordinator-workflow/v1":
            raise ValueError("unsupported coordinator workflow schema")
        if type(self.run) is not TrainingRunRef or type(self.provider) is not ProviderRef:
            raise TypeError("workflow run/provider types are invalid")
        if type(self.phase) is not WorkflowPhaseV1:
            raise TypeError("phase must be exact WorkflowPhaseV1")
        exact_integer(self.revision, "revision")
        for name in (
            "plan_fingerprint", "preflight_digest", "provider_context_digest",
            "provider_descriptor_digest",
        ):
            digest_text(getattr(self, name), name)
        if self.preparation_digest is not None:
            digest_text(self.preparation_digest, "preparation_digest")
        for values, name, validator in (
            (self.run_observation_digests, "run_observation_digest", digest_text),
            (self.verification_receipt_digests, "verification_receipt_digest", digest_text),
            (self.diagnostic_codes, "diagnostic_code", safe_ref),
        ):
            if type(values) is not tuple:
                raise TypeError(f"{name} history must be exact tuple")
            for value in values:
                validator(value, name)
        if type(self.verification_receipts) is not tuple or any(
            type(value) is not AuthenticatedArtifactVerificationReceiptV1
            for value in self.verification_receipts
        ):
            raise TypeError("verification receipt history must contain exact envelopes")
        derived_receipt_digests = tuple(
            value.authenticated_receipt_digest for value in self.verification_receipts
        )
        if derived_receipt_digests != self.verification_receipt_digests:
            raise ValueError("verification receipt digest history is not derived")
        if type(self.provider_run_observations) is not tuple or any(type(x) is not AuthenticatedProviderRunObservationV1 for x in self.provider_run_observations):raise TypeError("provider observations must be exact envelopes")
        if tuple(x.authenticated_observation_digest for x in self.provider_run_observations)!=self.run_observation_digests:raise ValueError("provider observation digest history is not derived")
        object.__setattr__(self, "verified_artifacts",
                           _validated_artifacts(self.verified_artifacts))
        if self.artifact_manifest_digest is not None:
            digest_text(self.artifact_manifest_digest, "artifact_manifest_digest")
        if (self.artifact_manifest is None) != (self.artifact_manifest_digest is None):
            raise ValueError("artifact manifest and digest must be recorded together")
        if self.artifact_manifest is not None and (
            type(self.artifact_manifest) is not ArtifactManifestV1
            or self.artifact_manifest.manifest_digest != self.artifact_manifest_digest
        ):
            raise ValueError("artifact manifest digest mismatch")

        for name, kind in (
            ("stage", EffectKind.STAGE), ("submit", EffectKind.SUBMIT),
            ("cancel", EffectKind.CANCEL),
        ):
            intent = getattr(self, name)
            if intent is not None:
                if type(intent) is not EffectIntentV1 or intent.kind is not kind:
                    raise TypeError(f"{name} intent type is invalid")
                preparation = parse_exact_command(intent.canonical_command_bytes).preparation
                if (
                    preparation.provider, preparation.project_ref, preparation.run_id,
                    preparation.plan_fingerprint, preparation.preparation_digest,
                ) != (
                    self.provider, self.run.project_ref, self.run.run_id,
                    self.plan_fingerprint, self.preparation_digest,
                ):
                    raise ValueError(f"{name} intent does not bind workflow lineage")
        if self.stage is not None and self.preparation_digest is None:
            raise ValueError("effect intent requires preparation")
        if self.provider_stage_ref is not None:
            bound = self.provider_stage_ref
            if type(bound) is not BoundProviderStageRefV1 or self.stage is None:
                raise TypeError("bound stage reference requires stage intent")
            if not self.stage.advanced or (
                bound.effect_id, bound.command_digest, bound.preparation_digest,
            ) != (self.stage.effect_id, self.stage.command_digest,
                   self.preparation_digest):
                raise ValueError("bound stage reference does not match workflow stage")
            latest_binding = self.stage.foundation_bindings[-1]
            latest_outcome = self.stage.foundation_outcomes[-1]
            if (bound.command_bytes_digest, bound.foundation_binding_digest,
                    bound.foundation_outcome_digest) != (
                    latest_binding.command_bytes_digest, latest_binding.binding_digest,
                    latest_outcome.outcome_digest,
            ) or _fresh_found_reference(latest_binding,bound.authenticated_receipt_digest,EffectKind.STAGE)!=bound.reference:
                raise ValueError("bound stage evidence lineage mismatch")
        if self.submit is not None and self.provider_stage_ref is None:
            raise ValueError("submit intent requires bound stage reference")
        if self.provider_run_ref is not None:
            bound = self.provider_run_ref
            if type(bound) is not BoundProviderRunRefV1 or self.submit is None:
                raise TypeError("bound run reference requires submit intent")
            if not self.submit.advanced or (
                bound.effect_id, bound.command_digest, bound.preparation_digest,
            ) != (self.submit.effect_id, self.submit.command_digest,
                   self.preparation_digest):
                raise ValueError("bound run reference does not match workflow submit")
            latest_binding = self.submit.foundation_bindings[-1]
            latest_outcome = self.submit.foundation_outcomes[-1]
            if (bound.command_bytes_digest, bound.foundation_binding_digest,
                    bound.foundation_outcome_digest) != (
                    latest_binding.command_bytes_digest, latest_binding.binding_digest,
                    latest_outcome.outcome_digest,
            ) or _fresh_found_reference(latest_binding,bound.authenticated_receipt_digest,EffectKind.SUBMIT)!=bound.reference:
                raise ValueError("bound run evidence lineage mismatch")
        if self.cancel is not None:
            if self.provider_run_ref is None or self.pre_cancel_phase not in {
                WorkflowPhaseV1.QUEUED, WorkflowPhaseV1.RUNNING,
            }:
                raise ValueError("cancel intent requires bound run and exact origin")
            target = parse_exact_command(
                self.cancel.canonical_command_bytes
            ).operation.effect.cancel_target
            if target.provider_job_ref != self.provider_run_ref.reference.provider_job_ref:
                raise ValueError("cancel intent targets different provider run")
        elif self.pre_cancel_phase is not None:
            raise ValueError("pre_cancel_phase requires cancel intent")
        if self.bound_cancellation is not None:
            if type(self.bound_cancellation) is not BoundCancellationRefV1 or self.cancel is None:
                raise TypeError("bound cancellation requires cancel intent")
            if self.provider_run_ref is None or (
                self.bound_cancellation.effect_id != self.cancel.effect_id
                or self.bound_cancellation.command_digest != self.cancel.command_digest
                or self.bound_cancellation.preparation_digest != self.preparation_digest
                or self.bound_cancellation.target_run_binding_digest != self.provider_run_ref.binding_digest
            ):
                raise ValueError("bound cancellation does not bind workflow")
            latest_binding = self.cancel.foundation_bindings[-1]
            latest_outcome = self.cancel.foundation_outcomes[-1]
            if (self.bound_cancellation.command_bytes_digest,
                    self.bound_cancellation.foundation_binding_digest,
                    self.bound_cancellation.foundation_outcome_digest) != (
                    latest_binding.command_bytes_digest, latest_binding.binding_digest,
                    latest_outcome.outcome_digest,
            ) or _fresh_found_reference(latest_binding,self.bound_cancellation.authenticated_receipt_digest,EffectKind.CANCEL)!=self.bound_cancellation.reference:
                raise ValueError("bound cancellation evidence lineage mismatch")
        self._validate_phase_fields()

    def _validate_phase_fields(self) -> None:
        phase = self.phase
        def last_is(intent, disposition):
            return bool(intent and intent.foundation_outcomes
                        and intent.foundation_outcomes[-1].disposition is disposition)
        progression = (
            self.preparation_digest, self.stage, self.submit, self.cancel,
            self.provider_stage_ref, self.provider_run_ref, self.bound_cancellation,
            self.pre_cancel_phase,
        )
        if phase in {WorkflowPhaseV1.PLANNED, WorkflowPhaseV1.PREPARING}:
            if (any(value is not None for value in progression)
                    or self.run_observation_digests
                    or self.verification_receipt_digests or self.diagnostic_codes
                    or self.artifact_manifest is not None or self.verified_artifacts):
                raise ValueError("planned/preparing contains premature state")
            return
        if phase not in {WorkflowPhaseV1.FAILED} and (
            self.preparation_digest is None or self.stage is None
        ):
            raise ValueError("workflow phase requires preparation and stage intent")
        if phase is WorkflowPhaseV1.STAGE_INTENT_RECORDED:
            if (self.stage.advanced or self.provider_stage_ref is not None or self.submit is not None
                    or self.run_observation_digests):
                raise ValueError("stage intent phase fields invalid")
        elif phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED:
            if (not self.stage.advanced or not last_is(self.stage, FoundationDispositionV1.INDETERMINATE)
                    or self.provider_stage_ref is not None
                    or self.submit is not None or self.run_observation_digests):
                raise ValueError("stage reconcile phase fields invalid")
        elif phase is WorkflowPhaseV1.STAGED:
            if (self.provider_stage_ref is None or not last_is(self.stage, FoundationDispositionV1.FOUND)
                    or self.submit is not None
                    or self.run_observation_digests):
                raise ValueError("staged phase fields invalid")
        elif phase is WorkflowPhaseV1.SUBMIT_INTENT_RECORDED:
            if (self.provider_stage_ref is None or self.submit is None or self.submit.advanced
                    or self.provider_run_ref is not None
                    or self.run_observation_digests):
                raise ValueError("submit intent phase fields invalid")
        elif phase is WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED:
            if (self.submit is None or not self.submit.advanced
                    or not last_is(self.submit, FoundationDispositionV1.INDETERMINATE)
                    or self.provider_run_ref is not None
                    or self.run_observation_digests):
                raise ValueError("submit reconcile phase fields invalid")
        elif phase in {WorkflowPhaseV1.QUEUED, WorkflowPhaseV1.RUNNING}:
            if self.provider_run_ref is None or not last_is(self.submit, FoundationDispositionV1.FOUND):
                raise ValueError("active run phase fields invalid")
            if self.cancel is not None and (
                not last_is(self.cancel, FoundationDispositionV1.DEFINITELY_ABSENT)
                or self.bound_cancellation is not None
            ):
                raise ValueError("restored cancellation history is invalid")
        elif phase is WorkflowPhaseV1.CANCEL_INTENT_RECORDED:
            if self.cancel is None or self.cancel.advanced or self.bound_cancellation is not None:
                raise ValueError("cancel intent phase fields invalid")
        elif phase is WorkflowPhaseV1.CANCEL_REQUESTED:
            if (self.cancel is None or not last_is(self.cancel, FoundationDispositionV1.FOUND)
                    or self.bound_cancellation is None):
                raise ValueError("requested cancellation fields invalid")
        elif phase is WorkflowPhaseV1.CANCEL_RECONCILE_REQUIRED:
            if (self.cancel is None
                    or not last_is(self.cancel, FoundationDispositionV1.INDETERMINATE)
                    or self.bound_cancellation is not None):
                raise ValueError("cancel reconciliation fields invalid")
        elif phase is WorkflowPhaseV1.SUCCEEDED_UNVERIFIED:
            if self.provider_run_ref is None or not self.run_observation_digests:
                raise ValueError("succeeded-unverified fields invalid")
        elif phase is WorkflowPhaseV1.VERIFICATION_FAILED:
            if (self.artifact_manifest is None or self.verified_artifacts
                    or not self.verification_receipt_digests
                    or not self.diagnostic_codes):
                raise ValueError("verification-failed fields invalid")
        elif phase is WorkflowPhaseV1.VERIFIED:
            if (self.artifact_manifest is None
                    or self.verified_artifacts != self.artifact_manifest.artifacts
                    or not self.verification_receipt_digests):
                raise ValueError("verified fields invalid")
        elif phase is WorkflowPhaseV1.FAILED:
            if not self.diagnostic_codes:
                raise ValueError("failed phase requires diagnostic")
        elif phase is WorkflowPhaseV1.CANCELLED:
            if self.provider_run_ref is None or not self.run_observation_digests:
                raise ValueError("cancelled fields invalid")
            if self.cancel is not None and not self.cancel.advanced:
                raise ValueError("internal cancellation requires advanced intent")
        elif phase is WorkflowPhaseV1.CONTRADICTED:
            active = self.cancel or self.submit or self.stage
            if (not self.stage.advanced or not self.diagnostic_codes
                    or not last_is(active, FoundationDispositionV1.CONTRADICTED)):
                raise ValueError("contradicted fields invalid")
        if phase not in {WorkflowPhaseV1.VERIFICATION_FAILED,
                         WorkflowPhaseV1.VERIFIED} and (
            self.artifact_manifest is not None or self.verified_artifacts
            or self.verification_receipt_digests
        ):
            raise ValueError("verification state appears outside verification phase")

    @property
    def latest_run_evidence_digest(self):
        return self.run_observation_digests[-1] if self.run_observation_digests else None

    @property
    def verification_receipt_digest(self):
        return (self.verification_receipt_digests[-1]
                if self.verification_receipt_digests else None)

    @property
    def diagnostic_code(self):
        return self.diagnostic_codes[-1] if self.diagnostic_codes else None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "run": self.run.to_dict(),
            "plan_fingerprint": self.plan_fingerprint,
            "preflight_digest": self.preflight_digest,
            "provider": self.provider.to_dict(),
            "provider_context_digest": self.provider_context_digest,
            "provider_descriptor_digest": self.provider_descriptor_digest,
            "phase": self.phase.value, "revision": self.revision,
            "preparation_digest": self.preparation_digest,
            "stage_intent_digest": None if self.stage is None else self.stage.intent_digest,
            "submit_intent_digest": None if self.submit is None else self.submit.intent_digest,
            "cancel_intent_digest": None if self.cancel is None else self.cancel.intent_digest,
            "provider_stage_binding_digest": None if self.provider_stage_ref is None else self.provider_stage_ref.binding_digest,
            "provider_run_binding_digest": None if self.provider_run_ref is None else self.provider_run_ref.binding_digest,
            "bound_cancellation_digest": None if self.bound_cancellation is None else self.bound_cancellation.binding_digest,
            "pre_cancel_phase": None if self.pre_cancel_phase is None else self.pre_cancel_phase.value,
            "run_observation_digests": list(self.run_observation_digests),
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "verified_artifacts": [item.to_dict() for item in self.verified_artifacts],
            "verification_receipt_digests": list(self.verification_receipt_digests),
            "diagnostic_codes": list(self.diagnostic_codes),
        }

    @property
    def record_digest(self):
        return domain_digest(
            "synaptic-coordinator-workflow/v1", canonical_bytes(self.to_dict())
        )


@dataclass(frozen=True, slots=True)
class WorkflowStorePageV1:
    records: tuple[WorkflowRecordV1, ...]
    has_more: bool

    def __post_init__(self) -> None:
        if type(self.records) is not tuple or any(
            type(record) is not WorkflowRecordV1 for record in self.records
        ):
            raise TypeError("records must be an exact tuple of WorkflowRecordV1")
        if type(self.has_more) is not bool:
            raise TypeError("has_more must be an exact boolean")


__all__ = [
    "ArtifactManifestV1", "ArtifactVerificationContentV1",
    "AuthenticatedArtifactVerificationReceiptV1", "AuthenticatedFoundationRecordAssessmentV1",
    "AuthenticatedProviderLogPageV1", "AuthenticatedProviderRunObservationV1", "BoundCancellationRefV1",
    "BoundProviderRunRefV1", "BoundProviderStageRefV1", "EffectIntentV1",
    "FoundationDispositionV1", "FoundationEffectBindingV1", "FoundationRecordAssessmentContentV1",
    "FoundationEffectOutcomeV1",
    "ProviderExecutionBindingV1", "ProviderLogPageContentV1", "ProviderLogQueryV1", "ProviderReadPurposeV1", "ProviderRunObservationContentV1", "ProviderRunReadRequestV1",
    "ProviderRunPhaseV1", "ReceiptAssessmentV1", "ReceiptFreshnessV1", "VerificationVerdictV1", "WorkflowPhaseV1",
    "WorkflowRecordV1", "WorkflowStorePageV1",
]
