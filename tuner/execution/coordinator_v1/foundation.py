"""Provider-neutral composition over the authenticated Foundation V2 core."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, replace

from tuner.execution.foundation_v2.authority import AuthenticatedGrantV2, AuthenticatedReconciliationGrantV1
from tuner.execution.foundation_v2.canonical import DiagnosticCode, FoundationError, canonical_bytes as encode_canonical, digest_text, domain_digest, exact_integer, parse_canonical_object, safe_ref
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.repository import DispatchState, EffectRecordV2, ReconciliationOwnershipV2, _revalidate_effect_record_v2_canonical

from .model import AuthenticatedFoundationRecordAssessmentV1, FoundationRecordAssessmentContentV1, ReceiptAssessmentV1, ReceiptFreshnessV1


def _claim_document(claim):
    return {"owner_ref":claim.owner_ref,"generation":claim.generation,"ownership_epoch":claim.ownership_epoch,"claimed_at_epoch":claim.claimed_at_epoch,"target_digest":claim.target_digest,"grant_ref":claim.grant_ref,"grant_digest":claim.grant_digest,"grant_lineage":[value.to_dict()|{"binding_digest":value.binding_digest} for value in claim.grant_lineage],"active":claim.active,"completed":claim.completed,"claim_digest":claim.claim_digest}


def _record_snapshot(record):
    command=parse_exact_command(record.command_bytes)
    return {"schema_version":"synaptic-foundation-effect-snapshot/v1","command":command.to_dict(),"command_bytes_digest":domain_digest("synaptic-foundation-command-bytes/v1",record.command_bytes),"grant":parse_canonical_object(record.grant.canonical_bytes,name="authenticated grant"),"dispatch":record.dispatch.value,"state":record.state.value,"attempt_count":record.attempt_count,"dispatch_epoch":record.dispatch_epoch,"receipts":[parse_canonical_object(value.canonical_bytes,name="receipt") for value in record.results],"receipt_admissions":[value.to_dict()|{"admission_digest":value.admission_digest} for value in record.receipt_admissions],"invalid_evidence":[parse_canonical_object(value.canonical_bytes,name="invalid evidence") for value in record.invalid_evidence],"invalid_evidence_admissions":[value.to_dict()|{"admission_digest":value.admission_digest} for value in record.invalid_evidence_admissions],"terminal_content_digests":list(record.terminal_content_digests),"invalid_codes":[value.value for value in record.invalid_codes],"reconciliation":None if record.reconciliation is None else _claim_document(record.reconciliation),"reconciliation_claims":[_claim_document(value) for value in record.reconciliation_claims],"b2_record_digest":record.record_digest}


def _validate_record(record,receipt_authority,invalid_authority,grant_authority):
    try:
        validated=_revalidate_effect_record_v2_canonical(record,receipt_authority,invalid_authority,grant_authority)
        snapshot=_record_snapshot(validated)
        if snapshot["b2_record_digest"]!=validated.record_digest:raise ValueError
        return validated
    except Exception:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID) from None


class FoundationRecordAssessmentAuthorityV1:
    __slots__=("authority_ref","key_ref","assessor_ref","assessor_version","_key","_clock","_receipts","_invalid","_grants")
    def __init__(self,authority_ref,key_ref,key,*,assessor_ref,assessor_version,clock,receipt_authority,invalid_evidence_authority,grant_authority):
        self.authority_ref=safe_ref(authority_ref,"authority_ref");self.key_ref=safe_ref(key_ref,"key_ref");self.assessor_ref=safe_ref(assessor_ref,"assessor_ref");self.assessor_version=safe_ref(assessor_version,"assessor_version")
        if type(key) is not bytes or len(key)<32:raise ValueError("assessment authority key too short")
        self._key=bytes(key);self._clock=clock;self._receipts=receipt_authority;self._invalid=invalid_evidence_authority;self._grants=grant_authority
    def __repr__(self):return f"FoundationRecordAssessmentAuthorityV1(authority_ref={self.authority_ref!r}, key_ref={self.key_ref!r}, key=<redacted>)"
    def _tag(self,digest):return hmac.new(self._key,b"foundation-assessment-v1\0"+bytes.fromhex(digest),hashlib.sha256).hexdigest()
    def assess(self,record):
        record=_validate_record(record,self._receipts,self._invalid,self._grants)
        try:
            assessed_at=self._clock.now_iso();safe_ref(assessed_at,"assessed_at");command=parse_exact_command(record.command_bytes);snapshot=_record_snapshot(record)
            assessments=tuple(ReceiptAssessmentV1(receipt.authenticated_receipt_digest,admission.admission_digest,admission.source_kind,admission.source_owner_ref,admission.source_generation,admission.source_ownership_epoch,ReceiptFreshnessV1.FRESH if admission.freshness.value=="fresh" else ReceiptFreshnessV1.STALE,admission.finality_verified,tuple(code.value for code in admission.generated_invalid_codes)) for receipt,admission in zip(record.results,record.receipt_admissions,strict=True))
            content=FoundationRecordAssessmentContentV1("synaptic-foundation-record-assessment-content/v1",command.operation.effect.effect_id,command.digest,snapshot["command_bytes_digest"],record.record_digest,domain_digest("synaptic-foundation-record-evidence/v1",encode_canonical(snapshot)),tuple(value.authenticated_receipt_digest for value in record.results),record.terminal_content_digests,assessments,tuple(value.admission_digest for value in record.invalid_evidence_admissions),tuple(value.value for value in record.invalid_codes),self.assessor_ref,self.assessor_version,assessed_at)
            envelope=AuthenticatedFoundationRecordAssessmentV1(content,self.authority_ref,self.key_ref,self._tag(content.content_digest))
            return AuthenticatedFoundationRecordAssessmentV1.parse(envelope.canonical_bytes)
        except FoundationError:raise
        except Exception:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID) from None
    def authenticate(self,assessment):
        try:
            if type(assessment) is not AuthenticatedFoundationRecordAssessmentV1:return False
            owned=AuthenticatedFoundationRecordAssessmentV1.parse(assessment.canonical_bytes)
            return owned==assessment and owned.canonical_bytes==assessment.canonical_bytes and owned.authority_ref==self.authority_ref and owned.key_ref==self.key_ref and owned.content.assessor_ref==self.assessor_ref and owned.content.assessor_version==self.assessor_version and hmac.compare_digest(owned.tag,self._tag(owned.content.content_digest))
        except Exception:return False


@dataclass(frozen=True,slots=True)
class QuiescenceRecoveryRequestV1:
    schema_version:str;effect_id:str;command_digest:str;command_bytes_digest:str;executor_digest:str;foundation_record_digest:str;dispatch_source_digest:str;dispatch_state:str;authenticated_execution_grant_digest:str;provider_id:str;profile_ref:str;account_ref:str;namespace_ref:str;attempt_count:int;dispatch_epoch:int
    def __post_init__(self):
        if self.schema_version!="synaptic-quiescence-recovery-request/v1":raise ValueError("unsupported quiescence recovery request")
        for name in ("effect_id","provider_id","profile_ref","account_ref","namespace_ref"):safe_ref(getattr(self,name),name)
        for name in ("command_digest","command_bytes_digest","executor_digest","foundation_record_digest","dispatch_source_digest","authenticated_execution_grant_digest"):digest_text(getattr(self,name),name)
        if self.dispatch_state!=DispatchState.ORPHANED_UNPROVEN.value:raise ValueError("recovery requires orphaned dispatch")
        exact_integer(self.attempt_count,"attempt_count",minimum=1);exact_integer(self.dispatch_epoch,"dispatch_epoch",minimum=1)
    def to_dict(self):return {name:getattr(self,name) for name in self.__dataclass_fields__}
    @property
    def canonical_bytes(self):return encode_canonical(self.to_dict())
    @property
    def request_digest(self):return domain_digest("synaptic-quiescence-recovery-request/v1",self.canonical_bytes)


def _capture(operation):
    failed=False;code=DiagnosticCode.AUTHORITY_INVALID;value=None
    try:value=operation()
    except FoundationError as error:failed=True;code=error.code
    except Exception:failed=True
    if failed:raise FoundationError(code)
    return value


class ComposedEffectFoundationV1:
    def __init__(self,repository,broker,reconciliation,*,grant_authority,receipt_authority,invalid_evidence_authority,assessment_authority,trusted_quiescence_evidence):
        self._repo=repository;self._broker=broker;self._reconciliation=reconciliation;self._grants=grant_authority;self._receipts=receipt_authority;self._invalid=invalid_evidence_authority;self._assessments=assessment_authority;self._quiescence=trusted_quiescence_evidence
    def _validated(self,record):return _validate_record(record,self._receipts,self._invalid,self._grants)
    def _durable(self,effect_id,returned):
        returned=self._validated(returned);reloaded=self._validated(_capture(lambda:self._repo.get(effect_id)))
        if returned!=reloaded or returned.record_digest!=reloaded.record_digest:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        if returned.command.operation.effect.effect_id!=effect_id:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return reloaded
    def get(self,effect_id):
        try:safe_ref(effect_id,"effect_id")
        except Exception:raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        record=_capture(lambda:self._repo.get(effect_id))
        if record is None:return None
        record=self._validated(record)
        if record.command.operation.effect.effect_id!=effect_id:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return record
    def assess(self,record):return self._assessments.assess(self._validated(record))
    def authenticate(self,assessment):
        try:return self._assessments.authenticate(assessment) is True
        except Exception:return False
    def execute(self,command_bytes,grant,*,now_epoch):
        try:
            if type(command_bytes) is not bytes or type(grant) is not AuthenticatedGrantV2:raise ValueError
            effect_id=parse_exact_command(command_bytes).operation.effect.effect_id
        except Exception:raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        return self._durable(effect_id,_capture(lambda:self._broker.execute(command_bytes,grant,now_epoch=now_epoch)))
    def reconcile(self,command_bytes,grant,*,now_epoch,continuation=None):
        try:
            if type(command_bytes) is not bytes or type(grant) is not AuthenticatedReconciliationGrantV1 or (continuation is not None and type(continuation) is not ReconciliationOwnershipV2):raise ValueError
            effect_id=parse_exact_command(command_bytes).operation.effect.effect_id
        except Exception:raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        return self._durable(effect_id,_capture(lambda:self._reconciliation.reconcile(command_bytes,grant,now_epoch=now_epoch,continuation=continuation)))
    def recover_orphan(self,effect_id,*,now_epoch):
        before=self.get(effect_id)
        if before is None:raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
        if before.dispatch is DispatchState.QUIESCENCE_PROVEN:return before
        if before.dispatch is not DispatchState.ORPHANED_UNPROVEN:raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
        command=parse_exact_command(before.command_bytes);preparation=command.preparation
        request=QuiescenceRecoveryRequestV1("synaptic-quiescence-recovery-request/v1",effect_id,command.digest,domain_digest("synaptic-foundation-command-bytes/v1",before.command_bytes),command.executor.digest,before.record_digest,before.dispatch_source_digest,before.dispatch.value,before.grant.authenticated_grant_digest,preparation.provider.provider_id,preparation.provider.profile_ref,preparation.scope.account_ref,preparation.scope.namespace_ref,before.attempt_count,before.dispatch_epoch)
        proof=_capture(lambda:self._quiescence.obtain(request,now_epoch=now_epoch))
        failed=None;returned=None
        try:returned=self._repo.prove_quiescence(effect_id,proof,now_epoch=now_epoch)
        except FoundationError as error:failed=FoundationError(error.code)
        except Exception:failed=FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        if failed is not None:
            try:converged=self.get(effect_id)
            except FoundationError:converged=None
            if converged==replace(before,dispatch=DispatchState.QUIESCENCE_PROVEN):return converged
            raise failed
        after=self._durable(effect_id,returned)
        if after!=replace(before,dispatch=DispatchState.QUIESCENCE_PROVEN):raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        return after


__all__=["ComposedEffectFoundationV1","FoundationRecordAssessmentAuthorityV1","QuiescenceRecoveryRequestV1"]
