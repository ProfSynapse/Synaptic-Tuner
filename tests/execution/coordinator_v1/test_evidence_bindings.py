from dataclasses import replace
import pytest
from tuner.execution.coordinator_v1.model import EffectIntentV1, WorkflowPhaseV1
from tuner.execution.coordinator_v1.state_machine import WorkflowTransitionError
from tuner.execution.foundation_v2.canonical import DiagnosticCode
from tuner.execution.foundation_v2.repository import DispatchState, EffectState, InvalidEvidenceAdmissionV2, ReceiptAdmissionV2, ReceiptFreshnessV2
from tuner.execution.foundation_v2.repository import ReconciliationGrantBindingV2, ReconciliationOwnershipV2
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.receipts import InvalidEvidenceAuthorityV2, InvalidEvidenceContentV2, InvalidEvidenceSiteV2, ReceiptAuthorityV2, ReceiptContentV2
from .test_state_machine import AssessmentAuth, Auth, PROVIDER_RUN, STAGE_REF, _apply_stage, apply_stage_effect_record, assessment, intent, record, stage_source

class CapturingAuth(Auth):
 def __init__(self): super().__init__(); self.command_bytes=None
 def authenticate_grant(self,grant,command_bytes): self.command_bytes=command_bytes; return True

def test_grant_authentication_receives_exact_retained_command_bytes():
 current,raw=stage_source(); authority=CapturingAuth()
 result=apply_stage_effect_record(current,record(raw,EffectState.FOUND,STAGE_REF),authority)
 assert authority.command_bytes is raw.canonical_command_bytes
 assert result.provider_stage_ref.command_bytes_digest==result.stage.foundation_bindings[-1].command_bytes_digest

def test_foundation_assessment_requires_exact_true_authentication():
 current,raw=stage_source();foundation=record(raw,EffectState.FOUND,STAGE_REF)
 with pytest.raises(WorkflowTransitionError,match="assessment authentication"):
  _apply_stage(current,foundation,assessment(foundation),Auth(),AssessmentAuth(False))

def test_forged_record_command_is_rejected_before_reduction():
 current,raw=stage_source(); other=intent("submit")
 forged=record(other,EffectState.FOUND,PROVIDER_RUN)
 with pytest.raises(WorkflowTransitionError,match="command bytes"):
  apply_stage_effect_record(current,forged,Auth())

def test_terminal_and_invalid_code_histories_are_snapshot_bound():
 current,raw=stage_source(); foundation=record(raw,EffectState.FOUND,STAGE_REF)
 with pytest.raises(ValueError):replace(foundation,terminal_content_digests=())
 indeterminate=record(raw,EffectState.INDETERMINATE)
 reconciled=apply_stage_effect_record(current,indeterminate,Auth())
 with pytest.raises(ValueError):replace(indeterminate,invalid_codes=(DiagnosticCode.EVIDENCE_INVALID,))

def test_exact_foundation_replay_after_phase_advancement_is_identity():
 current,raw=stage_source(); foundation=record(raw,EffectState.FOUND,STAGE_REF)
 staged=apply_stage_effect_record(current,foundation,Auth())
 assert apply_stage_effect_record(staged,foundation,Auth()) is staged

@pytest.mark.parametrize("dispatch",[DispatchState.OWNED_NOT_STARTED,DispatchState.OWNED_IN_FLIGHT])
def test_unstable_owned_dispatch_is_rejected(dispatch):
 current,raw=stage_source(); unstable=replace(record(raw,EffectState.INDETERMINATE),dispatch=dispatch)
 with pytest.raises(WorkflowTransitionError,match="not stable"):
  apply_stage_effect_record(current,unstable,Auth())

def test_unproven_final_absence_fails_closed_to_indeterminate():
 current,raw=stage_source(); supplied=record(raw,EffectState.DEFINITELY_ABSENT);admission=replace(supplied.receipt_admissions[0],finality_verified=False,generated_invalid_codes=(DiagnosticCode.FINALITY_UNPROVEN,))
 unproven=replace(supplied,state=EffectState.INDETERMINATE,receipt_admissions=(admission,),terminal_content_digests=(),invalid_codes=(DiagnosticCode.FINALITY_UNPROVEN,))
 result=apply_stage_effect_record(current,unproven,Auth())
 assert result.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
 assert result.stage.foundation_bindings[-1].invalid_codes==("finality_unproven",)

def test_stale_dispatch_found_is_retained_but_does_not_become_found_state():
 current,raw=stage_source(); unresolved=record(raw,EffectState.INDETERMINATE);base=replace(unresolved,results=(),receipt_admissions=(),state=EffectState.UNRESOLVED)
 g1=ReconciliationGrantBindingV2("grant-r1","4"*64,0,None);old=ReconciliationOwnershipV2("owner-a",1,1,10,"1"*64,"grant-r1","4"*64,(g1,),active=False)
 g2=ReconciliationGrantBindingV2("grant-r2","5"*64,0,None);active=ReconciliationOwnershipV2("owner-b",1,2,11,"2"*64,"grant-r2","5"*64,(g2,))
 content=ReceiptContentV2(raw.effect_id,raw.command_digest,ObservationDisposition.FOUND,"3"*64,1,STAGE_REF,None,None,None,"dispatch",base.grant.content.grant_ref,1,1,base.dispatch_source_digest,base.grant.content.grant_ref,base.grant.authenticated_grant_digest)
 receipt=ReceiptAuthorityV2("receipt-authority",b"r"*32).issue(content)
 admission=ReceiptAdmissionV2(receipt.authenticated_receipt_digest,"dispatch",base.grant.content.grant_ref,1,1,base.dispatch_source_digest,base.grant.content.grant_ref,base.grant.authenticated_grant_digest,"reconciliation",active.owner_ref,active.generation,active.ownership_epoch,active.claim_digest,active.grant_ref,active.grant_digest,ReceiptFreshnessV2.STALE,False,(DiagnosticCode.STALE_RESULT,))
 stale=replace(base,state=EffectState.INDETERMINATE,results=(receipt,),receipt_admissions=(admission,),terminal_content_digests=(content.semantic_digest,),invalid_codes=(DiagnosticCode.STALE_RESULT,),reconciliation=active,reconciliation_claims=(old,active))
 result=apply_stage_effect_record(current,stale,Auth())
 assert result.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
 assert result.stage.foundation_bindings[-1].terminal_content_digests==(content.semantic_digest,)

def test_stale_same_semantic_found_cannot_replace_fresh_bound_receipt():
 from tuner.execution.coordinator_v1.model import AuthenticatedFoundationRecordAssessmentV1, ReceiptAssessmentV1, ReceiptFreshnessV1
 current,raw=stage_source();base=record(raw,EffectState.FOUND,STAGE_REF);fresh=base.results[0]
 content=replace(fresh.content,observation_digest="4"*64)
 stale=ReceiptAuthorityV2("receipt-authority",b"s"*32).issue(content)
 binding=ReconciliationGrantBindingV2("grant-r1","4"*64,1,None);claim=ReconciliationOwnershipV2("owner-a",1,1,10,"1"*64,"grant-r1","4"*64,(binding,),active=False,completed=True)
 second_admission=ReceiptAdmissionV2(stale.authenticated_receipt_digest,"dispatch",base.grant.content.grant_ref,1,1,base.dispatch_source_digest,base.grant.content.grant_ref,base.grant.authenticated_grant_digest,"reconciliation",claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest,claim.grant_ref,claim.grant_digest,ReceiptFreshnessV2.STALE,False,(DiagnosticCode.STALE_RESULT,))
 foundation=replace(base,results=(fresh,stale),receipt_admissions=base.receipt_admissions+(second_admission,),invalid_codes=(DiagnosticCode.STALE_RESULT,),reconciliation=claim,reconciliation_claims=(claim,))
 supplied=assessment(foundation);first,second=supplied.content.receipt_assessments
 assessed=replace(supplied.content,receipt_assessments=(ReceiptAssessmentV1(first.authenticated_receipt_digest,first.receipt_admission_digest,first.source_kind,first.source_owner_ref,first.source_generation,first.source_ownership_epoch,ReceiptFreshnessV1.FRESH,False,()),ReceiptAssessmentV1(second.authenticated_receipt_digest,second.receipt_admission_digest,second.source_kind,second.source_owner_ref,second.source_generation,second.source_ownership_epoch,ReceiptFreshnessV1.STALE,False,("stale_result",))))
 envelope=AuthenticatedFoundationRecordAssessmentV1.parse(AuthenticatedFoundationRecordAssessmentV1(assessed,"authority-a","key-a","c"*64).canonical_bytes)
 result=_apply_stage(current,foundation,envelope,Auth(),AssessmentAuth())
 assert result.provider_stage_ref.authenticated_receipt_digest==fresh.authenticated_receipt_digest
 assert result.stage.foundation_bindings[-1].authenticated_receipt_digests==(fresh.authenticated_receipt_digest,stale.authenticated_receipt_digest)

def test_coordinator_derives_invalid_only_from_authenticated_chained_ledger():
 current,raw=stage_source();base=record(raw,EffectState.INDETERMINATE);grant=base.grant
 content=InvalidEvidenceContentV2(raw.effect_id,raw.command_digest,InvalidEvidenceSiteV2.DISPATCH_OBSERVATION,"dispatch",grant.content.grant_ref,1,base.dispatch_epoch,base.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest,"6"*64)
 evidence=InvalidEvidenceAuthorityV2("invalid-authority",b"i"*32).issue(content)
 admission=InvalidEvidenceAdmissionV2(evidence.authenticated_evidence_digest,1,None,content.site,content.source_kind,content.source_owner_ref,content.source_generation,content.source_ownership_epoch,content.source_claim_digest,content.source_grant_ref,content.source_grant_digest)
 foundation=replace(base,invalid_codes=(DiagnosticCode.EVIDENCE_INVALID,),invalid_evidence=(evidence,),invalid_evidence_admissions=(admission,))
 result=apply_stage_effect_record(current,foundation,Auth())
 assert result.stage.foundation_bindings[-1].invalid_codes==("evidence_invalid",)
 class RejectInvalid(Auth):
  def authenticate_invalid_evidence(self,value):return False
 with pytest.raises(WorkflowTransitionError,match="invalid evidence authentication"):_apply_stage(current,foundation,assessment(foundation),RejectInvalid(),AssessmentAuth())
 with pytest.raises(ValueError):replace(foundation,invalid_evidence_admissions=(replace(admission,sequence=2),))
 duplicate=replace(admission,sequence=2,prior_admission_digest=admission.admission_digest)
 object.__setattr__(foundation,"invalid_codes",(DiagnosticCode.EVIDENCE_INVALID,DiagnosticCode.EVIDENCE_INVALID))
 object.__setattr__(foundation,"invalid_evidence",(evidence,evidence))
 object.__setattr__(foundation,"invalid_evidence_admissions",(admission,duplicate))
 with pytest.raises(WorkflowTransitionError,match="duplicate invalid evidence"):_apply_stage(current,foundation,assessment(foundation),Auth(),AssessmentAuth())
