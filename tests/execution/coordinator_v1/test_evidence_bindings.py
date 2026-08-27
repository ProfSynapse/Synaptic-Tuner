from dataclasses import replace
import pytest
from tuner.execution.coordinator_v1.model import EffectIntentV1, WorkflowPhaseV1
from tuner.execution.coordinator_v1.state_machine import WorkflowTransitionError
from tuner.execution.foundation_v2.canonical import DiagnosticCode
from tuner.execution.foundation_v2.repository import DispatchState, EffectState
from tuner.execution.foundation_v2.repository import ReconciliationOwnershipV2
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.receipts import ReceiptAuthorityV1, ReceiptContentV1
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
 with pytest.raises(WorkflowTransitionError,match="terminal/state"):
  apply_stage_effect_record(current,replace(foundation,terminal_content_digests=()),Auth())
 indeterminate=record(raw,EffectState.INDETERMINATE)
 reconciled=apply_stage_effect_record(current,indeterminate,Auth())
 advanced=replace(indeterminate,invalid_codes=(DiagnosticCode.EVIDENCE_INVALID,))
 again=apply_stage_effect_record(reconciled,advanced,Auth())
 assert again.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
 assert again.stage.foundation_bindings[-1].invalid_codes==("evidence_invalid",)

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
 current,raw=stage_source(); supplied=record(raw,EffectState.DEFINITELY_ABSENT)
 unproven=replace(supplied,state=EffectState.INDETERMINATE,terminal_content_digests=(),invalid_codes=(DiagnosticCode.FINALITY_UNPROVEN,))
 result=apply_stage_effect_record(current,unproven,Auth())
 assert result.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
 assert result.stage.foundation_bindings[-1].invalid_codes==("finality_unproven",)

def test_stale_dispatch_found_is_retained_but_does_not_become_found_state():
 current,raw=stage_source(); base=record(raw,EffectState.INDETERMINATE)
 old=ReconciliationOwnershipV2("owner-a",1,1,10,"1"*64,"grant-r1",active=False)
 active=ReconciliationOwnershipV2("owner-b",1,2,11,"2"*64,"grant-r2")
 content=ReceiptContentV1(raw.effect_id,raw.command_digest,ObservationDisposition.FOUND,"3"*64,1,STAGE_REF,None,None,None,"dispatch",base.grant.content.grant_ref,1,1,base.dispatch_source_digest)
 receipt=ReceiptAuthorityV1("receipt-authority",b"r"*32).issue(content)
 stale=replace(base,state=EffectState.INDETERMINATE,results=(receipt,),terminal_content_digests=(content.semantic_digest,),invalid_codes=(DiagnosticCode.STALE_RESULT,),reconciliation=active,reconciliation_claims=(old,active))
 result=apply_stage_effect_record(current,stale,Auth())
 assert result.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
 assert result.stage.foundation_bindings[-1].terminal_content_digests==(content.semantic_digest,)

def test_stale_same_semantic_found_cannot_replace_fresh_bound_receipt():
 from tuner.execution.coordinator_v1.model import AuthenticatedFoundationRecordAssessmentV1, ReceiptAssessmentV1, ReceiptFreshnessV1
 current,raw=stage_source();base=record(raw,EffectState.FOUND,STAGE_REF);fresh=base.results[0]
 content=replace(fresh.content,observation_digest="4"*64)
 stale=ReceiptAuthorityV1("receipt-authority",b"s"*32).issue(content)
 foundation=replace(base,results=(fresh,stale),invalid_codes=(DiagnosticCode.STALE_RESULT,))
 supplied=assessment(foundation);first,second=supplied.content.receipt_assessments
 assessed=replace(supplied.content,receipt_assessments=(ReceiptAssessmentV1(first.authenticated_receipt_digest,first.source_kind,first.source_owner_ref,first.source_generation,first.source_ownership_epoch,ReceiptFreshnessV1.FRESH,False,()),ReceiptAssessmentV1(second.authenticated_receipt_digest,second.source_kind,second.source_owner_ref,second.source_generation,second.source_ownership_epoch,ReceiptFreshnessV1.STALE,False,("stale_result",))))
 envelope=AuthenticatedFoundationRecordAssessmentV1.parse(AuthenticatedFoundationRecordAssessmentV1(assessed,"authority-a","key-a","c"*64).canonical_bytes)
 result=_apply_stage(current,foundation,envelope,Auth(),AssessmentAuth())
 assert result.provider_stage_ref.authenticated_receipt_digest==fresh.authenticated_receipt_digest
 assert result.stage.foundation_bindings[-1].authenticated_receipt_digests==(fresh.authenticated_receipt_digest,stale.authenticated_receipt_digest)
