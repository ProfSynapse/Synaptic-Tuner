from dataclasses import replace
import pytest
from tuner.execution.coordinator_v1.model import ProviderRunPhaseV1, WorkflowPhaseV1
from tuner.execution.coordinator_v1.state_machine import WorkflowTransitionError, apply_provider_observation
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.repository import EffectState, ReconciliationGrantBindingV2, ReconciliationOwnershipV2
from .test_state_machine import Auth, apply_stage_effect_record, record, stage_source
from .test_state_machine import ObservationAuth, observation, queued_evidence

def test_observation_replay_is_identity_and_revision_only_tracks_new_evidence():
 current,foundation=queued_evidence(); request,envelope=observation(current,foundation,ProviderRunPhaseV1.RUNNING,{"phase":"running"})
 running=apply_provider_observation(current,request,envelope,ObservationAuth())
 assert running.revision==current.revision+1
 assert apply_provider_observation(running,request,envelope,ObservationAuth()) is running

def test_running_to_queued_regression_is_closed():
 current,foundation=queued_evidence();request,envelope=observation(current,foundation,ProviderRunPhaseV1.RUNNING,{"phase":"running"});running=apply_provider_observation(current,request,envelope,ObservationAuth())
 request,queued_observation=observation(running,foundation,ProviderRunPhaseV1.QUEUED,{"phase":"queued"})
 with pytest.raises(WorkflowTransitionError): apply_provider_observation(running,request,queued_observation,ObservationAuth())
 assert running.phase is WorkflowPhaseV1.RUNNING

def test_reconciliation_interrupt_resume_transfer_completion_retry_history():
 current,raw=stage_source(); foundation=record(raw,EffectState.INDETERMINATE)
 g1=ReconciliationGrantBindingV2("grant-r1","3"*64,1,None);genesis=ReconciliationOwnershipV2("owner-a",1,1,10,"1"*64,"grant-r1","3"*64,(g1,))
 foundation=replace(foundation,reconciliation=genesis,reconciliation_claims=(genesis,))
 current=apply_stage_effect_record(current,foundation,Auth())
 interrupted=replace(genesis,active=False); foundation=replace(foundation,reconciliation=interrupted,reconciliation_claims=(interrupted,));current=apply_stage_effect_record(current,foundation,Auth())
 g2=ReconciliationGrantBindingV2("grant-r2","4"*64,1,g1.binding_digest);resumed=replace(interrupted,active=True,grant_ref="grant-r2",grant_digest="4"*64,grant_lineage=(g1,g2)); foundation=replace(foundation,reconciliation=resumed,reconciliation_claims=(resumed,));current=apply_stage_effect_record(current,foundation,Auth())
 interrupted=replace(resumed,active=False); foundation=replace(foundation,reconciliation=interrupted,reconciliation_claims=(interrupted,));current=apply_stage_effect_record(current,foundation,Auth())
 g3=ReconciliationGrantBindingV2("grant-r3","5"*64,1,None);transferred=ReconciliationOwnershipV2("owner-b",1,2,11,"2"*64,"grant-r3","5"*64,(g3,));foundation=replace(foundation,reconciliation=transferred,reconciliation_claims=(interrupted,transferred));current=apply_stage_effect_record(current,foundation,Auth())
 completed=replace(transferred,active=False,completed=True);foundation=replace(foundation,reconciliation=completed,reconciliation_claims=(interrupted,completed));current=apply_stage_effect_record(current,foundation,Auth())
 g4=ReconciliationGrantBindingV2("grant-r4","6"*64,1,None);retried=ReconciliationOwnershipV2("owner-b",2,3,12,"2"*64,"grant-r4","6"*64,(g4,));foundation=replace(foundation,reconciliation=retried,reconciliation_claims=(interrupted,completed,retried));current=apply_stage_effect_record(current,foundation,Auth())
 assert current.stage.foundation_bindings[-1].active_reconciliation_claim_digest==retried.claim_digest

def test_resume_rejects_structurally_valid_rewritten_retained_grant_lineage():
 current,raw=stage_source();foundation=record(raw,EffectState.INDETERMINATE)
 original=ReconciliationGrantBindingV2("grant-r1","3"*64,1,None);genesis=ReconciliationOwnershipV2("owner-a",1,1,10,"1"*64,"grant-r1","3"*64,(original,))
 foundation=replace(foundation,reconciliation=genesis,reconciliation_claims=(genesis,));current=apply_stage_effect_record(current,foundation,Auth())
 interrupted=replace(genesis,active=False);foundation=replace(foundation,reconciliation=interrupted,reconciliation_claims=(interrupted,));current=apply_stage_effect_record(current,foundation,Auth())
 rewritten=ReconciliationGrantBindingV2("rewritten-r1","7"*64,1,None);leaf=ReconciliationGrantBindingV2("grant-r2","4"*64,1,rewritten.binding_digest)
 malicious=replace(interrupted,active=True,grant_ref=leaf.grant_ref,grant_digest=leaf.grant_digest,grant_lineage=(rewritten,leaf))
 structurally_valid=replace(foundation,reconciliation=malicious,reconciliation_claims=(malicious,))
 with pytest.raises(WorkflowTransitionError,match="append exactly one"):apply_stage_effect_record(current,structurally_valid,Auth())
