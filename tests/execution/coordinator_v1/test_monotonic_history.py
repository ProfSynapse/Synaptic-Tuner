from dataclasses import replace
import pytest
from tuner.execution.coordinator_v1.model import ProviderRunPhaseV1, WorkflowPhaseV1
from tuner.execution.coordinator_v1.state_machine import WorkflowTransitionError, apply_provider_observation
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.repository import EffectState, ReconciliationOwnershipV2
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
 genesis=ReconciliationOwnershipV2("owner-a",1,1,10,"1"*64,"grant-r1")
 foundation=replace(foundation,reconciliation=genesis,reconciliation_claims=(genesis,))
 current=apply_stage_effect_record(current,foundation,Auth())
 interrupted=replace(genesis,active=False); foundation=replace(foundation,reconciliation=interrupted,reconciliation_claims=(interrupted,));current=apply_stage_effect_record(current,foundation,Auth())
 resumed=replace(interrupted,active=True,grant_ref="grant-r2"); foundation=replace(foundation,reconciliation=resumed,reconciliation_claims=(resumed,));current=apply_stage_effect_record(current,foundation,Auth())
 interrupted=replace(resumed,active=False); foundation=replace(foundation,reconciliation=interrupted,reconciliation_claims=(interrupted,));current=apply_stage_effect_record(current,foundation,Auth())
 transferred=ReconciliationOwnershipV2("owner-b",1,2,11,"2"*64,"grant-r3");foundation=replace(foundation,reconciliation=transferred,reconciliation_claims=(interrupted,transferred));current=apply_stage_effect_record(current,foundation,Auth())
 completed=replace(transferred,active=False,completed=True);foundation=replace(foundation,reconciliation=completed,reconciliation_claims=(interrupted,completed));current=apply_stage_effect_record(current,foundation,Auth())
 retried=ReconciliationOwnershipV2("owner-b",2,3,12,"2"*64,"grant-r4");foundation=replace(foundation,reconciliation=retried,reconciliation_claims=(interrupted,completed,retried));current=apply_stage_effect_record(current,foundation,Auth())
 assert current.stage.foundation_bindings[-1].active_reconciliation_claim_digest==retried.claim_digest
