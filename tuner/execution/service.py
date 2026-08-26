"""Read/lifecycle recording service; provider mutation belongs to MutationBroker."""
from __future__ import annotations
from dataclasses import replace
from typing import Callable
from .contracts import *
from .lifecycle import initial_record

class LifecycleService:
    def __init__(self,repository:LifecycleRepository,*,clock:Callable[[],str]):
        if not isinstance(repository,LifecycleRepository):raise TypeError("repository must implement LifecycleRepository")
        self._repository=repository;self._clock=clock
    def _now(self):return timestamp(self._clock(),"clock")
    def plan(self,*,project_ref,run_id):return self._repository.create(initial_record(project_ref=project_ref,run_id=run_id,occurred_at=self._now()))
    def load(self,*,project_ref,run_id):
        record=self._repository.load(project_ref,run_id)
        if record is None:raise RunNotFound("run was not found")
        return record
    def authorize(self,*,project_ref,run_id,expected_revision,binding):
        if binding.project_ref!=project_ref:raise AuthorizationMismatch("grant project mismatch")
        return self._repository.append(project_ref,run_id,expected_revision=expected_revision,event=LifecycleEvent(EventCode.AUTHORITY_ACCEPTED,self._now(),MessageCode.AUTHORITY_BOUND,grant_binding=binding))
    def begin_preparation(self,*,project_ref,run_id,expected_revision):return self._append(project_ref,run_id,expected_revision,EventCode.PREPARATION_STARTED,MessageCode.PREPARING)
    def complete_preparation(self,*,project_ref,run_id,expected_revision):return self._append(project_ref,run_id,expected_revision,EventCode.PREPARATION_COMPLETED,MessageCode.READY)
    def record_effect_observation(self,*,project_ref,run_id,expected_revision,observation):
        record=self.load(project_ref=project_ref,run_id=run_id);claimed=next((e for e in record.effects if e.identity==observation.identity),None)
        if claimed is None:raise RunNotFound("effect was not found")
        if claimed.state in {EffectState.FOUND,EffectState.DEFINITELY_ABSENT}:return record
        state,code,msg={EffectDisposition.FOUND:(EffectState.FOUND,EventCode.EFFECT_FOUND,MessageCode.EFFECT_CONFIRMED),EffectDisposition.DEFINITELY_ABSENT:(EffectState.DEFINITELY_ABSENT,EventCode.EFFECT_DEFINITELY_ABSENT,MessageCode.EFFECT_ABSENT),EffectDisposition.INDETERMINATE:(EffectState.INDETERMINATE,EventCode.EFFECT_INDETERMINATE,MessageCode.EFFECT_OUTCOME_UNKNOWN)}[observation.disposition]
        effect=replace(claimed,state=state,provider_job_ref=observation.provider_job_ref,receipt_digest=observation.receipt_digest)
        return self._repository.append(project_ref,run_id,expected_revision=expected_revision,event=LifecycleEvent(code,self._now(),msg,effect=effect))
    def record_provider_phase(self,*,project_ref,run_id,expected_revision,provider_phase):
        code={ProviderRunPhase.QUEUED:EventCode.PROVIDER_QUEUED,ProviderRunPhase.RUNNING:EventCode.PROVIDER_RUNNING,ProviderRunPhase.SUCCEEDED:EventCode.PROVIDER_SUCCEEDED,ProviderRunPhase.FAILED:EventCode.PROVIDER_FAILED,ProviderRunPhase.CANCELLED:EventCode.PROVIDER_CANCELLED,ProviderRunPhase.UNKNOWN:EventCode.PROVIDER_UNKNOWN}[provider_phase]
        msg=MessageCode.SEMANTIC_VERIFICATION_PENDING if provider_phase is ProviderRunPhase.SUCCEEDED else MessageCode.PROVIDER_STATE_OBSERVED
        return self._append(project_ref,run_id,expected_revision,code,msg)
    def record_verification(self,*,project_ref,run_id,expected_revision,verification):
        code,msg={VerificationStatus.VERIFYING:(EventCode.VERIFICATION_STARTED,MessageCode.SEMANTIC_VERIFICATION_STARTED),VerificationStatus.VERIFIED:(EventCode.VERIFICATION_VERIFIED,MessageCode.SEMANTIC_VERIFICATION_PASSED),VerificationStatus.INVALID:(EventCode.VERIFICATION_INVALID,MessageCode.SEMANTIC_VERIFICATION_FAILED),VerificationStatus.INCONCLUSIVE:(EventCode.VERIFICATION_INCONCLUSIVE,MessageCode.SEMANTIC_VERIFICATION_INCONCLUSIVE)}[verification]
        return self._append(project_ref,run_id,expected_revision,code,msg)
    def reopen_verification(self,*,project_ref,run_id,expected_revision):return self._append(project_ref,run_id,expected_revision,EventCode.VERIFICATION_REOPENED,MessageCode.SEMANTIC_VERIFICATION_REOPENED)
    def list_runs(self,*,project_ref,limit=50,cursor=None):return self._repository.list_runs(project_ref,limit=limit,cursor=cursor)
    def _append(self,p,r,v,c,m):return self._repository.append(p,r,expected_revision=v,event=LifecycleEvent(c,self._now(),m))
__all__=["LifecycleService"]
