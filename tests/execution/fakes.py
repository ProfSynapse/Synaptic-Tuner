from __future__ import annotations
from dataclasses import replace
from datetime import datetime,timezone
from threading import RLock
from tuner.execution.contracts import *
from tuner.execution.contracts import _AttemptAdmission,_AttemptDisposition
from tuner.execution.lifecycle import apply_event

class InMemoryLifecycleRepository:
    def __init__(self,clock=lambda:"2026-08-25T12:00:00Z"):
        self.records={};self.order=[];self.consumed=set();self.lock=RLock();self.clock=clock
    def create(self,r):
        with self.lock:self.records[(r.project_ref,r.run_id)]=r;self.order.append((r.project_ref,r.run_id));return r
    def load(self,p,r):return self.records.get((p,r))
    def append(self,p,r,*,expected_revision,event):
        with self.lock:
            old=self.records[(p,r)]
            if old.revision!=expected_revision:raise RevisionConflict("revision")
            new=apply_event(old,event);self.records[(p,r)]=new;return new
    def compare_and_consume_attempt(self,p,r,*,expected_revision,grant_ref,canonical_command):
        with self.lock:
            old=self.records[(p,r)];cmd=canonical_command;raw=cmd.canonical_bytes
            for e in old.effects:
                if e.identity.effect_id==cmd.effect.effect_id or (e.identity.kind is cmd.effect.kind and e.identity.effect_key==cmd.effect.effect_key):
                    if e.command_digest==cmd.digest and e.canonical_command==raw:return _AttemptAdmission(old,e,_AttemptDisposition.LOOKUP_ONLY)
                    raise EffectCollision("canonical command collision")
            if old.revision!=expected_revision:raise RevisionConflict("revision")
            g=old.grant_binding
            now_text=self.clock();now=datetime.fromisoformat(now_text.replace("Z","+00:00"))
            if g is None or g.grant_ref!=grant_ref or grant_ref in self.consumed or g.project_ref!=p or g.operation_key!=cmd.effect.effect_key or g.effect_kind is not cmd.effect.kind or g.scope!=cmd.effect.scope or g.plan_fingerprint!=cmd.plan_fingerprint or g.source_digest!=cmd.source_digest or g.workload_digest!=cmd.workload_digest or g.artifact_slot_ref!=cmd.artifact_slot_ref or g.allowed_secret_refs_digest!=cmd.allowed_secret_refs_digest or g.quote_digest!=cmd.quote_digest or g.resource_digest!=cmd.resource_digest or g.operation_binding_digest!=cmd.operation_binding_digest or not datetime.fromisoformat(g.issued_at.replace("Z","+00:00"))<=now<datetime.fromisoformat(g.expires_at.replace("Z","+00:00")):raise AuthorizationMismatch("grant mismatch or expired")
            if cmd.effect.kind is EffectKind.CANCEL and not any(e.identity.kind is EffectKind.SUBMIT and e.state is EffectState.FOUND and e.provider_job_ref==cmd.target_provider_job_ref for e in old.effects):raise AuthorizationMismatch("unconfirmed cancel target")
            effect=EffectRecord(cmd.effect,g.fingerprint,EffectState.ATTEMPTED,grant_ref=grant_ref,command_digest=cmd.digest,canonical_command=raw,attempt_count=1)
            event=LifecycleEvent(EventCode.EFFECT_ATTEMPTED,now_text,MessageCode.EFFECT_MUTATION_ATTEMPTED,effect=effect)
            new=apply_event(old,event);self.records[(p,r)]=new;self.consumed.add(grant_ref);return _AttemptAdmission(new,effect,_AttemptDisposition.EXECUTE_NOW)
    def record_attempt_outcome(self,p,r,*,expected_revision,command_digest,observation):
        with self.lock:
            old=self.records[(p,r)]
            if old.revision!=expected_revision:raise RevisionConflict("revision")
            e=next((x for x in old.effects if x.identity==observation.identity),None)
            if e is None or e.command_digest!=command_digest:raise EffectCollision("outcome binding mismatch")
            wanted={EffectDisposition.FOUND:EffectState.FOUND,EffectDisposition.DEFINITELY_ABSENT:EffectState.DEFINITELY_ABSENT,EffectDisposition.INDETERMINATE:EffectState.INDETERMINATE}[observation.disposition]
            if e.state in {EffectState.FOUND,EffectState.DEFINITELY_ABSENT}:
                identical=(e.state is wanted and e.provider_job_ref==observation.provider_job_ref and e.receipt_digest==observation.receipt_digest)
                if identical:return old
                raise InvalidTransition("closed outcome conflict")
            if e.state not in {EffectState.ATTEMPTED,EffectState.INDETERMINATE}:raise InvalidTransition("effect is not outcome-recordable")
            updated=replace(e,state=wanted,provider_job_ref=observation.provider_job_ref,receipt_digest=observation.receipt_digest)
            code,msg={EffectState.FOUND:(EventCode.EFFECT_FOUND,MessageCode.EFFECT_CONFIRMED),EffectState.DEFINITELY_ABSENT:(EventCode.EFFECT_DEFINITELY_ABSENT,MessageCode.EFFECT_ABSENT),EffectState.INDETERMINATE:(EventCode.EFFECT_INDETERMINATE,MessageCode.EFFECT_OUTCOME_UNKNOWN)}[wanted]
            event=LifecycleEvent(code,self.clock(),msg,effect=updated);new=apply_event(old,event);self.records[(p,r)]=new;return new
    def list_runs(self,p,*,limit,cursor=None):
        vals=[self.records[k] for k in self.order if k[0]==p];return RunPage(tuple(vals[:limit]))
