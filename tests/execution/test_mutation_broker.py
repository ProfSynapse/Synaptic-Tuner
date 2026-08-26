from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pytest
from tuner.execution._effect_executor import _ProviderEffectExecutor
from tuner.execution.broker import MutationBroker,MutationCommandV1
from tuner.execution.contracts import *
from tuner.execution.operation import ModalStageTargetV1,OperationBindingV1
from tuner.execution.service import LifecycleService
from tests.execution.fakes import InMemoryLifecycleRepository
D="a"*64;NOW="2026-08-25T12:00:00Z";LATER="2026-08-25T13:00:00Z"
def ident(kind=EffectKind.SUBMIT,key="op",eid="e"):return EffectIdentity(eid,key,kind,ExecutionScope("modal","acct","env"))
def command(kind=EffectKind.SUBMIT,target=None,nonce="nonce",key="op",eid="e"):
    operation=OperationBindingV1(project_ref="p",run_id="r",effect=ident(kind,key,eid),grant_ref="g-"+key,plan_fingerprint=D,execution_source_digest=D,workload_digest=D,deployment_attestation_digest=D,artifact_contract_digest=D,log_policy_digest=D,invocation_intent_digest=D,resource_digest=D,quote_digest=D,secret_requirements_digest=D,invocation_arguments_digest=D,invocation_nonce=nonce,stage_target=ModalStageTargetV1("slot","cv","av",f"operations/{eid}/output",1,"key"),target_provider_job_ref=target)
    return MutationCommandV1(operation,D,D)
def grant(c,**changes):
    operation=replace(c.operation,**changes) if changes else c.operation
    return GrantBinding.from_operation(operation,issued_at=NOW,expires_at=LATER)
class Driver:
    def __init__(self,crash=False,observed_effect=None,disposition=EffectDisposition.FOUND):self.calls=0;self.crash=crash;self.observed_effect=observed_effect or command().effect;self.disposition=disposition
    def execute_once(self,raw):
        self.calls+=1
        if self.crash:raise RuntimeError("crash")
        return EffectObservation(self.observed_effect,self.disposition,"fc-1",D) if self.disposition is EffectDisposition.FOUND else EffectObservation(self.observed_effect,self.disposition)
def ready(c,repo=None):
    repo=repo or InMemoryLifecycleRepository();svc=LifecycleService(repo,clock=lambda:NOW);r=svc.plan(project_ref="p",run_id="r");r=svc.authorize(project_ref="p",run_id="r",expected_revision=r.revision,binding=grant(c));return repo,r
def test_sequential_and_concurrent_replay_execute_once():
    c=command();repo,r=ready(c);d=Driver();b=MutationBroker(repo,_ProviderEffectExecutor(d));b.execute(c,expected_revision=r.revision);b.execute(c,expected_revision=r.revision+2);assert d.calls==1
    c2=command(nonce="n2",key="op2",eid="e2");repo2,r2=ready(c2);d2=Driver(observed_effect=c2.effect);b2=MutationBroker(repo2,_ProviderEffectExecutor(d2))
    with ThreadPoolExecutor(max_workers=2) as x:list(x.map(lambda _:b2.execute(c2,expected_revision=r2.revision),range(2)))
    assert d2.calls==1
def test_same_key_different_command_collides():
    c=command();repo,r=ready(c);d=Driver();b=MutationBroker(repo,_ProviderEffectExecutor(d));b.execute(c,expected_revision=r.revision)
    with pytest.raises(EffectCollision):b.execute(command(nonce="different"),expected_revision=r.revision+2)
def test_crash_windows_never_retry():
    c=command();repo,r=ready(c);d=Driver(crash=True);b=MutationBroker(repo,_ProviderEffectExecutor(d))
    with pytest.raises(RuntimeError):b.execute(c,expected_revision=r.revision)
    d.crash=False;obs=b.execute(c,expected_revision=r.revision+1);assert obs.disposition is EffectDisposition.INDETERMINATE and d.calls==1
def test_expired_and_mismatched_grants_rejected():
    c=command();repo,r=ready(c,InMemoryLifecycleRepository(clock=lambda:LATER));d=Driver()
    with pytest.raises(AuthorizationMismatch):MutationBroker(repo,_ProviderEffectExecutor(d)).execute(c,expected_revision=r.revision)
    assert d.calls==0
def test_all_pre_stage_execution_fields_are_authorized_by_operation_binding():
    base=command()
    mutations=[
        dict(plan_fingerprint="b"*64),dict(deployment_attestation_digest="b"*64),
        dict(invocation_arguments_digest="b"*64),dict(invocation_nonce="other"),
        dict(execution_source_digest="b"*64),dict(workload_digest="b"*64),
        dict(secret_requirements_digest="b"*64),
        dict(quote_digest="b"*64),dict(resource_digest="b"*64),
    ]
    for changes in mutations:
        repo,r=ready(base);altered=replace(base,operation=replace(base.operation,**changes));driver=Driver()
        with pytest.raises(AuthorizationMismatch):MutationBroker(repo,_ProviderEffectExecutor(driver)).execute(altered,expected_revision=r.revision)
        assert driver.calls==0
    repo,r=ready(base);altered=replace(base,operation=replace(base.operation,stage_target=replace(base.operation.stage_target,artifact_slot_ref="other")));driver=Driver()
    with pytest.raises(AuthorizationMismatch):MutationBroker(repo,_ProviderEffectExecutor(driver)).execute(altered,expected_revision=r.revision)
    assert driver.calls==0
def test_final_command_digest_binds_bundle_and_stage_claim_without_a_fixed_point():
    base=command();altered=replace(base,bundle_digest="b"*64,stage_claim_digest="c"*64)
    assert base.operation_binding_digest==altered.operation_binding_digest
    assert base.digest!=altered.digest
    assert base.stage_claim_digest not in base.operation_binding_digest
    assert altered.stage_claim_digest not in altered.operation_binding_digest
def test_repository_clock_once_and_issue_expiry_boundaries():
    c=command();calls=[]
    def clock():calls.append(NOW);return NOW
    repo,r=ready(c,InMemoryLifecycleRepository(clock=clock));repo.compare_and_consume_attempt("p","r",expected_revision=r.revision,grant_ref=c.grant_ref,canonical_command=c);assert calls==[NOW]
    before="2026-08-25T11:59:59Z";repo2,r2=ready(c,InMemoryLifecycleRepository(clock=lambda:before));driver=Driver()
    with pytest.raises(AuthorizationMismatch):MutationBroker(repo2,_ProviderEffectExecutor(driver)).execute(c,expected_revision=r2.revision)
def test_outcomes_persist_exactly_and_closed_conflicts_reject():
    for disposition,state in [(EffectDisposition.FOUND,EffectState.FOUND),(EffectDisposition.DEFINITELY_ABSENT,EffectState.DEFINITELY_ABSENT),(EffectDisposition.INDETERMINATE,EffectState.INDETERMINATE)]:
        c=command(nonce=disposition.value);repo,r=ready(c);driver=Driver(observed_effect=c.effect,disposition=disposition);broker=MutationBroker(repo,_ProviderEffectExecutor(driver));assert broker.execute(c,expected_revision=r.revision).disposition is disposition;effect=repo.load("p","r").effects[0];assert effect.state is state
        if state in {EffectState.FOUND,EffectState.DEFINITELY_ABSENT}:
            current=repo.load("p","r");same=EffectObservation(c.effect,disposition,"fc-1",D) if disposition is EffectDisposition.FOUND else EffectObservation(c.effect,disposition);assert repo.record_attempt_outcome("p","r",expected_revision=current.revision,command_digest=c.digest,observation=same) is current
            conflict=EffectObservation(c.effect,EffectDisposition.INDETERMINATE)
            with pytest.raises(InvalidTransition):repo.record_attempt_outcome("p","r",expected_revision=current.revision,command_digest=c.digest,observation=conflict)
def test_indeterminate_reconciles_without_executor_retry():
    c=command();repo,r=ready(c);driver=Driver(observed_effect=c.effect,disposition=EffectDisposition.INDETERMINATE);MutationBroker(repo,_ProviderEffectExecutor(driver)).execute(c,expected_revision=r.revision);current=repo.load("p","r");repo.record_attempt_outcome("p","r",expected_revision=current.revision,command_digest=c.digest,observation=EffectObservation(c.effect,EffectDisposition.FOUND,"fc-late",D));assert driver.calls==1
def test_outcome_requires_exact_revision_effect_and_command():
    c=command();repo,r=ready(c);admission=repo.compare_and_consume_attempt("p","r",expected_revision=r.revision,grant_ref=c.grant_ref,canonical_command=c);obs=EffectObservation(c.effect,EffectDisposition.INDETERMINATE)
    with pytest.raises(RevisionConflict):repo.record_attempt_outcome("p","r",expected_revision=r.revision,command_digest=c.digest,observation=obs)
    with pytest.raises(EffectCollision):repo.record_attempt_outcome("p","r",expected_revision=admission.record.revision,command_digest="b"*64,observation=obs)
    wrong=EffectObservation(ident(eid="wrong"),EffectDisposition.INDETERMINATE)
    with pytest.raises(EffectCollision):repo.record_attempt_outcome("p","r",expected_revision=admission.record.revision,command_digest=c.digest,observation=wrong)
def test_cancel_requires_exact_confirmed_target():
    submit=command();repo,r=ready(submit);d=Driver();MutationBroker(repo,_ProviderEffectExecutor(d)).execute(submit,expected_revision=r.revision)
    svc=LifecycleService(repo,clock=lambda:NOW);current=repo.load("p","r");current=svc.record_provider_phase(project_ref="p",run_id="r",expected_revision=current.revision,provider_phase=ProviderRunPhase.RUNNING)
    c=command(EffectKind.CANCEL,"fc-other",key="cancel",eid="ce");current=svc.authorize(project_ref="p",run_id="r",expected_revision=current.revision,binding=grant(c))
    with pytest.raises(AuthorizationMismatch):MutationBroker(repo,_ProviderEffectExecutor(d)).execute(c,expected_revision=current.revision)
    assert d.calls==1
def test_exact_target_cancel_executes_once():
    submit=command();repo,r=ready(submit);MutationBroker(repo,_ProviderEffectExecutor(Driver())).execute(submit,expected_revision=r.revision)
    svc=LifecycleService(repo,clock=lambda:NOW);current=repo.load("p","r");current=svc.record_provider_phase(project_ref="p",run_id="r",expected_revision=current.revision,provider_phase=ProviderRunPhase.RUNNING)
    c=command(EffectKind.CANCEL,"fc-1",key="cancel",eid="ce");current=svc.authorize(project_ref="p",run_id="r",expected_revision=current.revision,binding=grant(c));driver=Driver(observed_effect=c.effect)
    assert MutationBroker(repo,_ProviderEffectExecutor(driver)).execute(c,expected_revision=current.revision).disposition is EffectDisposition.FOUND
    assert driver.calls==1
def test_public_provider_has_no_mutation_surface():
    import tuner.execution.providers.modal as modal
    assert "submit" not in modal.__all__ and "cancel" not in modal.__all__
    import tuner.execution as public
    assert not hasattr(public,"MutationPermit") and not hasattr(public,"_ProviderEffectExecutor")
