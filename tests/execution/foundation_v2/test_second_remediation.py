import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pytest
from tuner.execution.foundation_v2.authority import ReconciliationGrantContentV1
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import DiagnosticCode,FoundationError,canonical_bytes
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1,StageCommandV2,build_stage_command,build_submit_command,parse_exact_command
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ProviderObservationV1,ObservationDisposition
from tuner.execution.foundation_v2.receipts import ReceiptContentV1
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.repository import DispatchState,EffectState
from tuner.execution.foundation_v2.references import ProviderStageRefV1,StagePredecessorV2
from tuner.execution.foundation_v2.lifecycle import LifecyclePhaseV2,LifecycleStateV2,transition
from tuner.execution.foundation_v2.registry import LazyProviderRegistryV2,ProviderRegistrationV2
from synaptic_tuner.api.v1.providers import ProviderCapabilities,ProviderDescriptor,ProviderRef
from .helpers import *

def bound_executor(command,executor):
    original=executor.execute_once
    def call(payload,request):
        result=original(payload,request)
        return replace(result,effect_id=command.operation.effect.effect_id)
    executor.execute_once=call;return executor

def test_sealed_envelopes_are_final_immutable_and_independently_reparsed():
    p=prep();q=payload(EffectKind.STAGE,p);c=stage_command(p)
    for base in (type(p),type(q),type(c)):
        with pytest.raises(TypeError):type("Evil",(base,),{})
    for obj in (p,q,c):
        with pytest.raises(AttributeError):obj._raw=b"{}"
    assert parse_exact_command(c.canonical_bytes).canonical_bytes==c.canonical_bytes

def test_exact_command_rejects_proxy_subclass_and_embedded_tampering():
    c=stage_command();d=c.to_dict();d["preparation"]["workload_digest"]=D[12]
    with pytest.raises(ValueError):parse_exact_command(canonical_bytes(d))
    d=c.to_dict();d["payload"]["payload_kind"]="submit-payload/v2"
    with pytest.raises(ValueError,match="payload schema"):parse_exact_command(canonical_bytes(d))
    with pytest.raises(TypeError):StageCommandV2(c.canonical_bytes,_issuer=object())
    class Proxy:
        canonical_bytes=c.canonical_bytes
    with pytest.raises((TypeError,ValueError)):parse_exact_command(Proxy())

def test_effect_and_operation_are_recomputed_not_caller_supplied():
    c=stage_command();d=c.to_dict();d["effect"]["effect_id"]="stage-"+D[12]
    with pytest.raises(ValueError):parse_exact_command(canonical_bytes(d))
    d=c.to_dict();d["operation"]["source_digest"]=D[12]
    with pytest.raises(ValueError):parse_exact_command(canonical_bytes(d))

def test_grant_reconstructs_command_and_enforces_time_epoch_revocation():
    c=stage_command();_,a,_,_,_=environment(Executor());g=execution_grant(a,c)
    assert a.verify(g,c.canonical_bytes,now_epoch=150)
    assert not a.verify(g,c.canonical_bytes,now_epoch=200)
    a.revoke("grant");assert not a.verify(g,c.canonical_bytes,now_epoch=150)
    other=stage_command(prep(source_digest=D[12]));assert not a.verify(g,other.canonical_bytes,now_epoch=150)

def test_no_per_call_executor_and_broker_owned_reparsed_payload():
    c=stage_command();ex=bound_executor(c,Executor());repo,a,r,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r);g=execution_grant(a,c)
    result=b.execute(c.canonical_bytes,g,now_epoch=150)
    assert result.state is EffectState.FOUND and ex.calls==1
    assert ex.payloads[0] is not c.payload and ex.payloads[0].canonical_bytes==c.payload.canonical_bytes

def test_atomic_one_attempt_and_dispatch_crash_is_orphaned_without_retry():
    c=stage_command();ex=Executor(fail=True);repo,a,r,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r);g=execution_grant(a,c)
    with pytest.raises(FoundationError) as err:b.execute(c.canonical_bytes,g,now_epoch=150)
    assert err.value.code is DiagnosticCode.EFFECT_AMBIGUOUS and "secret" not in str(err.value)
    rec=repo.get(c.operation.effect.effect_id);assert rec.dispatch is DispatchState.ORPHANED_UNPROVEN and rec.attempt_count==1
    b.execute(c.canonical_bytes,g,now_epoch=150);assert ex.calls==1
    with pytest.raises(FoundationError):repo.prove_quiescence(c.operation.effect.effect_id,object(),now_epoch=150)
    proof=v.proof(rec,"quiescent");assert repo.prove_quiescence(c.operation.effect.effect_id,proof,now_epoch=150).dispatch is DispatchState.QUIESCENCE_PROVEN

def test_concurrent_admission_executes_once():
    c=stage_command();ex=bound_executor(c,Executor());repo,a,r,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r);g=execution_grant(a,c)
    with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(lambda _:b.execute(c.canonical_bytes,g,now_epoch=150),range(32)))
    assert ex.calls==1 and repo.get(c.operation.effect.effect_id).attempt_count==1

def test_absence_without_strong_finality_reduces_to_indeterminate_then_conflicts_when_proven():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.DEFINITELY_ABSENT));repo,a,r,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r);g=execution_grant(a,c)
    rec=b.execute(c.canonical_bytes,g,now_epoch=150);assert rec.state is EffectState.INDETERMINATE
    proof=v.proof(rec,"final_absent");obs=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.DEFINITELY_ABSENT,D[11],1,finality_proof=proof);receipt=r.issue(dispatch_receipt_content(c,obs,rec));rec=repo.append_result(c.operation.effect.effect_id,receipt,proof,now_epoch=150);assert rec.state is EffectState.DEFINITELY_ABSENT
    found=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.FOUND,D[11],1,stage_ref=ProviderStageRefV1("docker","local","acct","ns","other-stage"));found_receipt=r.issue(dispatch_receipt_content(c,found,rec));rec=repo.append_result(c.operation.effect.effect_id,found_receipt,None,now_epoch=150);assert rec.state is EffectState.CONTRADICTED and len(rec.results)==3
    assert repo.append_result(c.operation.effect.effect_id,found_receipt,None,now_epoch=150) is rec

def reconciliation_grant(a,c,adapter,owner="owner",epoch=1):
    p=c.preparation;e=c.operation.effect;x=ReconciliationGrantContentV1("reconcile",c.digest,e.effect_id,p.preparation_digest,adapter.digest,p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,owner,1,epoch,D[9],D[10],100,200,a.epoch,a.revocation_generation);return a.issue_reconciliation(x)

def test_reconciliation_only_relinquished_or_quiescence_and_no_duplicate_lookup():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r);g=execution_grant(a,c);b.execute(c.canonical_bytes,g,now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");obs=observation_for(c);adapter=Adapter(obs);service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r);rg=reconciliation_grant(a,c,ad)
    rec=service.reconcile(c.canonical_bytes,rg,now_epoch=150);assert rec.state is EffectState.FOUND and adapter.calls==1
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,rg,now_epoch=151)
    assert error.value.code is DiagnosticCode.EFFECT_INELIGIBLE and adapter.calls==1

def test_reconciliation_interruption_exact_resume_and_concurrent_claim_rejected():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r);g=execution_grant(a,c);b.execute(c.canonical_bytes,g,now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");adapter=Adapter(None,fail=True);service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r);rg=reconciliation_grant(a,c,ad)
    with pytest.raises(FoundationError):service.reconcile(c.canonical_bytes,rg,now_epoch=150)
    claim=repo.get(c.operation.effect.effect_id).reconciliation
    adapter.fail=False;adapter.observation=observation_for(c)
    with pytest.raises(FoundationError):service.reconcile(c.canonical_bytes,rg,now_epoch=151)
    resumed_content=replace(rg.content,grant_ref="resume")
    resumed_grant=a.issue_reconciliation(resumed_content)
    rec=service.reconcile(c.canonical_bytes,resumed_grant,now_epoch=151,resume=claim);assert rec.state is EffectState.FOUND and rec.reconciliation.claimed_at_epoch==150

def test_reconciliation_transfer_requires_authenticated_worker_quiescence():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r);g=execution_grant(a,c);b.execute(c.canonical_bytes,g,now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");adapter=Adapter(None,fail=True);service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r);rg=reconciliation_grant(a,c,ad)
    with pytest.raises(FoundationError):service.reconcile(c.canonical_bytes,rg,now_epoch=150)
    rec=repo.get(c.operation.effect.effect_id)
    transfer_content=replace(rg.content,grant_ref="transfer",owner_ref="new",ownership_epoch=2);transfer_grant=a.issue_reconciliation(transfer_content)
    with pytest.raises(FoundationError):repo.transfer_reconciliation(c.canonical_bytes,transfer_grant,proof=object(),now_epoch=160)
    proof=v.proof(rec,"quiescent");_,claim=repo.transfer_reconciliation(c.canonical_bytes,transfer_grant,proof=proof,now_epoch=160);assert claim.owner_ref=="new" and claim.ownership_epoch==2

def test_submit_repository_cas_requires_exact_durable_authenticated_stage_result():
    stage=stage_command();ex=bound_executor(stage,Executor());repo,a,r,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r);sg=execution_grant(a,stage);stage_record=b.execute(stage.canonical_bytes,sg,now_epoch=150)
    p=stage.preparation;ref=StagePredecessorV2(p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,p.project_ref,p.run_id,p.plan_fingerprint,p.preparation_digest,p.workload_digest,stage.operation.effect.effect_id,stage_record.results[0].authenticated_receipt_digest,stage_record.record_digest);submit=build_submit_command(p,"nonce",payload(EffectKind.SUBMIT,p),descriptor(),ref);sex=bound_executor(submit,Executor());sb=EffectBrokerV2(repo,ExecutorResolver(sex),a,r);grant=execution_grant(a,submit,"submit")
    assert sb.execute(submit.canonical_bytes,grant,now_epoch=150).state is EffectState.FOUND
    other=type(repo)(r,v,v,a);other_broker=EffectBrokerV2(other,ExecutorResolver(sex),a,r)
    with pytest.raises(FoundationError):other_broker.execute(submit.canonical_bytes,grant,now_epoch=150)

def test_strict_lifecycle_kind_matrix_and_verification_only_success():
    cancel=LifecycleStateV2(LifecyclePhaseV2.RECONCILING,EffectKind.CANCEL)
    assert transition(cancel,LifecycleStateV2(LifecyclePhaseV2.CANCELLED)).phase is LifecyclePhaseV2.CANCELLED
    with pytest.raises(ValueError):transition(cancel,LifecycleStateV2(LifecyclePhaseV2.STAGED))
    with pytest.raises(ValueError):transition(LifecycleStateV2(LifecyclePhaseV2.RUNNING),LifecycleStateV2(LifecyclePhaseV2.SUCCEEDED))
    with pytest.raises(ValueError):LifecycleStateV2(LifecyclePhaseV2.SUBMISSION_AMBIGUOUS,EffectKind.CANCEL)
    with pytest.raises(ValueError):transition(LifecycleStateV2(LifecyclePhaseV2.STAGING),LifecycleStateV2(LifecyclePhaseV2.RECONCILE_REQUIRED,EffectKind.CANCEL))
    assert transition(LifecycleStateV2(LifecyclePhaseV2.VERIFYING),LifecycleStateV2(LifecyclePhaseV2.SUCCEEDED)).phase is LifecyclePhaseV2.SUCCEEDED

def test_registry_returns_references_without_invocation():
    calls=[]
    def factory():calls.append(1);raise AssertionError
    provider=ProviderDescriptor("synaptic-provider-descriptor/v1","docker","Docker","1.0.0",ProviderCapabilities(True,True,True,True,True,False));registration=ProviderRegistrationV2(provider,descriptor(),AdapterDescriptorV1("docker","lookup","1.0.0"),factory,factory);registry=LazyProviderRegistryV2();registry.register(registration);ref=ProviderRef("docker","local")
    assert registry.executor_factory(ref) is factory and registry.adapter_factory(ref) is factory and calls==[]
