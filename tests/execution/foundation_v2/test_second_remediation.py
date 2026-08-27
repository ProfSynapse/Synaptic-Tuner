import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pytest
from tuner.execution.foundation_v2.authority import AuthenticatedGrantV2, ReconciliationGrantContentV1
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import DiagnosticCode,FoundationError,canonical_bytes
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1,StageCommandV2,build_stage_command,build_submit_command,parse_exact_command
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ProviderObservationV1,ObservationDisposition
from tuner.execution.foundation_v2.receipts import InvalidEvidenceContentV2, InvalidEvidenceSiteV2, ReceiptContentV2
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.repository import DispatchState,EffectState,ReceiptAdmissionV2,ReceiptFreshnessV2,ReconciliationGrantBindingV2
from tuner.execution.foundation_v2.references import ProviderStageRefV1,StagePredecessorV2
from tuner.execution.foundation_v2.lifecycle import LifecyclePhaseV2,LifecycleStateV2,transition
from tuner.execution.foundation_v2.registry import LazyProviderRegistryV2,ProviderRegistrationV2
from synaptic_tuner.api.v1.providers import ProviderCapabilities,ProviderDescriptor,ProviderRef
from .helpers import *

def assert_closed_error(error,code):
    raised=error.value
    assert raised.code is code and "secret-provider-credential" not in str(raised) and "secret-provider-credential" not in repr(raised)
    assert raised.__cause__ is None and raised.__context__ is None

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
    c=stage_command();_,a,_,invalid,_,_=environment(Executor());g=execution_grant(a,c)
    assert a.verify(g,c.canonical_bytes,now_epoch=150)
    assert not a.verify(g,c.canonical_bytes,now_epoch=200)
    a.revoke("grant");assert not a.verify(g,c.canonical_bytes,now_epoch=150)
    other=stage_command(prep(source_digest=D[12]));assert not a.verify(g,other.canonical_bytes,now_epoch=150)

def test_historical_execution_grant_authentication_binds_command_without_replaying_policy_time():
    c=stage_command();_,a,_,invalid,_,_=environment(Executor());g=execution_grant(a,c)
    assert a.authenticate(g,c.canonical_bytes) is True
    a.revoke(g.content.grant_ref)
    assert a.authenticate(g,c.canonical_bytes) is True and a.verify(g,c.canonical_bytes,now_epoch=250) is False
    assert a.authenticate(replace(g,tag=D[0]),c.canonical_bytes) is False
    altered_content=replace(g.content,policy_digest=D[0]);altered=AuthenticatedGrantV2(altered_content,g.authority_ref,g.tag)
    assert a.authenticate(altered,c.canonical_bytes) is False
    assert a.authenticate(g,stage_command(prep(source_digest=D[12])).canonical_bytes) is False

def test_no_per_call_executor_and_broker_owned_reparsed_payload():
    c=stage_command();ex=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r,invalid);g=execution_grant(a,c)
    result=b.execute(c.canonical_bytes,g,now_epoch=150)
    assert result.state is EffectState.FOUND and ex.calls==1
    assert ex.payloads[0] is not c.payload and ex.payloads[0].canonical_bytes==c.payload.canonical_bytes

def test_atomic_one_attempt_and_dispatch_crash_is_orphaned_without_retry():
    c=stage_command();ex=Executor(fail=True);repo,a,r,invalid,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r,invalid);g=execution_grant(a,c)
    with pytest.raises(FoundationError) as err:b.execute(c.canonical_bytes,g,now_epoch=150)
    assert err.value.code is DiagnosticCode.EFFECT_AMBIGUOUS and "secret" not in str(err.value)
    rec=repo.get(c.operation.effect.effect_id);assert rec.dispatch is DispatchState.ORPHANED_UNPROVEN and rec.attempt_count==1
    b.execute(c.canonical_bytes,g,now_epoch=150);assert ex.calls==1
    with pytest.raises(FoundationError):repo.prove_quiescence(c.operation.effect.effect_id,object(),now_epoch=150)
    proof=v.proof(rec,"quiescent");assert repo.prove_quiescence(c.operation.effect.effect_id,proof,now_epoch=150).dispatch is DispatchState.QUIESCENCE_PROVEN

def test_concurrent_admission_executes_once():
    c=stage_command();ex=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r,invalid);g=execution_grant(a,c)
    with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(lambda _:b.execute(c.canonical_bytes,g,now_epoch=150),range(32)))
    assert ex.calls==1 and repo.get(c.operation.effect.effect_id).attempt_count==1

def test_absence_without_strong_finality_reduces_to_indeterminate_then_conflicts_when_proven():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.DEFINITELY_ABSENT));repo,a,r,invalid,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r,invalid);g=execution_grant(a,c)
    rec=b.execute(c.canonical_bytes,g,now_epoch=150);assert rec.state is EffectState.INDETERMINATE
    proof=v.proof(rec,"final_absent");obs=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.DEFINITELY_ABSENT,D[11],1,finality_proof=proof);receipt=r.issue(dispatch_receipt_content(c,obs,rec));rec=repo.append_result(c.operation.effect.effect_id,receipt,proof,now_epoch=150);assert rec.state is EffectState.DEFINITELY_ABSENT
    found=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.FOUND,D[11],1,stage_ref=ProviderStageRefV1("docker","local","acct","ns","other-stage"));found_receipt=r.issue(dispatch_receipt_content(c,found,rec));rec=repo.append_result(c.operation.effect.effect_id,found_receipt,None,now_epoch=150);assert rec.state is EffectState.CONTRADICTED and len(rec.results)==3
    assert repo.append_result(c.operation.effect.effect_id,found_receipt,None,now_epoch=150) is rec

def reconciliation_grant(a,c,adapter,owner="owner",epoch=1):
    p=c.preparation;e=c.operation.effect;x=ReconciliationGrantContentV1("reconcile",c.digest,e.effect_id,p.preparation_digest,adapter.digest,p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,owner,1,epoch,D[9],D[10],100,200,a.epoch,a.revocation_generation);return a.issue_reconciliation(x)

def test_reconciliation_only_relinquished_or_quiescence_and_no_duplicate_lookup():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r,invalid);g=execution_grant(a,c);b.execute(c.canonical_bytes,g,now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");obs=observation_for(c);adapter=Adapter(obs);service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r,invalid);rg=reconciliation_grant(a,c,ad)
    rec=service.reconcile(c.canonical_bytes,rg,now_epoch=150);assert rec.state is EffectState.FOUND and adapter.calls==1
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,rg,now_epoch=151)
    assert error.value.code is DiagnosticCode.EFFECT_INELIGIBLE and adapter.calls==1

def test_reconciliation_interruption_exact_resume_and_concurrent_claim_rejected():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r,invalid);g=execution_grant(a,c);b.execute(c.canonical_bytes,g,now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");adapter=Adapter(None,fail=True);service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r,invalid);rg=reconciliation_grant(a,c,ad)
    with pytest.raises(FoundationError):service.reconcile(c.canonical_bytes,rg,now_epoch=150)
    claim=repo.get(c.operation.effect.effect_id).reconciliation
    adapter.fail=False;adapter.observation=observation_for(c)
    with pytest.raises(FoundationError):service.reconcile(c.canonical_bytes,rg,now_epoch=151)
    resumed_content=replace(rg.content,grant_ref="resume")
    resumed_grant=a.issue_reconciliation(resumed_content)
    rec=service.reconcile(c.canonical_bytes,resumed_grant,now_epoch=151,continuation=claim);assert rec.state is EffectState.FOUND and rec.reconciliation.claimed_at_epoch==150

def test_reconciliation_transfer_requires_authenticated_worker_quiescence():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r,invalid);g=execution_grant(a,c);b.execute(c.canonical_bytes,g,now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");adapter=Adapter(None,fail=True);service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r,invalid);rg=reconciliation_grant(a,c,ad)
    with pytest.raises(FoundationError):service.reconcile(c.canonical_bytes,rg,now_epoch=150)
    rec=repo.get(c.operation.effect.effect_id)
    transfer_content=replace(rg.content,grant_ref="transfer",owner_ref="new",ownership_epoch=2);transfer_grant=a.issue_reconciliation(transfer_content)
    with pytest.raises(FoundationError):repo.transfer_reconciliation(c.canonical_bytes,transfer_grant,proof=object(),now_epoch=160)
    proof=v.proof(rec,"quiescent");_,claim=repo.transfer_reconciliation(c.canonical_bytes,transfer_grant,proof=proof,now_epoch=160);assert claim.owner_ref=="new" and claim.ownership_epoch==2

def test_submit_repository_cas_requires_exact_durable_authenticated_stage_result():
    stage=stage_command();ex=bound_executor(stage,Executor());repo,a,r,invalid,v,resolver=environment(ex);b=EffectBrokerV2(repo,resolver,a,r,invalid);sg=execution_grant(a,stage);stage_record=b.execute(stage.canonical_bytes,sg,now_epoch=150)
    p=stage.preparation;ref=StagePredecessorV2(p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,p.project_ref,p.run_id,p.plan_fingerprint,p.preparation_digest,p.workload_digest,stage.operation.effect.effect_id,stage_record.results[0].authenticated_receipt_digest,stage_record.record_digest);submit=build_submit_command(p,"nonce",payload(EffectKind.SUBMIT,p),descriptor(),ref);sex=bound_executor(submit,Executor());sb=EffectBrokerV2(repo,ExecutorResolver(sex),a,r,invalid);grant=execution_grant(a,submit,"submit")
    assert sb.execute(submit.canonical_bytes,grant,now_epoch=150).state is EffectState.FOUND
    other=type(repo)(r,invalid,v,v,a);other_broker=EffectBrokerV2(other,ExecutorResolver(sex),a,r,invalid)
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

def test_receipt_and_admission_are_committed_atomically_once():
    c=stage_command();ex=bound_executor(c,Executor());repo,a,r,invalid,v,_=environment(ex);g=execution_grant(a,c)
    record,_=repo.consume_attempt(c.canonical_bytes,g,now_epoch=150);record=repo.begin_dispatch(c.operation.effect.effect_id)
    receipt=r.issue(dispatch_receipt_content(c,observation_for(c),record))
    class CountingDict(dict):
        writes=0
        def __setitem__(self,key,value):self.writes+=1;super().__setitem__(key,value)
    repo._records=CountingDict(repo._records)
    completed=repo.complete_dispatch(c.operation.effect.effect_id,receipt,None,now_epoch=150)
    assert repo._records.writes==1 and completed.dispatch is DispatchState.RELINQUISHED
    assert len(completed.results)==len(completed.receipt_admissions)==1
    admission=completed.receipt_admissions[0]
    assert admission.receipt_digest==receipt.authenticated_receipt_digest and admission.freshness is ReceiptFreshnessV2.FRESH

def test_complete_record_digest_binds_authority_evidence_state_and_claim_fields():
    c=stage_command();repo,a,r,invalid,v,_=environment(Executor());grant=execution_grant(a,c);initial,_=repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150)
    variants=(replace(initial,grant=execution_grant(a,c,"other-grant")),replace(initial,dispatch=DispatchState.OWNED_IN_FLIGHT),replace(initial,attempt_count=2),replace(initial,dispatch_epoch=2))
    assert all(value.record_digest!=initial.record_digest for value in variants)
    with pytest.raises(ValueError):replace(initial,state=EffectState.INDETERMINATE,invalid_codes=(DiagnosticCode.EVIDENCE_INVALID,))
    repo.begin_dispatch(c.operation.effect.effect_id);current=repo.get(c.operation.effect.effect_id);receipt=r.issue(dispatch_receipt_content(c,observation_for(c),current));completed=repo.complete_dispatch(c.operation.effect.effect_id,receipt,None,now_epoch=150)
    assert completed.record_digest not in {initial.record_digest,current.record_digest}
    with pytest.raises(ValueError):replace(completed,state=EffectState.INDETERMINATE)

def test_admission_cardinality_authority_freshness_finality_and_codes_are_closed():
    c=stage_command();ex=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(ex)
    record=EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150);admission=record.receipt_admissions[0]
    with pytest.raises(ValueError):replace(record,receipt_admissions=())
    with pytest.raises(ValueError):replace(record,receipt_admissions=(replace(admission,source_owner_ref="other"),))
    with pytest.raises(ValueError):replace(admission,freshness=ReceiptFreshnessV2.STALE)
    with pytest.raises(ValueError):replace(record,receipt_admissions=(replace(admission,finality_verified=True),))
    with pytest.raises(ValueError):replace(record,receipt_admissions=(replace(admission,generated_invalid_codes=(DiagnosticCode.FINALITY_UNPROVEN,)),))
    with pytest.raises(ValueError):replace(record,receipt_admissions=(replace(admission,expected_grant_digest=D[0]),))
    assert admission.admission_digest!=replace(admission,source_owner_ref="other").admission_digest
    assert admission.admission_digest!=replace(admission,expected_grant_ref="other-grant",expected_grant_digest=D[0]).admission_digest

def test_finality_assessment_uses_pre_admission_record_and_fails_closed():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));base,a,r,invalid,recovery,_=environment(ex)
    seen=[]
    class Finality:
        def verify_finality(self,proof,record,receipt,*,now_epoch):seen.append((len(record.results),len(record.receipt_admissions)));return object()
    repo=type(base)(r,invalid,recovery,Finality(),a);EffectBrokerV2(repo,ExecutorResolver(ex),a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    proof=type("Proof",(),{"proof_digest":D[12]})()
    before=repo.get(c.operation.effect.effect_id);obs=observation_for(c,ObservationDisposition.DEFINITELY_ABSENT,finality_proof=proof);receipt=r.issue(dispatch_receipt_content(c,obs,before))
    after=repo.append_result(c.operation.effect.effect_id,receipt,proof,now_epoch=150)
    assert seen==[(len(before.results),len(before.receipt_admissions))]
    assert after.state is EffectState.INDETERMINATE and after.receipt_admissions[-1].finality_verified is False
    assert after.receipt_admissions[-1].generated_invalid_codes==(DiagnosticCode.FINALITY_UNPROVEN,)
    with pytest.raises(ValueError):replace(after,invalid_codes=(DiagnosticCode.EVIDENCE_INVALID,DiagnosticCode.FINALITY_UNPROVEN))
    with pytest.raises(ValueError):replace(after,invalid_codes=())
    with pytest.raises(ValueError):replace(after,invalid_codes=(DiagnosticCode.FINALITY_UNPROVEN,DiagnosticCode.FINALITY_UNPROVEN))
    with pytest.raises(ValueError):replace(after,invalid_codes=(DiagnosticCode.STALE_RESULT,))

def test_exact_active_continuation_survives_expiry_without_claim_mutation_and_mismatch_has_zero_lookup():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");grant=reconciliation_grant(a,c,ad);record,claim,lookup=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150)
    continued,same,repeat=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=250,continuation=claim)
    assert continued is record and same is claim and lookup is repeat is True and continued.reconciliation_claims==(claim,)
    adapter=Adapter(observation_for(c));service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r,invalid)
    altered_binding=replace(claim.grant_lineage[-1],grant_digest=D[0]);altered=replace(claim,grant_digest=D[0],grant_lineage=(altered_binding,))
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,grant,now_epoch=250,continuation=altered)
    assert error.value.code is DiagnosticCode.RECONCILIATION_CONFLICT and adapter.calls==0

def test_completed_indeterminate_claim_retries_next_generation_and_concurrent_completion_is_idempotent():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");grant=reconciliation_grant(a,c,ad);_,claim,_=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150)
    found=observation_for(c,result_epoch=1);receipt=r.issue(reconciliation_receipt_content(found,claim))
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(lambda _:repo.complete_reconciliation(c.operation.effect.effect_id,claim,receipt,None,now_epoch=150),range(16)))
    assert all(value is records[0] for value in records) and records[0].reconciliation.completed
    # A separate indeterminate completion is retry-eligible at the next generation/epoch.
    c2=stage_command(prep(source_digest=D[12]));ex2=bound_executor(c2,Executor(ObservationDisposition.INDETERMINATE));repo2,a2,r2,invalid2,v2,resolver2=environment(ex2);EffectBrokerV2(repo2,resolver2,a2,r2,invalid2).execute(c2.canonical_bytes,execution_grant(a2,c2),now_epoch=150)
    grant1=reconciliation_grant(a2,c2,ad);_,claim1,_=repo2.acquire_reconciliation(c2.canonical_bytes,grant1,now_epoch=150)
    indeterminate=ProviderObservationV1(c2.operation.effect.effect_id,c2.digest,c2.executor.digest,ObservationDisposition.INDETERMINATE,D[11],1);receipt2=r2.issue(reconciliation_receipt_content(indeterminate,claim1));repo2.complete_reconciliation(c2.operation.effect.effect_id,claim1,receipt2,None,now_epoch=150)
    retry_content=replace(grant1.content,grant_ref="retry",generation=2,ownership_epoch=2);retry=a2.issue_reconciliation(retry_content)
    retried,claim2,lookup=repo2.acquire_reconciliation(c2.canonical_bytes,retry,now_epoch=151)
    assert lookup and claim2.generation==2 and claim2.ownership_epoch==2 and retried.reconciliation_claims[-2].completed
    late=observation_for(c2,result_epoch=1);late_receipt=r2.issue(reconciliation_receipt_content(late,claim1))
    retained=repo2.complete_reconciliation(c2.operation.effect.effect_id,claim1,late_receipt,None,now_epoch=151)
    assert retained.reconciliation==claim2 and retained.receipt_admissions[-1].freshness is ReceiptFreshnessV2.STALE
    assert retained.receipt_admissions[-1].generated_invalid_codes==(DiagnosticCode.STALE_RESULT,)
    assert (retained.receipt_admissions[-1].expected_grant_ref,retained.receipt_admissions[-1].expected_grant_digest)==(claim2.grant_ref,claim2.grant_digest)
    with pytest.raises(FoundationError) as error:repo2.complete_reconciliation(c2.operation.effect.effect_id,replace(claim1,ownership_epoch=99),late_receipt,None,now_epoch=151)
    assert error.value.code is DiagnosticCode.RECONCILIATION_CONFLICT

def test_resume_preserves_claim_identity_while_admissions_bind_pre_and_post_grants():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");grant=reconciliation_grant(a,c,ad);_,claim,_=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150)
    before_obs=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.INDETERMINATE,D[11],1);before=r.issue(reconciliation_receipt_content(before_obs,claim));repo.append_result(c.operation.effect.effect_id,before,None,now_epoch=150)
    delayed_obs=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.INDETERMINATE,D[13],1);delayed=r.issue(reconciliation_receipt_content(delayed_obs,claim))
    stopped=repo.interrupt_reconciliation(c.operation.effect.effect_id,claim);resumed_grant=a.issue_reconciliation(replace(grant.content,grant_ref="resumed"));_,resumed,_=repo.acquire_reconciliation(c.canonical_bytes,resumed_grant,now_epoch=151,continuation=stopped)
    assert resumed.claim_digest==claim.claim_digest and (resumed.grant_ref,resumed.grant_digest)!=(claim.grant_ref,claim.grant_digest)
    delayed_record=repo.append_result(c.operation.effect.effect_id,delayed,None,now_epoch=151)
    assert delayed_record.receipt_admissions[-1].freshness is ReceiptFreshnessV2.STALE
    assert (delayed_record.receipt_admissions[-1].source_grant_ref,delayed_record.receipt_admissions[-1].source_grant_digest)==(claim.grant_ref,claim.grant_digest)
    after_obs=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.INDETERMINATE,D[12],1);after=r.issue(reconciliation_receipt_content(after_obs,resumed));record=repo.append_result(c.operation.effect.effect_id,after,None,now_epoch=151)
    pre,delayed_admission,post=record.receipt_admissions[-3:]
    assert (pre.expected_grant_ref,pre.expected_grant_digest)==(claim.grant_ref,claim.grant_digest)
    assert (post.expected_grant_ref,post.expected_grant_digest)==(resumed.grant_ref,resumed.grant_digest)
    assert delayed_admission.expected_grant_ref==resumed.grant_ref and post.freshness is ReceiptFreshnessV2.FRESH

def test_future_same_claim_grant_is_rejected_before_its_activation_index():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");grant=reconciliation_grant(a,c,ad);_,claim,_=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150)
    future_grant=a.issue_reconciliation(replace(grant.content,grant_ref="future-grant"))
    future_binding=ReconciliationGrantBindingV2(future_grant.content.grant_ref,future_grant.authenticated_grant_digest,2,claim.grant_lineage[-1].binding_digest)
    future_claim=replace(claim,grant_ref=future_binding.grant_ref,grant_digest=future_binding.grant_digest,grant_lineage=claim.grant_lineage+(future_binding,))
    receipt=r.issue(reconciliation_receipt_content(observation_for(c),future_claim))
    before=repo.get(c.operation.effect.effect_id)
    with pytest.raises(FoundationError) as error:repo.append_result(c.operation.effect.effect_id,receipt,None,now_epoch=150)
    assert error.value.code is DiagnosticCode.BINDING_MISMATCH and repo.get(c.operation.effect.effect_id) is before

def test_multiple_resumes_at_same_admission_index_use_last_grant_and_lineage_shapes_are_closed():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");grant=reconciliation_grant(a,c,ad);_,claim,_=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150);stable=claim.claim_digest
    stopped=repo.interrupt_reconciliation(c.operation.effect.effect_id,claim);grant2=a.issue_reconciliation(replace(grant.content,grant_ref="resume-two"));_,claim2,_=repo.acquire_reconciliation(c.canonical_bytes,grant2,now_epoch=151,continuation=stopped)
    stopped2=repo.interrupt_reconciliation(c.operation.effect.effect_id,claim2);grant3=a.issue_reconciliation(replace(grant.content,grant_ref="resume-three"));_,claim3,_=repo.acquire_reconciliation(c.canonical_bytes,grant3,now_epoch=152,continuation=stopped2)
    assert claim3.claim_digest==stable and [x.activated_at_admission_index for x in claim3.grant_lineage]==[1,1,1]
    obs=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.INDETERMINATE,D[12],1);receipt=r.issue(reconciliation_receipt_content(obs,claim3));record=repo.append_result(c.operation.effect.effect_id,receipt,None,now_epoch=152)
    assert (record.receipt_admissions[-1].expected_grant_ref,record.receipt_admissions[-1].expected_grant_digest)==(claim3.grant_ref,claim3.grant_digest)
    lineage=claim3.grant_lineage;final_count=len(record.receipt_admissions)
    invalid_claims=(
        lambda:replace(claim3,grant_lineage=()),
        lambda:replace(claim3,grant_lineage=list(lineage)),
        lambda:replace(claim3,grant_lineage=lineage+(lineage[-1],)),
        lambda:replace(claim3,grant_lineage=lineage+(ReconciliationGrantBindingV2("future",D[0],final_count+1,lineage[-1].binding_digest),),grant_ref="future",grant_digest=D[0]),
        lambda:replace(claim3,grant_lineage=(replace(lineage[0],activated_at_admission_index=2),replace(lineage[1],activated_at_admission_index=1),lineage[2])),
        lambda:replace(claim3,grant_ref="wrong-leaf"),
    )
    for mutate in invalid_claims:
        with pytest.raises((TypeError,ValueError)):mutated=mutate();replace(record,reconciliation=mutated,reconciliation_claims=(mutated,))
    with pytest.raises(ValueError):ReconciliationGrantBindingV2("bad",D[0],-1,None)
    with pytest.raises(ValueError):replace(claim3,grant_ref=lineage[0].grant_ref,grant_digest=lineage[0].grant_digest,grant_lineage=tuple(reversed(lineage)))
    truncated=replace(claim3,grant_ref=lineage[0].grant_ref,grant_digest=lineage[0].grant_digest,grant_lineage=(lineage[0],))
    with pytest.raises(ValueError):replace(record,reconciliation=truncated,reconciliation_claims=(truncated,))
    replaced=replace(claim3,grant_ref="replacement",grant_digest=D[0],grant_lineage=lineage[:-1]+(ReconciliationGrantBindingV2("replacement",D[0],1,lineage[-2].binding_digest),))
    with pytest.raises(ValueError):replace(record,reconciliation=replaced,reconciliation_claims=(replaced,))
    historical=replace(claim3,active=False);transfer_a=ReconciliationGrantBindingV2("transfer-a",D[0],final_count,None);transfer_lineage=(transfer_a,ReconciliationGrantBindingV2("transfer-b",D[1],final_count,transfer_a.binding_digest));transferred=type(claim3)("other-owner",claim3.generation,claim3.ownership_epoch+1,153,D[2],"transfer-b",D[1],transfer_lineage)
    with pytest.raises(ValueError):replace(record,reconciliation=transferred,reconciliation_claims=(historical,transferred))

def test_distinct_concurrent_receipts_from_same_claim_are_both_atomically_reduced():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    ad=AdapterDescriptorV1("docker","lookup","1.0.0");grant=reconciliation_grant(a,c,ad);_,claim,_=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150)
    first=observation_for(c,result_epoch=1)
    second=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.FOUND,D[12],1,stage_ref=ProviderStageRefV1("docker","local","acct","ns","other-stage"))
    receipts=(r.issue(reconciliation_receipt_content(first,claim)),r.issue(reconciliation_receipt_content(second,claim)))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda receipt:repo.complete_reconciliation(c.operation.effect.effect_id,claim,receipt,None,now_epoch=150),receipts))
    final=repo.get(c.operation.effect.effect_id)
    assert final.reconciliation.completed and final.state is EffectState.CONTRADICTED
    assert len(final.results)==len(final.receipt_admissions)==3 and len(final.terminal_content_digests)==2
    assert tuple(admission.receipt_digest for admission in final.receipt_admissions[-2:])==tuple(receipt.authenticated_receipt_digest for receipt in final.results[-2:])
    assert {receipt.authenticated_receipt_digest for receipt in final.results[-2:]}=={receipt.authenticated_receipt_digest for receipt in receipts}

def test_receipt_source_grant_is_signed_and_cross_grant_substitution_is_rejected_atomically():
    c=stage_command();repo,a,r,invalid,v,_=environment(Executor());grant=execution_grant(a,c);repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150);current=repo.begin_dispatch(c.operation.effect.effect_id)
    content=dispatch_receipt_content(c,observation_for(c),current)
    authentic=r.issue(content);tampered=replace(authentic,content=replace(content,source_grant_digest=D[0]))
    with pytest.raises(FoundationError) as error:repo.complete_dispatch(c.operation.effect.effect_id,tampered,None,now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID
    forged=r.issue(replace(content,source_grant_ref="other-grant",source_grant_digest=D[0]))
    with pytest.raises(FoundationError) as error:repo.complete_dispatch(c.operation.effect.effect_id,forged,None,now_epoch=150)
    assert error.value.code is DiagnosticCode.BINDING_MISMATCH
    retained=repo.get(c.operation.effect.effect_id);assert retained.dispatch is DispatchState.OWNED_IN_FLIGHT and retained.results==retained.receipt_admissions==()

def test_authenticated_invalid_evidence_is_atomic_chained_redacted_and_digest_bound():
    c=stage_command();repo,a,r,invalid,v,_=environment(Executor());grant=execution_grant(a,c);repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150);current=repo.begin_dispatch(c.operation.effect.effect_id)
    content=InvalidEvidenceContentV2(c.operation.effect.effect_id,c.digest,InvalidEvidenceSiteV2.DISPATCH_OBSERVATION,"dispatch",grant.content.grant_ref,1,current.dispatch_epoch,current.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest,D[12])
    evidence=invalid.issue(content)
    with pytest.raises(FoundationError):repo.complete_invalid_dispatch(c.operation.effect.effect_id,replace(evidence,tag=D[0]))
    assert repo.get(c.operation.effect.effect_id)==current
    completed=repo.complete_invalid_dispatch(c.operation.effect.effect_id,evidence)
    assert completed.dispatch is DispatchState.RELINQUISHED and completed.state is EffectState.INDETERMINATE
    assert completed.invalid_codes==(DiagnosticCode.EVIDENCE_INVALID,) and completed.invalid_evidence==(evidence,)
    admission=completed.invalid_evidence_admissions[0]
    assert admission.sequence==1 and admission.prior_admission_digest is None
    assert b"secret" not in evidence.canonical_bytes and completed.record_digest!=current.record_digest
    with pytest.raises(ValueError):replace(completed,invalid_evidence_admissions=(replace(admission,sequence=2),))
    with pytest.raises(ValueError):replace(completed,invalid_codes=())

@pytest.mark.parametrize("field,value",[("effect_id","other-effect"),("command_digest",D[0]),("source_owner_ref","other-owner"),("source_grant_ref","other-grant"),("source_grant_digest",D[1])])
def test_authenticated_invalid_evidence_binding_substitutions_fail_without_partial_state(field,value):
    c=stage_command();repo,a,r,invalid,v,_=environment(Executor());grant=execution_grant(a,c);repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150);current=repo.begin_dispatch(c.operation.effect.effect_id)
    content=InvalidEvidenceContentV2(c.operation.effect.effect_id,c.digest,InvalidEvidenceSiteV2.DISPATCH_RESOLUTION,"dispatch",grant.content.grant_ref,1,current.dispatch_epoch,current.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest,D[12])
    forged=invalid.issue(replace(content,**{field:value}))
    with pytest.raises(FoundationError):repo.complete_invalid_dispatch(c.operation.effect.effect_id,forged)
    retained=repo.get(c.operation.effect.effect_id);assert retained==current and retained.invalid_evidence==retained.invalid_evidence_admissions==()

def test_invalid_evidence_site_source_duplicate_and_interrupt_return_are_closed():
    c=stage_command();repo,a,r,invalid,v,_=environment(Executor());grant=execution_grant(a,c);repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150);current=repo.begin_dispatch(c.operation.effect.effect_id)
    args=(c.operation.effect.effect_id,c.digest,InvalidEvidenceSiteV2.DISPATCH_OBSERVATION,"reconciliation",grant.content.grant_ref,1,current.dispatch_epoch,current.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest,D[12])
    with pytest.raises(ValueError,match="site/source"):InvalidEvidenceContentV2(*args)
    valid=InvalidEvidenceContentV2(*args[:3],"dispatch",*args[4:]);evidence=invalid.issue(valid);completed=repo.complete_invalid_dispatch(c.operation.effect.effect_id,evidence);admission=completed.invalid_evidence_admissions[0]
    with pytest.raises(ValueError,match="duplicate invalid evidence"):replace(completed,invalid_codes=completed.invalid_codes+(DiagnosticCode.EVIDENCE_INVALID,),invalid_evidence=completed.invalid_evidence+(evidence,),invalid_evidence_admissions=completed.invalid_evidence_admissions+(replace(admission,sequence=2,prior_admission_digest=admission.admission_digest),))

    c2=stage_command(prep(source_digest=D[13]));ex=bound_executor(c2,Executor(ObservationDisposition.INDETERMINATE));repo2,a2,r2,invalid2,v2,resolver2=environment(ex);EffectBrokerV2(repo2,resolver2,a2,r2,invalid2).execute(c2.canonical_bytes,execution_grant(a2,c2),now_epoch=150)
    adapter=AdapterDescriptorV1("docker","lookup","1.0.0");recon_grant=reconciliation_grant(a2,c2,adapter);record,claim,_=repo2.acquire_reconciliation(c2.canonical_bytes,recon_grant,now_epoch=150)
    invalid_content=InvalidEvidenceContentV2(c2.operation.effect.effect_id,c2.digest,InvalidEvidenceSiteV2.RECONCILIATION_OBSERVATION,"reconciliation",claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest,recon_grant.content.grant_ref,recon_grant.authenticated_grant_digest,D[12])
    persisted=repo2.interrupt_invalid_reconciliation(c2.operation.effect.effect_id,claim,invalid2.issue(invalid_content))
    assert type(persisted) is type(record) and persisted.reconciliation.active is False and persisted.invalid_codes[-1] is DiagnosticCode.EVIDENCE_INVALID

def test_duplicate_admitted_receipt_completes_active_claim_then_is_idempotent():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    adapter=AdapterDescriptorV1("docker","lookup","1.0.0");grant=reconciliation_grant(a,c,adapter);_,claim,_=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150)
    observation=ProviderObservationV1(c.operation.effect.effect_id,c.digest,c.executor.digest,ObservationDisposition.INDETERMINATE,D[12],1);receipt=r.issue(reconciliation_receipt_content(observation,claim))
    admitted=repo.append_result(c.operation.effect.effect_id,receipt,None,now_epoch=150);assert admitted.reconciliation.active
    stopped=repo.interrupt_reconciliation(c.operation.effect.effect_id,claim)
    with pytest.raises(FoundationError) as error:repo.complete_reconciliation(c.operation.effect.effect_id,claim,receipt,None,now_epoch=150)
    assert error.value.code is DiagnosticCode.RECONCILIATION_CONFLICT
    resumed_grant=a.issue_reconciliation(replace(grant.content,grant_ref="resumed-duplicate"));_,resumed,_=repo.acquire_reconciliation(c.canonical_bytes,resumed_grant,now_epoch=151,continuation=stopped)
    completed=repo.complete_reconciliation(c.operation.effect.effect_id,claim,receipt,None,now_epoch=151);assert completed.reconciliation.completed and completed.reconciliation.claim_digest==resumed.claim_digest
    assert repo.complete_reconciliation(c.operation.effect.effect_id,claim,receipt,None,now_epoch=150) is completed

def test_post_provider_invalid_admission_failure_orphans_and_requires_quiescence():
    c=stage_command();executor=Executor();original=executor.execute_once
    def malformed(payload_value,request):return replace(original(payload_value,request),effect_id="wrong-effect")
    executor.execute_once=malformed;repo,a,r,invalid,v,resolver=environment(executor)
    class RejectInvalidRepository:
        def __getattr__(self,name):return getattr(repo,name)
        def complete_invalid_dispatch(self,effect_id,evidence):raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
    broker=EffectBrokerV2(RejectInvalidRepository(),resolver,a,r,invalid)
    with pytest.raises(FoundationError) as error:broker.execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID
    retained=repo.get(c.operation.effect.effect_id);assert retained.dispatch is DispatchState.ORPHANED_UNPROVEN and retained.invalid_evidence==()
    adapter=AdapterDescriptorV1("docker","lookup","1.0.0");recon_grant=reconciliation_grant(a,c,adapter)
    with pytest.raises(FoundationError) as blocked:repo.acquire_reconciliation(c.canonical_bytes,recon_grant,now_epoch=150)
    assert blocked.value.code is DiagnosticCode.EFFECT_INELIGIBLE
    repo.prove_quiescence(c.operation.effect.effect_id,v.proof(retained,"quiescent",150),now_epoch=150)
    recovered,claim,_=repo.acquire_reconciliation(c.canonical_bytes,recon_grant,now_epoch=150)
    assert recovered.dispatch is DispatchState.QUIESCENCE_PROVEN and claim.active

def test_post_provider_dispatch_completion_failure_is_orphaned():
    c=stage_command();executor=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(executor)
    class RejectCompletionRepository:
        def __getattr__(self,name):return getattr(repo,name)
        def complete_dispatch(self,effect_id,receipt,proof,*,now_epoch):raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
    with pytest.raises(FoundationError) as error:EffectBrokerV2(RejectCompletionRepository(),resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID and executor.calls==1
    retained=repo.get(c.operation.effect.effect_id)
    assert retained.dispatch is DispatchState.ORPHANED_UNPROVEN and retained.results==retained.receipt_admissions==()

def test_valid_provider_observation_receipt_signing_failure_orphans_without_invalid_evidence():
    c=stage_command();executor=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(executor)
    class RejectReceiptSigner:
        def issue(self,content):raise RuntimeError("secret-provider-credential")
    with pytest.raises(FoundationError) as error:EffectBrokerV2(repo,resolver,a,RejectReceiptSigner(),invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    raised=error.value
    assert raised.code is DiagnosticCode.AUTHORITY_INVALID and executor.calls==1
    assert "secret-provider-credential" not in str(raised) and "secret-provider-credential" not in repr(raised)
    assert raised.__cause__ is None and raised.__context__ is None
    retained=repo.get(c.operation.effect.effect_id)
    assert retained.dispatch is DispatchState.ORPHANED_UNPROVEN and retained.invalid_codes==()
    assert retained.invalid_evidence==retained.invalid_evidence_admissions==()
    adapter=AdapterDescriptorV1("docker","lookup","1.0.0");recon_grant=reconciliation_grant(a,c,adapter)
    with pytest.raises(FoundationError) as blocked:repo.acquire_reconciliation(c.canonical_bytes,recon_grant,now_epoch=150)
    assert blocked.value.code is DiagnosticCode.EFFECT_INELIGIBLE
    repo.prove_quiescence(c.operation.effect.effect_id,v.proof(retained,"quiescent",150),now_epoch=150)
    recovered,claim,_=repo.acquire_reconciliation(c.canonical_bytes,recon_grant,now_epoch=150)
    assert recovered.dispatch is DispatchState.QUIESCENCE_PROVEN and claim.active

@pytest.mark.parametrize("post_provider",[False,True])
def test_invalid_evidence_signing_failure_stabilizes_dispatch_by_phase(post_provider):
    c=stage_command();correct=descriptor();wrong=type(correct)("docker","wrong-executor","1.0.0")
    if post_provider:
        executor=Executor();original=executor.execute_once
        executor.execute_once=lambda payload_value,request:replace(original(payload_value,request),effect_id="wrong-effect")
    else:
        class ExecutorWithLateBindingFailure:
            provider_id="docker";profile_ref="local";account_ref="acct";namespace_ref="ns";effect_kinds=("stage","submit","cancel");payload_schemas=("stage-payload/v2","submit-payload/v2","cancel-payload/v2")
            def __init__(self):self.reads=0;self.calls=0
            @property
            def descriptor(self):self.reads+=1;return correct if self.reads<=2 else wrong
            def execute_once(self,payload_value,request):self.calls+=1;raise AssertionError("provider must not run")
        executor=ExecutorWithLateBindingFailure()
    repo,a,r,invalid,v,resolver=environment(executor)
    class RejectSigner:
        def issue(self,content):raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
    with pytest.raises(FoundationError) as error:EffectBrokerV2(repo,resolver,a,r,RejectSigner()).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID
    retained=repo.get(c.operation.effect.effect_id)
    assert retained.dispatch is (DispatchState.ORPHANED_UNPROVEN if post_provider else DispatchState.RELINQUISHED)
    assert retained.invalid_evidence==retained.invalid_evidence_admissions==()

@pytest.mark.parametrize("post_provider",[False,True])
def test_raw_invalid_persistence_failure_is_closed_and_stabilized_by_dispatch_phase(post_provider):
    c=stage_command();correct=descriptor();wrong=type(correct)("docker","wrong-executor","1.0.0")
    if post_provider:
        executor=Executor();original=executor.execute_once
        executor.execute_once=lambda payload_value,request:replace(original(payload_value,request),effect_id="wrong-effect")
    else:
        class ExecutorWithLateBindingFailure:
            provider_id="docker";profile_ref="local";account_ref="acct";namespace_ref="ns";effect_kinds=("stage","submit","cancel");payload_schemas=("stage-payload/v2","submit-payload/v2","cancel-payload/v2")
            def __init__(self):self.reads=0;self.calls=0
            @property
            def descriptor(self):self.reads+=1;return correct if self.reads<=2 else wrong
            def execute_once(self,payload_value,request):self.calls+=1;raise AssertionError("provider must not run")
        executor=ExecutorWithLateBindingFailure()
    repo,a,r,invalid,v,resolver=environment(executor)
    class RawFailureRepository:
        def __getattr__(self,name):return getattr(repo,name)
        def complete_invalid_dispatch(self,effect_id,evidence):raise RuntimeError("secret-provider-credential")
    with pytest.raises(FoundationError) as error:EffectBrokerV2(RawFailureRepository(),resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    raised=error.value
    assert raised.code is DiagnosticCode.AUTHORITY_INVALID
    assert "secret-provider-credential" not in str(raised) and "secret-provider-credential" not in repr(raised)
    assert raised.__cause__ is None and raised.__context__ is None
    retained=repo.get(c.operation.effect.effect_id)
    assert retained.dispatch is (DispatchState.ORPHANED_UNPROVEN if post_provider else DispatchState.RELINQUISHED)

@pytest.mark.parametrize("mode",["none","object","stale"])
def test_pre_provider_unverified_invalid_commit_relinquishes_without_provider_call(mode):
    c=stage_command();correct=descriptor();wrong=type(correct)("docker","wrong-executor","1.0.0")
    class FlakyExecutor:
        provider_id="docker";profile_ref="local";account_ref="acct";namespace_ref="ns";effect_kinds=("stage","submit","cancel");payload_schemas=("stage-payload/v2","submit-payload/v2","cancel-payload/v2")
        def __init__(self):self.reads=0;self.calls=0
        @property
        def descriptor(self):self.reads+=1;return correct if self.reads<=2 else wrong
        def execute_once(self,payload_value,request):self.calls+=1;raise AssertionError("provider must not run")
    executor=FlakyExecutor();repo,a,r,invalid,v,resolver=environment(executor)
    class MalformedCommitRepository:
        def __getattr__(self,name):return getattr(repo,name)
        def complete_invalid_dispatch(self,effect_id,evidence):
            if mode=="none":return None
            if mode=="object":return object()
            return repo.get(effect_id)
    broker=EffectBrokerV2(MalformedCommitRepository(),resolver,a,r,invalid)
    with pytest.raises(FoundationError) as error:broker.execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID and executor.calls==0
    retained=repo.get(c.operation.effect.effect_id);assert retained.dispatch is DispatchState.RELINQUISHED and retained.invalid_evidence==()

@pytest.mark.parametrize("mode",["raise","none","object","stale"])
def test_invalid_admission_failure_stabilizes_reconciliation_and_propagates_failure(mode):
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    malformed=replace(observation_for(c),effect_id="wrong-effect");adapter=Adapter(malformed)
    class RejectInvalidRepository:
        def __getattr__(self,name):return getattr(repo,name)
        def interrupt_invalid_reconciliation(self,effect_id,claim,evidence):
            if mode=="raise":raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            if mode=="none":return None
            if mode=="object":return object()
            return repo.get(effect_id)
    service=ReconciliationServiceV1(RejectInvalidRepository(),a,AdapterResolver(adapter),r,invalid);grant=reconciliation_grant(a,c,adapter.descriptor)
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,grant,now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID
    retained=repo.get(c.operation.effect.effect_id);assert retained.reconciliation.active is False and retained.invalid_evidence==()

def test_invalid_evidence_signing_failure_interrupts_reconciliation():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    malformed=replace(observation_for(c),effect_id="wrong-effect");adapter=Adapter(malformed)
    class RejectSigner:
        def issue(self,content):raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
    service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r,RejectSigner());grant=reconciliation_grant(a,c,adapter.descriptor)
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,grant,now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID
    retained=repo.get(c.operation.effect.effect_id)
    assert retained.reconciliation.active is False and retained.reconciliation.completed is False and retained.invalid_evidence==()

def test_raw_invalid_persistence_failure_is_closed_and_interrupts_reconciliation():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    malformed=replace(observation_for(c),effect_id="wrong-effect");adapter=Adapter(malformed)
    class RawFailureRepository:
        def __getattr__(self,name):return getattr(repo,name)
        def interrupt_invalid_reconciliation(self,effect_id,claim,evidence):raise RuntimeError("secret-provider-credential")
    service=ReconciliationServiceV1(RawFailureRepository(),a,AdapterResolver(adapter),r,invalid);grant=reconciliation_grant(a,c,adapter.descriptor)
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,grant,now_epoch=150)
    raised=error.value
    assert raised.code is DiagnosticCode.AUTHORITY_INVALID
    assert "secret-provider-credential" not in str(raised) and "secret-provider-credential" not in repr(raised)
    assert raised.__cause__ is None and raised.__context__ is None
    retained=repo.get(c.operation.effect.effect_id)
    assert retained.reconciliation.active is False and retained.reconciliation.completed is False

@pytest.mark.parametrize("mode",["none","object","stale","missing_receipt","missing_admission","wrong_state","reload_mismatch","raw"])
def test_dispatch_completion_boundary_rejects_malformed_results_and_preserves_only_durable_success(mode):
    c=stage_command();executor=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(executor)
    class CompletionRepository:
        def __init__(self):self.before=None;self.completed=None;self.mismatch_reads=0
        def __getattr__(self,name):return getattr(repo,name)
        def complete_dispatch(self,effect_id,receipt,proof,*,now_epoch):
            self.before=repo.get(effect_id)
            if mode=="raw":raise RuntimeError("secret-provider-credential")
            if mode=="none":return None
            if mode=="object":return object()
            if mode=="stale":return self.before
            self.completed=repo.complete_dispatch(effect_id,receipt,proof,now_epoch=now_epoch)
            if mode=="reload_mismatch":return self.completed
            malformed=deepcopy(self.completed)
            if mode=="missing_receipt":object.__setattr__(malformed,"results",())
            elif mode=="missing_admission":object.__setattr__(malformed,"receipt_admissions",())
            else:object.__setattr__(malformed,"dispatch",DispatchState.OWNED_IN_FLIGHT)
            return malformed
        def get(self,effect_id):
            if mode=="reload_mismatch" and self.completed is not None and self.mismatch_reads==0:
                self.mismatch_reads+=1;return self.before
            return repo.get(effect_id)
    proxy=CompletionRepository()
    with pytest.raises(FoundationError) as error:EffectBrokerV2(proxy,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID);assert executor.calls==1
    retained=repo.get(c.operation.effect.effect_id)
    if mode in {"missing_receipt","missing_admission","wrong_state","reload_mismatch"}:
        assert retained.dispatch is DispatchState.RELINQUISHED and retained.state is EffectState.FOUND
    else:
        assert retained.dispatch is DispatchState.ORPHANED_UNPROVEN and retained.results==retained.receipt_admissions==()

@pytest.mark.parametrize("tamper",["semantic_state","nested_admission"])
def test_dispatch_completion_replays_full_canonical_invariants_when_return_and_reload_match(tamper):
    c=stage_command();executor=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(executor)
    class TamperingRepository:
        def __init__(self):self.completed=None;self.digest=None
        def __getattr__(self,name):return getattr(repo,name)
        def complete_dispatch(self,effect_id,receipt,proof,*,now_epoch):
            self.completed=repo.complete_dispatch(effect_id,receipt,proof,now_epoch=now_epoch)
            if tamper=="semantic_state":object.__setattr__(self.completed,"state",EffectState.INDETERMINATE)
            else:object.__setattr__(self.completed.receipt_admissions[-1],"finality_verified",0)
            self.digest=self.completed.record_digest
            return self.completed
    proxy=TamperingRepository()
    with pytest.raises(FoundationError) as error:EffectBrokerV2(proxy,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    assert proxy.completed is repo.get(c.operation.effect.effect_id) and proxy.completed.record_digest==proxy.digest

@pytest.mark.parametrize("mode",["none","object","stale","missing_receipt","missing_admission","wrong_state","reload_mismatch","raw"])
def test_reconciliation_completion_boundary_rejects_malformed_results_and_stabilizes_claim(mode):
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    adapter=Adapter(observation_for(c))
    class CompletionRepository:
        def __init__(self):self.before=None;self.completed=None;self.mismatch_reads=0
        def __getattr__(self,name):return getattr(repo,name)
        def complete_reconciliation(self,effect_id,claim,receipt,proof,*,now_epoch):
            self.before=repo.get(effect_id)
            if mode=="raw":raise RuntimeError("secret-provider-credential")
            if mode=="none":return None
            if mode=="object":return object()
            if mode=="stale":return self.before
            self.completed=repo.complete_reconciliation(effect_id,claim,receipt,proof,now_epoch=now_epoch)
            if mode=="reload_mismatch":return self.completed
            malformed=deepcopy(self.completed)
            if mode=="missing_receipt":object.__setattr__(malformed,"results",malformed.results[:-1])
            elif mode=="missing_admission":object.__setattr__(malformed,"receipt_admissions",malformed.receipt_admissions[:-1])
            else:object.__setattr__(malformed,"reconciliation",replace(malformed.reconciliation,active=True,completed=False))
            return malformed
        def get(self,effect_id):
            if mode=="reload_mismatch" and self.completed is not None and self.mismatch_reads==0:
                self.mismatch_reads+=1;return self.before
            return repo.get(effect_id)
    grant=reconciliation_grant(a,c,adapter.descriptor);proxy=CompletionRepository();service=ReconciliationServiceV1(proxy,a,AdapterResolver(adapter),r,invalid)
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,grant,now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID);assert adapter.calls==1
    retained=repo.get(c.operation.effect.effect_id)
    if mode in {"missing_receipt","missing_admission","wrong_state","reload_mismatch"}:
        assert retained.reconciliation.completed and not retained.reconciliation.active and retained.state is EffectState.FOUND
    else:
        assert not retained.reconciliation.active and not retained.reconciliation.completed

@pytest.mark.parametrize("tamper",["semantic_state","nested_grant_lineage"])
def test_reconciliation_completion_replays_full_canonical_invariants_when_return_and_reload_match(tamper):
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    adapter=Adapter(observation_for(c))
    class TamperingRepository:
        def __init__(self):self.completed=None;self.digest=None
        def __getattr__(self,name):return getattr(repo,name)
        def complete_reconciliation(self,effect_id,claim,receipt,proof,*,now_epoch):
            self.completed=repo.complete_reconciliation(effect_id,claim,receipt,proof,now_epoch=now_epoch)
            if tamper=="semantic_state":object.__setattr__(self.completed,"state",EffectState.INDETERMINATE)
            else:object.__setattr__(self.completed.reconciliation.grant_lineage[-1],"activated_at_admission_index",True)
            self.digest=self.completed.record_digest
            return self.completed
    proxy=TamperingRepository();grant=reconciliation_grant(a,c,adapter.descriptor);service=ReconciliationServiceV1(proxy,a,AdapterResolver(adapter),r,invalid)
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,grant,now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    assert proxy.completed is repo.get(c.operation.effect.effect_id) and proxy.completed.record_digest==proxy.digest

def test_retained_receipt_authenticity_tamper_is_rejected_after_structural_reprojection():
    c=stage_command();ex=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(ex);record=EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    object.__setattr__(record.results[0],"tag",D[0]);object.__setattr__(record.receipt_admissions[0],"receipt_digest",record.results[0].authenticated_receipt_digest)
    with pytest.raises(FoundationError) as error:repo._transition_dispatch(c.operation.effect.effect_id,DispatchState.RELINQUISHED,DispatchState.RELINQUISHED)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)

def test_retained_invalid_evidence_authenticity_tamper_is_rejected_after_structural_reprojection():
    c=stage_command();executor=Executor();original=executor.execute_once;executor.execute_once=lambda payload_value,request:replace(original(payload_value,request),effect_id="wrong-effect")
    repo,a,r,invalid,v,resolver=environment(executor)
    with pytest.raises(FoundationError):EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    record=repo.get(c.operation.effect.effect_id);object.__setattr__(record.invalid_evidence[0],"tag",D[0]);object.__setattr__(record.invalid_evidence_admissions[0],"authenticated_evidence_digest",record.invalid_evidence[0].authenticated_evidence_digest)
    with pytest.raises(FoundationError) as error:repo._transition_dispatch(c.operation.effect.effect_id,DispatchState.RELINQUISHED,DispatchState.RELINQUISHED)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)

@pytest.mark.parametrize("tamper",["tag","content"])
def test_retained_execution_grant_authenticity_tamper_blocks_guarded_transition(tamper):
    c=stage_command();repo,a,r,invalid,v,resolver=environment(Executor());record,_=repo.consume_attempt(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    if tamper=="tag":object.__setattr__(record.grant,"tag",D[0])
    else:object.__setattr__(record.grant.content,"policy_digest",D[0])
    with pytest.raises(FoundationError) as error:repo.begin_dispatch(c.operation.effect.effect_id)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)

@pytest.mark.parametrize("outcome",[False,None,object(),"truthy"])
def test_retained_receipt_verifier_requires_exact_true(outcome):
    c=stage_command();ex=bound_executor(c,Executor());repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    class Verifier:
        def verify(self,receipt):return outcome
    repo._receipt=Verifier()
    with pytest.raises(FoundationError) as error:repo._transition_dispatch(c.operation.effect.effect_id,DispatchState.RELINQUISHED,DispatchState.RELINQUISHED)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)

def test_retained_verifier_exception_is_closed_and_all_authenticators_are_invoked_on_positive_record():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    malformed=replace(observation_for(c),effect_id="wrong-effect");adapter=Adapter(malformed);service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),r,invalid);grant=reconciliation_grant(a,c,adapter.descriptor)
    with pytest.raises(FoundationError):service.reconcile(c.canonical_bytes,grant,now_epoch=150)
    base_receipt,base_invalid,base_grants=repo._receipt,repo._invalid_evidence,repo._grants
    calls={"receipt":0,"invalid":0,"grant":0}
    class ReceiptProbe:
        def verify(self,value):calls["receipt"]+=1;return base_receipt.verify(value)
    class InvalidProbe:
        def verify(self,value):calls["invalid"]+=1;return base_invalid.verify(value)
    class GrantProbe:
        def authenticate(self,value,raw):calls["grant"]+=1;return base_grants.authenticate(value,raw)
    repo._receipt=ReceiptProbe();repo._invalid_evidence=InvalidProbe();repo._grants=GrantProbe();repo._revalidate_stored_record(repo.get(c.operation.effect.effect_id))
    assert calls=={"receipt":1,"invalid":1,"grant":1}
    class ExplodingReceipt:
        def verify(self,value):raise RuntimeError("secret-provider-credential")
    repo._receipt=ExplodingReceipt()
    with pytest.raises(FoundationError) as error:repo._revalidate_stored_record(repo.get(c.operation.effect.effect_id))
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)

def test_invalid_completion_return_and_reload_authenticity_tamper_cannot_authorize_stabilization():
    c=stage_command();executor=Executor();original=executor.execute_once;executor.execute_once=lambda payload_value,request:replace(original(payload_value,request),effect_id="wrong-effect")
    repo,a,r,invalid,v,resolver=environment(executor)
    class TamperingRepository:
        def __getattr__(self,name):return getattr(repo,name)
        def complete_invalid_dispatch(self,effect_id,evidence):
            record=repo.complete_invalid_dispatch(effect_id,evidence);object.__setattr__(record.invalid_evidence[0],"tag",D[0]);object.__setattr__(record.invalid_evidence_admissions[0],"authenticated_evidence_digest",record.invalid_evidence[0].authenticated_evidence_digest);return record
    with pytest.raises(FoundationError) as error:EffectBrokerV2(TamperingRepository(),resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    assert repo.get(c.operation.effect.effect_id).dispatch is DispatchState.RELINQUISHED

@pytest.mark.parametrize("entrypoint",["consume_same","prove_quiescence","append_result","complete_dispatch","acquire_reconciliation","transfer_reconciliation"])
def test_every_existing_record_mutation_family_revalidates_before_inference_or_write(entrypoint):
    c=stage_command()
    if entrypoint in {"prove_quiescence","complete_dispatch"}:
        repo,a,r,invalid,v,resolver=environment(Executor());grant=execution_grant(a,c);record,_=repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150);record=repo.begin_dispatch(c.operation.effect.effect_id)
        if entrypoint=="prove_quiescence":record=repo.orphan(c.operation.effect.effect_id);argument=v.proof(record,"quiescent",150)
        else:argument=r.issue(dispatch_receipt_content(c,observation_for(c),record))
        object.__setattr__(record.grant,"tag",D[0])
        invoke=(lambda:repo.prove_quiescence(c.operation.effect.effect_id,argument,now_epoch=150)) if entrypoint=="prove_quiescence" else (lambda:repo.complete_dispatch(c.operation.effect.effect_id,argument,None,now_epoch=150))
    else:
        disposition=ObservationDisposition.INDETERMINATE if entrypoint in {"acquire_reconciliation","transfer_reconciliation"} else ObservationDisposition.FOUND
        ex=bound_executor(c,Executor(disposition));repo,a,r,invalid,v,resolver=environment(ex);grant=execution_grant(a,c);record=EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,grant,now_epoch=150)
        if entrypoint=="consume_same":invoke=lambda:repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150)
        elif entrypoint=="append_result":invoke=lambda:repo.append_result(c.operation.effect.effect_id,record.results[0],None,now_epoch=150)
        else:
            adapter=AdapterDescriptorV1("docker","lookup","1.0.0");recon_grant=reconciliation_grant(a,c,adapter);_,claim,_=repo.acquire_reconciliation(c.canonical_bytes,recon_grant,now_epoch=150)
            if entrypoint=="acquire_reconciliation":invoke=lambda:repo.acquire_reconciliation(c.canonical_bytes,recon_grant,now_epoch=150,continuation=claim)
            else:
                record=repo.interrupt_reconciliation(c.operation.effect.effect_id,claim);transfer_grant=a.issue_reconciliation(replace(recon_grant.content,grant_ref="transfer-gate",owner_ref="other-owner",ownership_epoch=2));proof=v.proof(repo.get(c.operation.effect.effect_id),"quiescent",150);invoke=lambda:repo.transfer_reconciliation(c.canonical_bytes,transfer_grant,proof=proof,now_epoch=150)
        record=repo.get(c.operation.effect.effect_id);object.__setattr__(record.results[0],"tag",D[0]);object.__setattr__(record.receipt_admissions[0],"receipt_digest",record.results[0].authenticated_receipt_digest)
    retained=repo.get(c.operation.effect.effect_id);before_digest=retained.record_digest;before_history=(retained.results,retained.receipt_admissions,retained.invalid_evidence,retained.invalid_evidence_admissions,retained.reconciliation_claims)
    with pytest.raises(FoundationError) as error:invoke()
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    after=repo.get(c.operation.effect.effect_id);assert after is retained and after.record_digest==before_digest
    assert (after.results,after.receipt_admissions,after.invalid_evidence,after.invalid_evidence_admissions,after.reconciliation_claims)==before_history

def test_submit_attempt_revalidates_tampered_predecessor_before_digest_or_receipt_authority():
    stage=stage_command();ex=bound_executor(stage,Executor());repo,a,r,invalid,v,resolver=environment(ex);grant=execution_grant(a,stage);stage_record=EffectBrokerV2(repo,resolver,a,r,invalid).execute(stage.canonical_bytes,grant,now_epoch=150)
    p=stage.preparation;ref=StagePredecessorV2(p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,p.project_ref,p.run_id,p.plan_fingerprint,p.preparation_digest,p.workload_digest,stage.operation.effect.effect_id,stage_record.results[0].authenticated_receipt_digest,stage_record.record_digest);submit=build_submit_command(p,"mutation-gate",payload(EffectKind.SUBMIT,p),descriptor(),ref);submit_grant=execution_grant(a,submit,"submit-gate")
    object.__setattr__(stage_record.results[0],"tag",D[0]);object.__setattr__(stage_record.receipt_admissions[0],"receipt_digest",stage_record.results[0].authenticated_receipt_digest)
    before_digest=stage_record.record_digest
    with pytest.raises(FoundationError) as error:repo.consume_attempt(submit.canonical_bytes,submit_grant,now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    assert repo.get(stage.operation.effect.effect_id) is stage_record and stage_record.record_digest==before_digest and repo.get(submit.operation.effect.effect_id) is None

@pytest.mark.parametrize("operation",["append_result","complete_dispatch"])
def test_incoming_receipt_admission_requires_exact_true_without_partial_write(operation):
    c=stage_command();repo,a,r,invalid,v,resolver=environment(Executor());grant=execution_grant(a,c);record,_=repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150);record=repo.begin_dispatch(c.operation.effect.effect_id);receipt=r.issue(dispatch_receipt_content(c,observation_for(c),record))
    class TruthyReceiptVerifier:
        def verify(self,value):return object()
    repo._receipt=TruthyReceiptVerifier();before_digest=record.record_digest;before=(record.results,record.receipt_admissions,record.dispatch,tuple(repo._records))
    invoke=(lambda:repo.append_result(c.operation.effect.effect_id,receipt,None,now_epoch=150)) if operation=="append_result" else (lambda:repo.complete_dispatch(c.operation.effect.effect_id,receipt,None,now_epoch=150))
    with pytest.raises(FoundationError) as error:invoke()
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    after=repo.get(c.operation.effect.effect_id);assert after is record and after.record_digest==before_digest
    assert (after.results,after.receipt_admissions,after.dispatch,tuple(repo._records))==before

@pytest.mark.parametrize("existing",[False,True])
def test_execution_grant_admission_requires_exact_true_and_never_stores_or_mutates(existing):
    c=stage_command();repo,a,r,invalid,v,resolver=environment(Executor());grant=execution_grant(a,c);record=None
    if existing:record,_=repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150)
    class TruthyGrantVerifier:
        def __getattr__(self,name):return getattr(a,name)
        def verify(self,value,raw,*,now_epoch):return object()
    repo._grants=TruthyGrantVerifier();before_members=tuple(repo._records);before_digest=None if record is None else record.record_digest
    with pytest.raises(FoundationError) as error:repo.consume_attempt(c.canonical_bytes,grant,now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID);assert tuple(repo._records)==before_members
    if existing:assert repo.get(c.operation.effect.effect_id) is record and record.record_digest==before_digest
    else:assert repo.get(c.operation.effect.effect_id) is None

@pytest.mark.parametrize("operation",["acquire","transfer"])
def test_reconciliation_grant_admission_requires_exact_true_without_claim_mutation(operation):
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150);adapter=AdapterDescriptorV1("docker","lookup","1.0.0");grant=reconciliation_grant(a,c,adapter)
    if operation=="transfer":
        _,claim,_=repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150);repo.interrupt_reconciliation(c.operation.effect.effect_id,claim);record=repo.get(c.operation.effect.effect_id);proof=v.proof(record,"quiescent",150);grant=a.issue_reconciliation(replace(grant.content,grant_ref="truthy-transfer",owner_ref="other-owner",ownership_epoch=2))
    else:record=repo.get(c.operation.effect.effect_id);proof=None
    class TruthyGrantVerifier:
        def __getattr__(self,name):return getattr(a,name)
        def verify_reconciliation(self,value,*,now_epoch):return object()
    repo._grants=TruthyGrantVerifier();before_digest=record.record_digest;before=(record.reconciliation,record.reconciliation_claims,tuple(repo._records))
    invoke=(lambda:repo.acquire_reconciliation(c.canonical_bytes,grant,now_epoch=150)) if operation=="acquire" else (lambda:repo.transfer_reconciliation(c.canonical_bytes,grant,proof=proof,now_epoch=150))
    with pytest.raises(FoundationError) as error:invoke()
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    after=repo.get(c.operation.effect.effect_id);assert after is record and after.record_digest==before_digest
    assert (after.reconciliation,after.reconciliation_claims,tuple(repo._records))==before

def test_broker_service_grant_gate_rejects_truthy_nonbool_before_resolution_or_mutation():
    c=stage_command();executor=Executor();repo,a,r,invalid,v,resolver=environment(executor);invalid_grant=replace(execution_grant(a,c),tag=D[0])
    class TruthyAuthority:
        def verify(self,value,raw,*,now_epoch):return object()
    class CountingResolver:
        def __init__(self):self.calls=0
        def resolve(self,request):self.calls+=1;raise AssertionError("resolver must not run")
    counting=CountingResolver();before=tuple(repo._records)
    with pytest.raises(FoundationError) as error:EffectBrokerV2(repo,counting,TruthyAuthority(),r,invalid).execute(c.canonical_bytes,invalid_grant,now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    assert counting.calls==0 and executor.calls==0 and tuple(repo._records)==before and repo.get(c.operation.effect.effect_id) is None

def test_reconciliation_service_grant_gate_rejects_truthy_nonbool_before_resolution_or_lookup():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);record=EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150);adapter=Adapter(observation_for(c));grant=replace(reconciliation_grant(a,c,adapter.descriptor),tag=D[0])
    class TruthyAuthority:
        def verify_reconciliation(self,value,*,now_epoch):return object()
    class CountingResolver:
        def __init__(self):self.calls=0
        def resolve(self,request):self.calls+=1;raise AssertionError("resolver must not run")
    counting=CountingResolver();before_digest=record.record_digest;before=(record.reconciliation,record.reconciliation_claims,tuple(repo._records))
    with pytest.raises(FoundationError) as error:ReconciliationServiceV1(repo,TruthyAuthority(),counting,r,invalid).reconcile(c.canonical_bytes,grant,now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID)
    after=repo.get(c.operation.effect.effect_id);assert counting.calls==0 and adapter.calls==0 and after is record and after.record_digest==before_digest
    assert (after.reconciliation,after.reconciliation_claims,tuple(repo._records))==before

def test_valid_reconciliation_observation_receipt_signing_failure_interrupts_without_invalid_evidence():
    c=stage_command();ex=bound_executor(c,Executor(ObservationDisposition.INDETERMINATE));repo,a,r,invalid,v,resolver=environment(ex);initial=EffectBrokerV2(repo,resolver,a,r,invalid).execute(c.canonical_bytes,execution_grant(a,c),now_epoch=150)
    adapter=Adapter(observation_for(c))
    class RejectReceiptSigner:
        def issue(self,content):raise RuntimeError("secret-provider-credential")
    service=ReconciliationServiceV1(repo,a,AdapterResolver(adapter),RejectReceiptSigner(),invalid);grant=reconciliation_grant(a,c,adapter.descriptor)
    with pytest.raises(FoundationError) as error:service.reconcile(c.canonical_bytes,grant,now_epoch=150)
    assert_closed_error(error,DiagnosticCode.AUTHORITY_INVALID);assert adapter.calls==1
    retained=repo.get(c.operation.effect.effect_id)
    assert not retained.reconciliation.active and not retained.reconciliation.completed
    assert retained.invalid_codes==initial.invalid_codes and retained.invalid_evidence==initial.invalid_evidence and retained.invalid_evidence_admissions==initial.invalid_evidence_admissions
