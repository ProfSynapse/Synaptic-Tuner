from dataclasses import replace
import pytest
from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, ProviderPlanRef, TrainingPlan, TrainingPlanBasisV1
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, TrainingRunState, VerifiedArtifact
from tuner.execution.coordinator_v1.model import (ArtifactManifestV1, ArtifactVerificationContentV1,
 AuthenticatedArtifactVerificationReceiptV1, AuthenticatedFoundationRecordAssessmentV1,
 FoundationRecordAssessmentContentV1, ReceiptAssessmentV1, ReceiptFreshnessV1,
 AuthenticatedProviderRunObservationV1, ProviderRunObservationContentV1,
 EffectIntentV1, ProviderRunPhaseV1, ProviderRunReadRequestV1, VerificationVerdictV1, WorkflowPhaseV1, WorkflowRecordV1)
from tuner.execution.coordinator_v1.state_machine import (WorkflowTransitionError,
 apply_artifact_verification, apply_cancel_effect_record as _apply_cancel, apply_provider_observation,
 apply_reverification, apply_stage_effect_record as _apply_stage, apply_submit_effect_record as _apply_submit,
 begin_preparation, project_run_outcome, record_cancel_intent, record_stage_intent,
 record_submit_intent, provider_run_read_request)
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.canonical import DiagnosticCode, canonical_bytes, domain_digest, parse_canonical_object
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1, build_cancel_command, build_stage_command, build_submit_command, parse_exact_command
from tuner.execution.foundation_v2.executors import ExecutorDescriptorV1
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.receipts import ReceiptAuthorityV2, ReceiptContentV2
from tuner.execution.foundation_v2.references import CancellationRefV1, ExecutionScopeV1, ProviderRunRefV1, ProviderStageRefV1, ScopedProviderRunRefV1, StagePredecessorV2
from tuner.execution.foundation_v2.repository import DispatchState, EffectRecordV2, EffectState, ReceiptAdmissionV2, ReceiptFreshnessV2

D = tuple(c * 64 for c in "123456789abcdef")
PROVIDER = ProviderRef("provider-a", "profile-a"); SCOPE = ExecutionScopeV1("account-a", "namespace-a")
RUN = TrainingRunRef("run-a", "project-a")
DESC = ProviderDescriptor("synaptic-provider-descriptor/v1", "provider-a", "Provider A", "1.0.0", ProviderCapabilities(True, True, True, True, True, True))
BASIS = TrainingPlanBasisV1("synaptic-training-plan-basis/v1", "request-a", RUN.project_ref, *D[:5])
CONTEXT = ProviderPlanContextV1("synaptic-provider-plan-context/v1", PROVIDER, BASIS.basis_digest, DESC.descriptor_digest, D[5])
PLAN = TrainingPlan("synaptic-training-plan/v2", BASIS, ProviderPlanRef(CONTEXT.provider_context_digest))
STAGE_REF = ProviderStageRefV1("provider-a", "profile-a", "account-a", "namespace-a", "stage-a")
PROVIDER_RUN = ScopedProviderRunRefV1("provider-a", "profile-a", "account-a", "namespace-a", "job-a")

def prep(): return CanonicalPreparationV2.build(provider=PROVIDER, scope=SCOPE, project_ref=RUN.project_ref, run_id=RUN.run_id, plan_fingerprint=PLAN.plan_fingerprint, source_digest=BASIS.source_digest, workload_digest=BASIS.workload_digest, runtime_digest=BASIS.runtime_digest, resource_digest=D[7], artifact_contract_digest=BASIS.artifact_policy_digest, quote_digest=D[8], secret_requirements_digest=D[9])
def planned(): return WorkflowRecordV1.planned(run=RUN, plan=PLAN, preflight_digest=D[6], context=CONTEXT, provider=PROVIDER, descriptor=DESC)
def intent(kind):
 p=prep(); payload=CanonicalProviderPayloadV1.build(PROVIDER.provider_id, f"{kind}-payload/v2", p.workload_digest); ex=ExecutorDescriptorV1(PROVIDER.provider_id,"executor-a","1.0.0")
 if kind=="stage": cmd=build_stage_command(p,"nonce-stage",payload,ex)
 elif kind=="submit": cmd=build_submit_command(p,"nonce-submit",payload,ex,StagePredecessorV2(PROVIDER.provider_id,PROVIDER.profile_ref,SCOPE.account_ref,SCOPE.namespace_ref,RUN.project_ref,RUN.run_id,PLAN.plan_fingerprint,p.preparation_digest,p.workload_digest,"stage-effect",D[12],D[13]))
 else: cmd=build_cancel_command(p,"nonce-cancel",payload,ex,CancellationRefV1(ProviderRunRefV1(PROVIDER_RUN.provider_job_ref),D[14]))
 return EffectIntentV1.from_command_bytes(cmd.canonical_bytes)

class Auth:
 def __init__(self, allowed=True): self.allowed=allowed
 def authenticate_grant(self, value, command_bytes): return self.allowed and type(command_bytes) is bytes
 def authenticate_receipt(self, value): return self.allowed
 def authenticate_invalid_evidence(self,value):return self.allowed
class AssessmentAuth:
 def __init__(self,allowed=True):self.allowed=allowed
 def authenticate(self,value):return self.allowed
class Verifier:
 def __init__(self, allowed=True): self.allowed=allowed
 def authenticate(self, value): return self.allowed

def record(effect, state, reference=None):
 grant=GrantAuthorityV2("grant-authority",b"g"*32).issue(effect.canonical_command_bytes,grant_ref=f"grant-{effect.kind.value}",policy_digest=D[10],requirement_digest=D[11],not_before_epoch=1,expires_at_epoch=100)
 base=EffectRecordV2(effect.canonical_command_bytes,grant,DispatchState.RELINQUISHED,EffectState.UNRESOLVED,1)
 results=[]; admissions=[]; terminal=[]
 def append(content,*,final=False):
  receipt=ReceiptAuthorityV2("receipt-authority",b"r"*32).issue(content);expected=("dispatch",grant.content.grant_ref,1,1,base.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest);codes=()
  admission=ReceiptAdmissionV2(receipt.authenticated_receipt_digest,content.source_kind,content.source_owner_ref,content.source_generation,content.source_ownership_epoch,content.source_claim_digest,content.source_grant_ref,content.source_grant_digest,*expected,ReceiptFreshnessV2.FRESH,final,codes)
  results.append(receipt);admissions.append(admission)
 if state is EffectState.CONTRADICTED:
  found=ReceiptContentV2(effect.effect_id,effect.command_digest,ObservationDisposition.FOUND,D[0],1,reference if effect.kind.value=="stage" else None,reference if effect.kind.value=="submit" else None,reference if effect.kind.value=="cancel" else None,None,"dispatch",grant.content.grant_ref,1,1,base.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest)
  absent=ReceiptContentV2(effect.effect_id,effect.command_digest,ObservationDisposition.DEFINITELY_ABSENT,D[3],1,None,None,None,D[1],"dispatch",grant.content.grant_ref,1,1,base.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest)
  append(found);append(absent,final=True);terminal=[found.semantic_digest,absent.semantic_digest]
 if state in {EffectState.FOUND,EffectState.DEFINITELY_ABSENT}:
  disp=ObservationDisposition.FOUND if state is EffectState.FOUND else ObservationDisposition.DEFINITELY_ABSENT
  kw={"stage_ref":None,"provider_run":None,"cancellation":None}
  if state is EffectState.FOUND: kw[{"stage":"stage_ref","submit":"provider_run","cancel":"cancellation"}[effect.kind.value]]=reference
  content=ReceiptContentV2(effect.effect_id,effect.command_digest,disp,D[0],1,kw["stage_ref"],kw["provider_run"],kw["cancellation"],D[1] if state is EffectState.DEFINITELY_ABSENT else None,"dispatch",grant.content.grant_ref,1,1,base.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest)
  append(content,final=state is EffectState.DEFINITELY_ABSENT);terminal=[content.semantic_digest]
 if state is EffectState.INDETERMINATE:
  content=ReceiptContentV2(effect.effect_id,effect.command_digest,ObservationDisposition.INDETERMINATE,D[0],1,None,None,None,None,"dispatch",grant.content.grant_ref,1,1,base.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest);append(content)
 return EffectRecordV2(effect.canonical_command_bytes,grant,DispatchState.RELINQUISHED,state,1,results=tuple(results),receipt_admissions=tuple(admissions),terminal_content_digests=tuple(terminal))

def assessment(record):
 command=parse_exact_command(record.command_bytes);grant=record.grant
 claim=lambda c:{"owner_ref":c.owner_ref,"generation":c.generation,"ownership_epoch":c.ownership_epoch,"claimed_at_epoch":c.claimed_at_epoch,"target_digest":c.target_digest,"grant_ref":c.grant_ref,"grant_digest":c.grant_digest,"grant_lineage":[x.to_dict()|{"binding_digest":x.binding_digest} for x in c.grant_lineage],"active":c.active,"completed":c.completed,"claim_digest":c.claim_digest}
 snapshot={"schema_version":"synaptic-foundation-effect-snapshot/v1","command":command.to_dict(),"command_bytes_digest":domain_digest("synaptic-foundation-command-bytes/v1",record.command_bytes),"grant":parse_canonical_object(grant.canonical_bytes,name="grant"),"dispatch":record.dispatch.value,"state":record.state.value,"attempt_count":record.attempt_count,"dispatch_epoch":record.dispatch_epoch,"receipts":[parse_canonical_object(x.canonical_bytes,name="receipt") for x in record.results],"receipt_admissions":[x.to_dict()|{"admission_digest":x.admission_digest} for x in record.receipt_admissions],"invalid_evidence":[parse_canonical_object(x.canonical_bytes,name="invalid evidence") for x in record.invalid_evidence],"invalid_evidence_admissions":[x.to_dict()|{"admission_digest":x.admission_digest} for x in record.invalid_evidence_admissions],"terminal_content_digests":list(record.terminal_content_digests),"invalid_codes":[x.value for x in record.invalid_codes],"reconciliation":None if record.reconciliation is None else claim(record.reconciliation),"reconciliation_claims":[claim(x) for x in record.reconciliation_claims],"b2_record_digest":record.record_digest}
 values=[]
 for index,receipt in enumerate(record.results):
  c=receipt.content;stale=DiagnosticCode.STALE_RESULT in record.invalid_codes;final=c.disposition is ObservationDisposition.DEFINITELY_ABSENT and DiagnosticCode.FINALITY_UNPROVEN not in record.invalid_codes
  codes=[]
  if stale:codes.append("stale_result")
  if c.disposition is ObservationDisposition.DEFINITELY_ABSENT and not final:codes.append("finality_unproven")
  values.append(ReceiptAssessmentV1(receipt.authenticated_receipt_digest,record.receipt_admissions[index].admission_digest,c.source_kind,c.source_owner_ref,c.source_generation,c.source_ownership_epoch,ReceiptFreshnessV1.STALE if stale else ReceiptFreshnessV1.FRESH,final,tuple(codes)))
 content=FoundationRecordAssessmentContentV1("synaptic-foundation-record-assessment-content/v1",command.operation.effect.effect_id,command.digest,snapshot["command_bytes_digest"],record.record_digest,domain_digest("synaptic-foundation-record-evidence/v1",canonical_bytes(snapshot)),tuple(x.authenticated_receipt_digest for x in record.results),record.terminal_content_digests,tuple(values),tuple(x.admission_digest for x in record.invalid_evidence_admissions),tuple(x.value for x in record.invalid_codes),"assessor-a","1.0.0","2026-08-26T00:00:00Z")
 return AuthenticatedFoundationRecordAssessmentV1.parse(AuthenticatedFoundationRecordAssessmentV1(content,"authority-a","key-a","a"*64).canonical_bytes)

def apply_stage_effect_record(c,r,a):return _apply_stage(c,r,assessment(r),a,AssessmentAuth())
def apply_submit_effect_record(c,r,a):return _apply_submit(c,r,assessment(r),a,AssessmentAuth())
def apply_cancel_effect_record(c,r,a):return _apply_cancel(c,r,assessment(r),a,AssessmentAuth())

def stage_source():
 raw=intent("stage"); current=record_stage_intent(begin_preparation(planned()),prep(),raw); return current,raw
def staged():
 current,raw=stage_source(); return apply_stage_effect_record(current,record(raw,EffectState.FOUND,STAGE_REF),Auth())
def queued():
 return queued_evidence()[0]
def queued_evidence():
 current=staged(); raw=intent("submit"); current=record_submit_intent(current,raw); foundation=record(raw,EffectState.FOUND,PROVIDER_RUN); return apply_submit_effect_record(current,foundation,Auth()),foundation

class ObservationAuth:
 def __init__(self,allowed=True):self.allowed=allowed
 def authenticate(self,value):return self.allowed
def observation(current,foundation,phase,evidence,diagnostic=None):
 ass=assessment(foundation);request=provider_run_read_request(current,foundation,ass,Auth(),AssessmentAuth());ref=current.provider_run_ref.reference
 content=ProviderRunObservationContentV1("synaptic-provider-run-observation-content/v1",request.request_digest,current.record_digest,current.revision,current.run,current.provider_run_ref.binding_digest,ref.provider_id,ref.profile_ref,ref.account_ref,ref.namespace_ref,ref.provider_job_ref,phase,canonical_bytes(evidence),diagnostic,"observer-a","1.0.0","2026-08-26T00:00:00Z")
 envelope=AuthenticatedProviderRunObservationV1.parse(AuthenticatedProviderRunObservationV1(content,"authority-a","key-a","b"*64).canonical_bytes)
 return request,envelope

def read_request_variant(request,**changes):
 values={name:getattr(request,name) for name in ("source_workflow_record_digest","source_revision","run","provider_run","submit_command_bytes","foundation_record","assessment","foundation_binding","foundation_outcome","found_receipt_digest")};values.update(changes)
 doc={"schema_version":"synaptic-provider-run-read-request/v1","source_workflow_record_digest":values["source_workflow_record_digest"],"source_revision":values["source_revision"],"run":values["run"].to_dict(),"provider_run_binding_digest":values["provider_run"].binding_digest,"submit_command_bytes_digest":domain_digest("synaptic-foundation-command-bytes/v1",values["submit_command_bytes"]),"foundation_record_digest":values["foundation_record"].record_digest,"assessment_digest":values["assessment"].authenticated_assessment_digest,"foundation_binding_digest":values["foundation_binding"].binding_digest,"foundation_outcome_digest":values["foundation_outcome"].outcome_digest,"found_receipt_digest":values["found_receipt_digest"]}
 raw=canonical_bytes(doc)
 return ProviderRunReadRequestV1(**values,canonical_bytes=raw,request_digest=domain_digest("synaptic-provider-run-read-request/v1",raw))

def test_named_effect_events_derive_refs_and_phases():
 s=staged(); assert s.phase is WorkflowPhaseV1.STAGED and s.provider_stage_ref.reference==STAGE_REF
 q=queued(); assert q.phase is WorkflowPhaseV1.QUEUED and q.provider_run_ref.reference==PROVIDER_RUN
 assert q.stage.foundation_bindings and q.stage.foundation_outcomes

@pytest.mark.parametrize("state,phase",[(EffectState.INDETERMINATE,WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED),(EffectState.DEFINITELY_ABSENT,WorkflowPhaseV1.FAILED)])
def test_stage_disposition_mapping(state,phase):
 current,raw=stage_source(); assert apply_stage_effect_record(current,record(raw,state),Auth()).phase is phase

def test_authenticated_contradiction_is_terminal():
 current,raw=stage_source(); contradicted=apply_stage_effect_record(current,record(raw,EffectState.CONTRADICTED,STAGE_REF),Auth())
 assert contradicted.phase is WorkflowPhaseV1.CONTRADICTED
 with pytest.raises(WorkflowTransitionError): apply_stage_effect_record(contradicted,record(raw,EffectState.FOUND,STAGE_REF),Auth())

def test_foundation_authentication_and_history_are_closed():
 current,raw=stage_source()
 with pytest.raises(WorkflowTransitionError): apply_stage_effect_record(current,record(raw,EffectState.INDETERMINATE),Auth(False))
 foundation_record=record(raw,EffectState.INDETERMINATE)
 one=apply_stage_effect_record(current,foundation_record,Auth())
 assert apply_stage_effect_record(one,foundation_record,Auth()) is one

def test_provider_observation_forbids_regression_and_is_idempotent():
 q,foundation=queued_evidence();request,running=observation(q,foundation,ProviderRunPhaseV1.RUNNING,{"phase":"running"})
 r=apply_provider_observation(q,request,running,ObservationAuth()); assert r.phase is WorkflowPhaseV1.RUNNING
 assert apply_provider_observation(r,request,running,ObservationAuth()) is r
 mismatched=read_request_variant(request,foundation_record=record(intent("stage"),EffectState.INDETERMINATE))
 with pytest.raises(WorkflowTransitionError,match="read request Foundation record"):
  apply_provider_observation(r,mismatched,running,ObservationAuth())
 request,back=observation(r,foundation,ProviderRunPhaseV1.QUEUED,{"phase":"queued"})
 with pytest.raises(WorkflowTransitionError): apply_provider_observation(r,request,back,ObservationAuth())
 with pytest.raises(WorkflowTransitionError,match="authentication"):
  apply_provider_observation(q,*observation(q,foundation,ProviderRunPhaseV1.RUNNING,{"phase":"denied"}),ObservationAuth(False))

@pytest.mark.parametrize("component",["foundation_record","assessment","foundation_binding","foundation_outcome","found_receipt_digest"])
def test_provider_observation_rejects_request_not_equal_to_retained_submit_evidence(component):
 q,foundation=queued_evidence();request,envelope=observation(q,foundation,ProviderRunPhaseV1.RUNNING,{"phase":"running-mismatched-request"})
 alternatives={
  "foundation_record":record(intent("stage"),EffectState.INDETERMINATE),
  "assessment":replace(request.assessment,authority_ref="authority-b"),
  "foundation_binding":q.stage.foundation_bindings[-1],
  "foundation_outcome":q.stage.foundation_outcomes[-1],
  "found_receipt_digest":D[2],
 }
 mismatched=read_request_variant(request,**{component:alternatives[component]})
 with pytest.raises(WorkflowTransitionError,match="read request"):
  apply_provider_observation(q,mismatched,envelope,ObservationAuth())

def test_cancel_found_and_final_absence_restore_exact_origin():
 q,foundation=queued_evidence(); raw=intent("cancel"); pending=record_cancel_intent(q,raw)
 cref=CancellationRefV1(ProviderRunRefV1(PROVIDER_RUN.provider_job_ref),D[14])
 requested=apply_cancel_effect_record(pending,record(raw,EffectState.FOUND,cref),Auth())
 assert requested.phase is WorkflowPhaseV1.CANCEL_REQUESTED and requested.bound_cancellation
 request_obs,envelope=observation(requested,foundation,ProviderRunPhaseV1.RUNNING,{"phase":"running-after-cancel"})
 still_pending=apply_provider_observation(requested,request_obs,envelope,ObservationAuth())
 assert still_pending.phase is WorkflowPhaseV1.CANCEL_REQUESTED and still_pending.bound_cancellation==requested.bound_cancellation
 pending=record_cancel_intent(q,raw); restored=apply_cancel_effect_record(pending,record(raw,EffectState.DEFINITELY_ABSENT),Auth())
 assert restored.phase is WorkflowPhaseV1.QUEUED and restored.cancel.foundation_outcomes

def verification_receipt(current,manifest,verdict,tag="a"):
 evidence=canonical_bytes({"verification":"bounded"}); artifacts=lambda xs:[x.to_dict() for x in xs]
 content={"schema_version":"synaptic-artifact-verification-content/v1","source_workflow_record_digest":current.record_digest,"source_revision":current.revision,"run":current.run.to_dict(),"provider_run_binding_digest":current.provider_run_ref.binding_digest,"manifest_digest":manifest.manifest_digest,"artifact_source_digest":manifest.artifact_source_digest,"manifest_artifacts":artifacts(manifest.artifacts),"verified_artifacts":artifacts(manifest.artifacts if verdict is VerificationVerdictV1.VERIFIED else ()),"verdict":verdict.value,"diagnostic_code":None if verdict is VerificationVerdictV1.VERIFIED else "artifact_rejected","verifier_ref":"verifier-a","verifier_version":"1.0.0","canonical_evidence":{"verification":"bounded"},"evidence_digest":domain_digest("synaptic-artifact-verification-evidence/v1",evidence),"checked_at":"2026-08-26T00:00:00Z"}
 content["content_digest"]=domain_digest("synaptic-artifact-verification-content/v1",canonical_bytes({k:v for k,v in content.items() if k!="content_digest"}))
 envelope={"schema_version":"synaptic-authenticated-artifact-verification/v1","content":content,"authority_ref":"authority-a","key_ref":"key-a","tag":(tag*64)[:64]}
 return AuthenticatedArtifactVerificationReceiptV1.parse(canonical_bytes(envelope))

def test_authenticated_verification_and_reverification():
 q,foundation=queued_evidence();request,envelope=observation(q,foundation,ProviderRunPhaseV1.SUCCEEDED,{"phase":"succeeded"});done=apply_provider_observation(q,request,envelope,ObservationAuth())
 artifact=VerifiedArtifact("adapter",D[4],10); manifest=ArtifactManifestV1.build(run=RUN,provider_run=PROVIDER_RUN,artifacts=(artifact,),artifact_source_digest=D[5],canonical_evidence=canonical_bytes({"inventory":1}))
 receipt=verification_receipt(done,manifest,VerificationVerdictV1.VERIFIED)
 assert AuthenticatedArtifactVerificationReceiptV1.parse(receipt.canonical_bytes)==receipt
 verified=apply_artifact_verification(done,manifest,receipt,Verifier()); assert verified.phase is WorkflowPhaseV1.VERIFIED
 assert apply_artifact_verification(verified,manifest,receipt,Verifier()) is verified
 replay=verification_receipt(verified,manifest,VerificationVerdictV1.VERIFIED,"b")
 reverified=apply_reverification(verified,manifest,replay,Verifier()); assert reverified.revision==verified.revision+1
 assert project_run_outcome(reverified).state is TrainingRunState.SUCCEEDED
 with pytest.raises(WorkflowTransitionError): apply_reverification(verified,manifest,replay,Verifier(False))

def test_rejected_verification_is_retained_and_replays_after_failure():
 q,foundation=queued_evidence();request,envelope=observation(q,foundation,ProviderRunPhaseV1.SUCCEEDED,{"phase":"succeeded-rejected"});done=apply_provider_observation(q,request,envelope,ObservationAuth())
 artifact=VerifiedArtifact("adapter",D[4],10); manifest=ArtifactManifestV1.build(run=RUN,provider_run=PROVIDER_RUN,artifacts=(artifact,),artifact_source_digest=D[5],canonical_evidence=canonical_bytes({"inventory":2}))
 receipt=verification_receipt(done,manifest,VerificationVerdictV1.REJECTED,"c")
 failed=apply_artifact_verification(done,manifest,receipt,Verifier())
 assert failed.phase is WorkflowPhaseV1.VERIFICATION_FAILED and failed.verification_receipts==(receipt,)
 assert apply_artifact_verification(failed,manifest,receipt,Verifier()) is failed
