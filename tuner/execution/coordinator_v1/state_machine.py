"""Explicit evidence-driven coordinator reducers."""
from __future__ import annotations
from synaptic_tuner.api.v1.results import TrainingRunState
from synaptic_tuner.api.v1.runs_facade import RunOutcome
from tuner.execution.foundation_v2.authority import AuthenticatedGrantV2, GrantContentV2
from tuner.execution.foundation_v2.canonical import DiagnosticCode, canonical_bytes, domain_digest, exact_integer, parse_canonical_object, safe_ref
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.receipts import AuthenticatedInvalidEvidenceV2, AuthenticatedReceiptV2
from tuner.execution.foundation_v2.references import CancellationRefV1
from tuner.execution.foundation_v2.repository import DispatchState, EffectRecordV2, EffectState, ReconciliationOwnershipV2
from .model import (ArtifactManifestV1,AuthenticatedArtifactVerificationReceiptV1,AuthenticatedFoundationRecordAssessmentV1,AuthenticatedProviderRunObservationV1,BoundCancellationRefV1,
 BoundProviderRunRefV1,BoundProviderStageRefV1,EffectIntentV1,FoundationDispositionV1,
 FoundationEffectBindingV1,FoundationEffectOutcomeV1,ProviderReadPurposeV1,ProviderRunReadRequestV1,ProviderRunPhaseV1,
 ReceiptFreshnessV1,VerificationVerdictV1,WorkflowPhaseV1,WorkflowRecordV1)

class WorkflowTransitionError(ValueError): pass

def _claim_doc(c):
 return {"owner_ref":c.owner_ref,"generation":c.generation,"ownership_epoch":c.ownership_epoch,"claimed_at_epoch":c.claimed_at_epoch,"target_digest":c.target_digest,"grant_ref":c.grant_ref,"grant_digest":c.grant_digest,"grant_lineage":[x.to_dict()|{"binding_digest":x.binding_digest} for x in c.grant_lineage],"active":c.active,"completed":c.completed,"claim_digest":c.claim_digest}

def _validate_claims(record):
 claims=record.reconciliation_claims
 if type(claims) is not tuple or any(type(c) is not ReconciliationOwnershipV2 for c in claims): raise ValueError("reconciliation claims must be exact tuple")
 if claims:
  first=claims[0]
  if (first.generation,first.ownership_epoch)!=(1,1): raise ValueError("reconciliation genesis must be generation/epoch 1/1")
  for i,c in enumerate(claims):
   safe_ref(c.owner_ref,"owner_ref");safe_ref(c.grant_ref,"grant_ref");exact_integer(c.generation,"generation",minimum=1);exact_integer(c.ownership_epoch,"ownership_epoch",minimum=1);exact_integer(c.claimed_at_epoch,"claimed_at_epoch")
   if type(c.active) is not bool or type(c.completed) is not bool or (c.active and c.completed): raise ValueError("reconciliation flags invalid")
   if i<len(claims)-1 and c.active: raise ValueError("historical reconciliation owner remains active")
  for old,new in zip(claims,claims[1:]):
   if new.claimed_at_epoch<old.claimed_at_epoch or new.grant_ref==old.grant_ref: raise ValueError("reconciliation time/grant regressed")
   if old.completed:
    if (new.owner_ref,new.generation,new.ownership_epoch,new.target_digest)!=(old.owner_ref,old.generation+1,old.ownership_epoch+1,old.target_digest): raise ValueError("reconciliation retry invalid")
   else:
    if old.active or (new.generation,new.ownership_epoch)==(old.generation,old.ownership_epoch) or new.generation!=old.generation or new.ownership_epoch!=old.ownership_epoch+1 or new.owner_ref==old.owner_ref or new.target_digest==old.target_digest: raise ValueError("reconciliation transfer invalid")
 if record.reconciliation!=(claims[-1] if claims else None): raise ValueError("current reconciliation must equal final claim")
 if sum(1 for c in claims if c.active)>1: raise ValueError("multiple active reconciliation owners")
 return claims

def _derive_foundation(intent,record,assessment,authenticator,assessor,provider_run):
 if type(intent) is not EffectIntentV1 or type(record) is not EffectRecordV2: raise TypeError("exact intent and EffectRecordV2 required")
 if type(record.command_bytes) is not bytes or record.command_bytes!=intent.canonical_command_bytes: raise ValueError("record command bytes differ from intent")
 command=parse_exact_command(record.command_bytes); effect=command.operation.effect; prep=command.preparation
 if command.canonical_bytes!=record.command_bytes or (effect.kind,effect.effect_id,command.digest)!=(intent.kind,intent.effect_id,intent.command_digest): raise ValueError("record command identity mismatch")
 if record.attempt_count!=1 or record.dispatch_epoch!=1: raise ValueError("foundation attempt and dispatch epoch must equal one")
 if record.dispatch in {DispatchState.OWNED_NOT_STARTED,DispatchState.OWNED_IN_FLIGHT}: raise ValueError("foundation dispatch is not stable")
 if type(record.dispatch) is not DispatchState or type(record.state) is not EffectState: raise TypeError("foundation enums must be exact")
 if type(record.results) is not tuple or type(record.receipt_admissions) is not tuple or type(record.invalid_evidence) is not tuple or type(record.invalid_evidence_admissions) is not tuple or type(record.terminal_content_digests) is not tuple or type(record.invalid_codes) is not tuple: raise TypeError("foundation histories must be exact tuples")
 if len(record.results)!=len(record.receipt_admissions) or len(record.invalid_evidence)!=len(record.invalid_evidence_admissions):raise ValueError("foundation evidence ledgers lack exact cardinality")
 if any(type(c) is not DiagnosticCode for c in record.invalid_codes): raise TypeError("foundation invalid codes must be exact")
 claims=_validate_claims(record)
 if claims and record.dispatch not in {DispatchState.RELINQUISHED,DispatchState.QUIESCENCE_PROVEN}: raise ValueError("claims require stable reconciliation dispatch")
 if record.dispatch is DispatchState.ORPHANED_UNPROVEN and (record.state is not EffectState.UNRESOLVED or claims): raise ValueError("orphaned dispatch must remain unresolved without claims")
 if record.dispatch is DispatchState.QUIESCENCE_PROVEN and not claims and record.state is not EffectState.UNRESOLVED: raise ValueError("quiescence without reconciliation must remain unresolved")
 grant=record.grant
 if type(grant) is not AuthenticatedGrantV2 or type(grant.content) is not GrantContentV2 or authenticator.authenticate_grant(grant,record.command_bytes) is not True: raise ValueError("foundation grant authentication failed")
 gc=grant.content
 expected=(command.digest,effect.effect_id,prep.preparation_digest,prep.provider.provider_id,prep.provider.profile_ref,prep.scope.account_ref,prep.scope.namespace_ref,effect.kind.value,command.payload.payload_kind)
 actual=(gc.command_digest,gc.effect_id,gc.preparation_digest,gc.provider_id,gc.profile_ref,gc.account_ref,gc.namespace_ref,gc.effect_kind,gc.payload_schema)
 if actual!=expected: raise ValueError("grant does not bind command")
 if type(assessment) is not AuthenticatedFoundationRecordAssessmentV1:raise TypeError("exact foundation assessment required")
 parsed_assessment=AuthenticatedFoundationRecordAssessmentV1.parse(assessment.canonical_bytes)
 if parsed_assessment.canonical_bytes!=assessment.canonical_bytes or assessor.authenticate(parsed_assessment) is not True:raise ValueError("foundation assessment authentication failed")
 ac=parsed_assessment.content
 current_epoch=record.reconciliation.ownership_epoch if record.reconciliation is not None else record.dispatch_epoch
 receipts=[];seen=set();terminals=[];derived_invalid=[];derived_state=EffectState.UNRESOLVED;authenticated_absences=set();fresh_found_receipts=[]
 scope=(prep.provider.provider_id,prep.provider.profile_ref,prep.scope.account_ref,prep.scope.namespace_ref)
 for index,supplied in enumerate(record.results):
  if type(supplied) is not AuthenticatedReceiptV2: raise TypeError("receipt must be exact")
  receipt=AuthenticatedReceiptV2.parse(supplied.canonical_bytes)
  if receipt!=supplied or receipt.canonical_bytes!=supplied.canonical_bytes or authenticator.authenticate_receipt(receipt) is not True: raise ValueError("receipt authentication failed")
  if receipt.authenticated_receipt_digest in seen: raise ValueError("duplicate authenticated receipt")
  seen.add(receipt.authenticated_receipt_digest);c=receipt.content
  if (c.command_digest,c.effect_id)!=(intent.command_digest,intent.effect_id) or c.result_epoch!=c.source_ownership_epoch: raise ValueError("receipt command/epoch mismatch")
  if c.source_kind=="dispatch":
   if (c.source_owner_ref,c.source_generation,c.source_ownership_epoch,c.source_claim_digest,c.source_grant_ref,c.source_grant_digest)!=(gc.grant_ref,1,record.dispatch_epoch,record.dispatch_source_digest,gc.grant_ref,grant.authenticated_grant_digest): raise ValueError("dispatch receipt ownership mismatch")
  elif not any((x.owner_ref,x.generation,x.ownership_epoch,x.claim_digest)==(c.source_owner_ref,c.source_generation,c.source_ownership_epoch,c.source_claim_digest) and any(g.grant_ref==c.source_grant_ref and g.grant_digest==c.source_grant_digest for g in x.grant_lineage) for x in claims): raise ValueError("reconciliation receipt ownership mismatch")
  if index>=len(ac.receipt_assessments):raise ValueError("assessment receipt order incomplete")
  ra=ac.receipt_assessments[index]
  admission=record.receipt_admissions[index]
  if (ra.authenticated_receipt_digest,ra.receipt_admission_digest,ra.source_kind,ra.source_owner_ref,ra.source_generation,ra.source_ownership_epoch)!=(receipt.authenticated_receipt_digest,admission.admission_digest,c.source_kind,c.source_owner_ref,c.source_generation,c.source_ownership_epoch):raise ValueError("assessment receipt source/order mismatch")
  if c.source_ownership_epoch>current_epoch:raise ValueError("receipt source epoch exceeds current authority")
  stale=ra.freshness is ReceiptFreshnessV1.STALE
  generated=tuple(DiagnosticCode(x) for x in ra.generated_invalid_codes);derived_invalid.extend(generated)
  terminal=None;candidate=EffectState.INDETERMINATE
  if c.disposition is ObservationDisposition.FOUND:
   ref={EffectKind.STAGE:c.stage_ref,EffectKind.SUBMIT:c.provider_run,EffectKind.CANCEL:c.cancellation}[intent.kind]
   if ref is None or sum(x is not None for x in (c.stage_ref,c.provider_run,c.cancellation))!=1: raise ValueError("FOUND typed reference mismatch")
   if intent.kind in {EffectKind.STAGE,EffectKind.SUBMIT} and (ref.provider_id,ref.profile_ref,ref.account_ref,ref.namespace_ref)!=scope: raise ValueError("FOUND reference scope mismatch")
   if intent.kind is EffectKind.CANCEL:
    reason=command.to_dict()["cancellation"]["reason_digest"]
    if type(ref) is not CancellationRefV1 or ref.run.provider_job_ref!=effect.cancel_target.provider_job_ref or ref.reason_digest!=reason or provider_run is None or ref.run.provider_job_ref!=provider_run.reference.provider_job_ref: raise ValueError("cancel receipt binding mismatch")
   terminal=c.semantic_digest;candidate=EffectState.INDETERMINATE if stale else EffectState.FOUND
   if not stale:fresh_found_receipts.append(receipt)
  elif c.disposition is ObservationDisposition.DEFINITELY_ABSENT:
   if ra.finality_verified:
    authenticated_absences.add(receipt.authenticated_receipt_digest);terminal=c.semantic_digest;candidate=EffectState.INDETERMINATE if stale else EffectState.DEFINITELY_ABSENT
   else:
    candidate=EffectState.INDETERMINATE
  elif ra.finality_verified:raise ValueError("finality may authenticate only absence")
  expected_codes=[]
  if stale:expected_codes.append(DiagnosticCode.STALE_RESULT)
  if c.disposition is ObservationDisposition.DEFINITELY_ABSENT and not ra.finality_verified:expected_codes.append(DiagnosticCode.FINALITY_UNPROVEN)
  if tuple(generated)!=tuple(expected_codes):raise ValueError("assessment generated invalid codes disagree with receipt")
  if terminal is not None:
   if terminal not in terminals:
    if terminals and terminal!=terminals[-1]: derived_state=EffectState.CONTRADICTED
    terminals.append(terminal)
  if derived_state is EffectState.CONTRADICTED: pass
  elif derived_state in {EffectState.FOUND,EffectState.DEFINITELY_ABSENT} and candidate is EffectState.INDETERMINATE: pass
  else: derived_state=candidate
  receipts.append(receipt)
 if len(ac.receipt_assessments)!=len(receipts):raise ValueError("assessment receipt count mismatch")
 assessed_invalid=tuple(DiagnosticCode(x) for x in ac.invalid_codes)
 invalid_digests=[]
 invalid_evidence_digests=set()
 prior=None
 for index,(supplied,admission) in enumerate(zip(record.invalid_evidence,record.invalid_evidence_admissions,strict=True),1):
  if type(supplied) is not AuthenticatedInvalidEvidenceV2:raise TypeError("invalid evidence must be exact")
  evidence=AuthenticatedInvalidEvidenceV2.parse(supplied.canonical_bytes)
  if evidence!=supplied or authenticator.authenticate_invalid_evidence(evidence) is not True:raise ValueError("invalid evidence authentication failed")
  if evidence.authenticated_evidence_digest in invalid_evidence_digests:raise ValueError("duplicate invalid evidence")
  invalid_evidence_digests.add(evidence.authenticated_evidence_digest)
  if (admission.authenticated_evidence_digest,admission.sequence,admission.prior_admission_digest)!=(evidence.authenticated_evidence_digest,index,prior):raise ValueError("invalid evidence admission chain mismatch")
  if (evidence.content.effect_id,evidence.content.command_digest)!=(intent.effect_id,intent.command_digest):raise ValueError("invalid evidence command mismatch")
  ec=evidence.content
  site_kind="dispatch" if ec.site.value.startswith("dispatch_") else "reconciliation"
  if ec.source_kind!=site_kind:raise ValueError("invalid evidence site/source mismatch")
  if ec.source_kind=="dispatch":
   if (ec.source_owner_ref,ec.source_generation,ec.source_ownership_epoch,ec.source_claim_digest,ec.source_grant_ref,ec.source_grant_digest)!=(gc.grant_ref,1,record.dispatch_epoch,record.dispatch_source_digest,gc.grant_ref,grant.authenticated_grant_digest):raise ValueError("invalid dispatch evidence authority mismatch")
  elif not any((x.owner_ref,x.generation,x.ownership_epoch,x.claim_digest)==(ec.source_owner_ref,ec.source_generation,ec.source_ownership_epoch,ec.source_claim_digest) and any(g.grant_ref==ec.source_grant_ref and g.grant_digest==ec.source_grant_digest for g in x.grant_lineage) for x in claims):raise ValueError("invalid reconciliation evidence authority mismatch")
  prior=admission.admission_digest;invalid_digests.append(admission.admission_digest)
 if tuple(invalid_digests)!=ac.invalid_evidence_admission_digests:raise ValueError("assessment invalid evidence projection mismatch")
 if tuple(derived_invalid)!=tuple(x for x in assessed_invalid if x is not DiagnosticCode.EVIDENCE_INVALID) or sum(x is DiagnosticCode.EVIDENCE_INVALID for x in assessed_invalid)!=len(invalid_digests):raise ValueError("assessment invalid provenance mismatch")
 if any(x not in {DiagnosticCode.STALE_RESULT,DiagnosticCode.FINALITY_UNPROVEN,DiagnosticCode.EVIDENCE_INVALID} for x in assessed_invalid):raise ValueError("assessment generated unsupported invalid code")
 if derived_state is EffectState.UNRESOLVED and DiagnosticCode.EVIDENCE_INVALID in assessed_invalid:
  derived_state=EffectState.INDETERMINATE
 if tuple(terminals)!=record.terminal_content_digests or derived_state is not record.state: raise ValueError("supplied foundation terminal/state disagrees with authenticated reduction")
 if record.state in {EffectState.FOUND,EffectState.DEFINITELY_ABSENT,EffectState.CONTRADICTED} and record.reconciliation is not None and record.reconciliation.active:raise ValueError("terminal foundation state retains active reconciliation")
 if record.invalid_codes!=assessed_invalid or tuple(x.value for x in record.invalid_codes)!=ac.invalid_codes: raise ValueError("foundation invalid-code history disagrees with assessment")
 grant_doc=parse_canonical_object(grant.canonical_bytes,name="authenticated grant")
 snapshot={"schema_version":"synaptic-foundation-effect-snapshot/v1","command":command.to_dict(),"command_bytes_digest":domain_digest("synaptic-foundation-command-bytes/v1",record.command_bytes),"grant":grant_doc,"dispatch":record.dispatch.value,"state":record.state.value,"attempt_count":record.attempt_count,"dispatch_epoch":record.dispatch_epoch,"receipts":[parse_canonical_object(x.canonical_bytes,name="receipt") for x in receipts],"receipt_admissions":[x.to_dict()|{"admission_digest":x.admission_digest} for x in record.receipt_admissions],"invalid_evidence":[parse_canonical_object(x.canonical_bytes,name="invalid evidence") for x in record.invalid_evidence],"invalid_evidence_admissions":[x.to_dict()|{"admission_digest":x.admission_digest} for x in record.invalid_evidence_admissions],"terminal_content_digests":list(record.terminal_content_digests),"invalid_codes":[x.value for x in record.invalid_codes],"reconciliation":None if record.reconciliation is None else _claim_doc(record.reconciliation),"reconciliation_claims":[_claim_doc(x) for x in claims],"b2_record_digest":record.record_digest}
 record_evidence_digest=domain_digest("synaptic-foundation-record-evidence/v1",canonical_bytes(snapshot))
 projection=(ac.effect_id,ac.command_digest,ac.command_bytes_digest,ac.foundation_record_digest,ac.record_evidence_digest,ac.authenticated_receipt_digests,ac.terminal_content_digests,ac.invalid_evidence_admission_digests)
 expected_projection=(intent.effect_id,intent.command_digest,snapshot["command_bytes_digest"],record.record_digest,record_evidence_digest,tuple(x.authenticated_receipt_digest for x in receipts),record.terminal_content_digests,tuple(x.admission_digest for x in record.invalid_evidence_admissions))
 if projection!=expected_projection:raise ValueError("foundation assessment record projection mismatch")
 snapshot_bytes=canonical_bytes(snapshot);snapshot_digest=domain_digest("synaptic-foundation-effect-snapshot/v1",snapshot_bytes);grant_digest=domain_digest("synaptic-authenticated-effect-grant/v1",canonical_bytes(grant_doc));command_digest=domain_digest("synaptic-foundation-command-bytes/v1",record.command_bytes)
 binding_doc={"kind":intent.kind.value,"effect_id":intent.effect_id,"command_digest":intent.command_digest,"command_bytes_digest":command_digest,"preparation_digest":prep.preparation_digest,"grant_digest":grant_digest,"foundation_record_digest":record.record_digest,"snapshot_digest":snapshot_digest,"assessment_digest":parsed_assessment.authenticated_assessment_digest}
 binding=FoundationEffectBindingV1(intent.kind,intent.effect_id,intent.command_digest,command_digest,prep.preparation_digest,grant_digest,record.record_digest,snapshot_bytes,snapshot_digest,1,1,record.dispatch,record.state,tuple(x.authenticated_receipt_digest for x in receipts),record.terminal_content_digests,tuple(x.value for x in record.invalid_codes),tuple(x.claim_digest for x in claims),None if record.reconciliation is None else record.reconciliation.claim_digest,parsed_assessment.canonical_bytes,parsed_assessment.authenticated_assessment_digest,domain_digest("synaptic-foundation-binding/v1",canonical_bytes(binding_doc)))
 disposition={EffectState.FOUND:FoundationDispositionV1.FOUND,EffectState.DEFINITELY_ABSENT:FoundationDispositionV1.DEFINITELY_ABSENT,EffectState.CONTRADICTED:FoundationDispositionV1.CONTRADICTED,EffectState.INDETERMINATE:FoundationDispositionV1.INDETERMINATE,EffectState.UNRESOLVED:FoundationDispositionV1.INDETERMINATE}[record.state]
 found=None
 if disposition is FoundationDispositionV1.FOUND:
  retained_semantic=record.terminal_content_digests[-1] if record.terminal_content_digests else None
  establishing=[x for x in fresh_found_receipts if x.content.semantic_digest==retained_semantic]
  if not establishing or any(x.content.semantic_digest!=retained_semantic for x in fresh_found_receipts):raise ValueError("retained FOUND lacks coherent fresh establishing receipt")
  found=establishing[-1]
 absence=next((x for x in reversed(receipts) if x.authenticated_receipt_digest in authenticated_absences),None)
 stage=None if found is None else found.content.stage_ref;run=None if found is None else found.content.provider_run;cancel=None if found is None else found.content.cancellation
 outcome_doc={"binding_digest":binding.binding_digest,"disposition":disposition.value,"receipts":list(binding.authenticated_receipt_digests),"stage":None if stage is None else stage.to_dict(),"run":None if run is None else run.to_dict(),"cancel":None if cancel is None else {"run":cancel.run.to_dict(),"reason_digest":cancel.reason_digest},"finality":None if absence is None else absence.content.finality_proof_digest}
 outcome=FoundationEffectOutcomeV1(binding.binding_digest,intent.kind,intent.effect_id,intent.command_digest,prep.preparation_digest,record.record_digest,disposition,binding.authenticated_receipt_digests,tuple(x.content.content_digest for x in receipts),tuple(x.content.observation_digest for x in receipts),stage,run,cancel,None if absence is None else absence.content.finality_proof_digest,domain_digest("synaptic-foundation-outcome/v1",canonical_bytes(outcome_doc)))
 bound=None
 if disposition is FoundationDispositionV1.FOUND:
  common=(intent.effect_id,intent.command_digest,command_digest,prep.preparation_digest,binding.binding_digest,outcome.outcome_digest,found.authenticated_receipt_digest)
  if intent.kind is EffectKind.STAGE:
   doc={"reference":stage.to_dict(),"effect_id":common[0],"command_digest":common[1],"command_bytes_digest":common[2],"preparation_digest":common[3],"foundation_binding_digest":common[4],"foundation_outcome_digest":common[5],"authenticated_receipt_digest":common[6]};bound=BoundProviderStageRefV1(stage,*common,domain_digest("synaptic-stage-evidence-binding/v1",canonical_bytes(doc)))
  elif intent.kind is EffectKind.SUBMIT:
   doc={"reference":run.to_dict(),"effect_id":common[0],"command_digest":common[1],"command_bytes_digest":common[2],"preparation_digest":common[3],"foundation_binding_digest":common[4],"foundation_outcome_digest":common[5],"authenticated_receipt_digest":common[6]};bound=BoundProviderRunRefV1(run,*common,domain_digest("synaptic-submit-evidence-binding/v1",canonical_bytes(doc)))
  else:
   doc={"reference":{"run":cancel.run.to_dict(),"reason_digest":cancel.reason_digest},"effect_id":common[0],"command_digest":common[1],"command_bytes_digest":common[2],"preparation_digest":common[3],"foundation_binding_digest":common[4],"foundation_outcome_digest":common[5],"authenticated_receipt_digest":common[6],"target_run_binding_digest":provider_run.binding_digest};bound=BoundCancellationRefV1(cancel,*common,provider_run.binding_digest,domain_digest("synaptic-cancel-evidence-binding/v1",canonical_bytes(doc)))
 return binding,outcome,bound

def _replay_or_validate_new(intent,binding,outcome):
 for i,old in enumerate(intent.foundation_bindings):
  if old.binding_digest==binding.binding_digest:
   if old.canonical_snapshot_bytes!=binding.canonical_snapshot_bytes or intent.foundation_outcomes[i]!=outcome: raise WorkflowTransitionError("same foundation digest has different canonical evidence")
   return True
 if intent.foundation_bindings:
  old=intent.foundation_bindings[-1];a=parse_canonical_object(old.canonical_snapshot_bytes,name="old snapshot");b=parse_canonical_object(binding.canonical_snapshot_bytes,name="new snapshot")
  for name in ("command","command_bytes_digest","grant","attempt_count","dispatch_epoch"):
   if a[name]!=b[name]: raise WorkflowTransitionError(f"foundation {name} changed")
  for name in ("receipts","receipt_admissions","invalid_evidence","invalid_evidence_admissions","terminal_content_digests","invalid_codes"):
   if b[name][:len(a[name])]!=a[name]: raise WorkflowTransitionError(f"foundation {name} is nonmonotonic")
  old_claims=a["reconciliation_claims"];new_claims=b["reconciliation_claims"]
  if old_claims!=new_claims:
   if len(new_claims)==len(old_claims) and old_claims and new_claims[:-1]==old_claims[:-1]:
    before=old_claims[-1];after=new_claims[-1];identity=("owner_ref","generation","ownership_epoch","claimed_at_epoch","target_digest","claim_digest")
    if any(before[x]!=after[x] for x in identity):raise WorkflowTransitionError("reconciliation identity changed")
    transition=(before["active"],before["completed"],after["active"],after["completed"])
    if transition not in {(True,False,False,False),(False,False,True,False),(True,False,False,True)}:raise WorkflowTransitionError("reconciliation replacement invalid")
    before_lineage=before["grant_lineage"];after_lineage=after["grant_lineage"]
    if transition==(False,False,True,False):
     if before["grant_ref"]==after["grant_ref"] or before["grant_digest"]==after["grant_digest"]:raise WorkflowTransitionError("resume must change grant")
     if len(after_lineage)!=len(before_lineage)+1 or after_lineage[:-1]!=before_lineage:raise WorkflowTransitionError("resume must append exactly one grant binding")
     leaf=after_lineage[-1]
     if leaf["prior_binding_digest"]!=before_lineage[-1]["binding_digest"] or (after["grant_ref"],after["grant_digest"])!=(leaf["grant_ref"],leaf["grant_digest"]):raise WorkflowTransitionError("resume grant binding is not chained to retained lineage")
    elif before["grant_ref"]!=after["grant_ref"] or before["grant_digest"]!=after["grant_digest"] or before_lineage!=after_lineage:raise WorkflowTransitionError("interrupt/completion changed grant lineage")
   elif len(new_claims)==len(old_claims)+1 and new_claims[:-1]==old_claims: pass
   else:raise WorkflowTransitionError("reconciliation history regressed")
  dispatch={"orphaned_unproven":{"orphaned_unproven","quiescence_proven"},"quiescence_proven":{"quiescence_proven"},"relinquished":{"relinquished"}}
  if b["dispatch"] not in dispatch[a["dispatch"]]:raise WorkflowTransitionError("stable dispatch regressed")
  terminals={"found","definitely_absent","contradicted"}
  if a["state"]=="contradicted" and b["state"]!="contradicted": raise WorkflowTransitionError("contradiction regressed")
  if a["state"] in terminals-{"contradicted"} and b["state"] not in {a["state"],"contradicted"}: raise WorkflowTransitionError("terminal state regressed")
 return False

def begin_preparation(c):
 if c.phase is not WorkflowPhaseV1.PLANNED: raise WorkflowTransitionError("preparation requires planned")
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,WorkflowPhaseV1.PREPARING,c.revision+1,None,None,None,None,None,None,None,None,(),None,None,(),(),(),())

def record_stage_intent(c,p,i):
 if c.phase is not WorkflowPhaseV1.PREPARING: raise WorkflowTransitionError("stage intent requires preparing")
 if type(p) is not CanonicalPreparationV2 or type(i) is not EffectIntentV1 or i.kind is not EffectKind.STAGE or i.advanced: raise TypeError("fresh exact stage intent required")
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,WorkflowPhaseV1.STAGE_INTENT_RECORDED,c.revision+1,p.preparation_digest,i,None,None,None,None,None,None,(),None,None,(),(),(),())

def record_submit_intent(c,i):
 if c.phase is not WorkflowPhaseV1.STAGED: raise WorkflowTransitionError("submit intent requires staged")
 if type(i) is not EffectIntentV1 or i.kind is not EffectKind.SUBMIT or i.advanced: raise TypeError("fresh exact submit intent required")
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,WorkflowPhaseV1.SUBMIT_INTENT_RECORDED,c.revision+1,c.preparation_digest,c.stage,i,None,c.provider_stage_ref,None,None,None,c.run_observation_digests,None,None,(),(),(),c.diagnostic_codes)

def record_cancel_intent(c,i):
 if c.phase not in {WorkflowPhaseV1.QUEUED,WorkflowPhaseV1.RUNNING}: raise WorkflowTransitionError("cancel requires active run")
 if type(i) is not EffectIntentV1 or i.kind is not EffectKind.CANCEL or i.advanced: raise TypeError("fresh exact cancel intent required")
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,WorkflowPhaseV1.CANCEL_INTENT_RECORDED,c.revision+1,c.preparation_digest,c.stage,c.submit,i,c.provider_stage_ref,c.provider_run_ref,None,c.phase,c.run_observation_digests,None,None,(),(),(),c.diagnostic_codes,c.provider_run_observations)

def fail_before_effect(c,code):
 if c.phase not in {WorkflowPhaseV1.PLANNED,WorkflowPhaseV1.PREPARING,WorkflowPhaseV1.STAGED}: raise WorkflowTransitionError("not a pre-effect failure")
 safe_ref(code,"diagnostic_code")
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,WorkflowPhaseV1.FAILED,c.revision+1,c.preparation_digest,c.stage,c.submit,c.cancel,c.provider_stage_ref,c.provider_run_ref,c.bound_cancellation,c.pre_cancel_phase,c.run_observation_digests,c.artifact_manifest,c.artifact_manifest_digest,c.verified_artifacts,c.verification_receipts,c.verification_receipt_digests,c.diagnostic_codes+(code,))

def apply_stage_effect_record(c,r,assessment,a,assessor):
 try:b,o,ref=_derive_foundation(c.stage,r,assessment,a,assessor,c.provider_run_ref)
 except (TypeError,ValueError,KeyError) as e:raise WorkflowTransitionError(str(e)) from None
 if _replay_or_validate_new(c.stage,b,o):return c
 if c.phase not in {WorkflowPhaseV1.STAGE_INTENT_RECORDED,WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED}:raise WorkflowTransitionError("new stage evidence illegal")
 stage=EffectIntentV1(c.stage.kind,c.stage.effect_id,c.stage.command_digest,c.stage.canonical_command_bytes,c.stage.foundation_bindings+(b,),c.stage.foundation_outcomes+(o,));target={FoundationDispositionV1.FOUND:WorkflowPhaseV1.STAGED,FoundationDispositionV1.INDETERMINATE:WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED,FoundationDispositionV1.DEFINITELY_ABSENT:WorkflowPhaseV1.FAILED,FoundationDispositionV1.CONTRADICTED:WorkflowPhaseV1.CONTRADICTED}[o.disposition];codes=c.diagnostic_codes+(() if o.disposition not in {FoundationDispositionV1.DEFINITELY_ABSENT,FoundationDispositionV1.CONTRADICTED} else (f"stage_{o.disposition.value}",))
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,target,c.revision+1,c.preparation_digest,stage,None,None,ref if o.disposition is FoundationDispositionV1.FOUND else None,None,None,None,(),None,None,(),(),(),codes)

def apply_submit_effect_record(c,r,assessment,a,assessor):
 try:b,o,ref=_derive_foundation(c.submit,r,assessment,a,assessor,c.provider_run_ref)
 except (TypeError,ValueError,KeyError) as e:raise WorkflowTransitionError(str(e)) from None
 if _replay_or_validate_new(c.submit,b,o):return c
 if c.phase not in {WorkflowPhaseV1.SUBMIT_INTENT_RECORDED,WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED}:raise WorkflowTransitionError("new submit evidence illegal")
 submit=EffectIntentV1(c.submit.kind,c.submit.effect_id,c.submit.command_digest,c.submit.canonical_command_bytes,c.submit.foundation_bindings+(b,),c.submit.foundation_outcomes+(o,));target={FoundationDispositionV1.FOUND:WorkflowPhaseV1.QUEUED,FoundationDispositionV1.INDETERMINATE:WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED,FoundationDispositionV1.DEFINITELY_ABSENT:WorkflowPhaseV1.FAILED,FoundationDispositionV1.CONTRADICTED:WorkflowPhaseV1.CONTRADICTED}[o.disposition];codes=c.diagnostic_codes+(() if o.disposition not in {FoundationDispositionV1.DEFINITELY_ABSENT,FoundationDispositionV1.CONTRADICTED} else (f"submit_{o.disposition.value}",))
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,target,c.revision+1,c.preparation_digest,c.stage,submit,None,c.provider_stage_ref,ref if o.disposition is FoundationDispositionV1.FOUND else None,None,None,(),None,None,(),(),(),codes)

def apply_cancel_effect_record(c,r,assessment,a,assessor):
 try:b,o,ref=_derive_foundation(c.cancel,r,assessment,a,assessor,c.provider_run_ref)
 except (TypeError,ValueError,KeyError) as e:raise WorkflowTransitionError(str(e)) from None
 if _replay_or_validate_new(c.cancel,b,o):return c
 if c.phase not in {WorkflowPhaseV1.CANCEL_INTENT_RECORDED,WorkflowPhaseV1.CANCEL_RECONCILE_REQUIRED}:raise WorkflowTransitionError("new cancel evidence illegal")
 cancel=EffectIntentV1(c.cancel.kind,c.cancel.effect_id,c.cancel.command_digest,c.cancel.canonical_command_bytes,c.cancel.foundation_bindings+(b,),c.cancel.foundation_outcomes+(o,));target={FoundationDispositionV1.FOUND:WorkflowPhaseV1.CANCEL_REQUESTED,FoundationDispositionV1.INDETERMINATE:WorkflowPhaseV1.CANCEL_RECONCILE_REQUIRED,FoundationDispositionV1.DEFINITELY_ABSENT:c.pre_cancel_phase,FoundationDispositionV1.CONTRADICTED:WorkflowPhaseV1.CONTRADICTED}[o.disposition];codes=c.diagnostic_codes+(() if o.disposition not in {FoundationDispositionV1.DEFINITELY_ABSENT,FoundationDispositionV1.CONTRADICTED} else (f"cancel_{o.disposition.value}",))
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,target,c.revision+1,c.preparation_digest,c.stage,c.submit,cancel,c.provider_stage_ref,c.provider_run_ref,ref if o.disposition is FoundationDispositionV1.FOUND else None,c.pre_cancel_phase,c.run_observation_digests,None,None,(),(),(),codes,c.provider_run_observations)

_OBS={WorkflowPhaseV1.QUEUED:{ProviderRunPhaseV1.QUEUED:WorkflowPhaseV1.QUEUED,ProviderRunPhaseV1.RUNNING:WorkflowPhaseV1.RUNNING,ProviderRunPhaseV1.SUCCEEDED:WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,ProviderRunPhaseV1.FAILED:WorkflowPhaseV1.FAILED,ProviderRunPhaseV1.CANCELLED:WorkflowPhaseV1.CANCELLED},WorkflowPhaseV1.RUNNING:{ProviderRunPhaseV1.RUNNING:WorkflowPhaseV1.RUNNING,ProviderRunPhaseV1.SUCCEEDED:WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,ProviderRunPhaseV1.FAILED:WorkflowPhaseV1.FAILED,ProviderRunPhaseV1.CANCELLED:WorkflowPhaseV1.CANCELLED},WorkflowPhaseV1.CANCEL_REQUESTED:{ProviderRunPhaseV1.QUEUED:WorkflowPhaseV1.CANCEL_REQUESTED,ProviderRunPhaseV1.RUNNING:WorkflowPhaseV1.CANCEL_REQUESTED,ProviderRunPhaseV1.SUCCEEDED:WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,ProviderRunPhaseV1.FAILED:WorkflowPhaseV1.FAILED,ProviderRunPhaseV1.CANCELLED:WorkflowPhaseV1.CANCELLED}}
_READ_PHASES={
 ProviderReadPurposeV1.OBSERVE:{WorkflowPhaseV1.QUEUED,WorkflowPhaseV1.RUNNING,WorkflowPhaseV1.CANCEL_REQUESTED},
 ProviderReadPurposeV1.LOGS:{WorkflowPhaseV1.QUEUED,WorkflowPhaseV1.RUNNING,WorkflowPhaseV1.CANCEL_INTENT_RECORDED,WorkflowPhaseV1.CANCEL_REQUESTED,WorkflowPhaseV1.CANCEL_RECONCILE_REQUIRED,WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,WorkflowPhaseV1.VERIFICATION_FAILED,WorkflowPhaseV1.VERIFIED,WorkflowPhaseV1.FAILED,WorkflowPhaseV1.CANCELLED},
 ProviderReadPurposeV1.ARTIFACTS:{WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,WorkflowPhaseV1.VERIFICATION_FAILED,WorkflowPhaseV1.VERIFIED},
}
def provider_run_read_request(c,record,assessment,authenticator,assessor,*,purpose):
 if type(purpose) is not ProviderReadPurposeV1:raise TypeError("exact provider read purpose required")
 if c.phase not in _READ_PHASES[purpose]:raise WorkflowTransitionError("workflow is not eligible for provider read purpose")
 try:binding,outcome,bound=_derive_foundation(c.submit,record,assessment,authenticator,assessor,c.provider_run_ref)
 except (TypeError,ValueError,KeyError) as e:raise WorkflowTransitionError(str(e)) from None
 if outcome.disposition is not FoundationDispositionV1.FOUND or bound!=c.provider_run_ref or binding!=c.submit.foundation_bindings[-1] or outcome!=c.submit.foundation_outcomes[-1]:raise WorkflowTransitionError("read request foundation evidence differs from workflow")
 found=bound.authenticated_receipt_digest
 doc={"schema_version":"synaptic-provider-run-read-request/v1","purpose":purpose.value,"source_workflow_record_digest":c.record_digest,"source_revision":c.revision,"run":c.run.to_dict(),"provider_run_binding_digest":bound.binding_digest,"submit_command_bytes_digest":domain_digest("synaptic-foundation-command-bytes/v1",c.submit.canonical_command_bytes),"foundation_record_digest":record.record_digest,"assessment_digest":assessment.authenticated_assessment_digest,"foundation_binding_digest":binding.binding_digest,"foundation_outcome_digest":outcome.outcome_digest,"found_receipt_digest":found}
 raw=canonical_bytes(doc)
 return ProviderRunReadRequestV1(purpose,c.record_digest,c.revision,c.run,bound,c.submit.canonical_command_bytes,record,assessment,binding,outcome,found,raw,domain_digest("synaptic-provider-run-read-request/v1",raw))

def _validate_read_request(c,request):
 try:
  ProviderRunReadRequestV1(request.purpose,request.source_workflow_record_digest,request.source_revision,request.run,request.provider_run,request.submit_command_bytes,request.foundation_record,request.assessment,request.foundation_binding,request.foundation_outcome,request.found_receipt_digest,request.canonical_bytes,request.request_digest)
 except (TypeError,ValueError,KeyError) as e:raise WorkflowTransitionError(str(e)) from None
 if c.submit is None or not c.submit.foundation_bindings or not c.submit.foundation_outcomes or c.provider_run_ref is None:raise WorkflowTransitionError("workflow lacks retained submit evidence")
 binding=c.submit.foundation_bindings[-1];outcome=c.submit.foundation_outcomes[-1]
 if (request.run,request.provider_run)!=(c.run,c.provider_run_ref):raise WorkflowTransitionError("read request is cross-workflow")
 if request.submit_command_bytes!=c.submit.canonical_command_bytes:raise WorkflowTransitionError("read request submit command differs from workflow")
 command=parse_exact_command(request.submit_command_bytes)
 command_bytes_digest=domain_digest("synaptic-foundation-command-bytes/v1",request.submit_command_bytes)
 if (command.operation.effect.kind,command.operation.effect.effect_id,command.digest,command_bytes_digest)!=(EffectKind.SUBMIT,c.submit.effect_id,c.submit.command_digest,binding.command_bytes_digest):raise WorkflowTransitionError("read request submit identity differs from workflow")
 record=request.foundation_record;grant=record.grant
 if type(grant) is not AuthenticatedGrantV2 or type(grant.content) is not GrantContentV2:raise WorkflowTransitionError("read request foundation grant is not exact")
 claims=record.reconciliation_claims
 snapshot={"schema_version":"synaptic-foundation-effect-snapshot/v1","command":parse_exact_command(record.command_bytes).to_dict(),"command_bytes_digest":domain_digest("synaptic-foundation-command-bytes/v1",record.command_bytes),"grant":parse_canonical_object(grant.canonical_bytes,name="authenticated grant"),"dispatch":record.dispatch.value,"state":record.state.value,"attempt_count":record.attempt_count,"dispatch_epoch":record.dispatch_epoch,"receipts":[parse_canonical_object(x.canonical_bytes,name="receipt") for x in record.results],"receipt_admissions":[x.to_dict()|{"admission_digest":x.admission_digest} for x in record.receipt_admissions],"invalid_evidence":[parse_canonical_object(x.canonical_bytes,name="invalid evidence") for x in record.invalid_evidence],"invalid_evidence_admissions":[x.to_dict()|{"admission_digest":x.admission_digest} for x in record.invalid_evidence_admissions],"terminal_content_digests":list(record.terminal_content_digests),"invalid_codes":[x.value for x in record.invalid_codes],"reconciliation":None if record.reconciliation is None else _claim_doc(record.reconciliation),"reconciliation_claims":[_claim_doc(x) for x in claims],"b2_record_digest":record.record_digest}
 if canonical_bytes(snapshot)!=binding.canonical_snapshot_bytes or record.record_digest!=binding.foundation_record_digest:raise WorkflowTransitionError("read request Foundation record differs from retained evidence")
 if request.assessment.canonical_bytes!=binding.canonical_assessment_bytes or request.assessment.authenticated_assessment_digest!=binding.assessment_digest:raise WorkflowTransitionError("read request assessment differs from retained evidence")
 if request.foundation_binding!=binding or request.foundation_outcome!=outcome:raise WorkflowTransitionError("read request binding/outcome differs from retained evidence")
 if outcome.disposition is not FoundationDispositionV1.FOUND or request.found_receipt_digest!=c.provider_run_ref.authenticated_receipt_digest or request.found_receipt_digest not in outcome.authenticated_receipt_digests:raise WorkflowTransitionError("read request FOUND receipt differs from retained evidence")
 if (request.provider_run.foundation_binding_digest,request.provider_run.foundation_outcome_digest,request.provider_run.authenticated_receipt_digest)!=(binding.binding_digest,outcome.outcome_digest,request.found_receipt_digest):raise WorkflowTransitionError("read request bound run differs from retained evidence")
 prep=command.preparation;ref=request.provider_run.reference
 if (ref.provider_id,ref.profile_ref,ref.account_ref,ref.namespace_ref)!=(prep.provider.provider_id,prep.provider.profile_ref,prep.scope.account_ref,prep.scope.namespace_ref):raise WorkflowTransitionError("read request bound run scope differs from submit command")

def apply_provider_observation(c,request,o,authenticator):
 if type(request) is not ProviderRunReadRequestV1 or type(o) is not AuthenticatedProviderRunObservationV1:raise TypeError("exact read request and observation required")
 parsed=AuthenticatedProviderRunObservationV1.parse(o.canonical_bytes)
 if parsed.canonical_bytes!=o.canonical_bytes or authenticator.authenticate(parsed) is not True:raise WorkflowTransitionError("observation authentication failed")
 _validate_read_request(c,request)
 if request.purpose is not ProviderReadPurposeV1.OBSERVE:raise WorkflowTransitionError("provider read request purpose is not observe")
 if parsed.content.request_digest!=request.request_digest:raise WorkflowTransitionError("observation read request digest mismatch")
 digest=parsed.authenticated_observation_digest
 for old in c.provider_run_observations:
  if old.authenticated_observation_digest==digest:
   if old.canonical_bytes!=parsed.canonical_bytes:raise WorkflowTransitionError("observation digest collision")
   return c
 x=parsed.content;ref=c.provider_run_ref.reference
 if (request.source_workflow_record_digest,request.source_revision,x.source_workflow_record_digest,x.source_revision)!=(c.record_digest,c.revision,c.record_digest,c.revision):raise WorkflowTransitionError("read request or observation source is stale")
 expected=(request.request_digest,c.record_digest,c.revision,c.run,c.provider_run_ref.binding_digest,ref.provider_id,ref.profile_ref,ref.account_ref,ref.namespace_ref,ref.provider_job_ref)
 actual=(x.request_digest,x.source_workflow_record_digest,x.source_revision,x.run,x.provider_run_binding_digest,x.provider_id,x.profile_ref,x.account_ref,x.namespace_ref,x.provider_job_ref)
 if actual!=expected:raise WorkflowTransitionError("observation request/workflow/scope mismatch")
 target=_OBS.get(c.phase,{}).get(x.phase)
 if target is None:raise WorkflowTransitionError("new observation illegal")
 codes=c.diagnostic_codes+(() if x.diagnostic_code is None else (x.diagnostic_code,))
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,target,c.revision+1,c.preparation_digest,c.stage,c.submit,c.cancel,c.provider_stage_ref,c.provider_run_ref,c.bound_cancellation,c.pre_cancel_phase,c.run_observation_digests+(digest,),None,None,(),(),(),codes,c.provider_run_observations+(parsed,))

def _authenticated_verification(receipt,verifier):
 if type(receipt) is not AuthenticatedArtifactVerificationReceiptV1:raise TypeError("exact verification receipt required")
 parsed=AuthenticatedArtifactVerificationReceiptV1.parse(receipt.canonical_bytes)
 if parsed.canonical_bytes!=receipt.canonical_bytes or verifier.authenticate(parsed) is not True:raise WorkflowTransitionError("verification authentication failed")
 return parsed
def _verification_replay(c,r):
 for old in c.verification_receipts:
  if old.authenticated_receipt_digest==r.authenticated_receipt_digest:
   if old.canonical_bytes!=r.canonical_bytes:raise WorkflowTransitionError("verification digest collision")
   return True
 return False
def _validate_verification_semantics(c,m,r):
 if type(m) is not ArtifactManifestV1:raise TypeError("exact manifest required")
 if c.artifact_manifest is not None and m!=c.artifact_manifest:raise WorkflowTransitionError("manifest changed")
 x=r.content
 if (x.source_workflow_record_digest,x.source_revision,x.run,x.provider_run_binding_digest,x.manifest_digest,x.artifact_source_digest,x.manifest_artifacts)!=(c.record_digest,c.revision,c.run,c.provider_run_ref.binding_digest,m.manifest_digest,m.artifact_source_digest,m.artifacts) or (m.run,m.provider_run)!=(c.run,c.provider_run_ref.reference):raise WorkflowTransitionError("verification semantic binding mismatch")
def apply_artifact_verification(c,m,r,v):
 parsed=_authenticated_verification(r,v)
 if _verification_replay(c,parsed):return c
 if c.phase not in {WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,WorkflowPhaseV1.VERIFICATION_FAILED}:raise WorkflowTransitionError("new verification illegal")
 _validate_verification_semantics(c,m,parsed);receipts=c.verification_receipts+(parsed,);digests=c.verification_receipt_digests+(parsed.authenticated_receipt_digest,)
 if parsed.content.verdict is VerificationVerdictV1.VERIFIED:target=WorkflowPhaseV1.VERIFIED;artifacts=parsed.content.verified_artifacts;codes=c.diagnostic_codes
 else:target=WorkflowPhaseV1.VERIFICATION_FAILED;artifacts=();codes=c.diagnostic_codes+(parsed.content.diagnostic_code,)
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,target,c.revision+1,c.preparation_digest,c.stage,c.submit,c.cancel,c.provider_stage_ref,c.provider_run_ref,c.bound_cancellation,c.pre_cancel_phase,c.run_observation_digests,m,m.manifest_digest,artifacts,receipts,digests,codes,c.provider_run_observations)
def apply_reverification(c,m,r,v):
 parsed=_authenticated_verification(r,v)
 if _verification_replay(c,parsed):return c
 if c.phase is not WorkflowPhaseV1.VERIFIED:raise WorkflowTransitionError("new reverification illegal")
 _validate_verification_semantics(c,m,parsed)
 if parsed.content.verdict is not VerificationVerdictV1.VERIFIED or parsed.content.verified_artifacts!=c.verified_artifacts:raise WorkflowTransitionError("reverification inventory changed")
 return WorkflowRecordV1(c.schema_version,c.run,c.plan_fingerprint,c.preflight_digest,c.provider,c.provider_context_digest,c.provider_descriptor_digest,WorkflowPhaseV1.VERIFIED,c.revision+1,c.preparation_digest,c.stage,c.submit,c.cancel,c.provider_stage_ref,c.provider_run_ref,c.bound_cancellation,c.pre_cancel_phase,c.run_observation_digests,c.artifact_manifest,c.artifact_manifest_digest,c.verified_artifacts,c.verification_receipts+(parsed,),c.verification_receipt_digests+(parsed.authenticated_receipt_digest,),c.diagnostic_codes,c.provider_run_observations)

_PUBLIC={WorkflowPhaseV1.PLANNED:TrainingRunState.PLANNED,WorkflowPhaseV1.PREPARING:TrainingRunState.PLANNED,WorkflowPhaseV1.STAGE_INTENT_RECORDED:TrainingRunState.PLANNED,WorkflowPhaseV1.STAGED:TrainingRunState.PLANNED,WorkflowPhaseV1.SUBMIT_INTENT_RECORDED:TrainingRunState.PLANNED,WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED:TrainingRunState.RECONCILE_REQUIRED,WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED:TrainingRunState.RECONCILE_REQUIRED,WorkflowPhaseV1.CANCEL_RECONCILE_REQUIRED:TrainingRunState.RECONCILE_REQUIRED,WorkflowPhaseV1.CONTRADICTED:TrainingRunState.RECONCILE_REQUIRED,WorkflowPhaseV1.QUEUED:TrainingRunState.QUEUED,WorkflowPhaseV1.RUNNING:TrainingRunState.RUNNING,WorkflowPhaseV1.SUCCEEDED_UNVERIFIED:TrainingRunState.SUCCEEDED,WorkflowPhaseV1.VERIFICATION_FAILED:TrainingRunState.SUCCEEDED,WorkflowPhaseV1.VERIFIED:TrainingRunState.SUCCEEDED,WorkflowPhaseV1.FAILED:TrainingRunState.FAILED,WorkflowPhaseV1.CANCEL_INTENT_RECORDED:TrainingRunState.CANCEL_REQUESTED,WorkflowPhaseV1.CANCEL_REQUESTED:TrainingRunState.CANCEL_REQUESTED,WorkflowPhaseV1.CANCELLED:TrainingRunState.CANCELLED}
def project_run_outcome(r):
 if type(r) is not WorkflowRecordV1:raise TypeError("exact workflow required")
 return RunOutcome("synaptic-run-outcome/v1",r.run,_PUBLIC[r.phase],r.verified_artifacts if r.phase is WorkflowPhaseV1.VERIFIED else (),r.diagnostic_code)
__all__=["WorkflowTransitionError","begin_preparation","record_stage_intent","record_submit_intent","record_cancel_intent","fail_before_effect","apply_stage_effect_record","apply_submit_effect_record","apply_cancel_effect_record","provider_run_read_request","apply_provider_observation","apply_artifact_verification","apply_reverification","project_run_outcome"]
