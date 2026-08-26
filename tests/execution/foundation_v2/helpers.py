import hashlib,hmac
from dataclasses import dataclass,replace
from synaptic_tuner.api.v1.providers import ProviderRef
from tuner.execution.foundation_v2.authority import GrantAuthorityV2,ReconciliationGrantContentV1
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1,build_stage_command,build_submit_command,build_cancel_command
from tuner.execution.foundation_v2.executors import ExecutorDescriptorV1,AdapterDescriptorV1,mint_resolved_executor,mint_resolved_adapter
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ProviderObservationV1,ObservationDisposition
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.receipts import ReceiptAuthorityV1
from tuner.execution.foundation_v2.references import ExecutionScopeV1,ProviderRunRefV1,CancellationRefV1,ProviderStageRefV1,ScopedProviderRunRefV1
from tuner.execution.foundation_v2.repository import InMemoryEffectRepositoryV2
D=tuple(c*64 for c in "123456789abcdef")
def prep(**kw):
    x=dict(provider=ProviderRef("docker","local"),scope=ExecutionScopeV1("acct","ns"),project_ref="project",run_id="run",plan_fingerprint=D[0],source_digest=D[1],workload_digest=D[2],runtime_digest=D[3],resource_digest=D[4],artifact_contract_digest=D[5],quote_digest=D[6],secret_requirements_digest=D[7]);x.update(kw);return CanonicalPreparationV2.build(**x)
def descriptor():return ExecutorDescriptorV1("docker","executor","1.0.0")
def payload(kind,p=None):
    p=p or prep();return CanonicalProviderPayloadV1.build("docker",f"{kind.value}-payload/v2",p.workload_digest)
def stage_command(p=None):p=p or prep();return build_stage_command(p,"nonce",payload(EffectKind.STAGE,p),descriptor())
def cancel_command(p=None):p=p or prep();return build_cancel_command(p,"nonce",payload(EffectKind.CANCEL,p),descriptor(),CancellationRefV1(ProviderRunRefV1("job-1"),D[8]))
def observation_for(command,disposition=ObservationDisposition.FOUND,*,resolution_digest=D[11],result_epoch=1,finality_proof=None):
    kw={}
    if disposition is ObservationDisposition.FOUND:
        p=command.preparation
        if command.operation.effect.kind is EffectKind.STAGE:kw["stage_ref"]=ProviderStageRefV1(p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,"stage-output")
        elif command.operation.effect.kind is EffectKind.SUBMIT:kw["provider_run"]=ScopedProviderRunRefV1(p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,"job-1")
        else:kw["cancellation"]=CancellationRefV1(command.operation.effect.cancel_target,command.to_dict()["cancellation"]["reason_digest"])
    return ProviderObservationV1(command.operation.effect.effect_id,command.digest,command.executor.digest,disposition,resolution_digest,result_epoch,finality_proof=finality_proof,**kw)
def dispatch_receipt_content(command,observation,record):
    from tuner.execution.foundation_v2.receipts import ReceiptContentV1
    return ReceiptContentV1.from_observation(observation,source_kind="dispatch",source_owner_ref=record.grant.content.grant_ref,source_generation=1,source_ownership_epoch=record.dispatch_epoch,source_claim_digest=record.dispatch_source_digest)
def reconciliation_receipt_content(observation,claim):
    from tuner.execution.foundation_v2.receipts import ReceiptContentV1
    return ReceiptContentV1.from_observation(observation,source_kind="reconciliation",source_owner_ref=claim.owner_ref,source_generation=claim.generation,source_ownership_epoch=claim.ownership_epoch,source_claim_digest=claim.claim_digest)
@dataclass(frozen=True,slots=True)
class Proof:
    effect_id:str;command_digest:str;epoch:int;assertion:str;tag:str
    @property
    def proof_digest(self):return hashlib.sha256((self.effect_id+self.command_digest+str(self.epoch)+self.assertion+self.tag).encode()).hexdigest()
class StrongVerifier:
    def __init__(self,key=b"p"*32):self.key=key
    def proof(self,record,assertion,epoch=1):
        e=record.command.operation.effect.effect_id;c=record.command.digest;msg=f"{e}|{c}|{epoch}|{assertion}".encode();return Proof(e,c,epoch,assertion,hmac.new(self.key,msg,hashlib.sha256).hexdigest())
    def _verify(self,p,record,assertion):
        if not isinstance(p,Proof) or p.assertion!=assertion or p.effect_id!=record.command.operation.effect.effect_id or p.command_digest!=record.command.digest:return False
        msg=f"{p.effect_id}|{p.command_digest}|{p.epoch}|{p.assertion}".encode();return hmac.compare_digest(p.tag,hmac.new(self.key,msg,hashlib.sha256).hexdigest())
    def verify_quiescence(self,p,record,*,now_epoch):return self._verify(p,record,"quiescent") and p.epoch<=now_epoch
    def verify_finality(self,p,record,receipt,*,now_epoch):return self._verify(p,record,"final_absent") and p.proof_digest==receipt.content.finality_proof_digest and p.epoch<=now_epoch
class Executor:
    def __init__(self,disposition=ObservationDisposition.FOUND,fail=False,*,provider_id="docker",profile_ref="local",account_ref="acct",namespace_ref="ns"):
        self.calls=0;self.disposition=disposition;self.fail=fail;self.payloads=[]
        self.descriptor=ExecutorDescriptorV1(provider_id,"executor","1.0.0");self.provider_id=provider_id;self.profile_ref=profile_ref;self.account_ref=account_ref;self.namespace_ref=namespace_ref
        self.effect_kinds=("stage","submit","cancel");self.payload_schemas=("stage-payload/v2","submit-payload/v2","cancel-payload/v2")
    def execute_once(self,payload,request):
        self.calls+=1;self.payloads.append(payload)
        if self.fail:raise RuntimeError("secret-provider-body")
        kw={}
        if self.disposition is ObservationDisposition.FOUND:
            if request.effect_kind=="stage":kw["stage_ref"]=ProviderStageRefV1(self.provider_id,self.profile_ref,self.account_ref,self.namespace_ref,"stage-output")
            elif request.effect_kind=="submit":kw["provider_run"]=ScopedProviderRunRefV1(self.provider_id,self.profile_ref,self.account_ref,self.namespace_ref,"job-1")
            else:kw["cancellation"]=CancellationRefV1(ProviderRunRefV1("job-1"),D[8])
        return ProviderObservationV1("pending",request.command_digest,request.descriptor_digest,self.disposition,request.digest,1,**kw)
class ExecutorResolver:
    def __init__(self,executor):self.executor=executor;self.calls=0
    def resolve(self,request):self.calls+=1;return mint_resolved_executor(request,self.executor)
class Adapter:
    def __init__(self,observation,fail=False,*,provider_id="docker",profile_ref="local",account_ref="acct",namespace_ref="ns"):
        self.observation=observation;self.fail=fail;self.calls=0;self.descriptor=AdapterDescriptorV1(provider_id,"lookup","1.0.0");self.provider_id=provider_id;self.profile_ref=profile_ref;self.account_ref=account_ref;self.namespace_ref=namespace_ref;self.capabilities=("lookup",)
    def lookup(self,target,preparation):
        self.calls+=1
        if self.fail:raise RuntimeError("secret-adapter-body")
        return replace(self.observation,resolution_digest=target.resolution_digest,result_epoch=target.ownership_epoch)
class AdapterResolver:
    def __init__(self,adapter):self.adapter=adapter;self.calls=0
    def resolve(self,request):self.calls+=1;return mint_resolved_adapter(request,self.adapter)
def environment(executor):
    grant=GrantAuthorityV2("grants",b"g"*32);receipt=ReceiptAuthorityV1("receipts",b"r"*32);verifier=StrongVerifier();repo=InMemoryEffectRepositoryV2(receipt,verifier,verifier,grant);return repo,grant,receipt,verifier,ExecutorResolver(executor)
def execution_grant(authority,command,ref="grant"):return authority.issue(command.canonical_bytes,grant_ref=ref,policy_digest=D[9],requirement_digest=D[10],not_before_epoch=100,expires_at_epoch=200)
