from __future__ import annotations
from dataclasses import dataclass
from .contracts import *
from .contracts import _object,require_layout,strict_int
from .binding import ModalClientBinding,Readiness,readiness_report
from .control import CrossPlaneIdentityV1
from ...contracts import digest,safe_ref
@dataclass(frozen=True,slots=True)
class CompletionExpectationV1:
    binding:ModalClientBinding;control_volume_id:str;artifact_volume_id:str;output_prefix:str;key_ref:str;job_ref:str;effect_id:str;command_digest:str;plan_digest:str;deployment_attestation_digest:str;invocation_nonce:str;generation:int
    def __post_init__(self):
        require_layout(self.control_volume_id,self.artifact_volume_id,self.output_prefix)
        for n in ("control_volume_id","artifact_volume_id","output_prefix","key_ref","job_ref","effect_id","invocation_nonce"):object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        for n in ("command_digest","plan_digest","deployment_attestation_digest"):object.__setattr__(self,n,digest(getattr(self,n),n))
        object.__setattr__(self,"generation",strict_int(self.generation,"generation",minimum=1,maximum=2**31-1))
    @property
    def identity(self):return CrossPlaneIdentityV1(self.binding,self.control_volume_id,self.artifact_volume_id,self.job_ref,self.effect_id,self.command_digest,self.plan_digest,self.deployment_attestation_digest,self.invocation_nonce,self.generation,self.key_ref)
@dataclass(frozen=True,slots=True)
class CompletionManifestV1:
    members:tuple[ArtifactMemberV1,...];account_ref:str;workspace_ref:str;environment_ref:str;client_ref:str;sdk_version:str;control_volume_id:str;artifact_volume_id:str;job_ref:str;effect_id:str;command_digest:str;plan_digest:str;deployment_attestation_digest:str;invocation_nonce:str;generation:int;terminal_evidence_digest:str;log_chain_digest:str
    @classmethod
    def parse(cls,data:bytes,*,limit=65536):
        v=_object(data,limit);keys={"schema","members","account_ref","workspace_ref","environment_ref","client_ref","sdk_version","control_volume_id","artifact_volume_id","job_ref","effect_id","command_digest","plan_digest","deployment_attestation_digest","invocation_nonce","generation","terminal_evidence_digest","log_chain_digest"}
        if set(v)!=keys or v["schema"]!="synaptic.modal-completion/v1" or not isinstance(v["members"],list) or len(v["members"])!=5:raise ValueError("invalid completion manifest")
        members=[]
        for x in v["members"]:
            if not isinstance(x,dict) or set(x)!={"role","path","size","sha256","provider_entry_id"}:raise ValueError("invalid member")
            members.append(ArtifactMemberV1(ArtifactRole(x["role"]),x["path"],x["size"],x["sha256"],x["provider_entry_id"]))
        return cls(tuple(members),**{k:v[k] for k in keys-{"schema","members"}})
    def __post_init__(self):
        if len(self.members)!=5 or frozenset(x.role for x in self.members)!=EXACT_ARTIFACT_ROLES or len({x.role for x in self.members})!=5 or len({x.path for x in self.members})!=5 or len({x.provider_entry_id for x in self.members})!=5:raise ValueError("manifest requires unique exact five artifacts")
        object.__setattr__(self,"generation",strict_int(self.generation,"generation",minimum=1,maximum=2**31-1))
        require_layout(self.control_volume_id,self.artifact_volume_id)
        for n in ("account_ref","workspace_ref","environment_ref","client_ref","sdk_version","control_volume_id","artifact_volume_id","job_ref","effect_id","invocation_nonce"):object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        for n in ("command_digest","plan_digest","deployment_attestation_digest","terminal_evidence_digest","log_chain_digest"):object.__setattr__(self,n,digest(getattr(self,n),n))
    @property
    def artifact_set_digest(self):
        return sha(canonical_json([{"role":m.role.value,"path":m.path,"size":m.size,"sha256":m.sha256,"provider_entry_id":m.provider_entry_id} for m in sorted(self.members,key=lambda item:item.role.value)]))
def verify_artifacts(manifest,expected,facade,*,bounds=BoundsPolicyV1()):
    if type(expected) is not CompletionExpectationV1 or type(expected.identity) is not CrossPlaneIdentityV1 or not expected.identity.matches_record(manifest):raise ValueError("completion binding mismatch")
    prefix=expected.output_prefix+"/"
    try:listing=facade.list_prefix(expected.artifact_volume_id,prefix,max_entries=6)
    except Exception:raise ValueError("artifact listing unavailable") from None
    if not isinstance(listing,(list,tuple)):raise ValueError("invalid artifact listing")
    normalized=[]
    for entry in listing:
        if not isinstance(entry,(list,tuple)) or len(entry)!=3:raise ValueError("invalid artifact listing entry")
        path,size,provider_entry_id=entry
        if canonical_path(path)!=path:raise ValueError("invalid artifact listing entry")
        strict_int(size,"artifact listing size",maximum=bounds.max_artifact_bytes)
        normalized.append((path,size,safe_ref(provider_entry_id,"provider_entry_id")))
    listing=tuple(normalized)
    if len(listing)!=5 or len({x[0] for x in listing})!=5 or len({x[2] for x in listing})!=5 or len(set(listing))!=5:raise ValueError("artifact listing must contain unique exact five entries")
    actual={p:(s,i) for p,s,i in listing};declared={m.path:m for m in manifest.members}
    if set(actual)!=set(declared) or any(not p.startswith(prefix) for p in actual):raise ValueError("artifact relist mismatch")
    if sum(m.size for m in manifest.members)>bounds.max_artifact_total_bytes:raise ValueError("artifact inventory exceeds total bound")
    out=[]
    for path,m in declared.items():
        size,entry=actual[path]
        if size!=m.size or entry!=m.provider_entry_id or size>bounds.max_artifact_bytes:raise ValueError("artifact metadata changed")
        try:data=facade.read_complete(expected.artifact_volume_id,path,max_bytes=bounds.max_artifact_bytes)
        except Exception:raise ValueError("artifact unavailable") from None
        if len(data)!=m.size or sha(data)!=m.sha256:raise ValueError("artifact content changed or partial")
        out.append(m)
    return tuple(out)
class CompletionControlPlane:
    def __init__(self,store,authenticator,facade,terminal_plane,log_plane,*,bounds=BoundsPolicyV1()):self.store=store;self.auth=authenticator;self.facade=facade;self.terminal=terminal_plane;self.logs=log_plane;self.bounds=bounds
    def validate(self,requested_effect):
        try:expected=self.store.load_completion_expectation(requested_effect)
        except Exception:raise ValueError("completion expectation unavailable") from None
        if not isinstance(expected,CompletionExpectationV1) or requested_effect!=expected.effect_id:raise ValueError("completion expectation misrouted")
        report=readiness_report(expected.binding,self.facade)
        if report.status is not Readiness.READY:raise ValueError(report.reason_code)
        try:data=self.facade.read_complete(expected.control_volume_id,operation_path(expected.effect_id,"evidence","completion-manifest.v1.json"),max_bytes=self.bounds.max_control_bytes);tag=self.facade.read_complete(expected.control_volume_id,operation_path(expected.effect_id,"evidence","completion-manifest.v1.mac"),max_bytes=128)
        except Exception:raise ValueError("completion manifest unavailable") from None
        try:authenticated=self.auth.verify("modal-completion/v1",data,tag,expected.key_ref)
        except Exception:raise ValueError("completion authentication unavailable") from None
        if not authenticated:raise ValueError("completion authentication failed")
        manifest=CompletionManifestV1.parse(data,limit=self.bounds.max_control_bytes)
        terminal=self.terminal.validate(requested_effect);logs=self.logs.validate(requested_effect);identity=expected.identity
        if type(terminal.identity) is not CrossPlaneIdentityV1 or type(logs.identity) is not CrossPlaneIdentityV1 or terminal.identity!=identity or logs.identity!=identity:raise ValueError("completion cross-plane identity mismatch")
        if terminal.evidence.status_code!="completed" or terminal.evidence.log_chain_digest!=manifest.log_chain_digest or terminal.canonical_digest!=manifest.terminal_evidence_digest or logs.chain_digest!=manifest.log_chain_digest:raise ValueError("completion evidence digest mismatch")
        verify_artifacts(manifest,expected,self.facade,bounds=self.bounds)
        if terminal.evidence.artifact_set_digest!=manifest.artifact_set_digest:raise ValueError("completion artifact-set digest mismatch")
        return manifest
