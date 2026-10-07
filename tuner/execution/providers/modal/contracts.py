"""Deeply immutable, bounded Modal control-plane evidence values."""
from __future__ import annotations
import hashlib,json
from dataclasses import dataclass
from enum import Enum
from ...contracts import EffectIdentity,digest,required_text,safe_ref

class ArtifactRole(str,Enum):
    WORKLOAD_RECORD="workload_record";TRAINING_LINEAGE="training_lineage";TRAINING_METRICS="training_metrics";FINAL_MODEL="final_model";TOKENIZER="tokenizer"
EXACT_ARTIFACT_ROLES=frozenset(ArtifactRole)
RESERVED_PREFIXES=("input","control","evidence","logs","output")
class Readiness(str,Enum):READY="ready";NOT_READY="not_ready"
@dataclass(frozen=True,slots=True)
class BoundsPolicyV1:
    max_control_bytes:int=64*1024;max_bundle_bytes:int=8*1024*1024;max_artifact_bytes:int=64*1024*1024;max_artifact_total_bytes:int=256*1024*1024;max_log_chunk_bytes:int=64*1024;max_log_records:int=256;max_depth:int=8;max_string_bytes:int=16*1024
    def __post_init__(self):
        for n in self.__dataclass_fields__:
            v=getattr(self,n)
            strict_int(v,n,minimum=1)
def strict_int(value,name,*,minimum=0,maximum=2**63-1):
    if type(value) is not int or not minimum<=value<=maximum:raise ValueError(f"invalid {name}")
    return value
@dataclass(frozen=True,slots=True)
class StageReceiptV1:
    effect_id:str;operation_binding_digest:str;control_volume_id:str;artifact_volume_id:str;claim_digest:str;bundle_digest:str
    def __post_init__(self):
        require_layout(self.control_volume_id,self.artifact_volume_id)
        for n in ("effect_id","control_volume_id","artifact_volume_id"):object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        for n in ("operation_binding_digest","claim_digest","bundle_digest"):object.__setattr__(self,n,digest(getattr(self,n),n))
@dataclass(frozen=True,slots=True)
class ArtifactMemberV1:
    role:ArtifactRole;path:str;size:int;sha256:str;provider_entry_id:str
    def __post_init__(self):
        if not isinstance(self.role,ArtifactRole):raise TypeError("role must be ArtifactRole")
        path=canonical_path(self.path)
        parts=path.split("/")
        if len(parts)!=4 or parts[0]!="operations" or parts[2]!="output":raise ValueError("artifact must be under its operation output")
        operation_root(parts[1])
        object.__setattr__(self,"path",path);object.__setattr__(self,"sha256",digest(self.sha256,"sha256"));object.__setattr__(self,"provider_entry_id",safe_ref(self.provider_entry_id,"provider_entry_id"))
        object.__setattr__(self,"size",strict_int(self.size,"size"))
@dataclass(frozen=True,slots=True)
class TerminalEvidenceV1:
    status_code:str;account_ref:str;workspace_ref:str;environment_ref:str;client_ref:str;sdk_version:str;control_volume_id:str;artifact_volume_id:str;job_ref:str;effect_id:str;command_digest:str;plan_digest:str;deployment_attestation_digest:str;invocation_nonce:str;generation:int;artifact_set_digest:str;log_chain_digest:str
    @classmethod
    def parse(cls,data:bytes,*,limit:int=65536):
        value=_object(data,limit);required={"schema","status_code","account_ref","workspace_ref","environment_ref","client_ref","sdk_version","control_volume_id","artifact_volume_id","job_ref","effect_id","command_digest","plan_digest","deployment_attestation_digest","invocation_nonce","generation","artifact_set_digest","log_chain_digest"}
        if set(value)!=required or value["schema"]!="synaptic.modal-terminal/v1" or value["status_code"] not in {"completed","failed","cancelled"}:raise ValueError("invalid terminal evidence")
        return cls(**{k:value[k] for k in required-{"schema"}})
    def __post_init__(self):
        for n in ("status_code","account_ref","workspace_ref","environment_ref","client_ref","sdk_version","control_volume_id","artifact_volume_id","job_ref","effect_id","invocation_nonce"):object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        for n in ("command_digest","plan_digest","deployment_attestation_digest","artifact_set_digest","log_chain_digest"):object.__setattr__(self,n,digest(getattr(self,n),n))
        object.__setattr__(self,"generation",strict_int(self.generation,"generation",minimum=1,maximum=2**31-1))
        require_layout(self.control_volume_id,self.artifact_volume_id)
def canonical_path(v:str)->str:
    v=required_text(v,"path")
    if v.startswith("/") or "\\" in v or v.endswith("/") or any(x in {"",".",".."} for x in v.split("/")):raise ValueError("noncanonical path")
    return v
def operation_root(effect_id:str)->str:
    effect_id=safe_ref(effect_id,"effect_id")
    if "/" in effect_id:raise ValueError("effect_id must be one path component")
    return f"operations/{effect_id}"
def operation_path(effect_id:str,*members:str)->str:
    suffix="/".join(members)
    path=f"{operation_root(effect_id)}/{suffix}" if suffix else operation_root(effect_id)
    return canonical_path(path)
def require_layout(control_volume_id,artifact_volume_id,output_prefix=None):
    if safe_ref(control_volume_id,"control_volume_id")==safe_ref(artifact_volume_id,"artifact_volume_id"):raise ValueError("control and artifact volumes must differ")
    if output_prefix is not None:
        parts=canonical_path(output_prefix).split("/")
        if len(parts)!=3 or parts[0]!="operations" or parts[2]!="output" or output_prefix!=operation_path(parts[1],"output"):raise ValueError("artifact output prefix must be operation scoped")
def provider_entry_identity(volume_id:str,path:str,size:int)->str:
    volume_id=safe_ref(volume_id,"volume_id");path=canonical_path(path);strict_int(size,"size")
    return hashlib.sha256(b"synaptic.modal-volume-entry/v1\0"+volume_id.encode()+b"\0"+path.encode()+b"\0"+str(size).encode()).hexdigest()
def require_reserved_path(path,prefix):
    path=canonical_path(path)
    if not path.startswith(prefix+"/") or path.split("/",1)[0] not in RESERVED_PREFIXES:raise ValueError("path is outside reserved prefix")
    return path
def canonical_json(v:object)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def _object(data:bytes,limit:int):
    if not isinstance(data,bytes) or len(data)>limit:raise ValueError("record exceeds bound")
    try:v=json.loads(data)
    except Exception:raise ValueError("invalid canonical record") from None
    if not isinstance(v,dict) or canonical_json(v)!=data:raise ValueError("record must be canonical JSON object")
    return v
