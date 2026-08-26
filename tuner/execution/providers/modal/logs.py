from __future__ import annotations
import base64,hashlib,json,struct,time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from .contracts import BoundsPolicyV1,_object,canonical_json,canonical_path,operation_path,require_layout,sha,strict_int
from .binding import ModalClientBinding,Readiness,readiness_report
from .control import CrossPlaneIdentityV1
from ...contracts import digest,required_text,safe_ref
class LogCode(str,Enum):BOOTSTRAP="bootstrap";PROGRESS="progress";CHECKPOINT="checkpoint";COMPLETED="completed";FAILED="failed"
@dataclass(frozen=True,slots=True)
class StructuredLogChunkV1:
    generation:int;sequence:int;previous_digest:str;payload_digest:str;job_ref:str;effect_id:str;plan_digest:str;invocation_nonce:str;records:tuple[tuple[LogCode,str],...]
    @classmethod
    def parse(cls,data:bytes,*,bounds=BoundsPolicyV1()):
        v=_object(data,bounds.max_log_chunk_bytes);keys={"schema","generation","sequence","previous_digest","payload_digest","job_ref","effect_id","plan_digest","invocation_nonce","records"}
        if set(v)!=keys or v["schema"]!="synaptic.modal-log-chunk/v1" or not isinstance(v["records"],list) or len(v["records"])>bounds.max_log_records:raise ValueError("invalid log chunk")
        records=[]
        for r in v["records"]:
            if not isinstance(r,dict) or set(r)!={"code","message"}:raise ValueError("invalid log record")
            message=required_text(r["message"],"message")
            if len(message.encode())>bounds.max_string_bytes:raise ValueError("log message exceeds bound")
            records.append((LogCode(r["code"]),message))
        strict_int(v["generation"],"generation",minimum=1,maximum=2**31-1);strict_int(v["sequence"],"sequence",maximum=2**31-1)
        for n in ("previous_digest","payload_digest","plan_digest"):digest(v[n],n)
        for n in ("job_ref","effect_id","invocation_nonce"):safe_ref(v[n],n)
        payload=canonical_json(v["records"])
        if hashlib.sha256(payload).hexdigest()!=v["payload_digest"]:raise ValueError("log payload digest mismatch")
        return cls(v["generation"],v["sequence"],v["previous_digest"],v["payload_digest"],v["job_ref"],v["effect_id"],v["plan_digest"],v["invocation_nonce"],tuple(records))
    @property
    def chunk_digest(self):
        return sha(canonical_json({"schema":"synaptic.modal-log-chain-node/v1","generation":self.generation,"sequence":self.sequence,"previous_digest":self.previous_digest,"payload_digest":self.payload_digest,"job_ref":self.job_ref,"effect_id":self.effect_id,"plan_digest":self.plan_digest,"invocation_nonce":self.invocation_nonce}))
class CursorAuthenticator(Protocol):
    def sign(self,payload:bytes)->bytes:...
    def verify(self,payload:bytes,tag:bytes)->bool:...
class CursorService:
    _FMT=">BIIIQ32s";_PAYLOAD=struct.calcsize(_FMT);_TAG=32;ENCODED_SIZE=len(base64.urlsafe_b64encode(b"x"*(_PAYLOAD+_TAG)))
    def __init__(self,auth:CursorAuthenticator,clock=lambda:int(time.time())):self.auth=auth;self.clock=clock
    def issue(self,*,generation,chunk,record,expires_at,context_digest):
        strict_int(generation,"generation",minimum=1,maximum=2**32-1);strict_int(chunk,"chunk",maximum=2**32-1);strict_int(record,"record",maximum=2**32-1);strict_int(expires_at,"expires_at",minimum=1,maximum=2**64-1)
        raw=struct.pack(self._FMT,1,generation,chunk,record,expires_at,bytes.fromhex(digest(context_digest,"context_digest")))
        try:tag=self.auth.sign(raw)
        except Exception:raise ValueError("invalid cursor authenticator") from None
        if len(tag)!=self._TAG:raise ValueError("invalid cursor authenticator")
        return base64.urlsafe_b64encode(raw+tag).decode()
    def parse(self,token,*,generation,context_digest):
        context_digest=digest(context_digest,"context_digest")
        if not isinstance(token,str) or len(token)!=self.ENCODED_SIZE:raise ValueError("invalid cursor size")
        try:data=base64.b64decode(token,altchars=b"-_",validate=True)
        except Exception:raise ValueError("invalid cursor encoding") from None
        raw,tag=data[:self._PAYLOAD],data[self._PAYLOAD:]
        try:authenticated=self.auth.verify(raw,tag)
        except Exception:raise ValueError("invalid cursor authentication") from None
        if len(data)!=self._PAYLOAD+self._TAG or not authenticated:raise ValueError("invalid cursor authentication")
        version,g,c,r,expiry,ctx=struct.unpack(self._FMT,raw)
        if version!=1 or g!=generation or ctx.hex()!=context_digest or self.clock()>=expiry:raise ValueError("cursor mismatch or expired")
        return c,r
def validate_chain(chunks):
    previous="0"*64
    identity=None
    for index,chunk in enumerate(chunks):
        current=(chunk.generation,chunk.job_ref,chunk.effect_id,chunk.plan_digest,chunk.invocation_nonce)
        if identity is None:identity=current
        if current!=identity or chunk.sequence!=index or chunk.previous_digest!=previous:raise ValueError("log chain mismatch")
        previous=chunk.chunk_digest
    return previous
@dataclass(frozen=True,slots=True)
class LogExpectationV1:
    binding:ModalClientBinding;control_volume_id:str;artifact_volume_id:str;key_ref:str;job_ref:str;effect_id:str;command_digest:str;plan_digest:str;deployment_attestation_digest:str;invocation_nonce:str;generation:int
    def __post_init__(self):
        require_layout(self.control_volume_id,self.artifact_volume_id)
        for n in ("control_volume_id","artifact_volume_id","key_ref","job_ref","effect_id","invocation_nonce"):object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        for n in ("command_digest","plan_digest","deployment_attestation_digest"):object.__setattr__(self,n,digest(getattr(self,n),n))
        object.__setattr__(self,"generation",strict_int(self.generation,"generation",minimum=1,maximum=2**31-1))
    @property
    def identity(self):return CrossPlaneIdentityV1(self.binding,self.control_volume_id,self.artifact_volume_id,self.job_ref,self.effect_id,self.command_digest,self.plan_digest,self.deployment_attestation_digest,self.invocation_nonce,self.generation,self.key_ref)
@dataclass(frozen=True,slots=True)
class LogValidationResult:
    chain_digest:str;chunk_count:int;identity:CrossPlaneIdentityV1
    def __post_init__(self):
        object.__setattr__(self,"chain_digest",digest(self.chain_digest,"chain_digest"));object.__setattr__(self,"chunk_count",strict_int(self.chunk_count,"chunk_count",minimum=1))
        if type(self.identity) is not CrossPlaneIdentityV1:raise TypeError("invalid log validation identity")
class LogControlPlane:
    def __init__(self,store,authenticator,facade,*,bounds=BoundsPolicyV1()):self.store=store;self.auth=authenticator;self.facade=facade;self.bounds=bounds
    def validate(self,requested_effect):
        try:expected=self.store.load_log_expectation(requested_effect)
        except Exception:raise ValueError("log expectation unavailable") from None
        if not isinstance(expected,LogExpectationV1) or requested_effect!=expected.effect_id:raise ValueError("log expectation misrouted")
        report=readiness_report(expected.binding,self.facade)
        if report.status is not Readiness.READY:raise ValueError(report.reason_code)
        root=operation_path(expected.effect_id,"logs")
        try:data=self.facade.read_complete(expected.control_volume_id,root+"/log-metadata.v1.json",max_bytes=self.bounds.max_control_bytes);tag=self.facade.read_complete(expected.control_volume_id,root+"/log-metadata.v1.mac",max_bytes=128)
        except Exception:raise ValueError("log metadata unavailable") from None
        try:authenticated=self.auth.verify("modal-log-metadata/v1",data,tag,expected.key_ref)
        except Exception:raise ValueError("log metadata authentication unavailable") from None
        if not authenticated:raise ValueError("log metadata authentication failed")
        value=_object(data,self.bounds.max_control_bytes);keys={"schema","account_ref","workspace_ref","environment_ref","client_ref","sdk_version","control_volume_id","artifact_volume_id","job_ref","effect_id","command_digest","plan_digest","deployment_attestation_digest","invocation_nonce","generation","chain_digest","chunks"}
        if set(value)!=keys or value["schema"]!="synaptic.modal-log-metadata/v1" or not isinstance(value["chunks"],list) or not 1<=len(value["chunks"])<=self.bounds.max_log_records:raise ValueError("invalid log metadata")
        try:
            record_binding=ModalClientBinding(value["account_ref"],value["workspace_ref"],value["environment_ref"],value["client_ref"],value["sdk_version"])
            record_identity=CrossPlaneIdentityV1(record_binding,value["control_volume_id"],value["artifact_volume_id"],value["job_ref"],value["effect_id"],value["command_digest"],value["plan_digest"],value["deployment_attestation_digest"],value["invocation_nonce"],value["generation"],expected.key_ref)
        except Exception:raise ValueError("invalid log metadata identity") from None
        if type(record_identity) is not CrossPlaneIdentityV1 or record_identity!=expected.identity:raise ValueError("log metadata binding mismatch")
        inventory=[]
        for item in value["chunks"]:
            if not isinstance(item,dict) or set(item)!={"path","size","sha256","provider_entry_id"}:raise ValueError("invalid log inventory")
            path=item["path"];size=item["size"]
            if not isinstance(path,str) or canonical_path(path)!=path or not path.startswith(root+"/chunks/"):raise ValueError("invalid log inventory member")
            strict_int(size,"log chunk size",minimum=1,maximum=self.bounds.max_log_chunk_bytes)
            inventory.append((path,size,digest(item["sha256"],"sha256"),safe_ref(item["provider_entry_id"],"provider_entry_id")))
        if len({x[0] for x in inventory})!=len(inventory) or len({x[3] for x in inventory})!=len(inventory):raise ValueError("duplicate log inventory")
        try:listing=self.facade.list_prefix(expected.control_volume_id,root+"/chunks/",max_entries=self.bounds.max_log_records+1)
        except Exception:raise ValueError("log listing unavailable") from None
        if not isinstance(listing,(list,tuple)):raise ValueError("invalid log listing")
        normalized=[]
        for entry in listing:
            if not isinstance(entry,(list,tuple)) or len(entry)!=3:raise ValueError("invalid log listing entry")
            path,size,provider_entry_id=entry
            if not isinstance(path,str) or canonical_path(path)!=path or not path.startswith(root+"/chunks/"):raise ValueError("invalid log listing entry")
            strict_int(size,"log listing size",minimum=1,maximum=self.bounds.max_log_chunk_bytes)
            normalized.append((path,size,safe_ref(provider_entry_id,"provider_entry_id")))
        listing=tuple(normalized)
        if len(listing)!=len(inventory) or len(set(listing))!=len(listing) or len({x[0] for x in listing})!=len(listing) or len({x[2] for x in listing})!=len(listing):raise ValueError("duplicate or changed log listing")
        actual={p:(s,i) for p,s,i in listing};chunks=[]
        for path,size,dg,entry in inventory:
            if actual.get(path)!=(size,entry) or size>self.bounds.max_log_chunk_bytes:raise ValueError("log listing mismatch")
            try:raw=self.facade.read_complete(expected.control_volume_id,path,max_bytes=self.bounds.max_log_chunk_bytes)
            except Exception:raise ValueError("log chunk unavailable") from None
            if len(raw)!=size or sha(raw)!=dg:raise ValueError("log chunk changed or partial")
            chunk=StructuredLogChunkV1.parse(raw,bounds=self.bounds)
            if (chunk.generation,chunk.job_ref,chunk.effect_id,chunk.plan_digest,chunk.invocation_nonce)!=(expected.generation,expected.job_ref,expected.effect_id,expected.plan_digest,expected.invocation_nonce):raise ValueError("log chunk binding mismatch")
            chunks.append(chunk)
        chain=validate_chain(tuple(chunks))
        if chain!=value["chain_digest"]:raise ValueError("log chain digest mismatch")
        return LogValidationResult(chain,len(chunks),record_identity)
