"""Sealed exact commands embedding the complete independently reconstructable DAG."""
from __future__ import annotations
from .canonical import canonical_bytes,digest_text,domain_digest,exact_fields,parse_canonical_object,safe_ref
from .preparation import CanonicalPreparationV2
from .identities import EffectKind,derive_effect
from .operations import derive_operation
from .executors import ExecutorDescriptorV1
from .references import CancellationRefV1,ProviderRunRefV1,StagePredecessorV2
_P_FIELDS=frozenset({"schema_version","provider_id","payload_kind","input_digest"});_P_ISSUER=object();_C_ISSUER=object()
_MATRIX={EffectKind.STAGE:"stage-payload/v2",EffectKind.SUBMIT:"submit-payload/v2",EffectKind.CANCEL:"cancel-payload/v2"}
class CanonicalProviderPayloadV1:
    __slots__=("_raw","_sealed")
    def __init_subclass__(cls,**kw):raise TypeError("CanonicalProviderPayloadV1 is final")
    def __init__(self,raw,*,_issuer):
        if _issuer is not _P_ISSUER:raise TypeError("payloads are parser-minted")
        object.__setattr__(self,"_raw",bytes(raw));object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):raise AttributeError("payload is immutable")
    @classmethod
    def parse(cls,raw):
        d=parse_canonical_object(raw,name="payload");exact_fields(d,_P_FIELDS,"payload")
        if d["schema_version"]!="synaptic-provider-payload/v2":raise ValueError("payload schema unsupported")
        safe_ref(d["provider_id"],"provider_id");safe_ref(d["payload_kind"],"payload_kind");digest_text(d["input_digest"],"input_digest")
        return cls(raw,_issuer=_P_ISSUER)
    @classmethod
    def build(cls,provider_id,payload_kind,input_digest):return cls.parse(canonical_bytes({"schema_version":"synaptic-provider-payload/v2","provider_id":provider_id,"payload_kind":payload_kind,"input_digest":input_digest}))
    def _doc(self):return parse_canonical_object(self._raw,name="payload")
    @property
    def canonical_bytes(self):return bytes(self._raw)
    @property
    def payload_digest(self):return domain_digest("synaptic-provider-payload/v2",self._raw)
    def __getattr__(self,n):
        if n in {"provider_id","payload_kind","input_digest"}:return self._doc()[n]
        raise AttributeError(n)
    def to_dict(self):return self._doc()
    def __eq__(self,o):return type(o) is CanonicalProviderPayloadV1 and self._raw==o._raw

def _descriptor(raw):
    if not isinstance(raw,dict) or set(raw)!={"provider_id","executor_id","implementation_version"}:raise ValueError("executor malformed")
    return ExecutorDescriptorV1(**raw)
def _reconstruct(doc,kind):
    prep=CanonicalPreparationV2.parse(canonical_bytes(doc["preparation"]));payload=CanonicalProviderPayloadV1.parse(canonical_bytes(doc["payload"]));executor=_descriptor(doc["executor"])
    target=None
    if kind is EffectKind.CANCEL:
        c=doc["cancellation"]
        if not isinstance(c,dict) or set(c)!={"provider_job_ref","reason_digest"}:raise ValueError("cancellation malformed")
        target=ProviderRunRefV1(c["provider_job_ref"]);digest_text(c["reason_digest"],"reason_digest")
    effect=derive_effect(prep,kind,cancel_target=target);op=derive_operation(prep,effect,doc["operation"]["invocation_nonce"])
    if doc["effect"]!=effect.to_dict() or doc["operation"]!=op.to_dict():raise ValueError("command lineage reconstruction mismatch")
    if payload.provider_id!=prep.provider.provider_id or executor.provider_id!=prep.provider.provider_id:raise ValueError("provider binding mismatch")
    if payload.payload_kind!=_MATRIX[kind]:raise ValueError("effect kind and payload schema mismatch")
    if payload.input_digest!=prep.workload_digest:raise ValueError("payload input lineage mismatch")
    return prep,effect,op,payload,executor,target
class _View:
    @staticmethod
    def doc(raw):return parse_canonical_object(raw,name="command")
    @staticmethod
    def prep(raw,kind):return _reconstruct(_View.doc(raw),kind)[0]
    @staticmethod
    def effect(raw,kind):return _reconstruct(_View.doc(raw),kind)[1]
    @staticmethod
    def operation(raw,kind):return _reconstruct(_View.doc(raw),kind)[2]
    @staticmethod
    def payload(raw,kind):return _reconstruct(_View.doc(raw),kind)[3]
    @staticmethod
    def executor(raw,kind):return _reconstruct(_View.doc(raw),kind)[4]
def _props(cls,kind):
    cls.canonical_bytes=property(lambda s:bytes(s._raw));cls.preparation=property(lambda s:_View.prep(s._raw,kind));cls.operation=property(lambda s:_View.operation(s._raw,kind));cls.payload=property(lambda s:_View.payload(s._raw,kind));cls.executor=property(lambda s:_View.executor(s._raw,kind));cls.digest=property(lambda s:domain_digest(f"synaptic-{kind.value}-command/v2",s._raw));cls.to_dict=lambda s:_View.doc(s._raw)
class StageCommandV2:
    __slots__=("_raw","_sealed")
    def __init_subclass__(cls,**kw):raise TypeError("StageCommandV2 is final")
    def __init__(self,raw,*,_issuer):
        if _issuer is not _C_ISSUER:raise TypeError("commands are parser-minted")
        object.__setattr__(self,"_raw",bytes(raw));object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):raise AttributeError("command is immutable")
class SubmitCommandV2:
    __slots__=("_raw","_sealed")
    def __init_subclass__(cls,**kw):raise TypeError("SubmitCommandV2 is final")
    def __init__(self,raw,*,_issuer):
        if _issuer is not _C_ISSUER:raise TypeError("commands are parser-minted")
        object.__setattr__(self,"_raw",bytes(raw));object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):raise AttributeError("command is immutable")
    @property
    def stage_predecessor(self):
        return StagePredecessorV2(**self.to_dict()["stage_predecessor"])
class CancelCommandV2:
    __slots__=("_raw","_sealed")
    def __init_subclass__(cls,**kw):raise TypeError("CancelCommandV2 is final")
    def __init__(self,raw,*,_issuer):
        if _issuer is not _C_ISSUER:raise TypeError("commands are parser-minted")
        object.__setattr__(self,"_raw",bytes(raw));object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):raise AttributeError("command is immutable")
_props(StageCommandV2,EffectKind.STAGE);_props(SubmitCommandV2,EffectKind.SUBMIT);_props(CancelCommandV2,EffectKind.CANCEL)
CanonicalCommandV2=StageCommandV2|SubmitCommandV2|CancelCommandV2
def _base(prep,kind,nonce,payload,executor):
    prep=CanonicalPreparationV2.parse(prep.canonical_bytes);payload=CanonicalProviderPayloadV1.parse(payload.canonical_bytes);effect=derive_effect(prep,kind,cancel_target=None);op=derive_operation(prep,effect,nonce)
    return {"schema_version":f"synaptic-{kind.value}-command/v2","preparation":prep.to_dict(),"effect":effect.to_dict(),"operation":op.to_dict(),"payload":payload.to_dict(),"executor":executor.to_dict()}
def build_stage_command(prep,nonce,payload,executor):return parse_exact_command(canonical_bytes(_base(prep,EffectKind.STAGE,nonce,payload,executor)))
def build_submit_command(prep,nonce,payload,executor,predecessor):
    if type(predecessor) is not StagePredecessorV2:raise TypeError("exact stage predecessor required")
    d=_base(prep,EffectKind.SUBMIT,nonce,payload,executor);d["stage_predecessor"]=predecessor.to_dict();return parse_exact_command(canonical_bytes(d))
def build_cancel_command(prep,nonce,payload,executor,cancellation):
    if type(cancellation) is not CancellationRefV1:raise TypeError("exact cancellation required")
    effect=derive_effect(prep,EffectKind.CANCEL,cancel_target=cancellation.run);op=derive_operation(prep,effect,nonce)
    d={"schema_version":"synaptic-cancel-command/v2","preparation":prep.to_dict(),"effect":effect.to_dict(),"operation":op.to_dict(),"payload":payload.to_dict(),"executor":executor.to_dict(),"cancellation":{"provider_job_ref":cancellation.run.provider_job_ref,"reason_digest":cancellation.reason_digest}}
    return parse_exact_command(canonical_bytes(d))
def parse_exact_command(raw):
    d=parse_canonical_object(raw,name="command");schema=d.get("schema_version")
    mapping={"synaptic-stage-command/v2":(StageCommandV2,EffectKind.STAGE,frozenset({"schema_version","preparation","effect","operation","payload","executor"})),"synaptic-submit-command/v2":(SubmitCommandV2,EffectKind.SUBMIT,frozenset({"schema_version","preparation","effect","operation","payload","executor","stage_predecessor"})),"synaptic-cancel-command/v2":(CancelCommandV2,EffectKind.CANCEL,frozenset({"schema_version","preparation","effect","operation","payload","executor","cancellation"}))}
    if schema not in mapping:raise ValueError("unsupported exact command schema")
    cls,kind,fields=mapping[schema];exact_fields(d,fields,"command")
    _reconstruct(d,kind)
    if kind is EffectKind.SUBMIT:
        s=d["stage_predecessor"]
        if not isinstance(s,dict):raise ValueError("stage predecessor malformed")
        StagePredecessorV2(**s)
    return cls(raw,_issuer=_C_ISSUER)
