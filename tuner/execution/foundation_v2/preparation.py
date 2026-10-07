"""Sealed immutable canonical preparation byte envelopes."""
from __future__ import annotations
from synaptic_tuner.api.v1.providers import ProviderRef
from .canonical import canonical_bytes,digest_text,domain_digest,exact_fields,parse_canonical_object,safe_ref
from .references import ExecutionScopeV1
_FIELDS=frozenset({"schema_version","provider","scope","project_ref","run_id","plan_fingerprint","source_digest","workload_digest","runtime_digest","resource_digest","artifact_contract_digest","quote_digest","secret_requirements_digest","execution_binding_digest"});_ISSUER=object()
class CanonicalPreparationV2:
    __slots__=("_raw","_sealed")
    def __init_subclass__(cls,**kw): raise TypeError("CanonicalPreparationV2 is final")
    def __init__(self,raw:bytes,*,_issuer):
        if _issuer is not _ISSUER: raise TypeError("preparations are parser-minted")
        object.__setattr__(self,"_raw",bytes(raw));object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):
        if getattr(self,"_sealed",False): raise AttributeError("preparation is immutable")
        object.__setattr__(self,n,v)
    @classmethod
    def parse(cls,raw:bytes):
        doc=parse_canonical_object(raw,name="preparation");exact_fields(doc,_FIELDS,"preparation")
        if doc["schema_version"]!="synaptic-preparation/v2": raise ValueError("unsupported preparation schema")
        p=doc["provider"];s=doc["scope"]
        if not isinstance(p,dict) or set(p)!={"provider_id","profile_ref"}: raise ValueError("provider malformed")
        if not isinstance(s,dict) or set(s)!={"account_ref","namespace_ref"}: raise ValueError("scope malformed")
        ProviderRef.from_dict(p);ExecutionScopeV1(**s);safe_ref(doc["project_ref"],"project_ref");safe_ref(doc["run_id"],"run_id")
        for n in _FIELDS-{"schema_version","provider","scope","project_ref","run_id"}:digest_text(doc[n],n)
        return cls(raw,_issuer=_ISSUER)
    @classmethod
    def build(cls,*,provider,scope,project_ref,run_id,plan_fingerprint,source_digest,workload_digest,runtime_digest,resource_digest,artifact_contract_digest,quote_digest,secret_requirements_digest,execution_binding_digest):
        if type(provider) is not ProviderRef or type(scope) is not ExecutionScopeV1: raise TypeError("exact references required")
        return cls.parse(canonical_bytes({"schema_version":"synaptic-preparation/v2","provider":provider.to_dict(),"scope":scope.to_dict(),"project_ref":project_ref,"run_id":run_id,"plan_fingerprint":plan_fingerprint,"source_digest":source_digest,"workload_digest":workload_digest,"runtime_digest":runtime_digest,"resource_digest":resource_digest,"artifact_contract_digest":artifact_contract_digest,"quote_digest":quote_digest,"secret_requirements_digest":secret_requirements_digest,"execution_binding_digest":execution_binding_digest}))
    def _doc(self):return parse_canonical_object(self._raw,name="preparation")
    @property
    def canonical_bytes(self):return bytes(self._raw)
    @property
    def preparation_digest(self):return domain_digest("synaptic-preparation/v2",self._raw)
    @property
    def provider(self):return ProviderRef.from_dict(self._doc()["provider"])
    @property
    def scope(self):return ExecutionScopeV1(**self._doc()["scope"])
    def __getattr__(self,n):
        if n in _FIELDS-{"schema_version","provider","scope"}:return self._doc()[n]
        raise AttributeError(n)
    def to_dict(self):return self._doc()
    def __eq__(self,o):return type(o) is CanonicalPreparationV2 and self._raw==o._raw
    def __hash__(self):return hash(self._raw)
