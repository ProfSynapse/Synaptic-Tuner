"""Trusted resolution ports; callers never supply executors or adapters per call."""
from dataclasses import dataclass
from typing import Protocol
from .canonical import DiagnosticCode, FoundationError, canonical_bytes, domain_digest, safe_ref
@dataclass(frozen=True,slots=True)
class ExecutorDescriptorV1:
    provider_id:str;executor_id:str;implementation_version:str
    def __post_init__(self):
        for n in self.__dataclass_fields__:safe_ref(getattr(self,n),n)
    def to_dict(self):return {n:getattr(self,n) for n in self.__dataclass_fields__}
    @property
    def digest(self):return domain_digest("synaptic-executor-descriptor/v1",canonical_bytes(self.to_dict()))
@dataclass(frozen=True,slots=True)
class AdapterDescriptorV1:
    provider_id:str;adapter_id:str;implementation_version:str
    def __post_init__(self):
        for n in self.__dataclass_fields__:safe_ref(getattr(self,n),n)
    def to_dict(self):return {n:getattr(self,n) for n in self.__dataclass_fields__}
    @property
    def digest(self):return domain_digest("synaptic-adapter-descriptor/v1",canonical_bytes(self.to_dict()))
@dataclass(frozen=True,slots=True)
class ExecutionResolutionRequestV2:
    command_digest:str;descriptor_digest:str;provider_id:str;profile_ref:str;account_ref:str;namespace_ref:str;effect_kind:str;payload_schema:str;input_digest:str
    @property
    def digest(self):return domain_digest("synaptic-execution-resolution/v2",canonical_bytes({n:getattr(self,n) for n in self.__dataclass_fields__}))
_RESOLVED=object()
class ResolvedExecutorV2:
    __slots__=("request_digest","executor","descriptor_digest","provider_id","profile_ref","account_ref","namespace_ref","effect_kinds","payload_schemas")
    def __init__(self,request,executor,descriptor,effect_kinds,payload_schemas,*,_issuer):
        if _issuer is not _RESOLVED:raise TypeError("resolved executors are resolver-minted")
        self.request_digest=request.digest;self.executor=executor;self.descriptor_digest=descriptor.digest
        self.provider_id=request.provider_id;self.profile_ref=request.profile_ref;self.account_ref=request.account_ref;self.namespace_ref=request.namespace_ref
        self.effect_kinds=effect_kinds;self.payload_schemas=payload_schemas
def _runtime_binding(value,request,descriptor_type,capability):
    descriptor=getattr(value,"descriptor",None)
    requested_digest=getattr(request,"descriptor_digest",getattr(request,"adapter_digest",None))
    if type(descriptor) is not descriptor_type or descriptor.digest!=requested_digest:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    actual=(getattr(value,"provider_id",None),getattr(value,"profile_ref",None),getattr(value,"account_ref",None),getattr(value,"namespace_ref",None))
    expected=(request.provider_id,request.profile_ref,request.account_ref,request.namespace_ref)
    if actual!=expected:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    capabilities=getattr(value,capability,None)
    if type(capabilities) is not tuple or not capabilities or any(type(x) is not str for x in capabilities):raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    return descriptor,capabilities
def mint_resolved_executor(request,executor):
    descriptor,effect_kinds=_runtime_binding(executor,request,ExecutorDescriptorV1,"effect_kinds")
    payload_schemas=getattr(executor,"payload_schemas",None)
    if type(payload_schemas) is not tuple or request.effect_kind not in effect_kinds or request.payload_schema not in payload_schemas:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    return ResolvedExecutorV2(request,executor,descriptor,effect_kinds,payload_schemas,_issuer=_RESOLVED)
class ExecutorResolverV2(Protocol):
    def resolve(self,request:ExecutionResolutionRequestV2)->ResolvedExecutorV2:...
class EffectExecutorV2(Protocol):
    def execute_once(self,payload:object,request:ExecutionResolutionRequestV2)->object:...
@dataclass(frozen=True,slots=True)
class ReconciliationResolutionRequestV2:
    command_digest:str;adapter_digest:str;provider_id:str;profile_ref:str;account_ref:str;namespace_ref:str
    @property
    def digest(self):return domain_digest("synaptic-reconciliation-resolution/v2",canonical_bytes({n:getattr(self,n) for n in self.__dataclass_fields__}))
_RAD=object()
class ResolvedAdapterV2:
    __slots__=("request_digest","adapter","descriptor_digest","provider_id","profile_ref","account_ref","namespace_ref","capabilities")
    def __init__(self,request,adapter,descriptor,capabilities,*,_issuer):
        if _issuer is not _RAD:raise TypeError("resolved adapters are resolver-minted")
        self.request_digest=request.digest;self.adapter=adapter;self.descriptor_digest=descriptor.digest
        self.provider_id=request.provider_id;self.profile_ref=request.profile_ref;self.account_ref=request.account_ref;self.namespace_ref=request.namespace_ref;self.capabilities=capabilities
def mint_resolved_adapter(request,adapter):
    descriptor,capabilities=_runtime_binding(adapter,request,AdapterDescriptorV1,"capabilities")
    if "lookup" not in capabilities:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    return ResolvedAdapterV2(request,adapter,descriptor,capabilities,_issuer=_RAD)
class ReconciliationResolverV2(Protocol):
    def resolve(self,request:ReconciliationResolutionRequestV2)->ResolvedAdapterV2:...
