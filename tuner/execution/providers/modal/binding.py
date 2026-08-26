from __future__ import annotations
from dataclasses import dataclass
from .contracts import Readiness
from ...contracts import safe_ref
@dataclass(frozen=True,slots=True)
class ModalClientBinding:
    account_ref:str;workspace_ref:str;environment_ref:str;client_ref:str;sdk_version:str
    def __post_init__(self):
        for n in self.__dataclass_fields__:object.__setattr__(self,n,safe_ref(getattr(self,n),n))
@dataclass(frozen=True,slots=True)
class CapabilityProofV1:
    explicit_client:bool;authenticated_scope:bool;volume_v1_io:bool;volume_listing:bool;deployment_identity:bool;image_identity:bool;function_version:bool
    @property
    def complete(self):return all(getattr(self,n) is True for n in self.__dataclass_fields__)
@dataclass(frozen=True,slots=True)
class ReadinessReport:
    status:Readiness;reason_code:str
def readiness_report(binding:ModalClientBinding,facade)->ReadinessReport:
    if not isinstance(binding,ModalClientBinding):return ReadinessReport(Readiness.NOT_READY,"binding_invalid")
    if binding.sdk_version!="1.5.4":return ReadinessReport(Readiness.NOT_READY,"sdk_version_mismatch")
    try:scope=facade.bound_scope()
    except Exception:return ReadinessReport(Readiness.NOT_READY,"scope_unavailable")
    if scope!=(binding.account_ref,binding.workspace_ref,binding.environment_ref,binding.client_ref):return ReadinessReport(Readiness.NOT_READY,"scope_mismatch")
    try:proof=facade.capability_proof(binding)
    except Exception:return ReadinessReport(Readiness.NOT_READY,"capability_unavailable")
    if not isinstance(proof,CapabilityProofV1) or not proof.complete:return ReadinessReport(Readiness.NOT_READY,"capability_incomplete")
    return ReadinessReport(Readiness.READY,"ready")
def readiness(binding,facade):return readiness_report(binding,facade).status
