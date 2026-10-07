from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .binding import ModalClientBinding
from .contracts import BoundsPolicyV1,StageReceiptV1,_object,operation_path,require_layout,sha,strict_int
from ...contracts import EffectIdentity,EffectKind,digest,safe_ref
from ...operation import OperationBindingV1
class DurableExpectationStore(Protocol):
    def load_modal_expectation(self,effect_id:str):...
class HostAuthenticator(Protocol):
    def verify(self,purpose:str,payload:bytes,tag:bytes,key_ref:str)->bool:...
@dataclass(frozen=True,slots=True)
class CrossPlaneIdentityV1:
    binding:ModalClientBinding;control_volume_id:str;artifact_volume_id:str;job_ref:str;effect_id:str;command_digest:str;plan_digest:str;deployment_attestation_digest:str;invocation_nonce:str;generation:int;key_ref:str
    def __post_init__(self):
        if type(self.binding) is not ModalClientBinding:raise TypeError("binding must be ModalClientBinding")
        require_layout(self.control_volume_id,self.artifact_volume_id)
        for n in ("control_volume_id","artifact_volume_id","job_ref","effect_id","invocation_nonce","key_ref"):object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        for n in ("command_digest","plan_digest","deployment_attestation_digest"):object.__setattr__(self,n,digest(getattr(self,n),n))
        object.__setattr__(self,"generation",strict_int(self.generation,"generation",minimum=1,maximum=2**31-1))
    def matches_record(self,record)->bool:
        fields=("account_ref","workspace_ref","environment_ref","client_ref","sdk_version")
        return all(getattr(record,n)==getattr(self.binding,n) for n in fields) and all(getattr(record,n)==getattr(self,n) for n in ("control_volume_id","artifact_volume_id","job_ref","effect_id","command_digest","plan_digest","deployment_attestation_digest","invocation_nonce","generation"))
@dataclass(frozen=True,slots=True)
class StageExpectationV1:
    operation:OperationBindingV1;binding:ModalClientBinding;claim_digest:str;bundle_digest:str;bundle_size:int
    def __post_init__(self):
        if type(self.operation) is not OperationBindingV1:raise TypeError("operation must be OperationBindingV1")
        if self.effect.kind is not EffectKind.SUBMIT:raise ValueError("stage effect must be submit")
        if self.effect.scope.provider!="modal" or self.effect.scope.account_ref!=self.binding.account_ref or self.effect.scope.namespace_ref!=self.binding.environment_ref:raise ValueError("stage effect scope mismatch")
        require_layout(self.control_volume_id,self.artifact_volume_id,self.output_prefix)
        for n in ("claim_digest","bundle_digest"):object.__setattr__(self,n,digest(getattr(self,n),n))
        object.__setattr__(self,"bundle_size",strict_int(self.bundle_size,"bundle_size",maximum=2**63-1))
    @property
    def effect(self):return self.operation.effect
    @property
    def operation_binding_digest(self):return self.operation.digest
    @property
    def control_volume_id(self):return self.operation.stage_target.control_volume_id
    @property
    def artifact_volume_id(self):return self.operation.stage_target.artifact_volume_id
    @property
    def output_prefix(self):return self.operation.stage_target.output_prefix
    @property
    def key_ref(self):return self.operation.stage_target.key_ref
    @property
    def invocation_nonce(self):return self.operation.invocation_nonce
    @property
    def plan_digest(self):return self.operation.plan_fingerprint
    @classmethod
    def from_stage(cls,operation,binding,*,claim:bytes,bundle:bytes):
        if not isinstance(claim,bytes) or not isinstance(bundle,bytes):raise TypeError("claim and bundle must be bytes")
        return cls(operation,binding,sha(claim),sha(bundle),len(bundle))
class StageControlPlane:
    def __init__(self,store,authenticator,facade,*,bounds=BoundsPolicyV1()):self.store=store;self.auth=authenticator;self.facade=facade;self.bounds=bounds
    def validate(self,receipt:StageReceiptV1)->StageExpectationV1:
        try:expected=self.store.load_modal_expectation(receipt.effect_id)
        except Exception:raise ValueError("stage expectation unavailable") from None
        if not isinstance(expected,StageExpectationV1):raise ValueError("stage expectation unavailable")
        expected_receipt=StageReceiptV1(expected.effect.effect_id,expected.operation_binding_digest,expected.control_volume_id,expected.artifact_volume_id,expected.claim_digest,expected.bundle_digest)
        if type(receipt) is not StageReceiptV1 or receipt!=expected_receipt:raise ValueError("receipt mismatch")
        from .binding import Readiness,readiness_report
        report=readiness_report(expected.binding,self.facade)
        if report.status is not Readiness.READY:raise ValueError(report.reason_code)
        try:
            claim=self.facade.read_complete(expected.control_volume_id,operation_path(expected.effect.effect_id,"control","stage-claim.v1.json"),max_bytes=self.bounds.max_control_bytes);tag=self.facade.read_complete(expected.control_volume_id,operation_path(expected.effect.effect_id,"control","stage-claim.v1.mac"),max_bytes=128)
        except Exception:raise ValueError("stage claim unavailable") from None
        try:authenticated=self.auth.verify("modal-stage-claim/v1",claim,tag,expected.key_ref)
        except Exception:raise ValueError("stage authentication unavailable") from None
        if not authenticated:raise ValueError("stage authentication failed")
        value=_object(claim,self.bounds.max_control_bytes);expected_claim={"schema":"synaptic.modal-stage-claim/v1","effect_provider":expected.effect.scope.provider,"effect_account_ref":expected.effect.scope.account_ref,"effect_namespace_ref":expected.effect.scope.namespace_ref,"effect_id":expected.effect.effect_id,"effect_kind":expected.effect.kind.value,"operation_key":expected.effect.effect_key,"operation_binding_digest":expected.operation_binding_digest,"control_volume_id":expected.control_volume_id,"artifact_volume_id":expected.artifact_volume_id,"bundle_digest":expected.bundle_digest,"bundle_size":expected.bundle_size,"plan_digest":expected.plan_digest,"invocation_nonce":expected.invocation_nonce,"output_prefix":expected.output_prefix}
        strict_int(value.get("bundle_size"),"bundle_size",maximum=self.bounds.max_bundle_bytes)
        if value!=expected_claim:raise ValueError("stage claim binding mismatch")
        try:bundle=self.facade.read_complete(expected.artifact_volume_id,operation_path(expected.effect.effect_id,"input","bundle.bin"),max_bytes=self.bounds.max_bundle_bytes)
        except Exception:raise ValueError("stage bundle unavailable") from None
        if len(bundle)!=expected.bundle_size or sha(bundle)!=expected.bundle_digest or sha(claim)!=expected.claim_digest:raise ValueError("stage content mismatch")
        return expected
@dataclass(frozen=True,slots=True)
class TerminalExpectationV1:
    binding:ModalClientBinding;control_volume_id:str;artifact_volume_id:str;key_ref:str;job_ref:str;effect_id:str;command_digest:str;plan_digest:str;deployment_attestation_digest:str;invocation_nonce:str;generation:int
    def __post_init__(self):
        require_layout(self.control_volume_id,self.artifact_volume_id)
        for n in ("control_volume_id","artifact_volume_id","key_ref","job_ref","effect_id","invocation_nonce"):object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        for n in ("command_digest","plan_digest","deployment_attestation_digest"):object.__setattr__(self,n,digest(getattr(self,n),n))
        object.__setattr__(self,"generation",strict_int(self.generation,"generation",minimum=1,maximum=2**31-1))
    @property
    def identity(self):return CrossPlaneIdentityV1(self.binding,self.control_volume_id,self.artifact_volume_id,self.job_ref,self.effect_id,self.command_digest,self.plan_digest,self.deployment_attestation_digest,self.invocation_nonce,self.generation,self.key_ref)
@dataclass(frozen=True,slots=True)
class TerminalValidationResult:
    evidence:object;canonical_digest:str;canonical_size:int;identity:CrossPlaneIdentityV1
    def __post_init__(self):
        from .contracts import TerminalEvidenceV1
        if type(self.evidence) is not TerminalEvidenceV1 or type(self.identity) is not CrossPlaneIdentityV1:raise TypeError("invalid terminal validation result")
        object.__setattr__(self,"canonical_digest",digest(self.canonical_digest,"canonical_digest"));object.__setattr__(self,"canonical_size",strict_int(self.canonical_size,"canonical_size",minimum=1))
class TerminalControlPlane:
    def __init__(self,store,authenticator,facade,*,bounds=BoundsPolicyV1()):self.store=store;self.auth=authenticator;self.facade=facade;self.bounds=bounds
    def validate(self,requested_effect):
        from .binding import Readiness,readiness_report
        from .contracts import TerminalEvidenceV1
        try:expected=self.store.load_terminal_expectation(requested_effect)
        except Exception:raise ValueError("terminal expectation unavailable") from None
        if not isinstance(expected,TerminalExpectationV1) or requested_effect!=expected.effect_id:raise ValueError("terminal expectation misrouted")
        report=readiness_report(expected.binding,self.facade)
        if report.status is not Readiness.READY:raise ValueError(report.reason_code)
        try:
            data=self.facade.read_complete(expected.control_volume_id,operation_path(expected.effect_id,"evidence","terminal-evidence.v1.json"),max_bytes=self.bounds.max_control_bytes);tag=self.facade.read_complete(expected.control_volume_id,operation_path(expected.effect_id,"evidence","terminal-evidence.v1.mac"),max_bytes=128)
        except Exception:raise ValueError("terminal evidence unavailable") from None
        try:authenticated=self.auth.verify("modal-terminal/v1",data,tag,expected.key_ref)
        except Exception:raise ValueError("terminal authentication unavailable") from None
        if not authenticated:raise ValueError("terminal authentication failed")
        value=TerminalEvidenceV1.parse(data,limit=self.bounds.max_control_bytes)
        identity=expected.identity
        if type(identity) is not CrossPlaneIdentityV1 or not identity.matches_record(value):raise ValueError("terminal identity mismatch")
        return TerminalValidationResult(value,sha(data),len(data),identity)
