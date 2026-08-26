"""Bounded provider contracts; credentials and mutation authority stay opaque."""
from __future__ import annotations

import hashlib, json
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..contracts import EffectDisposition, EffectIdentity, EffectObservation, LifecyclePhase, ProviderRunPhase, digest, required_text, safe_ref, timestamp

# The generic staging envelope uses the strictest currently supported provider
# boundary.  Modal v1 transports one canonical bundle as standard Base64.
MAX_BUNDLE_BYTES = 8_388_608
MAX_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_SECRETS = 16
MAX_SECRET_KEYS = 32
MAX_ARGUMENTS = 32
MAX_ARGUMENT_BYTES = 16 * 1024
MAX_CURSOR_BYTES = 4096
MAX_LOG_LIMIT = 1000
MAX_LOG_MESSAGE_BYTES = 16 * 1024
MAX_ARTIFACTS = 10_000
MAX_ARTIFACT_SIZE = 16 * 1024 * 1024 * 1024
MAX_ARTIFACT_TOTAL = 128 * 1024 * 1024 * 1024


class ProviderOutcome(str, Enum):
    READY="ready"; FOUND="found"; REJECTED="rejected"; COLLISION="collision"; INDETERMINATE="indeterminate"; UNSUPPORTED="unsupported"


class ProviderContractError(RuntimeError): pass
class ProviderRejected(ProviderContractError): pass
class ProviderCollision(ProviderContractError): pass
class ProviderIndeterminate(ProviderContractError): pass
class ProviderUnsupported(ProviderContractError): pass


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    provider: str = "modal"
    sdk_version: str = "1.5.4"
    artifact_backend: str = "modal_volume_v1"
    hardware: tuple[str, ...] = ("A10",)
    submission_lookup: bool = True
    cancellation_lookup: bool = False
    function_call_logs: str = "fetch_at_least_once"


@dataclass(frozen=True, slots=True)
class CredentialHandle:
    ref: str
    account_ref: str
    workspace_ref: str
    environment_ref: str
    def __post_init__(self):
        for name in ("ref", "account_ref", "workspace_ref", "environment_ref"):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class SecretReference:
    name: str
    required_keys: tuple[str, ...]
    def __post_init__(self):
        object.__setattr__(self, "name", safe_ref(self.name, "secret_name"))
        keys=tuple(safe_ref(v,"secret_key") for v in self.required_keys)
        if not keys or len(keys)>MAX_SECRET_KEYS or len(set(keys))!=len(keys): raise ValueError("invalid required secret keys")
        object.__setattr__(self,"required_keys",keys)


@dataclass(frozen=True, slots=True)
class ProviderPlan:
    app_name: str
    function_name: str
    function_version: str
    image_id: str
    gpu: str="A10"
    timeout_seconds: int=3600
    secrets: tuple[SecretReference,...]=()
    def __post_init__(self):
        for n in ("app_name","function_name","function_version","image_id"):
            object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        if self.gpu!="A10": raise ValueError("Modal hardware must be A10")
        if not isinstance(self.timeout_seconds,int) or isinstance(self.timeout_seconds,bool) or not 1<=self.timeout_seconds<=MAX_TIMEOUT_SECONDS: raise ValueError("invalid timeout")
        secrets=tuple(self.secrets)
        if len(secrets)>MAX_SECRETS or len({s.name for s in secrets})!=len(secrets): raise ValueError("invalid secret references")
        object.__setattr__(self,"secrets",secrets)
    @property
    def fingerprint(self)->str:
        document={"app":self.app_name,"function":self.function_name,"function_version":self.function_version,"image_id":self.image_id,"gpu":self.gpu,"timeout":self.timeout_seconds,"secrets":[{"name":s.name,"required_keys":s.required_keys} for s in self.secrets]}
        return hashlib.sha256(b"synaptic.modal-plan/v1\0"+json.dumps(document,sort_keys=True,separators=(",",":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderPreflight:
    outcome: ProviderOutcome
    descriptor: ProviderDescriptor
    diagnostic_code: str|None=None


@dataclass(frozen=True, slots=True)
class StageBundle:
    payload: bytes
    sha256: str
    def __post_init__(self):
        if not isinstance(self.payload,bytes) or not self.payload or len(self.payload)>MAX_BUNDLE_BYTES: raise ValueError("invalid stage bundle")
        object.__setattr__(self,"sha256",digest(self.sha256,"bundle_sha256"))
        if hashlib.sha256(self.payload).hexdigest()!=self.sha256: raise ValueError("stage bundle digest mismatch")


@dataclass(frozen=True, slots=True)
class ArtifactSlot:
    control_volume_name: str
    artifact_volume_name: str
    output_prefix: str="output"
    def __post_init__(self):
        for n in ("control_volume_name","artifact_volume_name"):
            object.__setattr__(self,n,safe_ref(getattr(self,n),n))
        prefix=_artifact_path(self.output_prefix)
        if "/" in prefix: raise ValueError("output_prefix must be one segment")
        object.__setattr__(self,"output_prefix",prefix)
        if self.control_volume_name==self.artifact_volume_name: raise ValueError("control and artifact volumes must differ")


_RECEIPT_ISSUER=object()
class StageReceipt:
    __slots__=("effect","slot","claim_digest","recovered")
    def __init__(self,effect:EffectIdentity,slot:ArtifactSlot,claim_digest:str,recovered:bool=False,*,_issuer:object):
        if _issuer is not _RECEIPT_ISSUER: raise TypeError("stage receipts are provider-issued")
        self.effect=effect;self.slot=slot;self.claim_digest=digest(claim_digest,"claim_digest");self.recovered=bool(recovered)
    def __eq__(self,other):return isinstance(other,StageReceipt) and (self.effect,self.slot,self.claim_digest,self.recovered)==(other.effect,other.slot,other.claim_digest,other.recovered)
def _mint_stage_receipt(effect,slot,claim_digest,recovered=False):return StageReceipt(effect,slot,claim_digest,recovered,_issuer=_RECEIPT_ISSUER)


@dataclass(frozen=True, slots=True)
class ReconciliationContext:
    effect: EffectIdentity
    credential: CredentialHandle
    stage: StageReceipt
    def __post_init__(self):
        if self.stage.effect!=self.effect: raise ValueError("stage/effect mismatch")
        scope=self.effect.scope
        if scope.account_ref!=self.credential.account_ref or scope.namespace_ref!=self.credential.environment_ref: raise ValueError("credential scope mismatch")


@dataclass(frozen=True, slots=True)
class InvocationArgument:
    name: str
    value: str
    def __post_init__(self):
        object.__setattr__(self,"name",safe_ref(self.name,"argument_name"))
        object.__setattr__(self,"value",required_text(self.value,"argument_value"))


@dataclass(frozen=True, slots=True)
class SubmissionRequest:
    context: ReconciliationContext
    plan: ProviderPlan
    arguments: tuple[InvocationArgument,...]=()
    def __post_init__(self):
        args=tuple(self.arguments)
        if any(not isinstance(a,InvocationArgument) for a in args): raise TypeError("arguments must be InvocationArgument values")
        if len(args)>MAX_ARGUMENTS or len({a.name for a in args})!=len(args) or sum(len(a.name.encode())+len(a.value.encode()) for a in args)>MAX_ARGUMENT_BYTES: raise ValueError("invalid invocation arguments")
        object.__setattr__(self,"arguments",args)


@dataclass(frozen=True, slots=True)
class CancellationRequest:
    context: ReconciliationContext
    provider_job_ref: str
    plan_fingerprint: str
    def __post_init__(self):
        object.__setattr__(self,"provider_job_ref",safe_ref(self.provider_job_ref,"provider_job_ref"))
        object.__setattr__(self,"plan_fingerprint",digest(self.plan_fingerprint,"plan_fingerprint"))


@dataclass(frozen=True, slots=True)
class EffectReceipt:
    effect: EffectIdentity
    disposition: EffectDisposition
    provider_job_ref: str
    receipt_digest: str
    def as_observation(self): return EffectObservation(self.effect,self.disposition,self.provider_job_ref,self.receipt_digest)


@dataclass(frozen=True, slots=True)
class ProviderJob:
    provider_job_ref: str
    effect: EffectIdentity
    plan_fingerprint: str
    def __post_init__(self):
        object.__setattr__(self,"provider_job_ref",safe_ref(self.provider_job_ref,"provider_job_ref"))
        object.__setattr__(self,"plan_fingerprint",digest(self.plan_fingerprint,"plan_fingerprint"))


@dataclass(frozen=True, slots=True)
class ProviderObservation:
    provider_job: ProviderJob
    phase: ProviderRunPhase
    lifecycle_phase: LifecyclePhase
    diagnostic_code: str|None=None


@dataclass(frozen=True, slots=True)
class ProviderLogEntry:
    timestamp: str; message: str; fingerprint: str
    def __post_init__(self):
        object.__setattr__(self,"timestamp",timestamp(self.timestamp,"timestamp")); object.__setattr__(self,"fingerprint",digest(self.fingerprint,"fingerprint"))


@dataclass(frozen=True, slots=True)
class ProviderLogPage:
    entries: tuple[ProviderLogEntry,...]; next_cursor: str|None; delivery: str="at_least_once"; diagnostic_code: str="provider_logs_redacted_at_least_once"


class ProviderArtifact:
    __slots__=("path","size","provider_identity","provider_job_ref","effect_id","plan_fingerprint")
    def __init__(self,path,size,provider_identity,provider_job_ref,effect_id,plan_fingerprint,*,_issuer):
        if _issuer is not _RECEIPT_ISSUER: raise TypeError("artifacts are inventory-issued")
        self.path=_artifact_path(path);self.provider_identity=safe_ref(provider_identity,"provider_identity");self.provider_job_ref=safe_ref(provider_job_ref,"provider_job_ref");self.effect_id=safe_ref(effect_id,"effect_id");self.plan_fingerprint=digest(plan_fingerprint,"plan_fingerprint")
        if not isinstance(size,int) or isinstance(size,bool) or not 0<=size<=MAX_ARTIFACT_SIZE: raise ValueError("invalid artifact size")
        self.size=size
    def __eq__(self,other):return isinstance(other,ProviderArtifact) and (self.path,self.size,self.provider_identity,self.provider_job_ref,self.effect_id,self.plan_fingerprint)==(other.path,other.size,other.provider_identity,other.provider_job_ref,other.effect_id,other.plan_fingerprint)
def _mint_artifact(*args):return ProviderArtifact(*args,_issuer=_RECEIPT_ISSUER)


@dataclass(frozen=True, slots=True)
class ProviderArtifactInventory:
    artifacts: tuple[ProviderArtifact,...]
    structural_only: bool=True


@dataclass(frozen=True, slots=True)
class ArtifactReadBounds:
    max_bytes: int; expected_size: int; expected_sha256: str|None=None
    def __post_init__(self):
        if not 1<=self.max_bytes<=MAX_ARTIFACT_SIZE or not 0<=self.expected_size<=self.max_bytes: raise ValueError("invalid artifact bounds")
        if self.expected_sha256: object.__setattr__(self,"expected_sha256",digest(self.expected_sha256,"expected_sha256"))


def _artifact_path(value: str)->str:
    value=required_text(value,"artifact_path")
    if "\\" in value or value.startswith("/") or value.endswith("/") or any(p in {"",".",".."} for p in value.split("/")): raise ValueError("artifact path must be canonical relative POSIX")
    return value


@runtime_checkable
class ExecutionProvider(Protocol):
    def lookup_submission(self, auth: ReconciliationContext, effect: EffectIdentity)->EffectObservation: ...


__all__=[n for n in tuple(globals()) if (n.startswith("Provider") or n in {"ArtifactReadBounds","ArtifactSlot","CredentialHandle","ExecutionProvider","InvocationArgument","ReconciliationContext","SecretReference","StageBundle","StageReceipt"}) and n not in {"ProviderAuthorization"}]
