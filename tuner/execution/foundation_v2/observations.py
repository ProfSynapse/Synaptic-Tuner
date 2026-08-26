from dataclasses import dataclass
from enum import Enum
from .canonical import canonical_bytes,digest_text,domain_digest,safe_ref
from .references import ProviderStageRefV1,ScopedProviderRunRefV1,CancellationRefV1
class ObservationDisposition(str,Enum):FOUND="found";DEFINITELY_ABSENT="definitely_absent";INDETERMINATE="indeterminate"
@dataclass(frozen=True,slots=True)
class ProviderObservationV1:
    effect_id:str;command_digest:str;executor_descriptor_digest:str;disposition:ObservationDisposition;resolution_digest:str;result_epoch:int;stage_ref:ProviderStageRefV1|None=None;provider_run:ScopedProviderRunRefV1|None=None;cancellation:CancellationRefV1|None=None;finality_proof:object|None=None
    def __post_init__(self):
        safe_ref(self.effect_id,"effect_id");digest_text(self.command_digest,"command_digest");digest_text(self.executor_descriptor_digest,"executor_descriptor_digest");digest_text(self.resolution_digest,"resolution_digest")
        if type(self.result_epoch) is not int or self.result_epoch<1:raise ValueError("result_epoch must be positive")
        refs=(self.stage_ref,self.provider_run,self.cancellation)
        if self.disposition is not ObservationDisposition.FOUND and any(x is not None for x in refs):raise ValueError("non-found observations cannot carry references")
        if self.disposition is ObservationDisposition.FOUND and sum(x is not None for x in refs)!=1:raise ValueError("found observation requires exactly one typed reference")
        if self.disposition is ObservationDisposition.DEFINITELY_ABSENT and self.finality_proof is not None: digest_text(self.finality_proof.proof_digest,"finality_proof_digest")
    @property
    def finality_proof_digest(self):return None if self.finality_proof is None else self.finality_proof.proof_digest
    def to_dict(self):return {"schema_version":"synaptic-observation/v2","effect_id":self.effect_id,"command_digest":self.command_digest,"executor_descriptor_digest":self.executor_descriptor_digest,"disposition":self.disposition.value,"resolution_digest":self.resolution_digest,"result_epoch":self.result_epoch,"stage_ref":None if self.stage_ref is None else self.stage_ref.to_dict(),"provider_run":None if self.provider_run is None else self.provider_run.to_dict(),"cancellation":None if self.cancellation is None else {"run":self.cancellation.run.to_dict(),"reason_digest":self.cancellation.reason_digest},"finality_proof_digest":self.finality_proof_digest}
    @property
    def digest(self):return domain_digest("synaptic-observation/v2",canonical_bytes(self.to_dict()))
