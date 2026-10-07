"""Canonical authenticated provider receipts and bounded invalid evidence."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .canonical import canonical_bytes, digest_text, domain_digest, exact_fields, exact_integer, parse_canonical_object, safe_ref
from .observations import ObservationDisposition, ProviderObservationV1
from .references import CancellationRefV1, ProviderRunRefV1, ProviderStageRefV1, ScopedProviderRunRefV1

_CONTENT_FIELDS = frozenset({"schema_version", "effect_id", "command_digest", "disposition", "observation_digest", "result_epoch", "stage_ref", "provider_run", "cancellation", "finality_proof_digest", "source_kind", "source_owner_ref", "source_generation", "source_ownership_epoch", "source_claim_digest", "source_grant_ref", "source_grant_digest"})
_AUTH_FIELDS = frozenset({"schema_version", "content", "authority_ref", "tag"})


@dataclass(frozen=True, slots=True)
class ReceiptContentV2:
    effect_id: str
    command_digest: str
    disposition: ObservationDisposition
    observation_digest: str
    result_epoch: int
    stage_ref: ProviderStageRefV1 | None
    provider_run: ScopedProviderRunRefV1 | None
    cancellation: CancellationRefV1 | None
    finality_proof_digest: str | None
    source_kind: str
    source_owner_ref: str
    source_generation: int
    source_ownership_epoch: int
    source_claim_digest: str
    source_grant_ref: str
    source_grant_digest: str

    def __post_init__(self):
        safe_ref(self.effect_id, "effect_id"); safe_ref(self.source_owner_ref, "source_owner_ref"); safe_ref(self.source_grant_ref, "source_grant_ref")
        for name in ("command_digest", "observation_digest", "source_claim_digest", "source_grant_digest"): digest_text(getattr(self, name), name)
        if type(self.disposition) is not ObservationDisposition or self.source_kind not in {"dispatch", "reconciliation"}: raise ValueError("receipt disposition/source invalid")
        exact_integer(self.result_epoch, "result_epoch", minimum=1); exact_integer(self.source_generation, "source_generation", minimum=1); exact_integer(self.source_ownership_epoch, "source_ownership_epoch", minimum=1)
        refs = (self.stage_ref, self.provider_run, self.cancellation)
        for value, expected in zip(refs, (ProviderStageRefV1, ScopedProviderRunRefV1, CancellationRefV1), strict=True):
            if value is not None and type(value) is not expected: raise TypeError("receipt contains a non-exact typed reference")
        if self.disposition is ObservationDisposition.FOUND and sum(value is not None for value in refs) != 1: raise ValueError("found receipt requires exactly one typed reference")
        if self.disposition is not ObservationDisposition.FOUND and any(value is not None for value in refs): raise ValueError("non-found receipt cannot carry references")
        if self.finality_proof_digest is not None: digest_text(self.finality_proof_digest, "finality_proof_digest")
        if self.disposition is not ObservationDisposition.DEFINITELY_ABSENT and self.finality_proof_digest is not None: raise ValueError("only absence can carry finality evidence")
        if self.result_epoch != self.source_ownership_epoch: raise ValueError("result epoch must equal source ownership epoch")

    @classmethod
    def from_observation(cls, observation: ProviderObservationV1, *, source_kind: str, source_owner_ref: str,
                         source_generation: int, source_ownership_epoch: int, source_claim_digest: str,
                         source_grant_ref: str, source_grant_digest: str) -> "ReceiptContentV2":
        if type(observation) is not ProviderObservationV1: raise TypeError("exact provider observation required")
        return cls(observation.effect_id, observation.command_digest, observation.disposition, observation.digest,
                   observation.result_epoch, observation.stage_ref, observation.provider_run, observation.cancellation,
                   observation.finality_proof_digest, source_kind, source_owner_ref, source_generation,
                   source_ownership_epoch, source_claim_digest, source_grant_ref, source_grant_digest)

    def to_dict(self):
        return {"schema_version":"synaptic-receipt-content/v4","effect_id":self.effect_id,"command_digest":self.command_digest,"disposition":self.disposition.value,"observation_digest":self.observation_digest,"result_epoch":self.result_epoch,"stage_ref":None if self.stage_ref is None else self.stage_ref.to_dict(),"provider_run":None if self.provider_run is None else self.provider_run.to_dict(),"cancellation":None if self.cancellation is None else {"run":self.cancellation.run.to_dict(),"reason_digest":self.cancellation.reason_digest},"finality_proof_digest":self.finality_proof_digest,"source_kind":self.source_kind,"source_owner_ref":self.source_owner_ref,"source_generation":self.source_generation,"source_ownership_epoch":self.source_ownership_epoch,"source_claim_digest":self.source_claim_digest,"source_grant_ref":self.source_grant_ref,"source_grant_digest":self.source_grant_digest}
    @property
    def canonical_bytes(self): return canonical_bytes(self.to_dict())
    @property
    def content_digest(self): return domain_digest("synaptic-receipt-content/v4", self.canonical_bytes)
    @property
    def semantic_digest(self):
        return domain_digest("synaptic-terminal-semantics/v2", canonical_bytes({"disposition":self.disposition.value,"stage_ref":None if self.stage_ref is None else self.stage_ref.to_dict(),"provider_run":None if self.provider_run is None else self.provider_run.to_dict(),"cancellation":None if self.cancellation is None else {"run":self.cancellation.run.to_dict(),"reason_digest":self.cancellation.reason_digest}}))
    @classmethod
    def parse(cls, raw):
        doc=parse_canonical_object(raw,name="receipt content");exact_fields(doc,_CONTENT_FIELDS,"receipt content")
        if doc["schema_version"]!="synaptic-receipt-content/v4":raise ValueError("unsupported receipt content schema")
        stage,run,cancellation=doc["stage_ref"],doc["provider_run"],doc["cancellation"]
        if stage is not None:
            if not isinstance(stage,dict):raise ValueError("stage reference malformed")
            stage=ProviderStageRefV1(**stage)
        if run is not None:
            if not isinstance(run,dict):raise ValueError("run reference malformed")
            run=ScopedProviderRunRefV1(**run)
        if cancellation is not None:
            if not isinstance(cancellation,dict) or set(cancellation)!={"run","reason_digest"}:raise ValueError("cancellation reference malformed")
            run_doc=cancellation["run"]
            if not isinstance(run_doc,dict) or set(run_doc)!={"provider_job_ref"}:raise ValueError("cancellation run malformed")
            cancellation=CancellationRefV1(ProviderRunRefV1(**run_doc),cancellation["reason_digest"])
        return cls(doc["effect_id"],doc["command_digest"],ObservationDisposition(doc["disposition"]),doc["observation_digest"],doc["result_epoch"],stage,run,cancellation,doc["finality_proof_digest"],doc["source_kind"],doc["source_owner_ref"],doc["source_generation"],doc["source_ownership_epoch"],doc["source_claim_digest"],doc["source_grant_ref"],doc["source_grant_digest"])


@dataclass(frozen=True, slots=True)
class AuthenticatedReceiptV2:
    content: ReceiptContentV2
    authority_ref: str
    tag: str
    def __post_init__(self):
        if type(self.content) is not ReceiptContentV2:raise TypeError("exact receipt content required")
        safe_ref(self.authority_ref,"authority_ref");digest_text(self.tag,"tag")
    @property
    def canonical_bytes(self):return canonical_bytes({"schema_version":"synaptic-authenticated-receipt/v4","content":self.content.to_dict(),"authority_ref":self.authority_ref,"tag":self.tag})
    @property
    def authenticated_receipt_digest(self):return domain_digest("synaptic-authenticated-receipt/v4",self.canonical_bytes)
    @classmethod
    def parse(cls,raw):
        doc=parse_canonical_object(raw,name="authenticated receipt");exact_fields(doc,_AUTH_FIELDS,"authenticated receipt")
        if doc["schema_version"]!="synaptic-authenticated-receipt/v4" or not isinstance(doc["content"],dict):raise ValueError("unsupported authenticated receipt schema")
        return cls(ReceiptContentV2.parse(canonical_bytes(doc["content"])),doc["authority_ref"],doc["tag"])


class ReceiptAuthorityV2:
    __slots__=("authority_ref","_key")
    def __init__(self,ref,key):
        self.authority_ref=safe_ref(ref,"authority_ref")
        if type(key) is not bytes or len(key)<32:raise ValueError("receipt authority key too short")
        self._key=key
    def __repr__(self):return f"ReceiptAuthorityV2(authority_ref={self.authority_ref!r}, key=<redacted>)"
    def issue(self,content):
        if type(content) is not ReceiptContentV2:raise TypeError("exact receipt content required")
        owned=ReceiptContentV2.parse(content.canonical_bytes);tag=hmac.new(self._key,b"receipt-v4\0"+bytes.fromhex(owned.content_digest),hashlib.sha256).hexdigest()
        return AuthenticatedReceiptV2(owned,self.authority_ref,tag)
    def verify(self,receipt):
        try:
            if type(receipt) is not AuthenticatedReceiptV2:return False
            owned=AuthenticatedReceiptV2.parse(receipt.canonical_bytes);expected=self.issue(owned.content)
            return owned.authority_ref==self.authority_ref and hmac.compare_digest(owned.tag,expected.tag)
        except Exception:return False


class InvalidEvidenceSiteV2(str, Enum):
    DISPATCH_RESOLUTION="dispatch_resolution"
    DISPATCH_OBSERVATION="dispatch_observation"
    RECONCILIATION_RESOLUTION="reconciliation_resolution"
    RECONCILIATION_OBSERVATION="reconciliation_observation"


@dataclass(frozen=True, slots=True)
class InvalidEvidenceContentV2:
    effect_id:str;command_digest:str;site:InvalidEvidenceSiteV2;source_kind:str;source_owner_ref:str;source_generation:int;source_ownership_epoch:int;source_claim_digest:str;source_grant_ref:str;source_grant_digest:str;evidence_digest:str
    def __post_init__(self):
        safe_ref(self.effect_id,"effect_id");safe_ref(self.source_owner_ref,"source_owner_ref");safe_ref(self.source_grant_ref,"source_grant_ref")
        for n in ("command_digest","source_claim_digest","source_grant_digest","evidence_digest"):digest_text(getattr(self,n),n)
        if type(self.site) is not InvalidEvidenceSiteV2 or self.source_kind not in {"dispatch","reconciliation"}:raise ValueError("invalid evidence source/site invalid")
        expected_kind="dispatch" if self.site in {InvalidEvidenceSiteV2.DISPATCH_RESOLUTION,InvalidEvidenceSiteV2.DISPATCH_OBSERVATION} else "reconciliation"
        if self.source_kind!=expected_kind:raise ValueError("invalid evidence site/source mismatch")
        exact_integer(self.source_generation,"source_generation",minimum=1);exact_integer(self.source_ownership_epoch,"source_ownership_epoch",minimum=1)
    def to_dict(self):return {"schema_version":"synaptic-invalid-evidence-content/v2","effect_id":self.effect_id,"command_digest":self.command_digest,"site":self.site.value,"source_kind":self.source_kind,"source_owner_ref":self.source_owner_ref,"source_generation":self.source_generation,"source_ownership_epoch":self.source_ownership_epoch,"source_claim_digest":self.source_claim_digest,"source_grant_ref":self.source_grant_ref,"source_grant_digest":self.source_grant_digest,"evidence_digest":self.evidence_digest}
    @property
    def canonical_bytes(self):return canonical_bytes(self.to_dict())
    @property
    def content_digest(self):return domain_digest("synaptic-invalid-evidence-content/v2",self.canonical_bytes)
    @classmethod
    def parse(cls,raw):
        doc=parse_canonical_object(raw,name="invalid evidence content");exact_fields(doc,frozenset({"schema_version","effect_id","command_digest","site","source_kind","source_owner_ref","source_generation","source_ownership_epoch","source_claim_digest","source_grant_ref","source_grant_digest","evidence_digest"}),"invalid evidence content")
        if doc["schema_version"]!="synaptic-invalid-evidence-content/v2":raise ValueError("invalid evidence schema")
        return cls(doc["effect_id"],doc["command_digest"],InvalidEvidenceSiteV2(doc["site"]),doc["source_kind"],doc["source_owner_ref"],doc["source_generation"],doc["source_ownership_epoch"],doc["source_claim_digest"],doc["source_grant_ref"],doc["source_grant_digest"],doc["evidence_digest"])


@dataclass(frozen=True, slots=True)
class AuthenticatedInvalidEvidenceV2:
    content:InvalidEvidenceContentV2;authority_ref:str;tag:str
    def __post_init__(self):
        if type(self.content) is not InvalidEvidenceContentV2:raise TypeError("exact invalid evidence content required")
        safe_ref(self.authority_ref,"authority_ref");digest_text(self.tag,"tag")
    @property
    def canonical_bytes(self):return canonical_bytes({"schema_version":"synaptic-authenticated-invalid-evidence/v2","content":self.content.to_dict(),"authority_ref":self.authority_ref,"tag":self.tag})
    @property
    def authenticated_evidence_digest(self):return domain_digest("synaptic-authenticated-invalid-evidence/v2",self.canonical_bytes)
    @classmethod
    def parse(cls,raw):
        doc=parse_canonical_object(raw,name="authenticated invalid evidence");exact_fields(doc,_AUTH_FIELDS,"authenticated invalid evidence")
        if doc["schema_version"]!="synaptic-authenticated-invalid-evidence/v2" or not isinstance(doc["content"],dict):raise ValueError("invalid evidence envelope schema")
        return cls(InvalidEvidenceContentV2.parse(canonical_bytes(doc["content"])),doc["authority_ref"],doc["tag"])


class InvalidEvidenceAuthorityV2:
    __slots__=("authority_ref","_key")
    def __init__(self,ref,key):
        self.authority_ref=safe_ref(ref,"authority_ref")
        if type(key) is not bytes or len(key)<32:raise ValueError("invalid evidence authority key too short")
        self._key=key
    def __repr__(self):return f"InvalidEvidenceAuthorityV2(authority_ref={self.authority_ref!r}, key=<redacted>)"
    def issue(self,content):
        if type(content) is not InvalidEvidenceContentV2:raise TypeError("exact invalid evidence content required")
        owned=InvalidEvidenceContentV2.parse(content.canonical_bytes);tag=hmac.new(self._key,b"invalid-evidence-v2\0"+bytes.fromhex(owned.content_digest),hashlib.sha256).hexdigest()
        return AuthenticatedInvalidEvidenceV2(owned,self.authority_ref,tag)
    def verify(self,evidence):
        try:
            if type(evidence) is not AuthenticatedInvalidEvidenceV2:return False
            owned=AuthenticatedInvalidEvidenceV2.parse(evidence.canonical_bytes);expected=self.issue(owned.content)
            return owned.authority_ref==self.authority_ref and hmac.compare_digest(owned.tag,expected.tag)
        except Exception:return False


class QuiescenceProofV2(Protocol): proof_digest:str
class FinalityProofV2(Protocol): proof_digest:str
class RecoveryVerifierV2(Protocol):
    def verify_quiescence(self,proof:QuiescenceProofV2,record:object,*,now_epoch:int)->bool:...
class FinalityVerifierV2(Protocol):
    def verify_finality(self,proof:FinalityProofV2,record:object,receipt:AuthenticatedReceiptV2,*,now_epoch:int)->bool:...
