"""Atomic dispatch ownership, evidence ledger, and reconciliation ownership."""

from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock

from .canonical import DiagnosticCode, FoundationError, canonical_bytes, digest_text, domain_digest, exact_integer, parse_canonical_object, safe_ref
from .authority import AuthenticatedGrantV2, AuthenticatedReconciliationGrantV1, GrantContentV2, ReconciliationGrantContentV1
from .commands import CancelCommandV2, StageCommandV2, SubmitCommandV2, parse_exact_command
from .observations import ObservationDisposition
from .receipts import AuthenticatedInvalidEvidenceV2, AuthenticatedReceiptV2, InvalidEvidenceSiteV2
from .references import CancellationRefV1, ProviderRunRefV1, ProviderStageRefV1, ScopedProviderRunRefV1


class DispatchState(str, Enum):
    OWNED_NOT_STARTED = "owned_not_started"
    OWNED_IN_FLIGHT = "owned_in_flight"
    RELINQUISHED = "relinquished"
    ORPHANED_UNPROVEN = "orphaned_unproven"
    QUIESCENCE_PROVEN = "quiescence_proven"


class EffectState(str, Enum):
    UNRESOLVED = "unresolved"
    FOUND = "found"
    DEFINITELY_ABSENT = "definitely_absent"
    INDETERMINATE = "indeterminate"
    CONTRADICTED = "contradicted"


class ReceiptFreshnessV2(str, Enum):
    FRESH = "fresh"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class ReconciliationGrantBindingV2:
    grant_ref: str
    grant_digest: str
    activated_at_admission_index: int
    prior_binding_digest: str | None

    def __post_init__(self):
        safe_ref(self.grant_ref,"grant_ref");digest_text(self.grant_digest,"grant_digest")
        exact_integer(self.activated_at_admission_index,"activated_at_admission_index")
        if self.prior_binding_digest is not None:digest_text(self.prior_binding_digest,"prior_binding_digest")

    def to_dict(self):return {"grant_ref":self.grant_ref,"grant_digest":self.grant_digest,"activated_at_admission_index":self.activated_at_admission_index,"prior_binding_digest":self.prior_binding_digest}
    @property
    def binding_digest(self):return domain_digest("synaptic-reconciliation-grant-binding/v2",canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class ReceiptAdmissionV2:
    receipt_digest: str
    source_kind: str
    source_owner_ref: str
    source_generation: int
    source_ownership_epoch: int
    source_claim_digest: str
    source_grant_ref: str
    source_grant_digest: str
    expected_source_kind: str
    expected_source_owner_ref: str
    expected_source_generation: int
    expected_source_ownership_epoch: int
    expected_source_claim_digest: str
    expected_grant_ref: str
    expected_grant_digest: str
    freshness: ReceiptFreshnessV2
    finality_verified: bool
    generated_invalid_codes: tuple[DiagnosticCode, ...]

    def __post_init__(self):
        digest_text(self.receipt_digest, "receipt_digest")
        if self.source_kind not in {"dispatch", "reconciliation"}: raise ValueError("unsupported admission source")
        safe_ref(self.source_owner_ref, "source_owner_ref")
        exact_integer(self.source_generation, "source_generation", minimum=1)
        exact_integer(self.source_ownership_epoch, "source_ownership_epoch", minimum=1)
        digest_text(self.source_claim_digest, "source_claim_digest")
        safe_ref(self.source_grant_ref,"source_grant_ref");digest_text(self.source_grant_digest,"source_grant_digest")
        if self.expected_source_kind not in {"dispatch","reconciliation"}:raise ValueError("unsupported expected admission source")
        safe_ref(self.expected_source_owner_ref,"expected_source_owner_ref");exact_integer(self.expected_source_generation,"expected_source_generation",minimum=1);exact_integer(self.expected_source_ownership_epoch,"expected_source_ownership_epoch",minimum=1);digest_text(self.expected_source_claim_digest,"expected_source_claim_digest")
        safe_ref(self.expected_grant_ref,"expected_grant_ref");digest_text(self.expected_grant_digest,"expected_grant_digest")
        if type(self.freshness) is not ReceiptFreshnessV2 or type(self.finality_verified) is not bool: raise TypeError("admission freshness/finality must be exact")
        if type(self.generated_invalid_codes) is not tuple or any(type(code) is not DiagnosticCode for code in self.generated_invalid_codes): raise TypeError("admission codes must be exact tuple")
        expected = ((DiagnosticCode.STALE_RESULT,) if self.freshness is ReceiptFreshnessV2.STALE else ())
        if self.finality_verified and DiagnosticCode.FINALITY_UNPROVEN in self.generated_invalid_codes: raise ValueError("verified finality cannot be unproven")
        if tuple(code for code in self.generated_invalid_codes if code is DiagnosticCode.STALE_RESULT) != expected: raise ValueError("admission stale code mismatch")
        if any(code not in {DiagnosticCode.STALE_RESULT, DiagnosticCode.FINALITY_UNPROVEN} for code in self.generated_invalid_codes): raise ValueError("unsupported admission code")

    def to_dict(self):
        return {"schema_version":"synaptic-receipt-admission/v2","receipt_digest":self.receipt_digest,"source_kind":self.source_kind,"source_owner_ref":self.source_owner_ref,"source_generation":self.source_generation,"source_ownership_epoch":self.source_ownership_epoch,"source_claim_digest":self.source_claim_digest,"source_grant_ref":self.source_grant_ref,"source_grant_digest":self.source_grant_digest,"expected_source_kind":self.expected_source_kind,"expected_source_owner_ref":self.expected_source_owner_ref,"expected_source_generation":self.expected_source_generation,"expected_source_ownership_epoch":self.expected_source_ownership_epoch,"expected_source_claim_digest":self.expected_source_claim_digest,"expected_grant_ref":self.expected_grant_ref,"expected_grant_digest":self.expected_grant_digest,"freshness":self.freshness.value,"finality_verified":self.finality_verified,"generated_invalid_codes":[code.value for code in self.generated_invalid_codes]}

    @property
    def admission_digest(self): return domain_digest("synaptic-receipt-admission/v2", canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class InvalidEvidenceAdmissionV2:
    authenticated_evidence_digest: str
    sequence: int
    prior_admission_digest: str | None
    site: InvalidEvidenceSiteV2
    source_kind: str
    source_owner_ref: str
    source_generation: int
    source_ownership_epoch: int
    source_claim_digest: str
    source_grant_ref: str
    source_grant_digest: str
    def __post_init__(self):
        digest_text(self.authenticated_evidence_digest,"authenticated_evidence_digest");exact_integer(self.sequence,"sequence",minimum=1)
        if self.prior_admission_digest is not None:digest_text(self.prior_admission_digest,"prior_admission_digest")
        if type(self.site) is not InvalidEvidenceSiteV2 or self.source_kind not in {"dispatch","reconciliation"}:raise ValueError("invalid evidence admission source invalid")
        safe_ref(self.source_owner_ref,"source_owner_ref");safe_ref(self.source_grant_ref,"source_grant_ref")
        exact_integer(self.source_generation,"source_generation",minimum=1);exact_integer(self.source_ownership_epoch,"source_ownership_epoch",minimum=1)
        digest_text(self.source_claim_digest,"source_claim_digest");digest_text(self.source_grant_digest,"source_grant_digest")
    def to_dict(self):return {"schema_version":"synaptic-invalid-evidence-admission/v2","authenticated_evidence_digest":self.authenticated_evidence_digest,"sequence":self.sequence,"prior_admission_digest":self.prior_admission_digest,"site":self.site.value,"source_kind":self.source_kind,"source_owner_ref":self.source_owner_ref,"source_generation":self.source_generation,"source_ownership_epoch":self.source_ownership_epoch,"source_claim_digest":self.source_claim_digest,"source_grant_ref":self.source_grant_ref,"source_grant_digest":self.source_grant_digest}
    @property
    def admission_digest(self):return domain_digest("synaptic-invalid-evidence-admission/v2",canonical_bytes(self.to_dict()))


def _grant_document(grant):
    if type(grant) is not AuthenticatedGrantV2 or type(grant.content) is not GrantContentV2: raise TypeError("exact authenticated grant required")
    return parse_canonical_object(grant.canonical_bytes,name="authenticated grant")


def _execution_grant_digest(grant):return grant.authenticated_grant_digest


def _reconciliation_grant_digest(grant):
    if type(grant) is not AuthenticatedReconciliationGrantV1 or type(grant.content) is not ReconciliationGrantContentV1: raise TypeError("exact reconciliation grant required")
    return grant.authenticated_grant_digest


def _expected_admission_authority(record,index):
    eligible=[claim for claim in record.reconciliation_claims if claim.grant_lineage[0].activated_at_admission_index<=index]
    if not eligible:return ("dispatch",record.grant.content.grant_ref,1,record.dispatch_epoch,record.dispatch_source_digest,record.grant.content.grant_ref,_execution_grant_digest(record.grant))
    claim=eligible[-1];binding=next(value for value in reversed(claim.grant_lineage) if value.activated_at_admission_index<=index)
    return ("reconciliation",claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest,binding.grant_ref,binding.grant_digest)


@dataclass(frozen=True, slots=True)
class ReconciliationOwnershipV2:
    owner_ref: str
    generation: int
    ownership_epoch: int
    claimed_at_epoch: int
    target_digest: str
    grant_ref: str
    grant_digest: str
    grant_lineage: tuple[ReconciliationGrantBindingV2, ...]
    active: bool = True
    completed: bool = False

    def __post_init__(self):
        safe_ref(self.owner_ref,"owner_ref");safe_ref(self.grant_ref,"grant_ref")
        exact_integer(self.generation,"generation",minimum=1);exact_integer(self.ownership_epoch,"ownership_epoch",minimum=1);exact_integer(self.claimed_at_epoch,"claimed_at_epoch")
        digest_text(self.target_digest,"target_digest");digest_text(self.grant_digest,"grant_digest")
        if type(self.grant_lineage) is not tuple or not self.grant_lineage or any(type(x) is not ReconciliationGrantBindingV2 for x in self.grant_lineage):raise ValueError("claim grant lineage must be nonempty exact tuple")
        if (self.grant_ref,self.grant_digest)!=(self.grant_lineage[-1].grant_ref,self.grant_lineage[-1].grant_digest):raise ValueError("claim current grant differs from lineage leaf")
        if len({x.grant_ref for x in self.grant_lineage})!=len(self.grant_lineage) or len({x.grant_digest for x in self.grant_lineage})!=len(self.grant_lineage):raise ValueError("claim grant lineage duplicates authority")
        if any(new.activated_at_admission_index<old.activated_at_admission_index for old,new in zip(self.grant_lineage,self.grant_lineage[1:],strict=False)):raise ValueError("claim grant lineage activation regressed")
        for index,binding in enumerate(self.grant_lineage):
            expected=None if index==0 else self.grant_lineage[index-1].binding_digest
            if binding.prior_binding_digest!=expected:raise ValueError("claim grant lineage chain invalid")
        if type(self.active) is not bool or type(self.completed) is not bool or (self.active and self.completed): raise ValueError("reconciliation flags invalid")

    @property
    def claim_digest(self):
        return domain_digest("synaptic-reconciliation-claim/v2", canonical_bytes({
            "owner_ref": self.owner_ref, "generation": self.generation,
            "ownership_epoch": self.ownership_epoch,
            "claimed_at_epoch": self.claimed_at_epoch,
            "target_digest": self.target_digest,
        }))


@dataclass(frozen=True, slots=True)
class EffectRecordV2:
    command_bytes: bytes
    grant: object
    dispatch: DispatchState
    state: EffectState
    attempt_count: int
    dispatch_epoch: int = 1
    results: tuple = ()
    receipt_admissions: tuple[ReceiptAdmissionV2, ...] = ()
    terminal_content_digests: tuple[str, ...] = ()
    invalid_codes: tuple[DiagnosticCode, ...] = ()
    invalid_evidence: tuple[AuthenticatedInvalidEvidenceV2, ...] = ()
    invalid_evidence_admissions: tuple[InvalidEvidenceAdmissionV2, ...] = ()
    reconciliation: ReconciliationOwnershipV2 | None = None
    reconciliation_claims: tuple[ReconciliationOwnershipV2, ...] = ()

    def __post_init__(self):
        if type(self.command_bytes) is not bytes or type(self.dispatch) is not DispatchState or type(self.state) is not EffectState: raise TypeError("effect record core types invalid")
        _grant_document(self.grant); exact_integer(self.attempt_count,"attempt_count",minimum=1);exact_integer(self.dispatch_epoch,"dispatch_epoch",minimum=1)
        if type(self.results) is not tuple or type(self.receipt_admissions) is not tuple or len(self.results)!=len(self.receipt_admissions): raise ValueError("receipt admissions must be ordered one-to-one")
        if any(type(r) is not AuthenticatedReceiptV2 for r in self.results) or any(type(a) is not ReceiptAdmissionV2 for a in self.receipt_admissions): raise TypeError("receipt ledger values must be exact")
        for receipt,admission in zip(self.results,self.receipt_admissions,strict=True):
            content=receipt.content
            if (admission.receipt_digest,admission.source_kind,admission.source_owner_ref,admission.source_generation,admission.source_ownership_epoch,admission.source_claim_digest,admission.source_grant_ref,admission.source_grant_digest)!=(receipt.authenticated_receipt_digest,content.source_kind,content.source_owner_ref,content.source_generation,content.source_ownership_epoch,content.source_claim_digest,content.source_grant_ref,content.source_grant_digest): raise ValueError("receipt admission authority/order mismatch")
            if admission.finality_verified and content.disposition is not ObservationDisposition.DEFINITELY_ABSENT: raise ValueError("only absence admits finality")
            expected_codes=((DiagnosticCode.STALE_RESULT,) if admission.freshness is ReceiptFreshnessV2.STALE else ())+((DiagnosticCode.FINALITY_UNPROVEN,) if content.disposition is ObservationDisposition.DEFINITELY_ABSENT and not admission.finality_verified else ())
            if admission.generated_invalid_codes!=expected_codes: raise ValueError("receipt admission generated-code matrix mismatch")
        if len({r.authenticated_receipt_digest for r in self.results})!=len(self.results): raise ValueError("duplicate receipt ledger entry")
        if type(self.terminal_content_digests) is not tuple or any(not isinstance(x,str) for x in self.terminal_content_digests): raise TypeError("terminal history must be tuple")
        if type(self.invalid_codes) is not tuple or any(type(x) is not DiagnosticCode for x in self.invalid_codes): raise TypeError("invalid-code history must be exact tuple")
        generated=tuple(code for admission in self.receipt_admissions for code in admission.generated_invalid_codes)
        if tuple(code for code in self.invalid_codes if code is not DiagnosticCode.EVIDENCE_INVALID)!=generated:raise ValueError("record invalid-code history differs from admissions")
        if type(self.invalid_evidence) is not tuple or type(self.invalid_evidence_admissions) is not tuple or len(self.invalid_evidence)!=len(self.invalid_evidence_admissions):raise ValueError("invalid evidence ledger must be exact one-to-one")
        if any(type(x) is not AuthenticatedInvalidEvidenceV2 for x in self.invalid_evidence) or any(type(x) is not InvalidEvidenceAdmissionV2 for x in self.invalid_evidence_admissions):raise TypeError("invalid evidence ledger types invalid")
        if len({x.authenticated_evidence_digest for x in self.invalid_evidence})!=len(self.invalid_evidence):raise ValueError("duplicate invalid evidence ledger entry")
        if sum(code is DiagnosticCode.EVIDENCE_INVALID for code in self.invalid_codes)!=len(self.invalid_evidence):raise ValueError("invalid evidence code/ledger cardinality mismatch")
        prior=None
        for index,(evidence,admission) in enumerate(zip(self.invalid_evidence,self.invalid_evidence_admissions,strict=True),1):
            content=evidence.content
            if (admission.authenticated_evidence_digest,admission.sequence,admission.prior_admission_digest)!=(evidence.authenticated_evidence_digest,index,prior):raise ValueError("invalid evidence admission chain mismatch")
            if (admission.site,admission.source_kind,admission.source_owner_ref,admission.source_generation,admission.source_ownership_epoch,admission.source_claim_digest,admission.source_grant_ref,admission.source_grant_digest)!=(content.site,content.source_kind,content.source_owner_ref,content.source_generation,content.source_ownership_epoch,content.source_claim_digest,content.source_grant_ref,content.source_grant_digest):raise ValueError("invalid evidence admission projection mismatch")
            prior=admission.admission_digest
        terminals=[];derived_state=EffectState.UNRESOLVED
        for receipt,admission in zip(self.results,self.receipt_admissions,strict=True):
            content=receipt.content;terminal=None
            if content.disposition is ObservationDisposition.FOUND:
                terminal=content.semantic_digest;candidate=EffectState.FOUND if admission.freshness is ReceiptFreshnessV2.FRESH else EffectState.INDETERMINATE
            elif content.disposition is ObservationDisposition.DEFINITELY_ABSENT and admission.finality_verified:
                terminal=content.semantic_digest;candidate=EffectState.DEFINITELY_ABSENT if admission.freshness is ReceiptFreshnessV2.FRESH else EffectState.INDETERMINATE
            else:candidate=EffectState.INDETERMINATE
            if terminal is not None and terminal not in terminals:
                if terminals and terminal!=terminals[-1]:derived_state=EffectState.CONTRADICTED
                terminals.append(terminal)
            if derived_state is EffectState.CONTRADICTED:continue
            if derived_state in {EffectState.FOUND,EffectState.DEFINITELY_ABSENT} and candidate is EffectState.INDETERMINATE:continue
            derived_state=candidate
        if not self.results and self.invalid_codes:derived_state=EffectState.INDETERMINATE
        if tuple(terminals)!=self.terminal_content_digests or derived_state is not self.state:raise ValueError("effect state/terminal history is not derived from admissions")
        if type(self.reconciliation_claims) is not tuple or any(type(x) is not ReconciliationOwnershipV2 for x in self.reconciliation_claims): raise TypeError("claim history must be exact tuple")
        if self.reconciliation!=(self.reconciliation_claims[-1] if self.reconciliation_claims else None): raise ValueError("current reconciliation must equal final history entry")
        if self.reconciliation_claims:
            first=self.reconciliation_claims[0]
            if (first.generation,first.ownership_epoch)!=(1,1):raise ValueError("reconciliation history lacks exact genesis")
            if any(claim.active for claim in self.reconciliation_claims[:-1]):raise ValueError("historical reconciliation claim remains active")
            if any(binding.activated_at_admission_index>len(self.receipt_admissions) for claim in self.reconciliation_claims for binding in claim.grant_lineage):raise ValueError("claim grant activation exceeds admission history")
            for old,new in zip(self.reconciliation_claims,self.reconciliation_claims[1:],strict=False):
                if new.claimed_at_epoch<old.claimed_at_epoch or new.grant_ref==old.grant_ref or new.grant_digest==old.grant_digest:raise ValueError("reconciliation authority history regressed")
                if old.completed:
                    if (new.owner_ref,new.generation,new.ownership_epoch,new.target_digest)!=(old.owner_ref,old.generation+1,old.ownership_epoch+1,old.target_digest) or len(new.grant_lineage)!=1:raise ValueError("reconciliation retry history invalid")
                elif old.active or (new.generation,new.ownership_epoch)!=(old.generation,old.ownership_epoch+1) or new.owner_ref==old.owner_ref or new.target_digest==old.target_digest or len(new.grant_lineage)!=1:raise ValueError("reconciliation transfer history invalid")
        for index,admission in enumerate(self.receipt_admissions):
            authority=(admission.source_owner_ref,admission.source_generation,admission.source_ownership_epoch,admission.source_claim_digest)
            if admission.source_kind=="dispatch":
                expected=(self.grant.content.grant_ref,1,self.dispatch_epoch,self.dispatch_source_digest)
                if authority!=expected:raise ValueError("dispatch admission authority mismatch")
            elif not any(authority==(claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest) for claim in self.reconciliation_claims):raise ValueError("reconciliation admission authority missing from history")
            expected=_expected_admission_authority(self,index)
            stored=(admission.expected_source_kind,admission.expected_source_owner_ref,admission.expected_source_generation,admission.expected_source_ownership_epoch,admission.expected_source_claim_digest,admission.expected_grant_ref,admission.expected_grant_digest)
            if stored!=expected:raise ValueError("admission expected authority differs from grant lineage")
            actual=(admission.source_kind,*authority)
            actual_grant=(admission.source_grant_ref,admission.source_grant_digest)
            if admission.source_kind=="dispatch":source_grant=(self.grant.content.grant_ref,_execution_grant_digest(self.grant))
            else:
                source_claim=next(claim for claim in self.reconciliation_claims if authority==(claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest))
                source_grants={(binding.grant_ref,binding.grant_digest) for binding in source_claim.grant_lineage if binding.activated_at_admission_index<=index}
                if actual_grant not in source_grants:raise ValueError("admission actual grant differs from source authority")
                source_grant=actual_grant
            if actual_grant!=source_grant:raise ValueError("admission actual grant differs from source authority")
            if (admission.freshness is ReceiptFreshnessV2.FRESH)!=(actual==expected[:5] and actual_grant==expected[5:]):raise ValueError("admission freshness differs from reconstructed authority")

    @property
    def command(self):
        return parse_exact_command(self.command_bytes)

    @property
    def record_digest(self):
        grant_doc=_grant_document(self.grant)
        claim_doc=lambda c:{"owner_ref":c.owner_ref,"generation":c.generation,"ownership_epoch":c.ownership_epoch,"claimed_at_epoch":c.claimed_at_epoch,"target_digest":c.target_digest,"grant_ref":c.grant_ref,"grant_digest":c.grant_digest,"grant_lineage":[x.to_dict() for x in c.grant_lineage],"active":c.active,"completed":c.completed,"claim_digest":c.claim_digest}
        return domain_digest("synaptic-effect-record/v2", canonical_bytes({
            "schema_version":"synaptic-effect-record/v2","command":self.command.to_dict(),"command_bytes_digest":domain_digest("synaptic-foundation-command-bytes/v1",self.command_bytes),"grant":grant_doc,"authenticated_grant_digest":domain_digest("synaptic-authenticated-grant/v3",canonical_bytes(grant_doc)),
            "dispatch": self.dispatch.value,
            "state": self.state.value,
            "attempt_count": self.attempt_count,
            "dispatch_epoch": self.dispatch_epoch,
            "results": [{"receipt":parse_canonical_object(r.canonical_bytes,name="authenticated receipt"),"authenticated_receipt_digest":r.authenticated_receipt_digest} for r in self.results],
            "receipt_admissions":[a.to_dict()|{"admission_digest":a.admission_digest} for a in self.receipt_admissions],
            "terminal_content_digests": list(self.terminal_content_digests),
            "invalid_codes":[code.value for code in self.invalid_codes],
            "invalid_evidence":[{"evidence":parse_canonical_object(e.canonical_bytes,name="invalid evidence"),"authenticated_evidence_digest":e.authenticated_evidence_digest} for e in self.invalid_evidence],
            "invalid_evidence_admissions":[a.to_dict()|{"admission_digest":a.admission_digest} for a in self.invalid_evidence_admissions],
            "reconciliation":None if self.reconciliation is None else claim_doc(self.reconciliation),"reconciliation_claims":[claim_doc(c) for c in self.reconciliation_claims],
        }))

    @property
    def dispatch_source_digest(self):
        return domain_digest("synaptic-dispatch-source/v2", canonical_bytes({
            "command_digest": self.command.digest,
            "effect_id": self.command.operation.effect.effect_id,
            "grant_ref": self.grant.content.grant_ref,
            "generation": 1, "ownership_epoch": self.dispatch_epoch,
        }))


def _revalidate_effect_record_v2_canonical(record,receipt_authority,invalid_evidence_authority,grant_authority):
    if type(record) is not EffectRecordV2:raise TypeError("exact effect record required")
    if type(record.grant) is not AuthenticatedGrantV2 or type(record.grant.content) is not GrantContentV2:raise TypeError("exact authenticated grant required")
    GrantContentV2.__post_init__(record.grant.content);AuthenticatedGrantV2.__post_init__(record.grant)
    if grant_authority.authenticate(record.grant,record.command_bytes) is not True:raise ValueError("execution grant authentication failed")
    for receipt in record.results:
        raw=receipt.canonical_bytes;owned=AuthenticatedReceiptV2.parse(raw)
        if owned!=receipt or owned.canonical_bytes!=raw:raise ValueError("receipt canonical reconstruction mismatch")
        if receipt_authority.verify(owned) is not True:raise ValueError("receipt authentication failed")
    for admission in record.receipt_admissions:ReceiptAdmissionV2.__post_init__(admission)
    for evidence in record.invalid_evidence:
        raw=evidence.canonical_bytes;owned=AuthenticatedInvalidEvidenceV2.parse(raw)
        if owned!=evidence or owned.canonical_bytes!=raw:raise ValueError("invalid evidence canonical reconstruction mismatch")
        if invalid_evidence_authority.verify(owned) is not True:raise ValueError("invalid evidence authentication failed")
    for admission in record.invalid_evidence_admissions:InvalidEvidenceAdmissionV2.__post_init__(admission)
    for claim in record.reconciliation_claims:
        for binding in claim.grant_lineage:ReconciliationGrantBindingV2.__post_init__(binding)
        ReconciliationOwnershipV2.__post_init__(claim)
    EffectRecordV2.__post_init__(record)
    return record


class InMemoryEffectRepositoryV2:
    def __init__(self, receipt_authority, invalid_evidence_authority, recovery_verifier, finality_verifier, grant_authority):
        self._receipt = receipt_authority
        self._invalid_evidence = invalid_evidence_authority
        self._recovery = recovery_verifier
        self._finality = finality_verifier
        self._grants = grant_authority
        self._lock = RLock()
        self._records = {}

    def _revalidate_stored_record(self,record):
        valid=False
        try:_revalidate_effect_record_v2_canonical(record,self._receipt,self._invalid_evidence,self._grants);valid=True
        except Exception:valid=False
        if not valid:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        return record

    def consume_attempt(self, command_bytes, grant, *, now_epoch):
        command = parse_exact_command(bytes(command_bytes))
        effect_id = command.operation.effect.effect_id
        if self._grants.verify(grant, command.canonical_bytes, now_epoch=now_epoch) is not True:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        with self._lock:
            old = self._records.get(effect_id)
            if old is not None:
                old=self._revalidate_stored_record(old)
                if old.command_bytes != command.canonical_bytes or old.grant != grant:
                    raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
                return old, False
            if type(command) is SubmitCommandV2:
                pred = command.stage_predecessor
                stage = self._records.get(pred.stage_effect_id)
                if stage is None:
                    raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
                stage=self._revalidate_stored_record(stage)
                if stage.state is not EffectState.FOUND:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
                stage_command = stage.command
                stage_prep = stage_command.preparation
                current_prep = command.preparation
                expected = (
                    stage_prep.provider.provider_id, stage_prep.provider.profile_ref,
                    stage_prep.scope.account_ref, stage_prep.scope.namespace_ref,
                    stage_prep.project_ref, stage_prep.run_id, stage_prep.plan_fingerprint,
                    stage_prep.preparation_digest, stage_prep.workload_digest,
                    stage_command.operation.effect.effect_id, stage.record_digest,
                )
                actual = (
                    pred.provider_id, pred.profile_ref, pred.account_ref, pred.namespace_ref,
                    pred.project_ref, pred.run_id, pred.plan_fingerprint,
                    pred.preparation_digest, pred.workload_digest,
                    pred.stage_effect_id, pred.record_digest,
                )
                current = (
                    current_prep.provider.provider_id, current_prep.provider.profile_ref,
                    current_prep.scope.account_ref, current_prep.scope.namespace_ref,
                    current_prep.project_ref, current_prep.run_id, current_prep.plan_fingerprint,
                    current_prep.preparation_digest, current_prep.workload_digest,
                )
                receipt_matches = any(
                    receipt.authenticated_receipt_digest == pred.authenticated_receipt_digest
                    for receipt in stage.results
                )
                if actual != expected or current != expected[:9] or not receipt_matches:
                    raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            record = EffectRecordV2(
                command.canonical_bytes, grant, DispatchState.OWNED_NOT_STARTED,
                EffectState.UNRESOLVED, 1,
            )
            self._revalidate_stored_record(record)
            self._records[effect_id] = record
            return record, True

    def _transition_dispatch(self, effect_id, expected, target):
        with self._lock:
            record = self._revalidate_stored_record(self._records[effect_id])
            if record.dispatch is not expected:
                raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
            record = replace(record, dispatch=target)
            self._records[effect_id] = record
            return record

    def begin_dispatch(self, effect_id):
        return self._transition_dispatch(effect_id, DispatchState.OWNED_NOT_STARTED, DispatchState.OWNED_IN_FLIGHT)

    def relinquish(self, effect_id):
        return self._transition_dispatch(effect_id, DispatchState.OWNED_IN_FLIGHT, DispatchState.RELINQUISHED)

    def orphan(self, effect_id):
        return self._transition_dispatch(effect_id, DispatchState.OWNED_IN_FLIGHT, DispatchState.ORPHANED_UNPROVEN)

    def prove_quiescence(self, effect_id, proof, *, now_epoch):
        with self._lock:
            record = self._revalidate_stored_record(self._records[effect_id])
            try:
                verified = self._recovery.verify_quiescence(proof, record, now_epoch=now_epoch)
            except Exception:
                verified = False
            if record.dispatch is not DispatchState.ORPHANED_UNPROVEN or type(verified) is not bool or verified is not True:
                raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            record = replace(record, dispatch=DispatchState.QUIESCENCE_PROVEN)
            self._records[effect_id] = record
            return record

    @staticmethod
    def _validate_receipt_semantics(record, content, admission_index):
        command = record.command
        prep = command.preparation
        if content.effect_id != command.operation.effect.effect_id or content.command_digest != command.digest:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        if content.disposition is ObservationDisposition.FOUND:
            scope = (
                prep.provider.provider_id, prep.provider.profile_ref,
                prep.scope.account_ref, prep.scope.namespace_ref,
            )
            if type(command) is StageCommandV2:
                ref = content.stage_ref
                if type(ref) is not ProviderStageRefV1 or (ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref) != scope:
                    raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)
            elif type(command) is SubmitCommandV2:
                ref = content.provider_run
                if type(ref) is not ScopedProviderRunRefV1 or (ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref) != scope:
                    raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)
            elif type(command) is CancelCommandV2:
                cancellation = command.to_dict()["cancellation"]
                expected = CancellationRefV1(
                    ProviderRunRefV1(command.operation.effect.cancel_target.provider_job_ref),
                    cancellation["reason_digest"],
                )
                if type(content.cancellation) is not CancellationRefV1 or content.cancellation != expected:
                    raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)
            else:
                raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)

        if content.source_kind == "dispatch":
            expected = (
                record.grant.content.grant_ref, 1, record.dispatch_epoch,
                record.dispatch_source_digest,record.grant.content.grant_ref,_execution_grant_digest(record.grant),
            )
            actual = (
                content.source_owner_ref, content.source_generation,
                content.source_ownership_epoch, content.source_claim_digest,
                content.source_grant_ref,content.source_grant_digest,
            )
            if actual != expected:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        else:
            matching = any(
                claim.owner_ref == content.source_owner_ref
                and claim.generation == content.source_generation
                and claim.ownership_epoch == content.source_ownership_epoch
                and claim.claim_digest == content.source_claim_digest
                and any(binding.grant_ref==content.source_grant_ref and binding.grant_digest==content.source_grant_digest and binding.activated_at_admission_index<=admission_index for binding in claim.grant_lineage)
                for claim in record.reconciliation_claims
            )
            if not matching:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)

    def _reduce_receipt_locked(self, record, receipt, finality_proof, *, now_epoch):
        try:
            owned_receipt = AuthenticatedReceiptV2.parse(receipt.canonical_bytes)
        except Exception:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        if self._receipt.verify(owned_receipt) is not True:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        existing_index=next((index for index,value in enumerate(record.results) if value.authenticated_receipt_digest==owned_receipt.authenticated_receipt_digest),None)
        admission_index=len(record.receipt_admissions) if existing_index is None else existing_index
        self._validate_receipt_semantics(record, owned_receipt.content, admission_index)
        if existing_index is not None:
            return record
        prior_terminal_content = record.terminal_content_digests
        expected_authority=_expected_admission_authority(record,len(record.receipt_admissions))
        actual_authority=(owned_receipt.content.source_kind,owned_receipt.content.source_owner_ref,owned_receipt.content.source_generation,owned_receipt.content.source_ownership_epoch,owned_receipt.content.source_claim_digest)
        actual_grant=(owned_receipt.content.source_grant_ref,owned_receipt.content.source_grant_digest)
        generated_invalid_codes = ()
        stale = actual_authority != expected_authority[:5] or actual_grant != expected_authority[5:]
        if stale:
            generated_invalid_codes += (DiagnosticCode.STALE_RESULT,)

        disposition = owned_receipt.content.disposition
        semantic_terminal = disposition is ObservationDisposition.FOUND
        if disposition is ObservationDisposition.DEFINITELY_ABSENT:
            try:
                verified = finality_proof is not None and self._finality.verify_finality(
                    finality_proof, record, owned_receipt, now_epoch=now_epoch,
                )
            except Exception:
                verified = False
            finality_verified = type(verified) is bool and verified is True
            semantic_terminal = finality_verified
            if finality_verified and not stale:
                candidate = EffectState.DEFINITELY_ABSENT
            else:
                candidate = EffectState.INDETERMINATE
                generated_invalid_codes += (DiagnosticCode.FINALITY_UNPROVEN,)
        elif disposition is ObservationDisposition.FOUND and not stale:
            candidate = EffectState.FOUND
        else:
            candidate = EffectState.INDETERMINATE

        terminal = {EffectState.FOUND, EffectState.DEFINITELY_ABSENT}
        conflicting_content = semantic_terminal and any(
            digest != owned_receipt.content.semantic_digest for digest in prior_terminal_content
        )
        terminal_content = prior_terminal_content
        if semantic_terminal and owned_receipt.content.semantic_digest not in terminal_content:
            terminal_content += (owned_receipt.content.semantic_digest,)
        if record.state is EffectState.CONTRADICTED or conflicting_content:
            state = EffectState.CONTRADICTED
        elif record.state in terminal and candidate is EffectState.INDETERMINATE:
            state = record.state
        else:
            state = candidate
        content=owned_receipt.content
        admission=ReceiptAdmissionV2(owned_receipt.authenticated_receipt_digest,content.source_kind,content.source_owner_ref,content.source_generation,content.source_ownership_epoch,content.source_claim_digest,content.source_grant_ref,content.source_grant_digest,*expected_authority,ReceiptFreshnessV2.STALE if stale else ReceiptFreshnessV2.FRESH,finality_verified if disposition is ObservationDisposition.DEFINITELY_ABSENT else False,generated_invalid_codes)
        return replace(record,results=record.results+(owned_receipt,),receipt_admissions=record.receipt_admissions+(admission,),state=state,invalid_codes=record.invalid_codes+generated_invalid_codes,terminal_content_digests=terminal_content)

    def append_result(self, effect_id, receipt, finality_proof, *, now_epoch):
        with self._lock:
            current=self._revalidate_stored_record(self._records[effect_id])
            replacement=self._reduce_receipt_locked(current,receipt,finality_proof,now_epoch=now_epoch)
            if replacement is not current:self._records[effect_id]=replacement
            return replacement

    def complete_dispatch(self, effect_id, receipt, finality_proof, *, now_epoch):
        with self._lock:
            record = self._revalidate_stored_record(self._records[effect_id])
            if record.dispatch is not DispatchState.OWNED_IN_FLIGHT:
                raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
            reduced = self._reduce_receipt_locked(record, receipt, finality_proof, now_epoch=now_epoch)
            replacement = replace(reduced, dispatch=DispatchState.RELINQUISHED)
            self._records[effect_id] = replacement
            return replacement

    def _append_invalid_locked(self, record, evidence, *, expected_sites):
        try:owned=AuthenticatedInvalidEvidenceV2.parse(evidence.canonical_bytes)
        except Exception:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID) from None
        if self._invalid_evidence.verify(owned) is not True:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        content=owned.content;command=record.command
        site_kind="dispatch" if content.site in {InvalidEvidenceSiteV2.DISPATCH_RESOLUTION,InvalidEvidenceSiteV2.DISPATCH_OBSERVATION} else "reconciliation"
        if content.site not in expected_sites or content.source_kind!=site_kind or (content.effect_id,content.command_digest)!=(command.operation.effect.effect_id,command.digest):raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        if content.source_kind=="dispatch":
            expected=(record.grant.content.grant_ref,1,record.dispatch_epoch,record.dispatch_source_digest,record.grant.content.grant_ref,_execution_grant_digest(record.grant))
        else:
            claim=record.reconciliation
            if claim is None:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            expected=(claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest,claim.grant_ref,claim.grant_digest)
        actual=(content.source_owner_ref,content.source_generation,content.source_ownership_epoch,content.source_claim_digest,content.source_grant_ref,content.source_grant_digest)
        if actual!=expected:raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        if any(x.authenticated_evidence_digest==owned.authenticated_evidence_digest for x in record.invalid_evidence):raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
        prior=None if not record.invalid_evidence_admissions else record.invalid_evidence_admissions[-1].admission_digest
        admission=InvalidEvidenceAdmissionV2(owned.authenticated_evidence_digest,len(record.invalid_evidence_admissions)+1,prior,content.site,content.source_kind,content.source_owner_ref,content.source_generation,content.source_ownership_epoch,content.source_claim_digest,content.source_grant_ref,content.source_grant_digest)
        state=EffectState.INDETERMINATE if record.state is EffectState.UNRESOLVED else record.state
        return replace(record,state=state,invalid_codes=record.invalid_codes+(DiagnosticCode.EVIDENCE_INVALID,),invalid_evidence=record.invalid_evidence+(owned,),invalid_evidence_admissions=record.invalid_evidence_admissions+(admission,))

    def complete_invalid_dispatch(self, effect_id, evidence):
        with self._lock:
            record = self._revalidate_stored_record(self._records[effect_id])
            if record.dispatch is not DispatchState.OWNED_IN_FLIGHT:
                raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
            reduced=self._append_invalid_locked(record,evidence,expected_sites={InvalidEvidenceSiteV2.DISPATCH_RESOLUTION,InvalidEvidenceSiteV2.DISPATCH_OBSERVATION})
            replacement=replace(reduced,dispatch=DispatchState.RELINQUISHED)
            self._records[effect_id]=replacement
            return replacement

    def interrupt_invalid_reconciliation(self,effect_id,claim,evidence):
        with self._lock:
            record=self._revalidate_stored_record(self._records[effect_id])
            if record.reconciliation!=claim or not claim.active:raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            reduced=self._append_invalid_locked(record,evidence,expected_sites={InvalidEvidenceSiteV2.RECONCILIATION_RESOLUTION,InvalidEvidenceSiteV2.RECONCILIATION_OBSERVATION})
            stopped=replace(claim,active=False);claims=tuple(stopped if value==claim else value for value in reduced.reconciliation_claims)
            replacement=replace(reduced,reconciliation=stopped,reconciliation_claims=claims)
            self._records[effect_id]=replacement
            return replacement

    @staticmethod
    def _reconciliation_target(content):
        return domain_digest("synaptic-reconciliation-target/v2", canonical_bytes({
            "command_digest": content.command_digest, "effect_id": content.effect_id,
            "preparation_digest": content.preparation_digest, "adapter_digest": content.adapter_digest,
            "provider_id": content.provider_id, "profile_ref": content.profile_ref,
            "account_ref": content.account_ref, "namespace_ref": content.namespace_ref,
            "owner_ref": content.owner_ref, "policy_digest": content.policy_digest,
            "requirement_digest": content.requirement_digest,
        }))

    def _verify_reconciliation_binding(self, command, grant, *, now_epoch):
        content = grant.content
        prep = command.preparation
        effect_id = command.operation.effect.effect_id
        if self._grants.verify_reconciliation(grant, now_epoch=now_epoch) is not True:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        actual = (
            content.command_digest, content.effect_id, content.preparation_digest,
            content.provider_id, content.profile_ref, content.account_ref, content.namespace_ref,
        )
        expected = (
            command.digest, effect_id, prep.preparation_digest, prep.provider.provider_id,
            prep.provider.profile_ref, prep.scope.account_ref, prep.scope.namespace_ref,
        )
        if actual != expected:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return effect_id, content, self._reconciliation_target(content)

    def acquire_reconciliation(self, command_bytes, grant, *, now_epoch, continuation=None):
        if type(command_bytes) is not bytes or type(grant) is not AuthenticatedReconciliationGrantV1:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        if type(grant.content) is not ReconciliationGrantContentV1:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        try:
            command = parse_exact_command(command_bytes)
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        verification_epoch=continuation.claimed_at_epoch if type(continuation) is ReconciliationOwnershipV2 and continuation.active else now_epoch
        effect_id, content, target_digest = self._verify_reconciliation_binding(command, grant, now_epoch=verification_epoch)
        grant_digest=_reconciliation_grant_digest(grant)
        with self._lock:
            record = self._records.get(effect_id)
            if record is None:
                raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
            record=self._revalidate_stored_record(record)
            if record.dispatch not in {DispatchState.RELINQUISHED, DispatchState.QUIESCENCE_PROVEN}:
                raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
            recovery_eligible = record.dispatch is DispatchState.QUIESCENCE_PROVEN and record.state is EffectState.UNRESOLVED
            if record.state is not EffectState.INDETERMINATE and not recovery_eligible:
                raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
            old = record.reconciliation
            if old:
                if old.active:
                    if continuation!=old or (old.owner_ref,old.generation,old.ownership_epoch,old.target_digest,old.grant_ref,old.grant_digest)!=(content.owner_ref,content.generation,content.ownership_epoch,target_digest,content.grant_ref,grant_digest):
                        raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
                    return record,old,True
                if old.completed:
                    if continuation is not None: raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
                    if record.state is not EffectState.INDETERMINATE:
                        raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
                    if (
                        content.owner_ref != old.owner_ref
                        or content.generation != old.generation + 1
                        or content.ownership_epoch != old.ownership_epoch + 1
                        or target_digest != old.target_digest
                        or content.grant_ref == old.grant_ref
                        or grant_digest == old.grant_digest
                    ):
                        raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
                    claim = ReconciliationOwnershipV2(
                        content.owner_ref, content.generation, content.ownership_epoch,
                        now_epoch, target_digest, content.grant_ref, grant_digest,(ReconciliationGrantBindingV2(content.grant_ref,grant_digest,len(record.receipt_admissions),None),),
                    )
                    record = replace(
                        record, reconciliation=claim,
                        reconciliation_claims=record.reconciliation_claims + (claim,),
                    )
                    self._records[effect_id] = record
                    return record, claim, True
                same_target = (
                    old.owner_ref == content.owner_ref and old.generation == content.generation
                    and old.ownership_epoch == content.ownership_epoch and old.target_digest == target_digest
                )
                if continuation != old or old.completed or old.active or not same_target or old.grant_ref == content.grant_ref or old.grant_digest == grant_digest:
                    raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
                resumed = replace(old, active=True, grant_ref=content.grant_ref,grant_digest=grant_digest,grant_lineage=old.grant_lineage+(ReconciliationGrantBindingV2(content.grant_ref,grant_digest,len(record.receipt_admissions),old.grant_lineage[-1].binding_digest),))
                claims = tuple(resumed if claim == old else claim for claim in record.reconciliation_claims)
                record = replace(record, reconciliation=resumed, reconciliation_claims=claims)
                self._records[effect_id] = record
                return record, resumed, True
            if content.generation != 1 or content.ownership_epoch != 1:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            if continuation is not None:raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            claim = ReconciliationOwnershipV2(
                content.owner_ref, content.generation, content.ownership_epoch,
                now_epoch, target_digest, content.grant_ref, grant_digest,(ReconciliationGrantBindingV2(content.grant_ref,grant_digest,len(record.receipt_admissions),None),),
            )
            record = replace(record, reconciliation=claim, reconciliation_claims=(claim,))
            self._records[effect_id] = record
            return record, claim, True

    def interrupt_reconciliation(self, effect_id, claim):
        with self._lock:
            record = self._revalidate_stored_record(self._records[effect_id])
            if record.reconciliation != claim or not claim.active:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            stopped = replace(claim, active=False)
            claims = tuple(stopped if value == claim else value for value in record.reconciliation_claims)
            record = replace(record, reconciliation=stopped, reconciliation_claims=claims)
            self._records[effect_id] = record
            return stopped

    def transfer_reconciliation(self, command_bytes, grant, *, proof, now_epoch):
        command = parse_exact_command(bytes(command_bytes))
        effect_id, content, target_digest = self._verify_reconciliation_binding(command, grant, now_epoch=now_epoch)
        grant_digest=_reconciliation_grant_digest(grant)
        with self._lock:
            record = self._revalidate_stored_record(self._records[effect_id])
            old = record.reconciliation
            if old is None or old.active or old.completed:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            if content.generation != old.generation or content.ownership_epoch != old.ownership_epoch + 1:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            if content.owner_ref == old.owner_ref or content.grant_ref == old.grant_ref or grant_digest==old.grant_digest:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            try:
                verified = self._recovery.verify_quiescence(proof, record, now_epoch=now_epoch)
            except Exception:
                verified = False
            if type(verified) is not bool or verified is not True:
                raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            claim = ReconciliationOwnershipV2(
                content.owner_ref, content.generation, content.ownership_epoch,
                now_epoch, target_digest, content.grant_ref, grant_digest,(ReconciliationGrantBindingV2(content.grant_ref,grant_digest,len(record.receipt_admissions),None),),
            )
            record = replace(
                record, reconciliation=claim,
                reconciliation_claims=record.reconciliation_claims + (claim,),
            )
            self._records[effect_id] = record
            return record, claim

    def complete_reconciliation(self, effect_id, claim, receipt, proof, *, now_epoch):
        with self._lock:
            record = self._revalidate_stored_record(self._records[effect_id])
            if type(claim) is not ReconciliationOwnershipV2:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            identity=(claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest)
            source_claim=next((value for value in record.reconciliation_claims if (value.owner_ref,value.generation,value.ownership_epoch,value.claim_digest)==identity),None)
            if source_claim is None:raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            try:owned_receipt=AuthenticatedReceiptV2.parse(receipt.canonical_bytes)
            except Exception:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID) from None
            if self._receipt.verify(owned_receipt) is not True:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            content=owned_receipt.content
            existing_index=next((index for index,value in enumerate(record.results) if value.authenticated_receipt_digest==owned_receipt.authenticated_receipt_digest),None)
            admission_index=len(record.receipt_admissions) if existing_index is None else existing_index
            matching_grant=any(binding.grant_ref==content.source_grant_ref and binding.grant_digest==content.source_grant_digest and binding.activated_at_admission_index<=admission_index for binding in source_claim.grant_lineage)
            if (content.source_kind,content.source_owner_ref,content.source_generation,content.source_ownership_epoch,content.source_claim_digest)!=("reconciliation",*identity) or not matching_grant:raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            duplicate=existing_index is not None
            if duplicate:
                if source_claim!=record.reconciliation:raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
                if source_claim.completed:return record
                if not source_claim.active:raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
                completed=replace(source_claim,active=False,completed=True)
                claims=tuple(completed if value==source_claim else value for value in record.reconciliation_claims)
                replacement=replace(record,reconciliation=completed,reconciliation_claims=claims)
                self._records[effect_id]=replacement
                return replacement
            reduced = self._reduce_receipt_locked(record, receipt, proof, now_epoch=now_epoch)
            if source_claim is record.reconciliation and source_claim.active:
                completed=replace(source_claim,active=False,completed=True)
                claims=tuple(completed if value is source_claim else value for value in reduced.reconciliation_claims)
                replacement=replace(reduced,reconciliation=completed,reconciliation_claims=claims)
            else:replacement=reduced
            self._records[effect_id] = replacement
            return replacement

    def get(self, effect_id):
        with self._lock:
            return self._records.get(effect_id)
