"""Atomic dispatch ownership, evidence ledger, and reconciliation ownership."""

from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock

from .canonical import DiagnosticCode, FoundationError, canonical_bytes, domain_digest
from .authority import AuthenticatedReconciliationGrantV1, ReconciliationGrantContentV1
from .commands import CancelCommandV2, StageCommandV2, SubmitCommandV2, parse_exact_command
from .observations import ObservationDisposition
from .receipts import AuthenticatedReceiptV1
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


@dataclass(frozen=True, slots=True)
class ReconciliationOwnershipV2:
    owner_ref: str
    generation: int
    ownership_epoch: int
    claimed_at_epoch: int
    target_digest: str
    grant_ref: str
    active: bool = True
    completed: bool = False

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
    terminal_content_digests: tuple[str, ...] = ()
    invalid_codes: tuple[DiagnosticCode, ...] = ()
    reconciliation: ReconciliationOwnershipV2 | None = None
    reconciliation_claims: tuple[ReconciliationOwnershipV2, ...] = ()

    @property
    def command(self):
        return parse_exact_command(self.command_bytes)

    @property
    def record_digest(self):
        return domain_digest("synaptic-effect-record/v2", canonical_bytes({
            "command_digest": self.command.digest,
            "dispatch": self.dispatch.value,
            "state": self.state.value,
            "attempt_count": self.attempt_count,
            "dispatch_epoch": self.dispatch_epoch,
            "result_digests": [r.authenticated_receipt_digest for r in self.results],
            "terminal_content_digests": list(self.terminal_content_digests),
        }))

    @property
    def dispatch_source_digest(self):
        return domain_digest("synaptic-dispatch-source/v2", canonical_bytes({
            "command_digest": self.command.digest,
            "effect_id": self.command.operation.effect.effect_id,
            "grant_ref": self.grant.content.grant_ref,
            "generation": 1, "ownership_epoch": self.dispatch_epoch,
        }))


class InMemoryEffectRepositoryV2:
    def __init__(self, receipt_authority, recovery_verifier, finality_verifier, grant_authority):
        self._receipt = receipt_authority
        self._recovery = recovery_verifier
        self._finality = finality_verifier
        self._grants = grant_authority
        self._lock = RLock()
        self._records = {}

    def consume_attempt(self, command_bytes, grant, *, now_epoch):
        command = parse_exact_command(bytes(command_bytes))
        effect_id = command.operation.effect.effect_id
        if not self._grants.verify(grant, command.canonical_bytes, now_epoch=now_epoch):
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        with self._lock:
            old = self._records.get(effect_id)
            if old:
                if old.command_bytes != command.canonical_bytes or old.grant != grant:
                    raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
                return old, False
            if type(command) is SubmitCommandV2:
                pred = command.stage_predecessor
                stage = self._records.get(pred.stage_effect_id)
                if stage is None or stage.state is not EffectState.FOUND:
                    raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
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
            self._records[effect_id] = record
            return record, True

    def _transition_dispatch(self, effect_id, expected, target):
        with self._lock:
            record = self._records[effect_id]
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
            record = self._records[effect_id]
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
    def _validate_receipt_semantics(record, content):
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
                record.dispatch_source_digest,
            )
            actual = (
                content.source_owner_ref, content.source_generation,
                content.source_ownership_epoch, content.source_claim_digest,
            )
            if actual != expected:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        else:
            matching = any(
                claim.owner_ref == content.source_owner_ref
                and claim.generation == content.source_generation
                and claim.ownership_epoch == content.source_ownership_epoch
                and claim.claim_digest == content.source_claim_digest
                for claim in record.reconciliation_claims
            )
            if not matching:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)

    def _append_result_locked(self, effect_id, receipt, finality_proof, *, now_epoch):
        record = self._records[effect_id]
        try:
            owned_receipt = AuthenticatedReceiptV1.parse(receipt.canonical_bytes)
        except Exception:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        if not self._receipt.verify(owned_receipt):
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        self._validate_receipt_semantics(record, owned_receipt.content)
        if any(x.authenticated_receipt_digest == owned_receipt.authenticated_receipt_digest for x in record.results):
            return record

        prior_results = record.results
        prior_terminal_content = record.terminal_content_digests
        record = replace(record, results=prior_results + (owned_receipt,))
        self._records[effect_id] = record  # append authenticated evidence before reduction/verifier
        expected_epoch = (
            record.reconciliation.ownership_epoch
            if record.reconciliation is not None and record.reconciliation.active
            else record.dispatch_epoch
        )
        codes = record.invalid_codes
        stale = owned_receipt.content.source_ownership_epoch != expected_epoch
        if stale:
            codes += (DiagnosticCode.STALE_RESULT,)

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
                codes += (DiagnosticCode.FINALITY_UNPROVEN,)
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
        record = replace(record, state=state, invalid_codes=codes, terminal_content_digests=terminal_content)
        self._records[effect_id] = record
        return record

    def append_result(self, effect_id, receipt, finality_proof, *, now_epoch):
        with self._lock:
            return self._append_result_locked(effect_id, receipt, finality_proof, now_epoch=now_epoch)

    def complete_dispatch(self, effect_id, receipt, finality_proof, *, now_epoch):
        with self._lock:
            record = self._records[effect_id]
            if record.dispatch is not DispatchState.OWNED_IN_FLIGHT:
                raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
            record = self._append_result_locked(effect_id, receipt, finality_proof, now_epoch=now_epoch)
            record = replace(record, dispatch=DispatchState.RELINQUISHED)
            self._records[effect_id] = record
            return record

    def complete_invalid_dispatch(self, effect_id, code=DiagnosticCode.EVIDENCE_INVALID):
        with self._lock:
            record = self._records[effect_id]
            if record.dispatch is not DispatchState.OWNED_IN_FLIGHT:
                raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
            record = replace(
                record, dispatch=DispatchState.RELINQUISHED, state=EffectState.INDETERMINATE,
                invalid_codes=record.invalid_codes + (code,),
            )
            self._records[effect_id] = record
            return record

    def invalid(self, effect_id):
        with self._lock:
            record = self._records[effect_id]
            record = replace(record, invalid_codes=record.invalid_codes + (DiagnosticCode.EVIDENCE_INVALID,))
            self._records[effect_id] = record
            return record

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
        if not self._grants.verify_reconciliation(grant, now_epoch=now_epoch):
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

    def acquire_reconciliation(self, command_bytes, grant, *, now_epoch, resume=None):
        if type(command_bytes) is not bytes or type(grant) is not AuthenticatedReconciliationGrantV1:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        if type(grant.content) is not ReconciliationGrantContentV1:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        try:
            command = parse_exact_command(command_bytes)
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        effect_id, content, target_digest = self._verify_reconciliation_binding(command, grant, now_epoch=now_epoch)
        with self._lock:
            record = self._records.get(effect_id)
            if record is None:
                raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
            if record.dispatch not in {DispatchState.RELINQUISHED, DispatchState.QUIESCENCE_PROVEN}:
                raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
            recovery_eligible = record.dispatch is DispatchState.QUIESCENCE_PROVEN and record.state is EffectState.UNRESOLVED
            if record.state is not EffectState.INDETERMINATE and not recovery_eligible:
                raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
            old = record.reconciliation
            if old:
                if old.completed:
                    if record.state is not EffectState.INDETERMINATE:
                        raise FoundationError(DiagnosticCode.EFFECT_INELIGIBLE)
                    if (
                        content.owner_ref != old.owner_ref
                        or content.generation != old.generation + 1
                        or content.ownership_epoch != old.ownership_epoch + 1
                        or target_digest != old.target_digest
                        or content.grant_ref == old.grant_ref
                    ):
                        raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
                    claim = ReconciliationOwnershipV2(
                        content.owner_ref, content.generation, content.ownership_epoch,
                        now_epoch, target_digest, content.grant_ref,
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
                if resume != old or old.completed or old.active or not same_target or old.grant_ref == content.grant_ref:
                    raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
                resumed = replace(old, active=True, grant_ref=content.grant_ref)
                claims = tuple(resumed if claim == old else claim for claim in record.reconciliation_claims)
                record = replace(record, reconciliation=resumed, reconciliation_claims=claims)
                self._records[effect_id] = record
                return record, resumed, True
            if content.generation != 1 or content.ownership_epoch != 1:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            claim = ReconciliationOwnershipV2(
                content.owner_ref, content.generation, content.ownership_epoch,
                now_epoch, target_digest, content.grant_ref,
            )
            record = replace(record, reconciliation=claim, reconciliation_claims=(claim,))
            self._records[effect_id] = record
            return record, claim, True

    def interrupt_reconciliation(self, effect_id, claim):
        with self._lock:
            record = self._records[effect_id]
            if record.reconciliation != claim:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            stopped = replace(claim, active=False)
            claims = tuple(stopped if value == claim else value for value in record.reconciliation_claims)
            record = replace(record, reconciliation=stopped, reconciliation_claims=claims)
            self._records[effect_id] = record
            return stopped

    def transfer_reconciliation(self, command_bytes, grant, *, proof, now_epoch):
        command = parse_exact_command(bytes(command_bytes))
        effect_id, content, target_digest = self._verify_reconciliation_binding(command, grant, now_epoch=now_epoch)
        with self._lock:
            record = self._records[effect_id]
            old = record.reconciliation
            if old is None or old.active or old.completed:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            if content.generation != old.generation or content.ownership_epoch != old.ownership_epoch + 1:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            if content.owner_ref == old.owner_ref or content.grant_ref == old.grant_ref:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            try:
                verified = self._recovery.verify_quiescence(proof, record, now_epoch=now_epoch)
            except Exception:
                verified = False
            if type(verified) is not bool or verified is not True:
                raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            claim = ReconciliationOwnershipV2(
                content.owner_ref, content.generation, content.ownership_epoch,
                now_epoch, target_digest, content.grant_ref,
            )
            record = replace(
                record, reconciliation=claim,
                reconciliation_claims=record.reconciliation_claims + (claim,),
            )
            self._records[effect_id] = record
            return record, claim

    def complete_reconciliation(self, effect_id, claim, receipt, proof, *, now_epoch):
        with self._lock:
            record = self._records[effect_id]
            if record.reconciliation != claim or not claim.active:
                raise FoundationError(DiagnosticCode.RECONCILIATION_CONFLICT)
            record = self._append_result_locked(effect_id, receipt, proof, now_epoch=now_epoch)
            completed = replace(claim, active=False, completed=True)
            claims = tuple(completed if value == claim else value for value in record.reconciliation_claims)
            record = replace(record, reconciliation=completed, reconciliation_claims=claims)
            self._records[effect_id] = record
            return record

    def get(self, effect_id):
        with self._lock:
            return self._records.get(effect_id)
