"""Trusted resolution and one-owner reconciliation without lease transfer."""

from dataclasses import dataclass

from .broker import EffectBrokerV2
from .authority import AuthenticatedReconciliationGrantV1
from .canonical import DiagnosticCode, FoundationError, digest_text, exact_integer, safe_ref
from .commands import parse_exact_command
from .executors import AdapterDescriptorV1, ReconciliationResolutionRequestV2, ResolvedAdapterV2
from .observations import ProviderObservationV1
from .receipts import ReceiptContentV1
from .repository import EffectRecordV2, ReconciliationOwnershipV2


@dataclass(frozen=True, slots=True)
class ReconciliationTargetV1:
    command_bytes: bytes
    command_digest: str
    effect_id: str
    owner_ref: str
    generation: int
    ownership_epoch: int
    claimed_at_epoch: int
    resolution_digest: str

    def __post_init__(self):
        if type(self.command_bytes) is not bytes:
            raise TypeError("command bytes must be exact bytes")
        for name in ("command_digest", "resolution_digest"):
            digest_text(getattr(self, name), name)
        for name in ("effect_id", "owner_ref"):
            safe_ref(getattr(self, name), name)
        for name in ("generation", "ownership_epoch", "claimed_at_epoch"):
            exact_integer(getattr(self, name), name, minimum=1)


class ReconciliationServiceV1:
    def __init__(self, repository, grant_authority, resolver, receipt_authority):
        self._repo = repository
        self._grants = grant_authority
        self._resolver = resolver
        self._receipts = receipt_authority

    @staticmethod
    def _validate_resolved(resolved, request) -> None:
        if type(resolved) is not ResolvedAdapterV2:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        adapter = resolved.adapter
        descriptor = getattr(adapter, "descriptor", None)
        actual = (
            resolved.request_digest, resolved.descriptor_digest, resolved.provider_id,
            resolved.profile_ref, resolved.account_ref, resolved.namespace_ref,
            descriptor.digest if type(descriptor) is AdapterDescriptorV1 else None,
            getattr(adapter, "provider_id", None), getattr(adapter, "profile_ref", None),
            getattr(adapter, "account_ref", None), getattr(adapter, "namespace_ref", None),
            getattr(adapter, "capabilities", None),
        )
        expected = (
            request.digest, request.adapter_digest, request.provider_id,
            request.profile_ref, request.account_ref, request.namespace_ref,
            request.adapter_digest, request.provider_id, request.profile_ref,
            request.account_ref, request.namespace_ref, resolved.capabilities,
        )
        if actual != expected or "lookup" not in resolved.capabilities:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)

    def reconcile(self, command_bytes, grant, *, now_epoch, resume=None):
        if type(command_bytes) is not bytes or type(grant) is not AuthenticatedReconciliationGrantV1:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        if resume is not None and type(resume) is not ReconciliationOwnershipV2:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        raw = bytes(command_bytes)
        try:
            command = parse_exact_command(raw)
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        content = grant.content
        prep = command.preparation
        effect = command.operation.effect
        if not self._grants.verify_reconciliation(grant, now_epoch=now_epoch):
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        expected = (
            command.digest, effect.effect_id, prep.preparation_digest,
            prep.provider.provider_id, prep.provider.profile_ref,
            prep.scope.account_ref, prep.scope.namespace_ref,
        )
        actual = (
            content.command_digest, content.effect_id, content.preparation_digest,
            content.provider_id, content.profile_ref,
            content.account_ref, content.namespace_ref,
        )
        if actual != expected:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        request = ReconciliationResolutionRequestV2(
            command.digest, content.adapter_digest, prep.provider.provider_id,
            prep.provider.profile_ref, prep.scope.account_ref, prep.scope.namespace_ref,
        )
        try:
            resolved = self._resolver.resolve(request)
        except FoundationError:
            raise
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        self._validate_resolved(resolved, request)
        try:
            acquired = self._repo.acquire_reconciliation(
                raw, grant, now_epoch=now_epoch, resume=resume,
            )
        except FoundationError:
            raise
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        if type(acquired) is not tuple or len(acquired) != 3:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        record, claim, fresh = acquired
        if (
            type(record) is not EffectRecordV2
            or type(claim) is not ReconciliationOwnershipV2
            or type(fresh) is not bool
            or claim.owner_ref != content.owner_ref
            or claim.generation != content.generation
            or claim.ownership_epoch != content.ownership_epoch
        ):
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        if not fresh:
            return record
        try:
            target = ReconciliationTargetV1(
                raw, command.digest, effect.effect_id, content.owner_ref,
                content.generation, content.ownership_epoch, claim.claimed_at_epoch,
                request.digest,
            )
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        try:
            self._validate_resolved(resolved, request)
        except Exception:
            self._repo.invalid(effect.effect_id)
            self._repo.interrupt_reconciliation(effect.effect_id, claim)
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID) from None
        try:
            observation = resolved.adapter.lookup(target, type(prep).parse(prep.canonical_bytes))
        except Exception:
            self._repo.interrupt_reconciliation(effect.effect_id, claim)
            raise FoundationError(DiagnosticCode.RECONCILIATION_INTERRUPTED) from None
        try:
            EffectBrokerV2._validate_observation(command, request, observation, claim.ownership_epoch)
            receipt = self._receipts.issue(ReceiptContentV1.from_observation(
                observation, source_kind="reconciliation",
                source_owner_ref=claim.owner_ref,
                source_generation=claim.generation,
                source_ownership_epoch=claim.ownership_epoch,
                source_claim_digest=claim.claim_digest,
            ))
            completed = self._repo.complete_reconciliation(
                effect.effect_id, claim, receipt, observation.finality_proof, now_epoch=now_epoch,
            )
            if type(completed) is not EffectRecordV2:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            completed_command = completed.command
            if (
                completed_command.digest != command.digest
                or completed_command.operation.effect.effect_id != effect.effect_id
            ):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            return completed
        except FoundationError:
            try:
                self._repo.invalid(effect.effect_id)
                self._repo.interrupt_reconciliation(effect.effect_id, claim)
            except Exception:
                pass
            raise
        except Exception:
            self._repo.invalid(effect.effect_id)
            try:
                self._repo.interrupt_reconciliation(effect.effect_id, claim)
            except Exception:
                pass
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID) from None
