"""Trusted resolution and one-owner reconciliation without lease transfer."""

from dataclasses import dataclass

from .broker import EffectBrokerV2
from .authority import AuthenticatedReconciliationGrantV1
from .canonical import DiagnosticCode, FoundationError, canonical_bytes, digest_text, domain_digest, exact_integer, safe_ref
from .commands import parse_exact_command
from .executors import AdapterDescriptorV1, ReconciliationResolutionRequestV2, ResolvedAdapterV2
from .observations import ProviderObservationV1
from .receipts import InvalidEvidenceContentV2, InvalidEvidenceSiteV2, ReceiptContentV2
from .repository import EffectRecordV2, ReconciliationOwnershipV2, _revalidate_effect_record_v2_canonical


def _closed_code(error):
    return error.code if type(error) is FoundationError else DiagnosticCode.AUTHORITY_INVALID


def _stabilize_reconciliation(repository,effect_id,claim):
    failure_code=None
    try:repository.interrupt_reconciliation(effect_id,claim)
    except Exception as error:failure_code=_closed_code(error)
    return failure_code


def _claim_identity(claim):
    return (claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest)


def _is_interrupted(record,claim,receipt_authority,invalid_evidence_authority,grant_authority):
    try:
        _revalidate_effect_record_v2_canonical(record,receipt_authority,invalid_evidence_authority,grant_authority)
        retained=record.reconciliation if type(record) is EffectRecordV2 else None
        return type(retained) is ReconciliationOwnershipV2 and _claim_identity(retained)==_claim_identity(claim) and not retained.active and not retained.completed
    except Exception:return False


def _is_reconciliation_completion(record,claim,receipt,receipt_authority,invalid_evidence_authority,grant_authority):
    try:
        _revalidate_effect_record_v2_canonical(record,receipt_authority,invalid_evidence_authority,grant_authority)
        retained=record.reconciliation;digest=receipt.authenticated_receipt_digest
        if type(retained) is not ReconciliationOwnershipV2 or _claim_identity(retained)!=_claim_identity(claim) or retained.active or not retained.completed:return False
        if record.reconciliation_claims==() or record.reconciliation_claims[-1]!=retained:return False
        return sum(value.authenticated_receipt_digest==digest for value in record.results)==1 and sum(value.receipt_digest==digest for value in record.receipt_admissions)==1
    except Exception:return False


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
    def __init__(self, repository, grant_authority, resolver, receipt_authority, invalid_evidence_authority):
        self._repo = repository
        self._grants = grant_authority
        self._resolver = resolver
        self._receipts = receipt_authority
        self._invalid_evidence = invalid_evidence_authority

    def _reconciliation_invalid(self,record,claim,grant,site,evidence_digest):
        content=InvalidEvidenceContentV2(record.command.operation.effect.effect_id,record.command.digest,site,"reconciliation",claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest,grant.content.grant_ref,grant.authenticated_grant_digest,evidence_digest)
        return self._invalid_evidence.issue(content)

    def _persist_reconciliation_invalid(self,effect_id,claim,evidence):
        failure_code=None
        try:
            result=self._repo.interrupt_invalid_reconciliation(effect_id,claim,evidence)
            reloaded=self._repo.get(effect_id)
            _revalidate_effect_record_v2_canonical(result,self._receipts,self._invalid_evidence,self._grants)
            _revalidate_effect_record_v2_canonical(reloaded,self._receipts,self._invalid_evidence,self._grants)
            digest=evidence.authenticated_evidence_digest
            if type(result) is not EffectRecordV2 or type(reloaded) is not EffectRecordV2 or result!=reloaded or result.record_digest!=reloaded.record_digest:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            envelope_count=sum(value.authenticated_evidence_digest==digest for value in result.invalid_evidence)
            admission_count=sum(value.authenticated_evidence_digest==digest for value in result.invalid_evidence_admissions)
            invalid_count=sum(value is DiagnosticCode.EVIDENCE_INVALID for value in result.invalid_codes)
            retained=result.reconciliation
            same_claim=retained is not None and (retained.owner_ref,retained.generation,retained.ownership_epoch,retained.claim_digest)==(claim.owner_ref,claim.generation,claim.ownership_epoch,claim.claim_digest)
            if envelope_count!=1 or admission_count!=1 or invalid_count!=len(result.invalid_evidence) or len(result.invalid_evidence)!=len(result.invalid_evidence_admissions) or not same_claim or retained.active or retained.completed:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            return result
        except Exception as error:
            failure_code=_closed_code(error)
        stabilization_code=_stabilize_reconciliation(self._repo,effect_id,claim)
        if stabilization_code is not None:raise FoundationError(stabilization_code)
        raise FoundationError(failure_code)

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

    def reconcile(self, command_bytes, grant, *, now_epoch, continuation=None):
        if type(command_bytes) is not bytes or type(grant) is not AuthenticatedReconciliationGrantV1:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        if continuation is not None and type(continuation) is not ReconciliationOwnershipV2:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        raw = bytes(command_bytes)
        try:
            command = parse_exact_command(raw)
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        content = grant.content
        prep = command.preparation
        effect = command.operation.effect
        verification_epoch=continuation.claimed_at_epoch if continuation is not None and continuation.active else now_epoch
        if self._grants.verify_reconciliation(grant, now_epoch=verification_epoch) is not True:
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
                raw, grant, now_epoch=now_epoch, continuation=continuation,
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
        resolution_invalid=False
        try:self._validate_resolved(resolved, request)
        except Exception:resolution_invalid=True
        if resolution_invalid:
            evidence_failure=False
            try:evidence=self._reconciliation_invalid(record,claim,grant,InvalidEvidenceSiteV2.RECONCILIATION_RESOLUTION,domain_digest("synaptic-invalid-resolution/v2",canonical_bytes({"resolution_request_digest":request.digest})))
            except Exception:evidence_failure=True
            if evidence_failure:
                stabilization_code=_stabilize_reconciliation(self._repo,effect.effect_id,claim)
                if stabilization_code is not None:raise FoundationError(stabilization_code)
                raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            self._persist_reconciliation_invalid(effect.effect_id,claim,evidence)
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID) from None
        provider_failed=False
        try:observation = resolved.adapter.lookup(target, type(prep).parse(prep.canonical_bytes))
        except Exception:provider_failed=True
        if provider_failed:
            stabilization_code=_stabilize_reconciliation(self._repo,effect.effect_id,claim)
            if stabilization_code is not None:raise FoundationError(stabilization_code)
            raise FoundationError(DiagnosticCode.RECONCILIATION_INTERRUPTED)
        observation_invalid=False
        try:EffectBrokerV2._validate_observation(command, request, observation, claim.ownership_epoch)
        except Exception:observation_invalid=True
        if observation_invalid:
            evidence_failure=False
            try:evidence=self._reconciliation_invalid(record,claim,grant,InvalidEvidenceSiteV2.RECONCILIATION_OBSERVATION,domain_digest("synaptic-invalid-observation/v2",canonical_bytes({"resolution_request_digest":request.digest})))
            except Exception:evidence_failure=True
            if evidence_failure:
                stabilization_code=_stabilize_reconciliation(self._repo,effect.effect_id,claim)
                if stabilization_code is not None:raise FoundationError(stabilization_code)
                raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            self._persist_reconciliation_invalid(effect.effect_id,claim,evidence)
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)
        receipt_failure=False
        try:
            receipt = self._receipts.issue(ReceiptContentV2.from_observation(
                observation, source_kind="reconciliation",
                source_owner_ref=claim.owner_ref,
                source_generation=claim.generation,
                source_ownership_epoch=claim.ownership_epoch,
                source_claim_digest=claim.claim_digest,
                source_grant_ref=grant.content.grant_ref,
                source_grant_digest=grant.authenticated_grant_digest,
            ))
        except Exception:receipt_failure=True
        if receipt_failure:
            stabilization_code=_stabilize_reconciliation(self._repo,effect.effect_id,claim)
            if stabilization_code is not None:raise FoundationError(stabilization_code)
            interrupted=None
            try:interrupted=self._repo.get(effect.effect_id)
            except Exception:interrupted=None
            if not _is_interrupted(interrupted,claim,self._receipts,self._invalid_evidence,self._grants):raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        completion_failure=None;completed=None;reloaded=None
        try:completed=self._repo.complete_reconciliation(effect.effect_id,claim,receipt,observation.finality_proof,now_epoch=now_epoch)
        except Exception as error:completion_failure=_closed_code(error)
        if completion_failure is None:
            try:reloaded=self._repo.get(effect.effect_id)
            except Exception as error:completion_failure=_closed_code(error)
        completion_valid=False
        if completion_failure is None:
            try:
                _revalidate_effect_record_v2_canonical(reloaded,self._receipts,self._invalid_evidence,self._grants)
                completion_valid=_is_reconciliation_completion(completed,claim,receipt,self._receipts,self._invalid_evidence,self._grants) and completed==reloaded and completed.record_digest==reloaded.record_digest
            except Exception:completion_valid=False
            if not completion_valid:completion_failure=DiagnosticCode.AUTHORITY_INVALID
        if completion_failure is None:return completed
        durable=reloaded
        if type(durable) is not EffectRecordV2:
            try:durable=self._repo.get(effect.effect_id)
            except Exception:durable=None
        if _is_reconciliation_completion(durable,claim,receipt,self._receipts,self._invalid_evidence,self._grants):raise FoundationError(completion_failure)
        retained=None;durable_active=False
        try:
            _revalidate_effect_record_v2_canonical(durable,self._receipts,self._invalid_evidence,self._grants);retained=durable.reconciliation
            durable_active=type(retained) is ReconciliationOwnershipV2 and retained==claim and retained.active
        except Exception:durable_active=False
        if durable_active or durable is None:
            stabilization_code=_stabilize_reconciliation(self._repo,effect.effect_id,claim)
            if stabilization_code is not None:
                final=None
                try:final=self._repo.get(effect.effect_id)
                except Exception:final=None
                if _is_reconciliation_completion(final,claim,receipt,self._receipts,self._invalid_evidence,self._grants):raise FoundationError(completion_failure)
                raise FoundationError(stabilization_code)
            final=None
            try:final=self._repo.get(effect.effect_id)
            except Exception:final=None
            if not _is_interrupted(final,claim,self._receipts,self._invalid_evidence,self._grants):raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        raise FoundationError(completion_failure)
