"""Independent command reconstruction, trusted resolution, and dispatch ownership."""

from .canonical import DiagnosticCode, FoundationError, canonical_bytes, domain_digest
from .commands import CancelCommandV2, StageCommandV2, SubmitCommandV2, parse_exact_command
from .executors import ExecutionResolutionRequestV2, ExecutorDescriptorV1, ResolvedExecutorV2
from .observations import ObservationDisposition, ProviderObservationV1
from .receipts import InvalidEvidenceContentV2, InvalidEvidenceSiteV2, ReceiptContentV2
from .references import CancellationRefV1, ProviderRunRefV1, ProviderStageRefV1, ScopedProviderRunRefV1
from .repository import DispatchState, EffectRecordV2, EffectState, _revalidate_effect_record_v2_canonical


def _closed_code(error):
    return error.code if type(error) is FoundationError else DiagnosticCode.AUTHORITY_INVALID


def _stabilize_dispatch(repository,effect_id,*,post_provider):
    failure_code=None
    try:
        (repository.orphan if post_provider else repository.relinquish)(effect_id)
    except Exception as error:
        failure_code=_closed_code(error)
    return failure_code


def _is_dispatch_completion(record,receipt,receipt_authority,invalid_evidence_authority,grant_authority):
    try:
        _revalidate_effect_record_v2_canonical(record,receipt_authority,invalid_evidence_authority,grant_authority)
        digest=receipt.authenticated_receipt_digest
        if record.dispatch is not DispatchState.RELINQUISHED or record.state not in {EffectState.INDETERMINATE,EffectState.FOUND,EffectState.DEFINITELY_ABSENT,EffectState.CONTRADICTED}:return False
        return sum(value.authenticated_receipt_digest==digest for value in record.results)==1 and sum(value.receipt_digest==digest for value in record.receipt_admissions)==1
    except Exception:return False


class EffectBrokerV2:
    def __init__(self, repository, resolver, grant_authority, receipt_authority, invalid_evidence_authority):
        self._repo = repository
        self._resolver = resolver
        self._grants = grant_authority
        self._receipts = receipt_authority
        self._invalid_evidence = invalid_evidence_authority

    def _dispatch_invalid(self, record, grant, site, evidence_digest):
        content=InvalidEvidenceContentV2(record.command.operation.effect.effect_id,record.command.digest,site,"dispatch",grant.content.grant_ref,1,record.dispatch_epoch,record.dispatch_source_digest,grant.content.grant_ref,grant.authenticated_grant_digest,evidence_digest)
        return self._invalid_evidence.issue(content)

    def _persist_dispatch_invalid(self,effect_id,evidence,*,post_provider):
        failure_code=None
        try:
            result=self._repo.complete_invalid_dispatch(effect_id,evidence)
            reloaded=self._repo.get(effect_id)
            _revalidate_effect_record_v2_canonical(result,self._receipts,self._invalid_evidence,self._grants)
            _revalidate_effect_record_v2_canonical(reloaded,self._receipts,self._invalid_evidence,self._grants)
            digest=evidence.authenticated_evidence_digest
            if type(result) is not EffectRecordV2 or type(reloaded) is not EffectRecordV2 or result!=reloaded or result.record_digest!=reloaded.record_digest:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            envelope_count=sum(value.authenticated_evidence_digest==digest for value in result.invalid_evidence)
            admission_count=sum(value.authenticated_evidence_digest==digest for value in result.invalid_evidence_admissions)
            invalid_count=sum(value is DiagnosticCode.EVIDENCE_INVALID for value in result.invalid_codes)
            if envelope_count!=1 or admission_count!=1 or invalid_count!=len(result.invalid_evidence) or len(result.invalid_evidence)!=len(result.invalid_evidence_admissions) or result.dispatch is not DispatchState.RELINQUISHED or result.state is not EffectState.INDETERMINATE:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            return result
        except Exception as error:
            failure_code=_closed_code(error)
        stabilization_code=_stabilize_dispatch(self._repo,effect_id,post_provider=post_provider)
        if stabilization_code is not None:raise FoundationError(stabilization_code)
        raise FoundationError(failure_code)

    @staticmethod
    def _validate_resolved(resolved, request) -> None:
        if type(resolved) is not ResolvedExecutorV2:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        executor = resolved.executor
        descriptor = getattr(executor, "descriptor", None)
        actual = (
            resolved.request_digest, resolved.descriptor_digest, resolved.provider_id,
            resolved.profile_ref, resolved.account_ref, resolved.namespace_ref,
            descriptor.digest if type(descriptor) is ExecutorDescriptorV1 else None,
            getattr(executor, "provider_id", None), getattr(executor, "profile_ref", None),
            getattr(executor, "account_ref", None), getattr(executor, "namespace_ref", None),
        )
        expected = (
            request.digest, request.descriptor_digest, request.provider_id,
            request.profile_ref, request.account_ref, request.namespace_ref,
            request.descriptor_digest, request.provider_id, request.profile_ref,
            request.account_ref, request.namespace_ref,
        )
        if actual != expected:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        if request.effect_kind not in resolved.effect_kinds or request.payload_schema not in resolved.payload_schemas:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        if getattr(executor, "effect_kinds", None) != resolved.effect_kinds or getattr(executor, "payload_schemas", None) != resolved.payload_schemas:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)

    @staticmethod
    def _validate_observation(command, request, observation, result_epoch) -> None:
        effect = command.operation.effect
        prep = command.preparation
        if type(observation) is not ProviderObservationV1:
            raise ValueError("observation type")
        if (
            observation.effect_id, observation.command_digest,
            observation.executor_descriptor_digest, observation.resolution_digest,
            observation.result_epoch,
        ) != (effect.effect_id, command.digest, command.executor.digest, request.digest, result_epoch):
            raise ValueError("observation binding")
        if observation.disposition is not ObservationDisposition.FOUND:
            return
        scope = (prep.provider.provider_id, prep.provider.profile_ref, prep.scope.account_ref, prep.scope.namespace_ref)
        if type(command) is StageCommandV2:
            ref = observation.stage_ref
            if type(ref) is not ProviderStageRefV1 or (ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref) != scope:
                raise ValueError("stage reference")
        elif type(command) is SubmitCommandV2:
            ref = observation.provider_run
            if type(ref) is not ScopedProviderRunRefV1 or (ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref) != scope:
                raise ValueError("run reference")
        elif type(command) is CancelCommandV2:
            cancellation = command.to_dict()["cancellation"]
            expected = CancellationRefV1(ProviderRunRefV1(effect.cancel_target.provider_job_ref), cancellation["reason_digest"])
            if type(observation.cancellation) is not CancellationRefV1 or observation.cancellation != expected:
                raise ValueError("cancellation reference")
        else:
            raise ValueError("command type")

    def execute(self, command_bytes, grant, *, now_epoch):
        raw = bytes(command_bytes)
        command = parse_exact_command(raw)
        if self._grants.verify(grant, raw, now_epoch=now_epoch) is not True:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        prep = command.preparation
        effect = command.operation.effect
        request = ExecutionResolutionRequestV2(
            command.digest, command.executor.digest, prep.provider.provider_id,
            prep.provider.profile_ref, prep.scope.account_ref, prep.scope.namespace_ref,
            effect.kind.value, command.payload.payload_kind, command.payload.input_digest,
        )
        try:
            resolved = self._resolver.resolve(request)
            self._validate_resolved(resolved, request)
        except FoundationError:
            raise
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        record, admitted = self._repo.consume_attempt(raw, grant, now_epoch=now_epoch)
        if not admitted:
            return record
        record = self._repo.begin_dispatch(effect.effect_id)
        owned_command = parse_exact_command(raw)
        owned_payload = type(owned_command.payload).parse(owned_command.payload.canonical_bytes)
        resolution_invalid=False
        try:self._validate_resolved(resolved, request)
        except Exception:resolution_invalid=True
        if resolution_invalid:
            evidence_failure=False
            try:evidence=self._dispatch_invalid(record,grant,InvalidEvidenceSiteV2.DISPATCH_RESOLUTION,domain_digest("synaptic-invalid-resolution/v2",canonical_bytes({"resolution_request_digest":request.digest})))
            except Exception:evidence_failure=True
            if evidence_failure:
                stabilization_code=_stabilize_dispatch(self._repo,effect.effect_id,post_provider=False)
                if stabilization_code is not None:raise FoundationError(stabilization_code)
                raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            self._persist_dispatch_invalid(effect.effect_id,evidence,post_provider=False)
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID) from None
        provider_failed=False
        try:observation = resolved.executor.execute_once(owned_payload, request)
        except Exception:provider_failed=True
        if provider_failed:
            stabilization_code=_stabilize_dispatch(self._repo,effect.effect_id,post_provider=True)
            if stabilization_code is not None:raise FoundationError(stabilization_code)
            raise FoundationError(DiagnosticCode.EFFECT_AMBIGUOUS)
        observation_invalid=False
        try:self._validate_observation(command, request, observation, record.dispatch_epoch)
        except Exception:observation_invalid=True
        if observation_invalid:
            evidence_failure=False
            try:evidence=self._dispatch_invalid(record,grant,InvalidEvidenceSiteV2.DISPATCH_OBSERVATION,domain_digest("synaptic-invalid-observation/v2",canonical_bytes({"resolution_request_digest":request.digest})))
            except Exception:evidence_failure=True
            if evidence_failure:
                stabilization_code=_stabilize_dispatch(self._repo,effect.effect_id,post_provider=True)
                if stabilization_code is not None:raise FoundationError(stabilization_code)
                raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
            self._persist_dispatch_invalid(effect.effect_id,evidence,post_provider=True)
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID) from None
        receipt_failure=False
        try:
            receipt_content=ReceiptContentV2.from_observation(
                observation, source_kind="dispatch",
                source_owner_ref=grant.content.grant_ref, source_generation=1,
                source_ownership_epoch=record.dispatch_epoch,
                source_claim_digest=record.dispatch_source_digest,
                source_grant_ref=grant.content.grant_ref,
                source_grant_digest=grant.authenticated_grant_digest,
            )
            receipt=self._receipts.issue(receipt_content)
        except Exception:
            receipt_failure=True
        if receipt_failure:
            stabilization_code=_stabilize_dispatch(self._repo,effect.effect_id,post_provider=True)
            if stabilization_code is not None:raise FoundationError(stabilization_code)
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        completion_failure=None;completed=None;reloaded=None
        try:
            completed=self._repo.complete_dispatch(
                effect.effect_id, receipt, observation.finality_proof, now_epoch=now_epoch,
            )
        except Exception as error:
            completion_failure=_closed_code(error)
        if completion_failure is None:
            try:reloaded=self._repo.get(effect.effect_id)
            except Exception as error:completion_failure=_closed_code(error)
        completion_valid=False
        if completion_failure is None:
            try:
                _revalidate_effect_record_v2_canonical(reloaded,self._receipts,self._invalid_evidence,self._grants)
                completion_valid=_is_dispatch_completion(completed,receipt,self._receipts,self._invalid_evidence,self._grants) and completed==reloaded and completed.record_digest==reloaded.record_digest
            except Exception:completion_valid=False
            if not completion_valid:completion_failure=DiagnosticCode.AUTHORITY_INVALID
        if completion_failure is None:return completed
        durable=reloaded
        if type(durable) is not EffectRecordV2:
            try:durable=self._repo.get(effect.effect_id)
            except Exception:durable=None
        if _is_dispatch_completion(durable,receipt,self._receipts,self._invalid_evidence,self._grants):raise FoundationError(completion_failure)
        durable_active=False
        try:_revalidate_effect_record_v2_canonical(durable,self._receipts,self._invalid_evidence,self._grants);durable_active=durable.dispatch is DispatchState.OWNED_IN_FLIGHT
        except Exception:durable_active=False
        if durable_active or durable is None:
            stabilization_code=_stabilize_dispatch(self._repo,effect.effect_id,post_provider=True)
            if stabilization_code is not None:
                final=None
                try:final=self._repo.get(effect.effect_id)
                except Exception:final=None
                if _is_dispatch_completion(final,receipt,self._receipts,self._invalid_evidence,self._grants):raise FoundationError(completion_failure)
                raise FoundationError(stabilization_code)
            final=None
            try:final=self._repo.get(effect.effect_id)
            except Exception:final=None
            try:_revalidate_effect_record_v2_canonical(final,self._receipts,self._invalid_evidence,self._grants);final_valid=final.dispatch is DispatchState.ORPHANED_UNPROVEN
            except Exception:final_valid=False
            if not final_valid:raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        raise FoundationError(completion_failure)
