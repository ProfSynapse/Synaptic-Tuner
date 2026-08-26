"""Independent command reconstruction, trusted resolution, and dispatch ownership."""

from .canonical import DiagnosticCode, FoundationError
from .commands import CancelCommandV2, StageCommandV2, SubmitCommandV2, parse_exact_command
from .executors import ExecutionResolutionRequestV2, ExecutorDescriptorV1, ResolvedExecutorV2
from .observations import ObservationDisposition, ProviderObservationV1
from .receipts import ReceiptContentV1
from .references import CancellationRefV1, ProviderRunRefV1, ProviderStageRefV1, ScopedProviderRunRefV1


class EffectBrokerV2:
    def __init__(self, repository, resolver, grant_authority, receipt_authority):
        self._repo = repository
        self._resolver = resolver
        self._grants = grant_authority
        self._receipts = receipt_authority

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
        if not self._grants.verify(grant, raw, now_epoch=now_epoch):
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
        try:
            self._validate_resolved(resolved, request)
        except Exception:
            self._repo.complete_invalid_dispatch(effect.effect_id)
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID) from None
        try:
            observation = resolved.executor.execute_once(owned_payload, request)
        except Exception:
            self._repo.orphan(effect.effect_id)
            raise FoundationError(DiagnosticCode.EFFECT_AMBIGUOUS) from None
        try:
            self._validate_observation(command, request, observation, record.dispatch_epoch)
            receipt = self._receipts.issue(ReceiptContentV1.from_observation(
                observation, source_kind="dispatch",
                source_owner_ref=grant.content.grant_ref, source_generation=1,
                source_ownership_epoch=record.dispatch_epoch,
                source_claim_digest=record.dispatch_source_digest,
            ))
            return self._repo.complete_dispatch(
                effect.effect_id, receipt, observation.finality_proof, now_epoch=now_epoch,
            )
        except Exception:
            try:
                self._repo.complete_invalid_dispatch(effect.effect_id)
            except Exception:
                pass
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID) from None
