"""Explicit-client, one-attempt Modal transport for packaged training."""

from __future__ import annotations

from typing import Protocol

from tuner.execution.foundation_v2.canonical import safe_ref
from tuner.execution.foundation_v2.commands import (
    CancelCommandV2,
    StageCommandV2,
    SubmitCommandV2,
)
from tuner.execution.foundation_v2.observations import ObservationDisposition

from .coordinator_effects import ModalEffectOutcome
from .contracts import provider_entry_identity
from .facade import EXACT_MODAL_SDK_VERSION, ExplicitModal154ReadFacade
from .packaged_binding import ModalPackagedCommandBinding
from .packaged_deployment import ModalPackagedDeploymentObserver
from .packaged_dispatch import (
    ModalPackagedDispatchVerifier,
    parse_modal_packaged_dispatch,
)
from .packaged_staging import (
    ModalPackagedInputStager,
    ModalPackagedPreparedInputDescriptor,
    ModalPackagedStageMaterial,
    ModalPackagedStageReceipt,
)


class ModalPackagedStageSource(Protocol):
    def resolve(self, command_digest: str) -> ModalPackagedStageMaterial | None: ...


class ModalPackagedDispatchSource(Protocol):
    def resolve(self, command_digest: str) -> bytes | None: ...


class ModalPackagedStageReceiptCatalog(Protocol):
    def publish_if_absent(self, command_digest: str, receipt: ModalPackagedStageReceipt) -> bool: ...
    def resolve(self, command_digest: str) -> ModalPackagedStageReceipt | None: ...


class ModalPackagedCallCatalog(Protocol):
    def publish_if_absent(self, command_digest: str, provider_job_ref: str) -> bool: ...
    def resolve(self, command_digest: str) -> str | None: ...


class ModalPackagedHostTransport:
    """Perform only the exact effect selected by Foundation authority."""

    __slots__ = (
        "_sdk", "_client", "_facade", "_observer", "_stager",
        "_stage_source", "_dispatch_source", "_stage_receipts", "_calls",
        "_dispatch_verifier",
    )

    def __init__(
        self,
        *,
        sdk: object,
        client: object,
        facade: ExplicitModal154ReadFacade,
        deployment_observer: ModalPackagedDeploymentObserver,
        stager: ModalPackagedInputStager,
        stage_source: ModalPackagedStageSource,
        dispatch_source: ModalPackagedDispatchSource,
        stage_receipts: ModalPackagedStageReceiptCatalog,
        call_catalog: ModalPackagedCallCatalog,
        dispatch_verifier: ModalPackagedDispatchVerifier,
    ) -> None:
        if getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION or client is None:
            raise TypeError("exact SDK and explicit Modal client required")
        if type(facade) is not ExplicitModal154ReadFacade \
                or type(deployment_observer) is not ModalPackagedDeploymentObserver \
                or type(stager) is not ModalPackagedInputStager:
            raise TypeError("exact packaged Modal transport collaborators required")
        if facade.client is not client or deployment_observer.client is not client \
                or facade.binding != deployment_observer.client_binding:
            raise ValueError("packaged transport collaborators use different clients")
        collaborators = (
            (stage_source, "resolve"), (dispatch_source, "resolve"),
            (stage_receipts, "publish_if_absent"), (stage_receipts, "resolve"),
            (call_catalog, "publish_if_absent"), (call_catalog, "resolve"),
            (dispatch_verifier, "verify"),
        )
        if any(not hasattr(value, member) for value, member in collaborators):
            raise TypeError("complete packaged transport catalogs and verifier required")
        self._sdk, self._client, self._facade = sdk, client, facade
        self._observer, self._stager = deployment_observer, stager
        self._stage_source, self._dispatch_source = stage_source, dispatch_source
        self._stage_receipts, self._calls = stage_receipts, call_catalog
        self._dispatch_verifier = dispatch_verifier

    def _binding(self, value: object, command: object) -> ModalPackagedCommandBinding:
        if type(value) is not ModalPackagedCommandBinding:
            raise ValueError("exact packaged command binding required")
        rebuilt = value.reconstructed()
        if rebuilt != value or type(rebuilt.command) is not type(command) \
                or rebuilt.command.canonical_bytes != command.canonical_bytes:
            raise ValueError("packaged operational binding mismatch")
        self._facade.bound_scope()
        self._observer.observe(
            rebuilt.provider_facts,
            runtime_release=rebuilt.runtime_release,
            provider_binding=rebuilt.provider_binding,
        )
        return rebuilt

    @staticmethod
    def _stage_descriptor(
        binding: ModalPackagedCommandBinding,
        command: StageCommandV2,
    ) -> ModalPackagedPreparedInputDescriptor:
        return ModalPackagedPreparedInputDescriptor.create(
            binding.execution_binding,
            stage_effect_id=command.operation.effect.effect_id,
            artifact_volume_id=binding.provider_facts.artifact_volume_id,
        )

    @classmethod
    def _stage_receipt(
        cls,
        binding: ModalPackagedCommandBinding,
        command: StageCommandV2,
        receipt: object,
    ) -> ModalPackagedStageReceipt:
        expected = cls._stage_descriptor(binding, command)
        if type(receipt) is not ModalPackagedStageReceipt or (
            receipt.stage_effect_id,
            receipt.execution_binding_digest,
            receipt.artifact_volume_id,
            receipt.path,
            receipt.size_bytes,
            receipt.content_digest,
            receipt.provider_entry_id,
        ) != (
            command.operation.effect.effect_id,
            binding.execution_binding.binding_digest,
            binding.provider_facts.artifact_volume_id,
            expected.relative_path,
            expected.identity.size_bytes,
            expected.identity.content_digest,
            provider_entry_identity(
                binding.provider_facts.artifact_volume_id,
                expected.relative_path,
                expected.identity.size_bytes,
            ),
        ):
            raise ValueError("retained packaged stage receipt mismatch")
        return receipt

    def execute_once(self, binding: object, command: object) -> ModalEffectOutcome:
        rebuilt = self._binding(binding, command)
        if type(command) is StageCommandV2:
            material = self._stage_source.resolve(command.digest)
            expected_descriptor = self._stage_descriptor(rebuilt, command)
            if type(material) is not ModalPackagedStageMaterial \
                    or material.execution_binding_digest != rebuilt.execution_binding.binding_digest \
                    or material.descriptor != expected_descriptor:
                raise ValueError("retained packaged stage material mismatch")
            try:
                receipt = self._stage_receipt(
                    rebuilt, command, self._stager.stage_once(material),
                )
                self._stage_receipts.publish_if_absent(command.digest, receipt)
                retained = self._stage_receipts.resolve(command.digest)
                if retained != receipt:
                    raise ValueError
            except Exception:
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            return ModalEffectOutcome(ObservationDisposition.FOUND, receipt.provider_ref)
        if type(command) is SubmitCommandV2:
            payload = self._dispatch_source.resolve(command.digest)
            if type(payload) is not bytes:
                raise ValueError("retained packaged dispatch is unavailable")
            dispatch = parse_modal_packaged_dispatch(payload, self._dispatch_verifier)
            if dispatch.submit_command_bytes != command.canonical_bytes \
                    or dispatch.execution_binding != rebuilt.execution_binding \
                    or dispatch.provider_facts != rebuilt.provider_facts:
                raise ValueError("retained packaged dispatch mismatch")
            facts = rebuilt.provider_facts
            function = self._facade._function(
                app_name=facts.app_name, function_name=facts.function_name,
            )
            try:
                call = function.spawn(payload)
                provider_ref = safe_ref(
                    getattr(call, "object_id", None), "provider_job_ref",
                )
                self._calls.publish_if_absent(command.digest, provider_ref)
                if self._calls.resolve(command.digest) != provider_ref:
                    raise ValueError
            except Exception:
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            return ModalEffectOutcome(ObservationDisposition.FOUND, provider_ref)
        if type(command) is CancelCommandV2:
            target = safe_ref(
                command.to_dict()["cancellation"]["provider_job_ref"],
                "provider_job_ref",
            )
            try:
                call = self._sdk.FunctionCall.from_id(target, client=self._client)
                call.hydrate(self._client)
                if getattr(call, "is_hydrated", False) is not True \
                        or getattr(call, "object_id", None) != target:
                    raise ValueError
                call.cancel(terminate_containers=True)
            except Exception:
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            return ModalEffectOutcome(ObservationDisposition.FOUND, target)
        raise ValueError("unsupported packaged Modal effect")

    def lookup_once(self, binding: object, command: object) -> ModalEffectOutcome:
        rebuilt = self._binding(binding, command)
        if type(command) is StageCommandV2:
            receipt = self._stage_receipts.resolve(command.digest)
            if receipt is None:
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            receipt = self._stage_receipt(rebuilt, command, receipt)
            return ModalEffectOutcome(ObservationDisposition.FOUND, receipt.provider_ref)
        if type(command) is SubmitCommandV2:
            provider_ref = self._calls.resolve(command.digest)
            if provider_ref is None:
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            return ModalEffectOutcome(
                ObservationDisposition.FOUND,
                safe_ref(provider_ref, "provider_job_ref"),
            )
        if type(command) is CancelCommandV2:
            # Cancellation completion cannot be reconstructed from a missing
            # provider acknowledgement without inventing finality.
            return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
        raise ValueError("unsupported packaged Modal lookup")


__all__ = [
    "ModalPackagedCallCatalog",
    "ModalPackagedDispatchSource",
    "ModalPackagedHostTransport",
    "ModalPackagedStageReceiptCatalog",
    "ModalPackagedStageSource",
]
