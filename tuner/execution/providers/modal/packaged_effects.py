"""Foundation effect and reconciliation adapters for packaged Modal runs."""

from __future__ import annotations

from typing import Protocol

from tuner.execution.foundation_v2.canonical import DiagnosticCode, FoundationError, safe_ref
from tuner.execution.foundation_v2.commands import (
    CancelCommandV2,
    StageCommandV2,
    SubmitCommandV2,
)
from tuner.execution.foundation_v2.executors import (
    AdapterDescriptorV1,
    ExecutionResolutionRequestV2,
    ExecutorDescriptorV1,
    ReconciliationResolutionRequestV2,
    mint_resolved_adapter,
    mint_resolved_executor,
)
from tuner.execution.foundation_v2.observations import (
    ObservationDisposition,
    ProviderObservationV1,
)
from tuner.execution.foundation_v2.reconciliation import ReconciliationTargetV1
from tuner.execution.foundation_v2.references import (
    CancellationRefV1,
    ProviderRunRefV1,
    ProviderStageRefV1,
    ScopedProviderRunRefV1,
)

from .coordinator_effects import ModalEffectOutcome
from .packaged_binding import (
    ModalPackagedBindingCatalog,
    ModalPackagedCommandBinding,
)


class ModalPackagedBindingAuthority(Protocol):
    def authenticate(self, binding: ModalPackagedCommandBinding) -> bool: ...


class ModalPackagedOperationalTransport(Protocol):
    def execute_once(self, binding: object, command: object) -> ModalEffectOutcome: ...
    def lookup_once(self, binding: object, command: object) -> ModalEffectOutcome: ...


def _binding(
    catalog: ModalPackagedBindingCatalog,
    authority: ModalPackagedBindingAuthority,
    digest: str,
) -> tuple[ModalPackagedCommandBinding, object]:
    try:
        value = catalog.resolve(digest)
        if type(value) is not ModalPackagedCommandBinding:
            raise TypeError
        rebuilt = value.reconstructed()
        if rebuilt != value or rebuilt.command_digest != digest \
                or authority.authenticate(rebuilt) is not True:
            raise ValueError
        command = rebuilt.command
        facts = rebuilt.provider_facts
        preparation = command.preparation
        if (
            preparation.provider.provider_id != "modal"
            or preparation.scope.account_ref != facts.account_ref
            or preparation.run_id != rebuilt.execution_binding.run_ref
            or preparation.source_digest != rebuilt.execution_binding.binding_digest
            or preparation.workload_digest != rebuilt.execution_binding.workload_digest
        ):
            raise ValueError
        return rebuilt, command
    except FoundationError:
        raise
    except Exception:
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None


def _runtime(adapter, command) -> None:
    preparation = command.preparation
    if (
        adapter.provider_id,
        adapter.profile_ref,
        adapter.account_ref,
        adapter.namespace_ref,
    ) != (
        preparation.provider.provider_id,
        preparation.provider.profile_ref,
        preparation.scope.account_ref,
        preparation.scope.namespace_ref,
    ):
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH)


def _request(command, request: ExecutionResolutionRequestV2) -> None:
    preparation, effect = command.preparation, command.operation.effect
    actual = (
        command.digest, command.executor.digest,
        preparation.provider.provider_id, preparation.provider.profile_ref,
        preparation.scope.account_ref, preparation.scope.namespace_ref,
        effect.kind.value, command.payload.payload_kind, command.payload.input_digest,
    )
    expected = tuple(getattr(request, name) for name in request.__dataclass_fields__)
    if actual != expected:
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH)


def _observation(command, resolution_digest, epoch, outcome):
    if type(outcome) is not ModalEffectOutcome:
        raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)
    preparation, effect = command.preparation, command.operation.effect
    values: dict[str, object] = {}
    if outcome.disposition is ObservationDisposition.FOUND:
        if type(command) is StageCommandV2:
            values["stage_ref"] = ProviderStageRefV1(
                preparation.provider.provider_id, preparation.provider.profile_ref,
                preparation.scope.account_ref, preparation.scope.namespace_ref,
                outcome.provider_ref,
            )
        elif type(command) is SubmitCommandV2:
            values["provider_run"] = ScopedProviderRunRefV1(
                preparation.provider.provider_id, preparation.provider.profile_ref,
                preparation.scope.account_ref, preparation.scope.namespace_ref,
                outcome.provider_ref,
            )
        elif type(command) is CancelCommandV2:
            cancellation = command.to_dict()["cancellation"]
            if cancellation["provider_job_ref"] != outcome.provider_ref:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            values["cancellation"] = CancellationRefV1(
                ProviderRunRefV1(outcome.provider_ref), cancellation["reason_digest"],
            )
        else:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    return ProviderObservationV1(
        effect.effect_id, command.digest, command.executor.digest,
        outcome.disposition, resolution_digest, epoch,
        finality_proof=outcome.finality_proof, **values,
    )


class ModalPackagedFoundationEffectExecutor:
    effect_kinds = ("stage", "submit", "cancel")
    payload_schemas = ("stage-payload/v2", "submit-payload/v2", "cancel-payload/v2")

    def __init__(
        self,
        *,
        profile_ref: str,
        account_ref: str,
        namespace_ref: str,
        catalog: ModalPackagedBindingCatalog,
        authority: ModalPackagedBindingAuthority,
        transport: ModalPackagedOperationalTransport,
        executor_id: str = "modal-packaged-executor",
        implementation_version: str = "0.1.0",
    ) -> None:
        self.descriptor = ExecutorDescriptorV1(
            "modal", safe_ref(executor_id, "executor_id"),
            safe_ref(implementation_version, "implementation_version"),
        )
        self.provider_id = "modal"
        self.profile_ref = safe_ref(profile_ref, "profile_ref")
        self.account_ref = safe_ref(account_ref, "account_ref")
        self.namespace_ref = safe_ref(namespace_ref, "namespace_ref")
        if not hasattr(catalog, "resolve") or not hasattr(authority, "authenticate") \
                or not hasattr(transport, "execute_once"):
            raise TypeError("complete packaged effect collaborators required")
        self._catalog, self._authority, self._transport = catalog, authority, transport

    def execute_once(self, payload, request: ExecutionResolutionRequestV2):
        if type(request) is not ExecutionResolutionRequestV2:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        binding, command = _binding(self._catalog, self._authority, request.command_digest)
        _runtime(self, command)
        if command.executor != self.descriptor:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        _request(command, request)
        if type(payload) is not type(command.payload) or payload != command.payload:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return _observation(
            command, request.digest, 1,
            self._transport.execute_once(binding, command),
        )


class ModalPackagedFoundationReconciliationAdapter:
    capabilities = ("lookup",)

    def __init__(
        self,
        *,
        profile_ref: str,
        account_ref: str,
        namespace_ref: str,
        catalog: ModalPackagedBindingCatalog,
        authority: ModalPackagedBindingAuthority,
        transport: ModalPackagedOperationalTransport,
        implementation_version: str = "0.1.0",
    ) -> None:
        self.descriptor = AdapterDescriptorV1(
            "modal", "modal-packaged-lookup",
            safe_ref(implementation_version, "implementation_version"),
        )
        self.provider_id = "modal"
        self.profile_ref = safe_ref(profile_ref, "profile_ref")
        self.account_ref = safe_ref(account_ref, "account_ref")
        self.namespace_ref = safe_ref(namespace_ref, "namespace_ref")
        if not hasattr(catalog, "resolve") or not hasattr(authority, "authenticate") \
                or not hasattr(transport, "lookup_once"):
            raise TypeError("complete packaged reconciliation collaborators required")
        self._catalog, self._authority, self._transport = catalog, authority, transport

    def lookup(self, target: ReconciliationTargetV1, preparation):
        if type(target) is not ReconciliationTargetV1:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        binding, command = _binding(self._catalog, self._authority, target.command_digest)
        _runtime(self, command)
        if target.command_bytes != command.canonical_bytes \
                or preparation != command.preparation \
                or target.effect_id != command.operation.effect.effect_id:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return _observation(
            command, target.resolution_digest, target.ownership_epoch,
            self._transport.lookup_once(binding, command),
        )


class ModalPackagedFoundationExecutorResolver:
    def __init__(self, executor: ModalPackagedFoundationEffectExecutor) -> None:
        if type(executor) is not ModalPackagedFoundationEffectExecutor:
            raise TypeError("exact packaged Foundation executor required")
        self._executor = executor

    def resolve(self, request: ExecutionResolutionRequestV2):
        return mint_resolved_executor(request, self._executor)


class ModalPackagedFoundationReconciliationResolver:
    def __init__(self, adapter: ModalPackagedFoundationReconciliationAdapter) -> None:
        if type(adapter) is not ModalPackagedFoundationReconciliationAdapter:
            raise TypeError("exact packaged reconciliation adapter required")
        self._adapter = adapter

    def resolve(self, request: ReconciliationResolutionRequestV2):
        return mint_resolved_adapter(request, self._adapter)


__all__ = [
    "ModalPackagedBindingAuthority",
    "ModalPackagedFoundationEffectExecutor",
    "ModalPackagedFoundationExecutorResolver",
    "ModalPackagedFoundationReconciliationAdapter",
    "ModalPackagedFoundationReconciliationResolver",
    "ModalPackagedOperationalTransport",
]
