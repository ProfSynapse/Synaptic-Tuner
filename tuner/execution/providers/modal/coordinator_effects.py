"""Foundation effect and reconciliation adapters for Modal.

The adapters in this module do not grant mutation authority.  They consume an
authenticated, complete-command catalog and an operational transport supplied
by composition after provider authentication.  This keeps the Foundation
grant as the only mutation authority and prevents the legacy Modal lifecycle
from being nested below the coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tuner.execution.foundation_v2.canonical import (
    DiagnosticCode, FoundationError, canonical_bytes, domain_digest,
)
from tuner.execution.foundation_v2.commands import (
    CancelCommandV2,
    StageCommandV2,
    SubmitCommandV2,
    parse_exact_command,
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

from .binding import ModalClientBinding
from .coordinator_binding import ModalCommandBinding
from .resolution import VerifiedModalDeploymentIdentityV1


_KINDS = ("stage", "submit", "cancel")
_PAYLOADS = ("stage-payload/v2", "submit-payload/v2", "cancel-payload/v2")


class _AuthenticatedCommandCatalog(Protocol):
    """Consumer-owned authentication boundary for retained complete commands."""

    def resolve(self, command_digest: str) -> ModalCommandBinding: ...


class _CommandBindingAuthority(Protocol):
    def authenticate(self, binding: object) -> bool: ...


@dataclass(frozen=True, slots=True)
class ModalEffectOutcome:
    """Narrow result of one already-authorized provider operation or lookup."""

    disposition: ObservationDisposition
    provider_ref: str | None = None
    finality_proof: object | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not ObservationDisposition:
            raise TypeError("exact observation disposition required")
        if self.disposition is ObservationDisposition.FOUND:
            if type(self.provider_ref) is not str or not self.provider_ref:
                raise ValueError("found Modal outcome requires a provider reference")
            if self.finality_proof is not None:
                raise ValueError("found Modal outcome cannot carry absence proof")
        elif self.provider_ref is not None:
            raise ValueError("non-found Modal outcome cannot carry a provider reference")
        if (
            self.disposition is ObservationDisposition.DEFINITELY_ABSENT
            and self.finality_proof is None
        ):
            raise ValueError("definite absence requires finality proof")
        if (
            self.disposition is not ObservationDisposition.DEFINITELY_ABSENT
            and self.finality_proof is not None
        ):
            raise ValueError("only definite absence may carry finality proof")


class _ModalOperationalTransport(Protocol):
    """Provider-authenticated one-attempt transport; it owns no lifecycle state."""

    def execute_once(self, binding: object, command: object) -> ModalEffectOutcome: ...

    def lookup_once(self, binding: object, command: object) -> ModalEffectOutcome: ...


def _load_exact(
    catalog: _AuthenticatedCommandCatalog,
    authority: _CommandBindingAuthority,
    digest: str,
):
    try:
        retained = catalog.resolve(digest)
        if type(retained) is not ModalCommandBinding:
            raise TypeError
        # Reconstruct from the three immutable byte snapshots before asking the
        # authority to authenticate the complete canonical content.  Never use
        # derived properties from a caller-retained instance directly.
        binding = ModalCommandBinding(
            retained.command_bytes,
            retained.preparation_snapshot,
            retained.deployment_bytes,
        )
        if binding != retained or authority.authenticate(binding) is not True:
            raise ValueError
        raw = binding.command_bytes
        if type(raw) is not bytes:
            raise TypeError
        command = parse_exact_command(raw)
        if command.digest != digest:
            raise ValueError
        prep = command.preparation
        declared = (
            binding.command_digest,
            binding.provider_id,
            binding.profile_ref,
            binding.account_ref,
            binding.namespace_ref,
        )
        actual = (
            command.digest,
            prep.provider.provider_id,
            prep.provider.profile_ref,
            prep.scope.account_ref,
            prep.scope.namespace_ref,
        )
        if declared != actual:
            raise ValueError
        client = binding.client_binding
        deployment = binding.deployment
        if type(client) is not ModalClientBinding or type(deployment) is not VerifiedModalDeploymentIdentityV1:
            raise TypeError
        selection = deployment.selection
        namespace = domain_digest("synaptic-modal-namespace/v1", canonical_bytes({
            "workspace_ref": client.workspace_ref,
            "environment_ref": client.environment_ref,
        }))
        if (
            client.account_ref != prep.scope.account_ref
            or namespace != prep.scope.namespace_ref
            or selection.account_ref != client.account_ref
            or selection.workspace_ref != client.workspace_ref
            or selection.environment_ref != client.environment_ref
            or selection.client_ref != client.client_ref
            or selection.sdk_version != client.sdk_version
        ):
            raise ValueError
        return binding, command
    except FoundationError:
        raise
    except Exception:
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None


def _validate_request(command, request: ExecutionResolutionRequestV2) -> None:
    prep = command.preparation
    effect = command.operation.effect
    actual = (
        command.digest,
        command.executor.digest,
        prep.provider.provider_id,
        prep.provider.profile_ref,
        prep.scope.account_ref,
        prep.scope.namespace_ref,
        effect.kind.value,
        command.payload.payload_kind,
        command.payload.input_digest,
    )
    expected = tuple(getattr(request, name) for name in request.__dataclass_fields__)
    if actual != expected:
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH)


def _validate_runtime(adapter, command) -> None:
    prep = command.preparation
    if (
        adapter.provider_id,
        adapter.profile_ref,
        adapter.account_ref,
        adapter.namespace_ref,
    ) != (
        prep.provider.provider_id,
        prep.provider.profile_ref,
        prep.scope.account_ref,
        prep.scope.namespace_ref,
    ):
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH)


def _observation(command, resolution_digest: str, epoch: int, outcome: ModalEffectOutcome):
    prep = command.preparation
    effect = command.operation.effect
    values: dict[str, object] = {}
    if outcome.disposition is ObservationDisposition.FOUND:
        if type(command) is StageCommandV2:
            values["stage_ref"] = ProviderStageRefV1(
                prep.provider.provider_id,
                prep.provider.profile_ref,
                prep.scope.account_ref,
                prep.scope.namespace_ref,
                outcome.provider_ref,
            )
        elif type(command) is SubmitCommandV2:
            values["provider_run"] = ScopedProviderRunRefV1(
                prep.provider.provider_id,
                prep.provider.profile_ref,
                prep.scope.account_ref,
                prep.scope.namespace_ref,
                outcome.provider_ref,
            )
        elif type(command) is CancelCommandV2:
            cancellation = command.to_dict()["cancellation"]
            if outcome.provider_ref != cancellation["provider_job_ref"]:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            values["cancellation"] = CancellationRefV1(
                ProviderRunRefV1(outcome.provider_ref), cancellation["reason_digest"]
            )
        else:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    return ProviderObservationV1(
        effect.effect_id,
        command.digest,
        command.executor.digest,
        outcome.disposition,
        resolution_digest,
        epoch,
        finality_proof=outcome.finality_proof,
        **values,
    )


class ModalFoundationEffectExecutor:
    """Translate one Foundation dispatch into one operational transport call."""

    effect_kinds = _KINDS
    payload_schemas = _PAYLOADS

    def __init__(
        self, *, profile_ref: str, account_ref: str, namespace_ref: str,
        catalog: _AuthenticatedCommandCatalog, authority: _CommandBindingAuthority,
        transport: _ModalOperationalTransport,
    ) -> None:
        self.descriptor = ExecutorDescriptorV1(
            "modal", "modal-coordinator-executor", "0.1.0"
        )
        self.provider_id = "modal"
        self.profile_ref = profile_ref
        self.account_ref = account_ref
        self.namespace_ref = namespace_ref
        self._catalog = catalog
        self._authority = authority
        self._transport = transport

    def execute_once(self, payload, request: ExecutionResolutionRequestV2):
        if type(request) is not ExecutionResolutionRequestV2:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        binding, command = _load_exact(self._catalog, self._authority, request.command_digest)
        _validate_runtime(self, command)
        if command.executor != self.descriptor:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        _validate_request(command, request)
        if type(payload) is not type(command.payload) or payload != command.payload:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        outcome = self._transport.execute_once(binding, command)
        if type(outcome) is not ModalEffectOutcome:
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)
        return _observation(command, request.digest, 1, outcome)


class ModalFoundationReconciliationAdapter:
    """Conservative lookup of the exact retained Foundation command."""

    capabilities = ("lookup",)

    def __init__(
        self, *, profile_ref: str, account_ref: str, namespace_ref: str,
        catalog: _AuthenticatedCommandCatalog, authority: _CommandBindingAuthority,
        transport: _ModalOperationalTransport,
    ) -> None:
        self.descriptor = AdapterDescriptorV1(
            "modal", "modal-coordinator-lookup", "0.1.0"
        )
        self.provider_id = "modal"
        self.profile_ref = profile_ref
        self.account_ref = account_ref
        self.namespace_ref = namespace_ref
        self._catalog = catalog
        self._authority = authority
        self._transport = transport

    def lookup(self, target: ReconciliationTargetV1, preparation):
        if type(target) is not ReconciliationTargetV1:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        binding, command = _load_exact(
            self._catalog, self._authority, target.command_digest
        )
        _validate_runtime(self, command)
        if target.command_bytes != command.canonical_bytes:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        if preparation != command.preparation or target.effect_id != command.operation.effect.effect_id:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        outcome = self._transport.lookup_once(binding, command)
        if type(outcome) is not ModalEffectOutcome:
            raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)
        return _observation(command, target.resolution_digest, target.ownership_epoch, outcome)


class ModalFoundationExecutorResolver:
    def __init__(self, executor: ModalFoundationEffectExecutor) -> None:
        if type(executor) is not ModalFoundationEffectExecutor:
            raise TypeError("exact Modal Foundation executor required")
        self._executor = executor

    def resolve(self, request: ExecutionResolutionRequestV2):
        return mint_resolved_executor(request, self._executor)


class ModalFoundationReconciliationResolver:
    def __init__(self, adapter: ModalFoundationReconciliationAdapter) -> None:
        if type(adapter) is not ModalFoundationReconciliationAdapter:
            raise TypeError("exact Modal Foundation adapter required")
        self._adapter = adapter

    def resolve(self, request: ReconciliationResolutionRequestV2):
        return mint_resolved_adapter(request, self._adapter)


__all__: list[str] = []
