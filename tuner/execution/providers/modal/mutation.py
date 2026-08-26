"""Private exact Modal submission adapter used only behind MutationBroker."""
from __future__ import annotations

import hashlib

from ...broker import MutationCommandV1
from ...contracts import EffectDisposition, EffectObservation, safe_ref
from .facade import ExplicitModal154ReadFacade, ModalFacadeError
from .resolution import VerifiedModalDeploymentIdentityV1


class _ExplicitModal154FunctionMutator:
    """Perform one detached ``spawn`` against one immutable deployment."""

    __slots__ = ("_facade", "_deployment")

    def __init__(
        self,
        facade: ExplicitModal154ReadFacade,
        deployment: VerifiedModalDeploymentIdentityV1,
    ) -> None:
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact Modal 1.5.4 facade is required")
        if type(deployment) is not VerifiedModalDeploymentIdentityV1:
            raise TypeError("authenticated Modal deployment identity is required")
        selection = deployment.selection
        binding = facade.binding
        if (
            selection.account_ref != binding.account_ref
            or selection.workspace_ref != binding.workspace_ref
            or selection.environment_ref != binding.environment_ref
            or selection.client_ref != binding.client_ref
            or selection.sdk_version != binding.sdk_version
        ):
            raise ValueError("deployment does not bind the explicit Modal client")
        self._facade = facade
        self._deployment = deployment

    def execute_once(self, canonical_command: bytes) -> EffectObservation:
        command = MutationCommandV1.from_bytes(canonical_command)
        effect = command.effect
        binding = self._facade.binding
        if (
            effect.scope.provider != "modal"
            or effect.scope.account_ref != binding.account_ref
            or effect.scope.namespace_ref != binding.environment_ref
            or command.deployment_attestation_digest
            != self._deployment.attestation_digest
        ):
            raise ModalFacadeError("modal_mutation_binding_mismatch")
        selection = self._deployment.selection
        function = self._facade._function(
            app_name=selection.app_name,
            function_name=selection.function_name,
            function_version=selection.function_version,
        )
        try:
            call = function.spawn(canonical_command)
        except Exception:
            # Once spawn is entered, Modal may have accepted the invocation. The
            # same grant must reconcile; it may never silently resubmit.
            return EffectObservation(effect, EffectDisposition.INDETERMINATE)
        try:
            job_ref = safe_ref(call.object_id, "provider_job_ref")
        except Exception:
            return EffectObservation(effect, EffectDisposition.INDETERMINATE)
        receipt = hashlib.sha256(
            b"synaptic.modal-function-call/v1\0"
            + command.digest.encode("ascii") + b"\0" + job_ref.encode("ascii")
        ).hexdigest()
        return EffectObservation(effect, EffectDisposition.FOUND, job_ref, receipt)

    def lookup_handle(self, provider_job_ref: str):
        """Recreate a read/cancel handle without consulting ambient credentials."""
        provider_job_ref = safe_ref(provider_job_ref, "provider_job_ref")
        try:
            return self._facade.sdk.FunctionCall.from_id(
                provider_job_ref, client=self._facade.client
            )
        except Exception:
            raise ModalFacadeError("modal_function_call_unavailable") from None


__all__: list[str] = []
