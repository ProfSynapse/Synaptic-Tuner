"""Provider-free Modal planning and preparation for the generic coordinator.

This is an internal adapter, not an executable provider registration. It does
not load the SDK, create a client, authenticate a deployment, or authorize a
job. Its preflight deliberately refuses execution until the Foundation effect
and authenticated reader adapters are connected and independently verified.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping

from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1,
    ProviderPlanRef,
    ResolvedTrainingRequest,
    TrainingPlan,
    TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.providers import (
    ProviderCapabilities, ProviderDescriptor, ProviderRef,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.coordinator_v1.ports import CoordinatorClockPortV1
from tuner.execution.foundation_v2.canonical import (
    DiagnosticCode, FoundationError, canonical_bytes, digest_text, domain_digest,
    parse_canonical_object,
)
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
from tuner.execution.foundation_v2.executors import (
    AdapterDescriptorV1, ExecutorDescriptorV1,
)
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1

from .binding import ModalClientBinding
from .config import ModalProviderProfileV1, ModalRuntimeLockV1
from .deployment_v1 import ModalDeploymentSpecV1
from .resolution import ModalDeploymentSelectionV1


def _descriptor() -> ProviderDescriptor:
    # These describe this adapter, not the capabilities of the older Modal path.
    return ProviderDescriptor(
        "synaptic-provider-descriptor/v1", "modal", "Modal preparation", "0.1.0",
        ProviderCapabilities(False, False, False, False, False, False),
    )


class ModalPreparationAdapter:
    """Implement the coordinator's planning, binding, and preparation ports.

    One instance binds one resolved request and provider configuration. The
    consumer retains the configuration; only canonical digests and neutral
    references enter coordinator plans, preparations, and effect commands.
    Construction snapshots caller-owned values and reads the packaged runtime
    lock, but performs no provider I/O.
    """

    __slots__ = ("_configuration", "_basis", "_clock")

    def __init__(
        self, *, profile: ModalProviderProfileV1, binding: ModalClientBinding,
        resolved: ResolvedTrainingRequest, runtime_environment: Mapping[str, str],
        quote_digest: str, timeout_seconds: int, clock: CoordinatorClockPortV1,
    ) -> None:
        if type(profile) is not ModalProviderProfileV1:
            raise TypeError("exact Modal profile required")
        if type(binding) is not ModalClientBinding:
            raise TypeError("exact Modal binding required")
        if type(resolved) is not ResolvedTrainingRequest:
            raise TypeError("exact resolved request required")
        digest_text(quote_digest, "quote_digest")
        environment = dict(runtime_environment)
        # Reuse the declarative profile parser and secret/environment policy.
        document = {
            "schema_version": "synaptic-modal-provider/v1",
            "profile": profile.profile,
            "deployment": {
                "app_name": profile.app_name,
                "function_name": profile.function_name,
                "deployment_ref": profile.deployment_ref,
            },
            "runtime_lock": profile.runtime_lock_ref,
            "volumes": {
                "control_ref": profile.control_volume_ref,
                "artifact_ref": profile.artifact_volume_ref,
            },
            "secrets": [
                {"provider": "modal", "name": item.name,
                 "required_keys": list(item.required_keys)}
                for item in profile.secrets
            ],
        }
        profile = ModalProviderProfileV1.from_mapping(
            parse_canonical_object(canonical_bytes(document), name="profile")
        )
        lock = ModalRuntimeLockV1.packaged()
        for secret in profile.secrets:
            ModalDeploymentSpecV1(
                profile.deployment_ref, profile.function_name,
                lock.registry_reference, profile.control_volume_ref,
                profile.artifact_volume_ref, secret.name, secret.required_keys,
                environment, timeout_seconds,
            )
        selection = ModalDeploymentSelectionV1.from_profile(
            profile, binding=binding, runtime_environment=environment,
            timeout_seconds=timeout_seconds,
        )
        self._configuration = canonical_bytes({
            "profile": document, "selection": selection.to_dict(),
            "quote_digest": quote_digest,
        })
        self._basis = canonical_bytes(TrainingPlanBasisV1.from_resolved(resolved).to_dict())
        self._clock = clock

    def snapshot(self) -> bytes:
        """Non-secret configuration for consumer-owned retention; not authority."""
        return canonical_bytes({
            "schema_version": "synaptic-modal-preparation-snapshot/v1",
            "configuration": parse_canonical_object(self._configuration, name="configuration"),
            "basis": parse_canonical_object(self._basis, name="basis"),
        })

    @classmethod
    def restore(cls, snapshot: bytes, *, clock: CoordinatorClockPortV1):
        """Re-run construction policy instead of trusting retained digest fields."""
        document = parse_canonical_object(snapshot, name="Modal preparation snapshot")
        if (set(document) != {"schema_version", "configuration", "basis"}
                or document["schema_version"] != "synaptic-modal-preparation-snapshot/v1"):
            raise ValueError("invalid Modal preparation snapshot")
        configuration = document["configuration"]
        if type(configuration) is not dict or set(configuration) != {"profile", "selection", "quote_digest"}:
            raise ValueError("invalid retained Modal configuration")
        selection = ModalDeploymentSelectionV1.from_dict(configuration["selection"])
        basis = TrainingPlanBasisV1.from_dict(document["basis"]).to_dict()
        basis["schema_version"] = "synaptic-resolved-training-request/v1"
        restored = cls(
            profile=ModalProviderProfileV1.from_mapping(configuration["profile"]),
            binding=ModalClientBinding(
                selection.account_ref, selection.workspace_ref, selection.environment_ref,
                selection.client_ref, selection.sdk_version,
            ),
            resolved=ResolvedTrainingRequest.from_dict(basis),
            runtime_environment=selection.runtime_environment,
            quote_digest=configuration["quote_digest"],
            timeout_seconds=selection.timeout_seconds, clock=clock,
        )
        if restored.snapshot() != snapshot:
            raise ValueError("retained Modal configuration differs from reconstructed policy")
        return restored

    def _snapshot(self):
        document = parse_canonical_object(self._configuration, name="Modal configuration")
        selection = ModalDeploymentSelectionV1.from_dict(document["selection"])
        ModalRuntimeLockV1.packaged().validate_selection(selection)
        profile = ModalProviderProfileV1.from_mapping(document["profile"])
        basis = TrainingPlanBasisV1.from_dict(
            parse_canonical_object(self._basis, name="training basis")
        )
        provider = ProviderRef("modal", profile.profile)
        descriptor = _descriptor()
        profile_digest = domain_digest("synaptic-modal-coordinator-profile/v1", self._configuration)
        context = ProviderPlanContextV1(
            "synaptic-provider-plan-context/v1", provider, basis.basis_digest,
            descriptor.descriptor_digest, profile_digest,
        )
        # A workspace/environment tuple must not alias another tuple by joining
        # strings. Raw Modal namespace structure remains inside this adapter.
        namespace = domain_digest("synaptic-modal-namespace/v1", canonical_bytes({
            "workspace_ref": selection.workspace_ref,
            "environment_ref": selection.environment_ref,
        }))
        resource_digest = domain_digest("synaptic-modal-coordinator-resources/v1", canonical_bytes({
            "accelerator": selection.accelerator, "accelerator_count": 1,
            "timeout_seconds": selection.timeout_seconds, "max_retries": selection.max_retries,
        }))
        execution = ProviderExecutionBindingV1(
            provider, descriptor.descriptor_digest, profile_digest,
            ExecutionScopeV1(selection.account_ref, namespace),
            ExecutorDescriptorV1("modal", "modal-coordinator-executor", "0.1.0"),
            AdapterDescriptorV1("modal", "modal-coordinator-lookup", "0.1.0").digest,
            resource_digest, document["quote_digest"], profile.secret_requirements_digest,
        )
        plan = TrainingPlan(
            "synaptic-training-plan/v2", basis,
            ProviderPlanRef(context.provider_context_digest),
        )
        return context, execution, plan

    def describe(self, provider: ProviderRef) -> ProviderDescriptor:
        context, _, _ = self._snapshot()
        if type(provider) is not ProviderRef or provider != context.provider:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return _descriptor()

    def context(
        self, resolved: ResolvedTrainingRequest, provider: ProviderRef,
    ) -> ProviderPlanContextV1:
        context, _, plan = self._snapshot()
        if (type(resolved) is not ResolvedTrainingRequest
                or type(provider) is not ProviderRef or provider != context.provider
                or TrainingPlanBasisV1.from_resolved(resolved) != plan.basis):
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return context

    def resolve(
        self, provider: ProviderRef, context: ProviderPlanContextV1,
    ) -> ProviderExecutionBindingV1:
        expected, execution, _ = self._snapshot()
        if (type(provider) is not ProviderRef or type(context) is not ProviderPlanContextV1
                or provider != expected.provider or context != expected):
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return execution

    def preflight(self, plan: TrainingPlan) -> TrainingPreflight:
        _, _, expected = self._snapshot()
        if type(plan) is not TrainingPlan or plan != expected:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        checked_at = self._clock.now_iso()
        expiry = datetime.fromisoformat(checked_at.replace("Z", "+00:00")) + timedelta(seconds=1)
        return TrainingPreflight(
            plan.plan_fingerprint, False, checked_at, expiry.isoformat(),
            diagnostic_codes=("modal_coordinator_execution_unavailable",),
        )

    def prepare(
        self, plan: TrainingPlan, run: TrainingRunRef,
        binding: ProviderExecutionBindingV1,
    ) -> CanonicalPreparationV2:
        _, expected_binding, expected_plan = self._snapshot()
        if (type(plan) is not TrainingPlan or plan != expected_plan
                or type(run) is not TrainingRunRef or run.project_ref != plan.basis.project_ref
                or type(binding) is not ProviderExecutionBindingV1 or binding != expected_binding):
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return CanonicalPreparationV2.build(
            provider=binding.provider, scope=binding.scope,
            project_ref=run.project_ref, run_id=run.run_id,
            plan_fingerprint=plan.plan_fingerprint, source_digest=plan.basis.source_digest,
            workload_digest=plan.basis.workload_digest, runtime_digest=plan.basis.runtime_digest,
            resource_digest=binding.resource_digest,
            artifact_contract_digest=plan.basis.artifact_policy_digest,
            quote_digest=binding.quote_digest, secret_requirements_digest=binding.secret_requirements_digest,
            execution_binding_digest=binding.binding_digest,
        )

    def payload(
        self, preparation: CanonicalPreparationV2, kind: EffectKind,
    ) -> CanonicalProviderPayloadV1:
        if type(preparation) is not CanonicalPreparationV2 or type(kind) is not EffectKind:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        _, binding, plan = self._snapshot()
        sealed = CanonicalPreparationV2.parse(preparation.canonical_bytes)
        expected = self.prepare(
            plan, TrainingRunRef(sealed.run_id, sealed.project_ref), binding,
        )
        if sealed != expected:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return CanonicalProviderPayloadV1.build(
            binding.provider.provider_id, f"{kind.value}-payload/v2", sealed.workload_digest,
        )
