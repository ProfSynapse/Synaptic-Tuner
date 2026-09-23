"""Single-command host bridge from prepared public input to packaged coordinator material."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ProviderPlanRef, ResolvedTrainingRequest, TrainingPlan,
    TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import (
    AuthorizationRequirement, TrainingPreflight, TrainingRequest,
)
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.project.context import ProjectContext
from tuner.training import default_recipe_registry
from tuner.training.contracts import CanonicalDocument, ResolvedTrainingComponents
from tuner.training.packaged_boundary import (
    create_execution_training_service, derive_packaged_coordinator_material,
    parse_packaged_coordinator_material,
)


class _FixedPackagedResolver:
    def __init__(self, components: ResolvedTrainingComponents) -> None:
        self.components = components

    def resolve(self, request, *, context):
        return self.components


class ModalPackagedRequestsV1:
    """Preserve the exact prepared request and one allocated run in this process."""

    def __init__(self, *, prepared_request: TrainingRequest, run: TrainingRunRef,
                 components: ResolvedTrainingComponents, context: ProjectContext) -> None:
        if type(prepared_request) is not TrainingRequest or type(run) is not TrainingRunRef:
            raise TypeError("exact prepared request and run required")
        if type(components) is not ResolvedTrainingComponents or type(context) is not ProjectContext:
            raise TypeError("exact packaged components and project context required")
        if run.project_ref != prepared_request.project_ref:
            raise ValueError("run differs from prepared project")
        self._request = prepared_request
        self._run = run
        self._recipes = default_recipe_registry()
        self._service = create_execution_training_service(
            context=context, resolver=_FixedPackagedResolver(components),
            recipes=self._recipes,
        )
        self._material = None

    def load(self, canonical_json: str) -> TrainingRequest:
        if type(canonical_json) is not str or canonical_json != self._request.canonical_json:
            raise ValueError("submitted request differs from prepared input")
        if TrainingInputV1.from_json(canonical_json).canonical_json() != canonical_json:
            raise ValueError("submitted training input is not canonical")
        return self._request

    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest:
        if type(request) is not TrainingRequest or request != self._request:
            raise ValueError("request differs from prepared input")
        rich = self._service.resolve_selected(
            self._service.load(CanonicalDocument(request.canonical_json))
        )
        material = derive_packaged_coordinator_material(
            rich, self._recipes, request_id=request.request_id,
            project_ref=request.project_ref, run_id=self._run.run_id,
        )
        self._material = parse_packaged_coordinator_material(
            material.canonical_bytes, self._recipes,
        )
        return self._material.planning_request

    def for_plan(self, plan: TrainingPlan) -> TrainingRunRef:
        if type(plan) is not TrainingPlan or self._material is None:
            raise ValueError("plan has no retained packaged material")
        if (plan.basis != TrainingPlanBasisV1.from_resolved(self._material.planning_request)
                or plan.basis.request_id != self._request.request_id):
            raise ValueError("plan differs from retained packaged request")
        return self._run


class ModalPackagedPlanPortsV1:
    """Provider-neutral coordinator planning over one authenticated binding."""

    def __init__(self, *, provider: ProviderRef, descriptor: ProviderDescriptor,
                 binding: ProviderExecutionBindingV1, clock: object,
                 maximum_cost_minor_units: int, currency: str = "USD") -> None:
        if (type(provider) is not ProviderRef or type(descriptor) is not ProviderDescriptor
                or type(binding) is not ProviderExecutionBindingV1):
            raise TypeError("exact Modal plan inputs required")
        if provider.provider_id != "modal" or descriptor.provider_id != "modal":
            raise ValueError("Modal plan inputs differ from provider")
        if binding.provider != provider or binding.provider_descriptor_digest != descriptor.descriptor_digest:
            raise ValueError("Modal execution binding differs from provider")
        if type(maximum_cost_minor_units) is not int or maximum_cost_minor_units < 1:
            raise ValueError("bounded training cost is required")
        self._provider, self._descriptor, self._binding = provider, descriptor, binding
        self._clock, self._cost, self._currency = clock, maximum_cost_minor_units, currency
        self._expected_plan = None
        self._expected_context = None

    def describe(self, provider: ProviderRef) -> ProviderDescriptor:
        if provider != self._provider:
            raise ValueError("provider differs from Modal planning")
        return self._descriptor

    def context(self, resolved: ResolvedTrainingRequest,
                provider: ProviderRef) -> ProviderPlanContextV1:
        if type(resolved) is not ResolvedTrainingRequest or provider != self._provider:
            raise ValueError("resolved request differs from Modal planning")
        basis = TrainingPlanBasisV1.from_resolved(resolved)
        context = ProviderPlanContextV1(
            "synaptic-provider-plan-context/v1", provider,
            basis.basis_digest,
            self._descriptor.descriptor_digest, self._binding.profile_digest,
        )
        expected = TrainingPlan(
            "synaptic-training-plan/v2", basis,
            ProviderPlanRef(context.provider_context_digest),
        )
        if self._expected_plan is not None and self._expected_plan != expected:
            raise ValueError("Modal planning changed after its first resolution")
        self._expected_plan = expected
        self._expected_context = context
        return context

    def resolve(self, provider: ProviderRef,
                context: ProviderPlanContextV1) -> ProviderExecutionBindingV1:
        if (provider != self._provider or type(context) is not ProviderPlanContextV1
                or context != self._expected_context):
            raise ValueError("Modal plan context differs from execution binding")
        return self._binding

    def preflight(self, plan: TrainingPlan) -> TrainingPreflight:
        if type(plan) is not TrainingPlan or plan != self._expected_plan:
            raise ValueError("exact Modal plan required")
        now = self._clock.now()
        expiry = (datetime.fromisoformat(now.replace("Z", "+00:00"))
                  + timedelta(minutes=15)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return TrainingPreflight(
            plan.plan_fingerprint, True, now, expiry,
            (AuthorizationRequirement("training.start", True, self._cost, self._currency),),
        )

    def prepare(self, plan: TrainingPlan, run: TrainingRunRef,
                binding: ProviderExecutionBindingV1) -> CanonicalPreparationV2:
        if (type(plan) is not TrainingPlan or plan != self._expected_plan
                or type(run) is not TrainingRunRef
                or binding != self._binding or run.project_ref != plan.basis.project_ref):
            raise ValueError("Modal preparation differs from retained plan")
        return CanonicalPreparationV2.build(
            provider=binding.provider, scope=binding.scope,
            project_ref=run.project_ref, run_id=run.run_id,
            plan_fingerprint=plan.plan_fingerprint,
            source_digest=plan.basis.source_digest,
            workload_digest=plan.basis.workload_digest,
            runtime_digest=plan.basis.runtime_digest,
            resource_digest=binding.resource_digest,
            artifact_contract_digest=plan.basis.artifact_policy_digest,
            quote_digest=binding.quote_digest,
            secret_requirements_digest=binding.secret_requirements_digest,
            execution_binding_digest=binding.binding_digest,
        )

    def payload(self, preparation: CanonicalPreparationV2, kind):
        if type(preparation) is not CanonicalPreparationV2:
            raise TypeError("exact Modal preparation required")
        return CanonicalProviderPayloadV1.build(
            "modal", f"{kind.value}-payload/v2", preparation.workload_digest,
        )
