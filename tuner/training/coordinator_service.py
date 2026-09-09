"""Concrete provider-neutral training facade over the durable coordinator."""
from __future__ import annotations

from dataclasses import fields

from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ProviderPlanRef, ResolvedTrainingRequest,
    TrainingPlan, TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.training_facade import (
    Clock, TrainingPreflight, TrainingRequest, TrainingStart,
)
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.model import WorkflowRecordV1
from tuner.execution.coordinator_v1.ports import (
    PlanningPortV1, PlanningStorePortV1, RequestLoaderPortV1,
    RequestResolutionPortV1,
)


def _rebuild(value, expected):
    if type(value) is not expected:
        raise TypeError(f"exact {expected.__name__} required")
    if hasattr(value, "to_dict") and hasattr(expected, "from_dict"):
        rebuilt = expected.from_dict(value.to_dict())
    else:
        rebuilt = expected(**{field.name: getattr(value, field.name) for field in fields(expected)})
    if rebuilt != value:
        raise ValueError(f"{expected.__name__} reconstruction mismatch")
    return rebuilt


class CoordinatorTrainingService:
    """Compose planning ports with one authenticated durable coordinator."""

    __slots__ = ("_loader", "_resolver", "_planning", "_store", "_coordinator", "_clock")

    def __init__(self, *, loader: RequestLoaderPortV1,
                 resolver: RequestResolutionPortV1,
                 planning: PlanningPortV1,
                 planning_store: PlanningStorePortV1,
                 coordinator: TrainingCoordinatorV1, clock: Clock) -> None:
        for value, method in ((loader, "load"), (resolver, "resolve"),
                              (planning, "describe"), (planning, "context"),
                              (planning, "preflight"), (planning_store, "put_plan_if_absent"),
                              (planning_store, "get_plan"),
                              (planning_store, "put_context_if_absent"),
                              (planning_store, "get_context"), (clock, "now")):
            if not callable(getattr(value, method, None)):
                raise TypeError("coordinator training dependency is incomplete")
        if type(coordinator) is not TrainingCoordinatorV1:
            raise TypeError("exact TrainingCoordinatorV1 required")
        self._loader, self._resolver, self._planning = loader, resolver, planning
        self._store, self._coordinator, self._clock = planning_store, coordinator, clock

    def load(self, canonical_json: str) -> TrainingRequest:
        if type(canonical_json) is not str or not canonical_json:
            raise TypeError("canonical_json must be an exact nonempty string")
        request = _rebuild(self._loader.load(canonical_json), TrainingRequest)
        if request.canonical_json != canonical_json:
            raise ValueError("loader replaced the submitted canonical request")
        return request

    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest:
        request = _rebuild(request, TrainingRequest)
        resolved = _rebuild(self._resolver.resolve(request), ResolvedTrainingRequest)
        if (resolved.request_id, resolved.project_ref) != (request.request_id, request.project_ref):
            raise ValueError("resolution differs from its request")
        return resolved

    def _current(self, plan: TrainingPlan):
        plan = _rebuild(plan, TrainingPlan)
        retained = self._store.get_plan(plan.plan_fingerprint)
        if type(retained) is not TrainingPlan or _rebuild(retained, TrainingPlan) != plan:
            raise ValueError("retained training plan mismatch")
        context = self._store.get_context(plan.provider_plan.context_digest)
        context = _rebuild(context, ProviderPlanContextV1)
        descriptor = _rebuild(self._planning.describe(context.provider), ProviderDescriptor)
        if (context.provider_context_digest != plan.provider_plan.context_digest
                or context.basis_digest != plan.basis.basis_digest
                or descriptor.provider_id != context.provider.provider_id
                or descriptor.descriptor_digest != context.descriptor_digest):
            raise ValueError("current provider plan binding mismatch")
        return plan, context, descriptor

    def plan(self, resolved: ResolvedTrainingRequest, provider: ProviderRef) -> TrainingPlan:
        resolved = _rebuild(resolved, ResolvedTrainingRequest)
        provider = _rebuild(provider, ProviderRef)
        descriptor = _rebuild(self._planning.describe(provider), ProviderDescriptor)
        if descriptor.provider_id != provider.provider_id:
            raise ValueError("provider descriptor mismatch")
        basis = TrainingPlanBasisV1.from_resolved(resolved)
        context = _rebuild(self._planning.context(resolved, provider), ProviderPlanContextV1)
        if (context.provider != provider or context.basis_digest != basis.basis_digest
                or context.descriptor_digest != descriptor.descriptor_digest):
            raise ValueError("provider planning context mismatch")
        plan = TrainingPlan(
            "synaptic-training-plan/v2", basis,
            ProviderPlanRef(context.provider_context_digest),
        )
        context_result = self._store.put_context_if_absent(context)
        if type(context_result) is not bool:
            raise TypeError("planning store admission must return exact booleans")
        retained_context = self._store.get_context(context.provider_context_digest)
        if _rebuild(retained_context, ProviderPlanContextV1) != context:
            raise ValueError("planning store collision")
        plan_result = self._store.put_plan_if_absent(plan)
        if type(plan_result) is not bool:
            raise TypeError("planning store admission must return exact booleans")
        retained_plan = self._store.get_plan(plan.plan_fingerprint)
        if _rebuild(retained_plan, TrainingPlan) != plan:
            raise ValueError("planning store collision")
        return plan

    def preflight(self, plan: TrainingPlan) -> TrainingPreflight:
        plan, _, _ = self._current(plan)
        value = _rebuild(self._planning.preflight(plan), TrainingPreflight)
        if not value.binds(plan) or value.is_expired(self._clock.now()):
            raise ValueError("provider preflight is invalid or expired")
        return value

    def start(self, plan: TrainingPlan, preflight: TrainingPreflight) -> TrainingStart:
        plan, context, descriptor = self._current(plan)
        preflight = _rebuild(preflight, TrainingPreflight)
        if (preflight.ready is not True or not preflight.binds(plan)
                or preflight.is_expired(self._clock.now())):
            raise ValueError("training preflight is not startable")
        workflow = _rebuild(self._coordinator.start(plan, preflight), WorkflowRecordV1)
        if (workflow.run.project_ref != plan.basis.project_ref
                or workflow.plan_fingerprint != plan.plan_fingerprint
                or workflow.provider != context.provider
                or workflow.provider_context_digest != context.provider_context_digest
                or workflow.provider_descriptor_digest != descriptor.descriptor_digest):
            raise ValueError("durable workflow differs from the accepted plan")
        return TrainingStart(workflow.run, True)


__all__: list[str] = []
