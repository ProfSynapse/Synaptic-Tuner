from __future__ import annotations

import json

import pytest

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, ResolvedTrainingRequest
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.training_facade import TrainingPreflight, TrainingRequest
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.model import WorkflowRecordV1
from tuner.training.coordinator_service import CoordinatorTrainingService


D = tuple(character * 64 for character in "123456789abcdef")
PROVIDER = ProviderRef("fake", "profile")
DESCRIPTOR = ProviderDescriptor(
    "synaptic-provider-descriptor/v1", "fake", "Fake", "1.0.0",
    ProviderCapabilities(True, True, True, True, True, False),
)
REQUEST = TrainingRequest("request-1", "project-1", '{"method":"sft"}')
RESOLVED = ResolvedTrainingRequest(
    "synaptic-resolved-training-request/v1", "request-1", "project-1",
    D[0], D[1], D[2], D[3], D[4],
)


class Loader:
    value = REQUEST
    def load(self, value): return self.value


class Resolver:
    value = RESOLVED
    def resolve(self, value): return self.value


class Planning:
    def __init__(self): self.preflight_calls = 0
    def describe(self, provider): return DESCRIPTOR
    def context(self, resolved, provider):
        basis = __import__(
            "synaptic_tuner.api.v1.planning", fromlist=["TrainingPlanBasisV1"],
        ).TrainingPlanBasisV1.from_resolved(resolved)
        return ProviderPlanContextV1(
            "synaptic-provider-plan-context/v1", provider, basis.basis_digest,
            DESCRIPTOR.descriptor_digest, D[5],
        )
    def preflight(self, plan):
        self.preflight_calls += 1
        return TrainingPreflight(
            plan.plan_fingerprint, True, "2026-09-09T00:00:00Z",
            "2026-09-09T01:00:00Z",
        )


class Store:
    def __init__(self): self.plans = {}; self.contexts = {}; self.substitute = None
    def put_plan_if_absent(self, value):
        if value.plan_fingerprint in self.plans: return False
        self.plans[value.plan_fingerprint] = value; return True
    def get_plan(self, key): return self.substitute or self.plans.get(key)
    def put_context_if_absent(self, value):
        if value.provider_context_digest in self.contexts: return False
        self.contexts[value.provider_context_digest] = value; return True
    def get_context(self, key): return self.contexts.get(key)


class Clock:
    value = "2026-09-09T00:30:00Z"
    def now(self): return self.value


def service(monkeypatch):
    coordinator = object.__new__(TrainingCoordinatorV1)
    planning, store = Planning(), Store()
    value = CoordinatorTrainingService(
        loader=Loader(), resolver=Resolver(), planning=planning,
        planning_store=store, coordinator=coordinator, clock=Clock(),
    )
    return value, planning, store, coordinator


def test_load_resolve_plan_and_restart_safe_retention(monkeypatch):
    value, _, store, _ = service(monkeypatch)
    assert value.load(REQUEST.canonical_json) == REQUEST
    assert value.resolve(REQUEST) == RESOLVED
    first = value.plan(RESOLVED, PROVIDER)
    assert value.plan(RESOLVED, PROVIDER) == first
    assert store.get_plan(first.plan_fingerprint) == first
    assert store.get_context(first.provider_plan.context_digest).provider == PROVIDER


def test_loader_and_resolution_cannot_substitute_request_identity(monkeypatch):
    value, _, _, _ = service(monkeypatch)
    value._loader.value = TrainingRequest("request-1", "project-1", '{"method":"other"}')
    with pytest.raises(ValueError, match="replaced"):
        value.load(REQUEST.canonical_json)
    value._resolver.value = ResolvedTrainingRequest(
        "synaptic-resolved-training-request/v1", "other", "project-1",
        D[0], D[1], D[2], D[3], D[4],
    )
    with pytest.raises(ValueError, match="differs"):
        value.resolve(REQUEST)


def test_store_substitution_is_rejected(monkeypatch):
    value, _, store, _ = service(monkeypatch)
    plan = value.plan(RESOLVED, PROVIDER)
    other = ResolvedTrainingRequest(
        "synaptic-resolved-training-request/v1", "request-2", "project-1",
        D[0], D[1], D[2], D[3], D[4],
    )
    store.substitute = __import__(
        "synaptic_tuner.api.v1.planning", fromlist=["TrainingPlan"],
    ).TrainingPlan(
        "synaptic-training-plan/v2",
        __import__("synaptic_tuner.api.v1.planning", fromlist=["TrainingPlanBasisV1"]).TrainingPlanBasisV1.from_resolved(other),
        plan.provider_plan,
    )
    with pytest.raises(ValueError, match="retained"):
        value.preflight(plan)


def test_context_collision_stops_before_plan_retention(monkeypatch):
    value, _, store, _ = service(monkeypatch)
    original_put = store.put_context_if_absent
    def collide(context):
        original_put(context)
        store.contexts[context.provider_context_digest] = ProviderPlanContextV1(
            "synaptic-provider-plan-context/v1", context.provider,
            context.basis_digest, context.descriptor_digest, D[7],
        )
        return False
    store.put_context_if_absent = collide
    with pytest.raises(ValueError, match="collision"):
        value.plan(RESOLVED, PROVIDER)
    assert store.plans == {}


def test_failed_or_expired_preflight_never_calls_coordinator(monkeypatch):
    value, planning, _, _ = service(monkeypatch)
    plan = value.plan(RESOLVED, PROVIDER)
    calls = []
    monkeypatch.setattr(TrainingCoordinatorV1, "start", lambda *args: calls.append(1))
    failed = TrainingPreflight(
        plan.plan_fingerprint, False, "2026-09-09T00:00:00Z",
        "2026-09-09T01:00:00Z", diagnostic_codes=("not-ready",),
    )
    with pytest.raises(ValueError, match="not startable"):
        value.start(plan, failed)
    assert calls == [] and planning.preflight_calls == 0


def test_start_projects_exact_durable_workflow_and_restart(monkeypatch):
    value, _, store, _ = service(monkeypatch)
    plan = value.plan(RESOLVED, PROVIDER)
    checked = value.preflight(plan)
    context = store.get_context(plan.provider_plan.context_digest)
    workflow = WorkflowRecordV1.planned(
        run=__import__("synaptic_tuner.api.v1.results", fromlist=["TrainingRunRef"]).TrainingRunRef("run-1", "project-1"),
        plan=plan, preflight_digest=D[6], context=context,
        provider=PROVIDER, descriptor=DESCRIPTOR,
    )
    calls = []
    monkeypatch.setattr(TrainingCoordinatorV1, "start", lambda self, p, f: calls.append((p, f)) or workflow)
    assert value.start(plan, checked).accepted is True
    assert value.start(plan, checked).run == workflow.run
    assert len(calls) == 2


def test_real_coordinator_reconcile_required_is_durable_acceptance():
    from tests.execution.coordinator_v1.test_start_reconcile_service import (
        CONTEXT, DESC, PLAN, Harness, PlanningStore, preflight,
    )
    from tuner.execution.foundation_v2.observations import ObservationDisposition

    class ExistingStore(PlanningStore):
        def put_plan_if_absent(self, value): return False
        def put_context_if_absent(self, value): return False
    class ExistingPlanning:
        def describe(self, provider): return DESC
        def context(self, resolved, provider): raise AssertionError
        def preflight(self, plan): raise AssertionError
    class AugustClock:
        def now(self): return "2026-08-27T00:00:00Z"
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    value = CoordinatorTrainingService(
        loader=Loader(), resolver=Resolver(), planning=ExistingPlanning(),
        planning_store=ExistingStore(), coordinator=harness.service,
        clock=AugustClock(),
    )
    result = value.start(PLAN, preflight())
    assert result.accepted is True and result.run.project_ref == PLAN.basis.project_ref
    assert harness.workflows.get(result.run).phase.value == "stage_reconcile_required"
    calls = tuple(harness.executor.calls)
    restarted = value.start(PLAN, preflight())
    assert restarted == result
    assert tuple(harness.executor.calls) == calls == ("stage",)
