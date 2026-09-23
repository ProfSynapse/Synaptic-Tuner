from __future__ import annotations

from pathlib import Path

import pytest

from synaptic_tuner.api.v1.providers import (
    ProviderCapabilities, ProviderDescriptor, ProviderRef,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.foundation_v2.executors import ExecutorDescriptorV1
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.project.context import ProjectContext
from tuner.training.coordinator_service import CoordinatorTrainingService
from tuner.training.modal_host_requests import ModalPackagedPlanPortsV1, ModalPackagedRequestsV1
from tests.training.test_packaged_execution_material import packaged_fixture


class _Clock:
    def now(self):
        return "2026-09-22T12:00:00Z"


class _Store:
    def __init__(self):
        self.contexts = {}
        self.plans = {}

    def put_context_if_absent(self, context):
        self.contexts[context.provider_context_digest] = context
        return True

    def get_context(self, digest):
        return self.contexts.get(digest)

    def put_plan_if_absent(self, plan):
        self.plans[plan.plan_fingerprint] = plan
        return True

    def get_plan(self, digest):
        return self.plans.get(digest)


def test_packaged_request_uses_public_coordinator_ladder(tmp_path):
    public, components = packaged_fixture()
    request = TrainingRequest("request", "project", public.canonical_json())
    run = TrainingRunRef("run-packaged", "project")
    bridge = ModalPackagedRequestsV1(
        prepared_request=request, run=run, components=components,
        context=ProjectContext.standalone(engine_root=tmp_path),
    )
    provider = ProviderRef("modal", "qwen35-sft-v1")
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1", "modal", "Modal", "1.0.0",
        ProviderCapabilities(True, True, True, True, True, False),
    )
    binding = ProviderExecutionBindingV1(
        provider, descriptor.descriptor_digest, "a" * 64,
        ExecutionScopeV1("account", "namespace"),
        ExecutorDescriptorV1("modal", "packaged", "1.0.0"),
        "b" * 64, "c" * 64, "d" * 64, "e" * 64,
    )
    planning = ModalPackagedPlanPortsV1(
        provider=provider, descriptor=descriptor, binding=binding,
        clock=_Clock(), maximum_cost_minor_units=1000,
    )
    service = CoordinatorTrainingService(
        loader=bridge, resolver=bridge, planning=planning,
        planning_store=_Store(), coordinator=object.__new__(TrainingCoordinatorV1),
        clock=_Clock(),
    )
    loaded = service.load(public.canonical_json())
    resolved = service.resolve(loaded)
    plan = service.plan(resolved, provider)
    assert bridge.for_plan(plan) == run
    assert planning.resolve(provider, planning.context(resolved, provider)) == binding
    assert service.preflight(plan).authorization[0].operation == "training.start"
    with pytest.raises(ValueError, match="prepared input"):
        bridge.load(public.canonical_json() + " ")
