from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from typing import get_type_hints

import pytest

from synaptic_tuner.api.v1.docker import (
    DockerCoordinatorHostPortsV1,
    DockerSameProcessBindingStoreV1,
    DockerSameProcessLaunchV1,
    DockerSameProcessRuntimeV1,
    compose_docker_same_process_coordinator_v1,
)
from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, ProviderPlanRef, TrainingPlan
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.providers.docker_provider_v1.model import (
    DockerCommandBindingV1,
    DockerEffectIdentityV1,
)
from tuner.execution.coordinator_v1.model import WorkflowRecordV1
from tests.execution.providers.docker_provider_v1.conftest import Authority, BindingAuthority


def launch_for(profile, plan, run):
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider,
        plan.basis.basis_digest, profile.descriptor.descriptor_digest,
        profile.profile_digest,
    )
    bound_plan = TrainingPlan(
        "synaptic-training-plan/v2", plan.basis,
        ProviderPlanRef(context.provider_context_digest),
    )
    preflight = TrainingPreflight(
        bound_plan.plan_fingerprint, True,
        "2026-08-27T00:00:00Z", "2026-08-29T00:00:00Z",
    )
    return DockerSameProcessLaunchV1(profile, context, bound_plan, run, preflight)


def ports_for(seams, *, authority=None, source=None, control=None):
    catalog, images, source_seals, docker_control, _ = seams
    binding_authority = authority or catalog.binding_authority
    store = DockerSameProcessBindingStoreV1(binding_authority)
    return DockerCoordinatorHostPortsV1(
        store, binding_authority, images, source or source_seals,
        control or docker_control, Authority(),
    )


def test_public_launch_is_frozen_exact_and_profile_bound(profile, plan, run):
    launch = launch_for(profile, plan, run)
    with pytest.raises(FrozenInstanceError):
        launch.run = run
    with pytest.raises(ValueError):
        DockerSameProcessLaunchV1(
            profile, launch.context,
            TrainingPlan("synaptic-training-plan/v2", plan.basis, ProviderPlanRef("f" * 64)),
            run, launch.preflight,
        )
    with pytest.raises(TypeError):
        DockerSameProcessLaunchV1(profile, launch.context, launch.plan, run, object())


@pytest.mark.parametrize(
    "basis_field",
    ("workload_digest", "runtime_digest", "artifact_policy_digest"),
)
def test_public_launch_rejects_self_consistent_plan_that_differs_from_profile(
    profile, plan, run, basis_field,
):
    hostile_basis = replace(plan.basis, **{basis_field: "0" * 64})
    hostile_context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider,
        hostile_basis.basis_digest, profile.descriptor.descriptor_digest,
        profile.profile_digest,
    )
    hostile_plan = TrainingPlan(
        "synaptic-training-plan/v2", hostile_basis,
        ProviderPlanRef(hostile_context.provider_context_digest),
    )
    hostile_preflight = TrainingPreflight(
        hostile_plan.plan_fingerprint, True,
        "2026-08-27T00:00:00Z", "2026-08-29T00:00:00Z",
    )
    with pytest.raises(ValueError, match="Docker launch bindings differ"):
        DockerSameProcessLaunchV1(
            profile, hostile_context, hostile_plan, run, hostile_preflight,
        )


def test_public_and_concrete_runtime_return_exact_workflow_contract(
    profile, plan, run, seams,
):
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports_for(seams),
    )
    public_start = get_type_hints(DockerSameProcessRuntimeV1.start)
    public_reconcile = get_type_hints(DockerSameProcessRuntimeV1.reconcile)
    concrete_start = get_type_hints(type(runtime).start)
    concrete_reconcile = get_type_hints(type(runtime).reconcile)
    assert public_start["return"] is WorkflowRecordV1
    assert public_reconcile["return"] is WorkflowRecordV1
    assert concrete_start["return"] is WorkflowRecordV1
    assert concrete_reconcile["return"] is WorkflowRecordV1
    assert type(runtime.start()) is WorkflowRecordV1
    assert type(runtime.reconcile()) is WorkflowRecordV1


def test_composed_runtime_starts_once_and_exposes_only_exact_bindings(
    profile, plan, run, seams,
):
    ports = ports_for(seams)
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports,
    )
    assert sorted(name for name in dir(runtime) if not name.startswith("_")) == [
        "binding", "reconcile", "start",
    ]
    with pytest.raises(LookupError):
        runtime.binding("stage")
    first = runtime.start()
    before = tuple(seams[3].trace)
    second = runtime.start()
    assert first.phase.value == second.phase.value == "queued"
    assert before == tuple(seams[3].trace)
    assert tuple(event[0] for event in before) == ("create", "start")
    stage = runtime.binding("stage")
    submit = runtime.binding("submit")
    assert stage.content.effect_kind == "stage"
    assert submit.content.effect_kind == "submit"
    assert stage.content.command_digest != submit.content.command_digest
    assert ports.binding_authority.authenticate(stage) is True
    assert ports.binding_authority.authenticate(submit) is True
    for value in ("cancel", "", "STAGE", None):
        with pytest.raises((TypeError, ValueError)):
            runtime.binding(value)


def test_concurrent_same_runtime_start_has_one_create_and_start(
    profile, plan, run, seams,
):
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports_for(seams),
    )

    def attempt(_):
        try:
            return runtime.start()
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(attempt, range(8)))
    final = runtime.start()
    assert final.phase.value == "queued"
    assert any(value is not None and value.phase.value == "queued" for value in results)
    assert tuple(event[0] for event in seams[3].trace) == ("create", "start")
    assert seams[2].calls == 1


def test_binding_is_published_before_stage_and_submit_provider_effects(
    profile, plan, run, seams,
):
    catalog, images, source, control, _ = seams
    store = DockerSameProcessBindingStoreV1(catalog.binding_authority)

    class Source:
        def seal_read_only(self, request):
            assert store.resolve(request.identity.command_digest).content.effect_kind == "stage"
            return source.seal_read_only(request)

        def lookup(self, request):
            return source.lookup(request)

    class Control:
        def create_once(self, **values):
            labels = values["labels"]
            assert store.resolve(labels.command_digest).content.effect_kind == "submit"
            return control.create_once(**values)

        def start_once(self, container_ref, labels):
            return control.start_once(container_ref, labels)

        def lookup(self, request):
            return control.lookup(request)

    ports = DockerCoordinatorHostPortsV1(
        store, catalog.binding_authority, images, Source(), Control(), Authority(),
    )
    result = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports,
    ).start()
    assert result.phase.value == "queued"


def test_stage_reconciliation_uses_lookup_then_progresses_without_stage_retry(
    profile, plan, run, seams,
):
    seams[2].lost_return = True
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports_for(seams),
    )
    first = runtime.start()
    assert first.phase.value == "stage_reconcile_required"
    assert seams[2].calls == 1
    recovered = runtime.reconcile()
    assert recovered.phase.value == "queued"
    assert seams[2].calls == seams[2].lookup_calls == 1
    assert tuple(event[0] for event in seams[3].trace) == ("create", "start")


def test_host_exception_totalizes_without_raw_text(profile, plan, run, seams):
    def fail_create(**values):
        raise RuntimeError("secret-host-provider-body")

    seams[3].create_once = fail_create
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports_for(seams),
    )
    result = runtime.start()
    assert result.phase.value == "submit_reconcile_required"
    assert "secret-host-provider-body" not in repr(result)


def test_malformed_legacy_start_result_closes_without_raw_host_data(
    profile, plan, run, seams,
):
    seams[3].start_result = True
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports_for(seams),
    )
    with pytest.raises(Exception) as caught:
        runtime.start()
    assert "True" not in str(caught.value)
    assert tuple(event[0] for event in seams[3].trace) == ("create", "start")


def test_exact_binding_store_replays_and_rejects_conflict(profile, plan, run, seams):
    ports = ports_for(seams)
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports,
    )
    runtime.start()
    stage = runtime.binding("stage")
    assert ports.binding_store.publish_once(stage) == stage
    identity = DockerEffectIdentityV1(
        stage.content.command_digest, "conflicting-effect", "stage",
        stage.content.identity.plan,
    )
    conflict = ports.binding_authority.issue(
        DockerCommandBindingV1(identity, stage.content.command_bytes)
    )
    with pytest.raises(ValueError):
        ports.binding_store.publish_once(conflict)


def test_conflicting_publication_closes_before_any_provider_effect(
    profile, plan, run, seams,
):
    class ConflictAuthority(BindingAuthority):
        def __init__(self):
            self.store = None
            self.planted = False

        def issue(self, binding):
            expected = super().issue(binding)
            if not self.planted:
                self.planted = True
                identity = DockerEffectIdentityV1(
                    binding.command_digest, "conflicting-effect",
                    binding.effect_kind, binding.identity.plan,
                )
                conflicting = super().issue(
                    DockerCommandBindingV1(identity, binding.command_bytes)
                )
                self.store.publish_once(conflicting)
            return expected

    authority = ConflictAuthority()
    store = DockerSameProcessBindingStoreV1(authority)
    authority.store = store
    ports = DockerCoordinatorHostPortsV1(
        store, authority, seams[1], seams[2], seams[3], Authority(),
    )
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports,
    )
    with pytest.raises(Exception) as caught:
        runtime.start()
    assert "conflicting-effect" not in str(caught.value)
    assert seams[1].calls == seams[2].calls == 0
    assert seams[3].trace == []


def test_binding_authentication_cannot_mutate_probe_into_acceptance(
    profile, plan, run, seams,
):
    class MutatingAuthority(BindingAuthority):
        def authenticate(self, value):
            object.__setattr__(value.content.identity, "effect_id", "mutated-effect")
            return True

    authority = MutatingAuthority()
    store = DockerSameProcessBindingStoreV1(authority)
    ports = DockerCoordinatorHostPortsV1(
        store, authority, seams[1], seams[2], seams[3], Authority(),
    )
    runtime = compose_docker_same_process_coordinator_v1(
        launch_for(profile, plan, run), ports,
    )
    with pytest.raises(Exception):
        runtime.start()
    assert seams[1].calls == seams[2].calls == 0
    assert seams[3].trace == []


def test_docker_api_is_explicit_not_package_wide():
    import synaptic_tuner.api.v1 as api

    for name in (
        "DockerSameProcessLaunchV1", "DockerCoordinatorHostPortsV1",
        "DockerSameProcessBindingStoreV1",
        "compose_docker_same_process_coordinator_v1",
    ):
        assert name not in api.__all__
