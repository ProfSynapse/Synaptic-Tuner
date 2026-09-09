"""Shared provider-free fixtures for the Foundation-native Modal path."""

from dataclasses import replace

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.references import ProviderStageRefV1
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_bundle import ModalCoordinatorBundle
from tuner.execution.providers.modal.coordinator_launch import prepare_modal_foundation_launch
from tuner.execution.providers.modal.coordinator_staging import (
    modal_stage_provider_ref, prepare_modal_foundation_stage,
)

from tests.execution.providers.test_modal_coordinator_adapter import coordinator_harness
from tests.execution.providers.test_modal_coordinator_bundle import _fixture
from tests.execution.providers.test_modal_coordinator_launch import Authenticator, BindingAuthority


def real_launch_bundle_case(monkeypatch):
    """Run the generic coordinator through a real semantic stage bundle."""
    stage_template, material, recipes, policy, closure = _fixture()
    import tests.execution.providers.test_modal_coordinator_adapter as adapter_tests
    original_inputs = adapter_tests.inputs
    monkeypatch.setattr(
        adapter_tests, "inputs",
        lambda: original_inputs() | {"resolved": material.planning_request},
    )
    generic, harness, adapter, plan = coordinator_harness(monkeypatch)
    monkeypatch.setattr(
        generic, "RUN", TrainingRunRef(material.run_id, material.planning_request.project_ref),
    )
    deployment_bytes = stage_template.deployment_bytes
    authority = BindingAuthority(())
    authentic = Authenticator()
    staged = {}
    execute_once = harness.executor.execute_once

    def execute_with_bundle(payload, request):
        observation = execute_once(payload, request)
        if request.effect_kind == "stage":
            command = harness.executor.commands[request.command_digest]
            binding = ModalCommandBinding(
                command.canonical_bytes, adapter.snapshot(), deployment_bytes,
            )
            authority.values.add(binding.canonical_bytes)
            bundle = ModalCoordinatorBundle.build(
                binding, material, recipes, log_terminal_policy=policy,
                worker_closure_manifest=closure,
            )
            stage_material = prepare_modal_foundation_stage(
                binding, bundle.transport_bytes, authority, authentic,
                control_volume_id="control-id", artifact_volume_id="artifact-id",
                key_ref="stage-key",
            )
            staged.update(bundle=bundle, material=stage_material)
            observation = replace(
                observation,
                stage_ref=ProviderStageRefV1(
                    observation.stage_ref.provider_id,
                    observation.stage_ref.profile_ref,
                    observation.stage_ref.account_ref,
                    observation.stage_ref.namespace_ref,
                    modal_stage_provider_ref(stage_material),
                ),
            )
        return observation

    harness.executor.execute_once = execute_with_bundle
    workflow = harness.service.start(plan, generic.preflight())
    stage = parse_exact_command(workflow.stage.canonical_command_bytes)
    submit = parse_exact_command(workflow.submit.canonical_command_bytes)
    stage_record = harness.repository.get(stage.operation.effect.effect_id)
    assessment = harness.foundation.assess(stage_record)
    submit_binding = ModalCommandBinding(
        submit.canonical_bytes, adapter.snapshot(), deployment_bytes,
    )
    authority.values.add(submit_binding.canonical_bytes)
    envelope = prepare_modal_foundation_launch(
        submit_binding=submit_binding, stage_material=staged["material"],
        stage_record=stage_record, stage_assessment=assessment,
        foundation_authenticator=harness.authenticator,
        assessment_authenticator=harness.foundation,
        binding_authority=authority, stage_verifier=authentic, signer=authentic,
    )
    return {
        "harness": harness, "recipes": recipes, "bundle": staged["bundle"],
        "material": staged["material"], "envelope": envelope,
        "authority": authority, "authenticator": authentic,
    }
