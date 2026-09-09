"""Provider-free Foundation stage-to-submit launch admission tests."""

from dataclasses import replace
import copy
import hashlib

import pytest

from tests.execution.providers.test_modal_coordinator_adapter import coordinator_harness, inputs
from tests.execution.providers.test_modal_sdk154_adapter import verified
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.commands import build_submit_command, parse_exact_command
from tuner.execution.foundation_v2.references import ProviderStageRefV1
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_launch import (
    admit_modal_foundation_launch,
    prepare_modal_foundation_launch,
)
from tuner.execution.providers.modal.coordinator_staging import (
    modal_stage_provider_ref,
    prepare_modal_foundation_stage,
)
from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1


class BindingAuthority:
    def __init__(self, values):
        self.values = {value.canonical_bytes for value in values}

    def authenticate(self, value):
        return type(value) is ModalCommandBinding and value.canonical_bytes in self.values


class Authenticator:
    def sign(self, purpose, payload, key_ref):
        return hashlib.sha256(
            purpose.encode("ascii") + b"\0" + key_ref.encode("ascii") + b"\0" + payload
        ).digest()

    def verify(self, purpose, payload, tag, key_ref):
        return tag == self.sign(purpose, payload, key_ref)


def launch_case(monkeypatch):
    _, harness, adapter, plan = coordinator_harness(monkeypatch)
    values = inputs()
    selection = ModalDeploymentSelectionV1.from_profile(
        values["profile"], binding=values["binding"],
        runtime_environment=values["runtime_environment"],
        timeout_seconds=values["timeout_seconds"],
    )
    deployment_bytes = canonical_bytes(verified(selection).to_dict())
    authority = BindingAuthority(())
    authentic = Authenticator()
    staged = {}
    execute_once = harness.executor.execute_once

    def execute_with_material_reference(payload, request):
        observation = execute_once(payload, request)
        if request.effect_kind == "stage":
            command = harness.executor.commands[request.command_digest]
            binding = ModalCommandBinding(
                command.canonical_bytes, adapter.snapshot(), deployment_bytes,
            )
            authority.values.add(binding.canonical_bytes)
            material = prepare_modal_foundation_stage(
                binding, b"opaque-bundle", authority, authentic,
                control_volume_id="control-id", artifact_volume_id="artifact-id",
                key_ref="stage-key",
            )
            staged["material"] = material
            observation = replace(
                observation,
                stage_ref=ProviderStageRefV1(
                    observation.stage_ref.provider_id,
                    observation.stage_ref.profile_ref,
                    observation.stage_ref.account_ref,
                    observation.stage_ref.namespace_ref,
                    modal_stage_provider_ref(material),
                ),
            )
        return observation

    harness.executor.execute_once = execute_with_material_reference
    workflow = harness.service.start(plan, __import__(
        "tests.execution.coordinator_v1.test_start_reconcile_service",
        fromlist=["preflight"],
    ).preflight())
    stage = parse_exact_command(workflow.stage.canonical_command_bytes)
    submit = parse_exact_command(workflow.submit.canonical_command_bytes)
    stage_record = harness.repository.get(stage.operation.effect.effect_id)
    assessment = harness.foundation.assess(stage_record)
    submit_binding = ModalCommandBinding(
        submit.canonical_bytes, adapter.snapshot(), deployment_bytes,
    )
    authority.values.add(submit_binding.canonical_bytes)
    material = staged["material"]
    envelope = prepare_modal_foundation_launch(
        submit_binding=submit_binding, stage_material=material,
        stage_record=stage_record, stage_assessment=assessment,
        foundation_authenticator=harness.authenticator,
        assessment_authenticator=harness.foundation,
        binding_authority=authority, stage_verifier=authentic, signer=authentic,
    )
    return harness, stage, submit, stage_record, assessment, authority, authentic, material, envelope


def admit(case, envelope=None):
    harness, _, _, _, _, authority, authentic, _, original = case
    return admit_modal_foundation_launch(
        envelope or original, foundation_authenticator=harness.authenticator,
        assessment_authenticator=harness.foundation,
        binding_authority=authority, stage_verifier=authentic,
        launch_verifier=authentic,
    )


def test_launch_claim_and_admission_bind_exact_authenticated_predecessor(monkeypatch):
    case = launch_case(monkeypatch)
    _, stage, submit, stage_record, assessment, _, _, material, envelope = case
    claim = __import__("json").loads(envelope.claim)
    assert claim["schema_version"] == "synaptic.modal-launch-claim/v1"
    assert claim["submit_command"] == submit.to_dict()
    assert claim["stage_predecessor"] == submit.stage_predecessor.to_dict()
    assert claim["stage_record_digest"] == stage_record.record_digest
    assert claim["stage_assessment_digest"] == assessment.authenticated_assessment_digest
    admitted = admit(case)
    assert admitted.submit_command_bytes == submit.canonical_bytes
    assert admitted.stage_effect_id == stage.operation.effect.effect_id
    assert admitted.bundle == material.bundle


def test_forged_submit_predecessor_is_rejected_by_generic_stage_reduction(monkeypatch):
    case = launch_case(monkeypatch)
    harness, _, submit, stage_record, assessment, authority, authentic, material, _ = case
    forged_predecessor = replace(
        submit.stage_predecessor, authenticated_receipt_digest="f" * 64,
    )
    forged_command = build_submit_command(
        submit.preparation, submit.operation.invocation_nonce, submit.payload,
        submit.executor, forged_predecessor,
    )
    original_binding = case[-1].submit_binding
    forged_binding = ModalCommandBinding(
        forged_command.canonical_bytes, original_binding.preparation_snapshot,
        original_binding.deployment_bytes,
    )
    authority.values.add(forged_binding.canonical_bytes)
    with pytest.raises(ValueError, match="predecessor"):
        prepare_modal_foundation_launch(
            submit_binding=forged_binding, stage_material=material,
            stage_record=stage_record, stage_assessment=assessment,
            foundation_authenticator=harness.authenticator,
            assessment_authenticator=harness.foundation,
            binding_authority=authority, stage_verifier=authentic, signer=authentic,
        )


def test_valid_material_for_same_stage_command_cannot_replace_recorded_material(monkeypatch):
    case = launch_case(monkeypatch)
    harness, _, _, stage_record, assessment, authority, authentic, material_a, envelope = case
    material_b = prepare_modal_foundation_stage(
        material_a.binding, b"other-valid-opaque-bundle", authority, authentic,
        control_volume_id=material_a.control_volume_id,
        artifact_volume_id=material_a.artifact_volume_id,
        key_ref=material_a.key_ref,
    )
    assert modal_stage_provider_ref(material_a) != modal_stage_provider_ref(material_b)
    with pytest.raises(ValueError, match="does not identify retained material"):
        prepare_modal_foundation_launch(
            submit_binding=envelope.submit_binding, stage_material=material_b,
            stage_record=stage_record, stage_assessment=assessment,
            foundation_authenticator=harness.authenticator,
            assessment_authenticator=harness.foundation,
            binding_authority=authority, stage_verifier=authentic, signer=authentic,
        )


@pytest.mark.parametrize("field", ["key", "claim", "bundle", "launch_tag"])
def test_poisoned_stage_or_launch_inputs_fail_admission(monkeypatch, field):
    case = launch_case(monkeypatch)
    envelope = case[-1]
    material = envelope.stage_material
    if field == "key":
        material = replace(material, key_ref="other-key")
    elif field == "claim":
        material = replace(material, claim=canonical_bytes({"forged": True}))
    elif field == "bundle":
        material = replace(material, bundle=b"different")
    else:
        envelope = replace(envelope, claim_tag=b"forged")
    if field != "launch_tag":
        envelope = replace(envelope, stage_material=material)
    with pytest.raises(ValueError):
        admit(case, envelope)


def test_stage_record_or_retained_configuration_substitution_is_rejected(monkeypatch):
    case = launch_case(monkeypatch)
    harness, _, submit, stage_record, assessment, authority, authentic, material, _ = case
    # A submit binding backed by a different deployment/configuration cannot be
    # constructed against the retained snapshot, before launch signing.
    other_deployment = verified(replace(
        material.binding.deployment.selection, workspace_ref="other-workspace",
    ))
    with pytest.raises(ValueError):
        ModalCommandBinding(
            submit.canonical_bytes, material.binding.preparation_snapshot,
            canonical_bytes(other_deployment.to_dict()),
        )
    # Even a valid assessment for another authenticated stage record cannot be
    # paired with the retained command/material.
    changed_record = copy.copy(stage_record)
    object.__setattr__(changed_record, "terminal_content_digests", ())
    with pytest.raises(ValueError):
        prepare_modal_foundation_launch(
            submit_binding=case[-1].submit_binding, stage_material=material,
            stage_record=changed_record, stage_assessment=assessment,
            foundation_authenticator=harness.authenticator,
            assessment_authenticator=harness.foundation,
            binding_authority=authority, stage_verifier=authentic, signer=authentic,
        )
