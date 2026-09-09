"""Provider-free tests for Foundation-native Modal Volume staging."""

from dataclasses import replace
import hashlib
import json

import pytest

from tests.execution.providers.test_modal_coordinator_binding import case
from tests.execution.providers.test_modal_sdk154_adapter import (
    FakeVolume, SDK,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_staging import (
    ModalFoundationVolumeWriter,
    ModalStageMaterial,
    prepare_modal_foundation_stage,
)
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade, ModalFacadeError


CONTROL_ID = "control-id"
ARTIFACT_ID = "artifact-id"
KEY_REF = "stage-key"


class Authority:
    def __init__(self, expected, allowed=True):
        self.expected = expected
        self.allowed = allowed

    def authenticate(self, binding):
        return (self.allowed and type(binding) is ModalCommandBinding
                and binding.canonical_bytes == self.expected)


class Authenticator:
    def sign(self, purpose, payload, key_ref):
        assert (purpose, key_ref) == ("modal-stage-claim/v2", KEY_REF)
        return b"authenticated-tag"

    def verify(self, purpose, payload, tag, key_ref):
        return (purpose, key_ref, tag) == (
            "modal-stage-claim/v2", KEY_REF, b"authenticated-tag"
        )


def binding_case():
    adapter, command, deployment = case()
    return ModalCommandBinding(
        command.canonical_bytes, adapter.snapshot(),
        canonical_bytes(deployment.to_dict()),
    )


def facade(binding):
    FakeVolume.calls = []
    FakeVolume.registry = {
        "modal-control-v1": FakeVolume(CONTROL_ID),
        "modal-artifacts-v1": FakeVolume(ARTIFACT_ID),
    }
    client = object()
    session = binding.client_binding
    return ExplicitModal154ReadFacade(
        session, sdk=SDK, client=client,
        scope_observer=lambda supplied: (
            session.account_ref, session.workspace_ref,
            session.environment_ref, session.client_ref,
        ) if supplied is client else (),
        deployment_observer=lambda **_: binding.deployment.selection,
        volume_names={
            CONTROL_ID: "modal-control-v1", ARTIFACT_ID: "modal-artifacts-v1",
        },
    )


def material_case():
    binding = binding_case()
    authority = Authority(binding.canonical_bytes)
    material = prepare_modal_foundation_stage(
        binding, b"opaque-not-yet-remote-compatible-bundle", authority,
        Authenticator(), control_volume_id=CONTROL_ID,
        artifact_volume_id=ARTIFACT_ID, key_ref=KEY_REF,
    )
    return binding, authority, material


def test_claim_binds_complete_foundation_stage_and_opaque_bundle():
    binding, _, material = material_case()
    command = parse_exact_command(binding.command_bytes)
    claim = json.loads(material.claim)
    assert claim["schema_version"] == "synaptic.modal-stage-claim/v2"
    assert claim["command"] == command.to_dict()
    assert claim["command_digest"] == command.digest
    assert claim["binding_digest"] == binding.authenticated_binding_digest
    assert claim["effect_id"] == command.operation.effect.effect_id
    assert claim["preparation_digest"] == command.preparation.preparation_digest
    assert claim["bundle_size"] == len(material.bundle)


def test_writer_authenticates_then_writes_exact_three_paths_without_creation():
    binding, authority, material = material_case()
    modal = facade(binding)
    receipt = ModalFoundationVolumeWriter(
        modal, authority, Authenticator(),
    ).stage_once(material)
    effect_id = receipt.effect_id
    artifact = FakeVolume.registry["modal-artifacts-v1"].files
    control = FakeVolume.registry["modal-control-v1"].files
    assert artifact == {f"operations/{effect_id}/input/bundle.bin": material.bundle}
    assert control == {
        f"operations/{effect_id}/control/stage-claim.v2.json": material.claim,
        f"operations/{effect_id}/control/stage-claim.v2.mac": material.claim_tag,
    }
    assert all(call[1]["create_if_missing"] is False for call in FakeVolume.calls)
    calls = len(FakeVolume.calls)
    assert ModalFoundationVolumeWriter(modal, authority, Authenticator()).stage_once(material) == receipt
    assert len(FakeVolume.calls) > calls  # exact readback, never blind success


@pytest.mark.parametrize("failure", ["binding", "signature", "claim"])
def test_forged_material_stops_before_first_provider_call(failure):
    binding, authority, material = material_case()
    if failure == "binding":
        authority.allowed = False
    elif failure == "signature":
        material = replace(material, claim_tag=b"forged")
    else:
        material = replace(material, claim=canonical_bytes({"forged": True}))
    modal = facade(binding)
    with pytest.raises(ValueError):
        ModalFoundationVolumeWriter(modal, authority, Authenticator()).stage_once(material)
    assert FakeVolume.calls == []


def test_collision_or_wrong_configured_volume_name_never_overwrites():
    binding, authority, material = material_case()
    modal = facade(binding)
    effect_id = parse_exact_command(binding.command_bytes).operation.effect.effect_id
    artifact = FakeVolume.registry["modal-artifacts-v1"]
    path = f"operations/{effect_id}/input/bundle.bin"
    artifact.files[path] = b"different"
    with pytest.raises(ModalFacadeError, match="collision"):
        ModalFoundationVolumeWriter(modal, authority, Authenticator()).stage_once(material)
    assert artifact.files[path] == b"different"

    bad = ExplicitModal154ReadFacade(
        binding.client_binding, sdk=SDK, client=modal.client,
        scope_observer=lambda _: (
            binding.client_binding.account_ref, binding.client_binding.workspace_ref,
            binding.client_binding.environment_ref, binding.client_binding.client_ref,
        ),
        deployment_observer=lambda **_: binding.deployment.selection,
        volume_names={CONTROL_ID: "wrong-control", ARTIFACT_ID: "modal-artifacts-v1"},
    )
    with pytest.raises(ModalFacadeError, match="binding_mismatch"):
        ModalFoundationVolumeWriter(bad, authority, Authenticator()).stage_once(material)


def test_partial_exact_stage_resumes_only_missing_files():
    binding, authority, material = material_case()
    modal = facade(binding)
    effect_id = parse_exact_command(binding.command_bytes).operation.effect.effect_id
    FakeVolume.registry["modal-artifacts-v1"].files[
        f"operations/{effect_id}/input/bundle.bin"
    ] = material.bundle
    receipt = ModalFoundationVolumeWriter(
        modal, authority, Authenticator(),
    ).stage_once(material)
    assert receipt.bundle_digest == hashlib.sha256(material.bundle).hexdigest()
    assert len(FakeVolume.registry["modal-control-v1"].files) == 2
