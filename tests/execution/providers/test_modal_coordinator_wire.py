"""Provider-free tests for pure remote Modal launch admission."""

from dataclasses import replace

import pytest

from tests.execution.providers.test_modal_coordinator_launch import launch_case
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.coordinator_wire import (
    ModalWorkerLaunchExpectation, admit_modal_launch_wire,
)


def wire_case(monkeypatch):
    case = launch_case(monkeypatch)
    envelope = case[-1]
    material = envelope.stage_material
    submit = envelope.submit_binding
    command = __import__(
        "tuner.execution.foundation_v2.commands", fromlist=["parse_exact_command"],
    ).parse_exact_command(submit.command_bytes)
    prep = command.preparation
    selection = submit.deployment.selection
    expectation = ModalWorkerLaunchExpectation(
        command.canonical_bytes, submit.deployment_bytes,
        prep.provider.provider_id, prep.provider.profile_ref,
        prep.scope.account_ref, prep.scope.namespace_ref,
        selection.app_name, selection.function_name,
        material.control_volume_id, material.artifact_volume_id,
        "modal-control-v1", "modal-artifacts-v1", material.key_ref,
        __import__("hashlib").sha256(material.claim).hexdigest(),
        __import__("hashlib").sha256(material.bundle).hexdigest(), len(material.bundle),
        command.executor.executor_id, command.executor.implementation_version,
    )
    return case, envelope, material, expectation


def admit(values, **changes):
    _, envelope, material, expectation = values
    supplied = {
        "claim": envelope.claim, "claim_tag": envelope.claim_tag,
        "stage_claim": material.claim, "stage_claim_tag": material.claim_tag,
        "bundle": material.bundle, "expectation": expectation,
        "verifier": values[0][6],
    }
    supplied.update(changes)
    return admit_modal_launch_wire(**supplied)


def test_wire_admits_only_minimal_exact_worker_values(monkeypatch):
    values = wire_case(monkeypatch)
    result = admit(values)
    assert result.submit_command_bytes == values[3].submit_command_bytes
    assert result.stage_command_bytes == values[2].binding.command_bytes
    assert result.preparation_snapshot == values[2].binding.preparation_snapshot
    assert result.deployment_bytes == values[2].binding.deployment_bytes
    assert result.bundle == values[2].bundle
    assert result.launch_claim_sha256 == __import__("hashlib").sha256(
        values[1].claim,
    ).hexdigest()
    assert not hasattr(result, "stage_record") and not hasattr(result, "authority")


@pytest.mark.parametrize("field", ["claim_tag", "stage_claim_tag", "bundle"])
def test_wire_rejects_bad_signatures_or_bundle(monkeypatch, field):
    values = wire_case(monkeypatch)
    changed = b"forged" if field != "bundle" else b"other-bundle"
    with pytest.raises(ValueError):
        admit(values, **{field: changed})


def test_wire_rejects_other_valid_submit_target(monkeypatch):
    values = wire_case(monkeypatch)
    with pytest.raises(ValueError, match="worker expectation"):
        admit(values, expectation=replace(values[3], submit_command_bytes=b"other"))


def test_wire_rejects_host_stage_commitment_substitution(monkeypatch):
    values = wire_case(monkeypatch)
    envelope = values[1]
    document = __import__("json").loads(envelope.claim)
    document["stage_bound_reference_digest"] = "f" * 64
    forged = canonical_bytes(document)
    authenticator = values[0][6]
    tag = authenticator.sign("modal-launch-claim/v1", forged, values[2].key_ref)
    with pytest.raises(ValueError, match="stage result commitment"):
        admit(values, claim=forged, claim_tag=tag)


def test_wire_rejects_bounds_before_verification(monkeypatch):
    values = wire_case(monkeypatch)

    class Never:
        def verify(self, *args):
            raise AssertionError("verifier must not run")

    with pytest.raises(ValueError, match="exceeds bound"):
        admit_modal_launch_wire(
            b"x" * 65537, b"tag", values[2].claim, values[2].claim_tag,
            values[2].bundle, expectation=values[3], verifier=Never(),
        )


def test_wire_rejects_bool_bundle_size_even_for_one_byte_bundle(monkeypatch):
    values = wire_case(monkeypatch)
    envelope, material, authenticator = values[1], values[2], values[0][6]
    stage = __import__("json").loads(material.claim)
    stage["bundle_sha256"] = __import__("hashlib").sha256(b"x").hexdigest()
    stage["bundle_size"] = True
    stage_claim = canonical_bytes(stage)
    stage_tag = authenticator.sign("modal-stage-claim/v2", stage_claim, material.key_ref)
    launch = __import__("json").loads(envelope.claim)
    launch["stage_claim_sha256"] = __import__("hashlib").sha256(stage_claim).hexdigest()
    launch["stage_bundle_sha256"] = __import__("hashlib").sha256(b"x").hexdigest()
    launch["stage_bundle_size"] = True
    claim = canonical_bytes(launch)
    tag = authenticator.sign("modal-launch-claim/v1", claim, material.key_ref)
    with pytest.raises(ValueError):
        admit(values, claim=claim, claim_tag=tag, stage_claim=stage_claim,
              stage_claim_tag=stage_tag, bundle=b"x")


def test_wire_rejects_valid_signed_material_substitution_against_dispatch(monkeypatch):
    values = wire_case(monkeypatch)
    envelope, material, authenticator = values[1], values[2], values[0][6]
    replacement = b"X" * len(material.bundle)
    stage = __import__("json").loads(material.claim)
    stage["bundle_sha256"] = __import__("hashlib").sha256(replacement).hexdigest()
    stage_claim = canonical_bytes(stage)
    stage_tag = authenticator.sign("modal-stage-claim/v2", stage_claim, material.key_ref)
    launch = __import__("json").loads(envelope.claim)
    launch["stage_claim_sha256"] = __import__("hashlib").sha256(stage_claim).hexdigest()
    launch["stage_bundle_sha256"] = __import__("hashlib").sha256(replacement).hexdigest()
    predecessor = launch["stage_predecessor"]
    bound = {
        "reference": {"provider_id": predecessor["provider_id"],
                      "profile_ref": predecessor["profile_ref"],
                      "account_ref": predecessor["account_ref"],
                      "namespace_ref": predecessor["namespace_ref"],
                      "stage_ref": "modal-stage-claim:" + launch["stage_claim_sha256"]},
        "effect_id": predecessor["stage_effect_id"],
        "command_digest": launch["stage_command_digest"],
        "command_bytes_digest": __import__(
            "tuner.execution.foundation_v2.canonical", fromlist=["domain_digest"],
        ).domain_digest("synaptic-foundation-command-bytes/v1", material.binding.command_bytes),
        "preparation_digest": predecessor["preparation_digest"],
        "foundation_binding_digest": launch["stage_foundation_binding_digest"],
        "foundation_outcome_digest": launch["stage_outcome_digest"],
        "authenticated_receipt_digest": predecessor["authenticated_receipt_digest"],
    }
    launch["stage_bound_reference_digest"] = __import__(
        "tuner.execution.foundation_v2.canonical", fromlist=["domain_digest"],
    ).domain_digest("synaptic-stage-evidence-binding/v1", canonical_bytes(bound))
    claim = canonical_bytes(launch)
    tag = authenticator.sign("modal-launch-claim/v1", claim, material.key_ref)
    with pytest.raises(ValueError, match="worker expectation"):
        admit(values, claim=claim, claim_tag=tag, stage_claim=stage_claim,
              stage_claim_tag=stage_tag, bundle=replacement)


def test_wire_rejects_claim_volume_name_not_retained_profile(monkeypatch):
    values = wire_case(monkeypatch)
    envelope, material, authenticator = values[1], values[2], values[0][6]
    launch = __import__("json").loads(envelope.claim)
    launch["configured_control_volume_ref"] = "other-control"
    claim = canonical_bytes(launch)
    tag = authenticator.sign("modal-launch-claim/v1", claim, material.key_ref)
    with pytest.raises(ValueError, match="Volume names"):
        admit(values, claim=claim, claim_tag=tag)
