"""Retained configuration is reconstructed independently of submitted hashes."""

from dataclasses import FrozenInstanceError, replace

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef
from tests.execution.providers.test_modal_coordinator_adapter import composed, inputs, Clock
from tests.execution.providers.test_modal_sdk154_adapter import verified
from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.foundation_v2.commands import build_stage_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.providers.modal.coordinator_adapter import ModalPreparationAdapter
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1


def case():
    adapter, _, plan, execution = composed()
    prep = adapter.prepare(plan, TrainingRunRef("run-a", "project-a"), execution)
    command = build_stage_command(prep, "nonce-a", adapter.payload(prep, EffectKind.STAGE),
                                  execution.executor_descriptor)
    values = inputs()
    selection = ModalDeploymentSelectionV1.from_profile(
        values["profile"], binding=values["binding"],
        runtime_environment=values["runtime_environment"],
        timeout_seconds=values["timeout_seconds"],
    )
    return adapter, command, verified(selection)


def test_retained_binding_reconstructs_every_preparation_commitment():
    adapter, command, deployment = case()
    binding = ModalCommandBinding(command.canonical_bytes, adapter.snapshot(),
                                  canonical_bytes(deployment.to_dict()))
    assert binding.command_digest == command.digest
    assert binding.namespace_ref == command.preparation.scope.namespace_ref
    restored = ModalPreparationAdapter.restore(adapter.snapshot(), clock=Clock())
    assert restored.snapshot() == adapter.snapshot()
    assert restored.preflight(restored._snapshot()[2]).ready is False
    with pytest.raises(FrozenInstanceError):
        binding.command_bytes = b"changed"
    detached = binding.deployment
    object.__setattr__(detached.selection, "workspace_ref", "changed")
    assert binding.deployment.selection.workspace_ref == "workspace-a"


@pytest.mark.parametrize("field", ["source_digest", "workload_digest", "runtime_digest",
                                  "resolved_config_digest", "artifact_policy_digest"])
def test_changed_retained_basis_cannot_match_original_command(field):
    adapter, command, deployment = case()
    document = parse_canonical_object(adapter.snapshot(), name="snapshot")
    document["basis"][field] = "f" * 64
    with pytest.raises(ValueError, match="command differs"):
        ModalCommandBinding(command.canonical_bytes, canonical_bytes(document),
                            canonical_bytes(deployment.to_dict()))


@pytest.mark.parametrize("change", ["quote", "volume", "environment", "workspace", "timeout"])
def test_changed_retained_configuration_cannot_match_original_command(change):
    adapter, command, deployment = case()
    document = parse_canonical_object(adapter.snapshot(), name="snapshot")
    configuration = document["configuration"]
    if change == "quote":
        configuration["quote_digest"] = "f" * 64
    elif change == "volume":
        configuration["profile"]["volumes"]["control_ref"] = "other-control"
    elif change == "environment":
        configuration["selection"]["runtime_environment"]["LANG"] = "C"
    elif change == "workspace":
        configuration["selection"]["workspace_ref"] = "other-workspace"
    else:
        configuration["selection"]["timeout_seconds"] += 1
    with pytest.raises(ValueError):
        ModalCommandBinding(command.canonical_bytes, canonical_bytes(document),
                            canonical_bytes(deployment.to_dict()))


def test_self_consistent_other_deployment_is_not_retained_deployment():
    adapter, command, deployment = case()
    other = verified(replace(deployment.selection, workspace_ref="other-workspace"))
    with pytest.raises(ValueError, match="deployment differs"):
        ModalCommandBinding(command.canonical_bytes, adapter.snapshot(),
                            canonical_bytes(other.to_dict()))


def test_restore_rejects_secret_env_and_untrusted_runtime_lock():
    adapter, _, _ = case()
    document = parse_canonical_object(adapter.snapshot(), name="snapshot")
    document["configuration"]["selection"]["runtime_environment"]["HF_TOKEN"] = "synthetic"
    with pytest.raises(ValueError, match="named Modal Secrets"):
        ModalPreparationAdapter.restore(canonical_bytes(document), clock=Clock())
    document = parse_canonical_object(adapter.snapshot(), name="snapshot")
    document["configuration"]["selection"]["image_digest"] = "f" * 64
    with pytest.raises(ValueError):
        ModalPreparationAdapter.restore(canonical_bytes(document), clock=Clock())
