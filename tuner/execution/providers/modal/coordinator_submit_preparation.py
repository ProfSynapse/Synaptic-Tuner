"""Host-only semantic preparation of one authenticated Modal submit dispatch."""

from __future__ import annotations

from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.foundation_v2.commands import StageCommandV2, SubmitCommandV2, parse_exact_command
from tuner.training.recipes import RecipeRegistry

from .contracts import BoundsPolicyV1, sha
from .coordinator_binding import ModalCommandBinding
from .coordinator_bundle import ModalCoordinatorBundle
from .coordinator_dispatch import ModalWorkerDispatch, build_modal_worker_dispatch
from .coordinator_launch import ModalLaunchEnvelope, admit_modal_foundation_launch
from .coordinator_staging import ModalStageMaterial
from .coordinator_wire import ModalWorkerLaunchExpectation


def prepare_modal_submit_dispatch(
    envelope: ModalLaunchEnvelope, *, foundation_authenticator,
    assessment_authenticator, binding_authority, stage_verifier,
    launch_verifier, recipes: RecipeRegistry,
    bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalWorkerDispatch:
    """Authenticate host lineage, validate bundle semantics, then encode dispatch."""
    if type(envelope) is not ModalLaunchEnvelope:
        raise TypeError("exact Modal launch envelope required")
    if type(recipes) is not RecipeRegistry:
        raise TypeError("exact recipe registry required")

    material = envelope.stage_material
    submit_binding = envelope.submit_binding
    if type(material) is not ModalStageMaterial or type(submit_binding) is not ModalCommandBinding:
        raise TypeError("exact retained launch values required")
    if any(type(value) is not bytes for value in (
        envelope.claim, envelope.claim_tag, material.claim, material.claim_tag,
        material.bundle, submit_binding.command_bytes,
        submit_binding.preparation_snapshot, submit_binding.deployment_bytes,
    )):
        raise TypeError("exact immutable launch bytes required")
    # Snapshot every byte/value used after host admission. The admitted result
    # must project back to these bytes before semantic bundle work begins.
    launch_claim = bytes(envelope.claim)
    launch_tag = bytes(envelope.claim_tag)
    stage_claim = bytes(material.claim)
    stage_tag = bytes(material.claim_tag)
    bundle = bytes(material.bundle)
    command_bytes = bytes(submit_binding.command_bytes)
    snapshot_bytes = bytes(submit_binding.preparation_snapshot)
    deployment_bytes = bytes(submit_binding.deployment_bytes)
    control_id, artifact_id, key_ref = (
        material.control_volume_id, material.artifact_volume_id, material.key_ref,
    )

    admitted = admit_modal_foundation_launch(
        envelope, foundation_authenticator=foundation_authenticator,
        assessment_authenticator=assessment_authenticator,
        binding_authority=binding_authority, stage_verifier=stage_verifier,
        launch_verifier=launch_verifier, bounds=bounds,
    )
    admitted_projection = (
        admitted.submit_command_bytes, admitted.control_volume_id,
        admitted.artifact_volume_id, admitted.key_ref, admitted.bundle,
        admitted.stage_claim, admitted.stage_claim_tag,
    )
    if admitted_projection != (
        command_bytes, control_id, artifact_id, key_ref, bundle,
        stage_claim, stage_tag,
    ):
        raise ValueError("host launch admission differs from retained snapshot")

    stage_document = parse_canonical_object(stage_claim, name="stage claim")
    stage_command = parse_exact_command(canonical_bytes(stage_document.get("command")))
    submit_command = parse_exact_command(command_bytes)
    if type(stage_command) is not StageCommandV2 or type(submit_command) is not SubmitCommandV2:
        raise ValueError("exact Foundation stage and submit commands required")
    stage_binding = ModalCommandBinding(
        stage_command.canonical_bytes, snapshot_bytes, deployment_bytes,
    )
    parsed_bundle = ModalCoordinatorBundle.parse_transport(
        bundle, binding=stage_binding, recipes=recipes,
    )
    if parsed_bundle.transport_bytes != bundle or parsed_bundle.binding != stage_binding:
        raise ValueError("coordinator bundle differs from authenticated stage binding")

    submit_rebuilt = ModalCommandBinding(command_bytes, snapshot_bytes, deployment_bytes)
    snapshot = parse_canonical_object(snapshot_bytes, name="preparation snapshot")
    profile = snapshot["configuration"]["profile"]
    selection = submit_rebuilt.deployment.selection
    prep = submit_command.preparation
    expectation = ModalWorkerLaunchExpectation(
        submit_command.canonical_bytes, deployment_bytes,
        prep.provider.provider_id, prep.provider.profile_ref,
        prep.scope.account_ref, prep.scope.namespace_ref,
        selection.app_name, selection.function_name, control_id, artifact_id,
        profile["volumes"]["control_ref"], profile["volumes"]["artifact_ref"],
        key_ref, sha(stage_claim), sha(bundle), len(bundle),
        submit_command.executor.executor_id,
        submit_command.executor.implementation_version,
    )
    return build_modal_worker_dispatch(
        expectation, launch_claim, launch_tag, bounds=bounds,
    )


__all__: list[str] = []
