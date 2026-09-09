"""Explicit-client operational transport for Foundation-native Modal effects."""

from __future__ import annotations

from typing import Protocol

from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object, safe_ref
from tuner.execution.foundation_v2.commands import (
    CancelCommandV2, StageCommandV2, SubmitCommandV2, parse_exact_command,
)
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.training.recipes import RecipeRegistry

from .contracts import BoundsPolicyV1
from .coordinator_binding import ModalCommandBinding
from .coordinator_effects import ModalEffectOutcome
from .coordinator_launch import ModalLaunchEnvelope
from .coordinator_staging import (
    ModalFoundationVolumeWriter, ModalStageMaterial, _claim_document,
    modal_stage_provider_ref,
)
from .coordinator_submit_preparation import prepare_modal_submit_dispatch
from .facade import ExplicitModal154ReadFacade
from .config import ModalRuntimeLockV1
from .resolution import VerifiedModalDeploymentIdentityV1


class _StageMaterialSource(Protocol):
    def resolve(self, command_digest: str) -> ModalStageMaterial: ...


class _LaunchEnvelopeSource(Protocol):
    def resolve(self, command_digest: str) -> ModalLaunchEnvelope: ...


class ModalFoundationHostTransport:
    """Make one submit-spawn or cancel attempt; stage uses exact bounded writes."""

    def __init__(
        self, *, facade: ExplicitModal154ReadFacade,
        deployment: VerifiedModalDeploymentIdentityV1,
        stage_source: _StageMaterialSource, launch_source: _LaunchEnvelopeSource,
        binding_authority, foundation_authenticator, assessment_authenticator,
        stage_verifier, launch_verifier, recipes: RecipeRegistry,
        bounds: BoundsPolicyV1 = BoundsPolicyV1(),
    ) -> None:
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact explicit Modal facade required")
        if type(deployment) is not VerifiedModalDeploymentIdentityV1:
            raise TypeError("exact verified Modal deployment required")
        if type(recipes) is not RecipeRegistry:
            raise TypeError("exact recipe registry required")
        self._facade = facade
        self._deployment = deployment
        self._stage_source = stage_source
        self._launch_source = launch_source
        self._authority = binding_authority
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._stage_verifier = stage_verifier
        self._launch_verifier = launch_verifier
        self._recipes = recipes
        self._bounds = bounds

    def _binding(self, supplied: object, supplied_command: object):
        if type(supplied) is not ModalCommandBinding:
            raise ValueError("exact Modal command binding required")
        rebuilt = ModalCommandBinding(
            supplied.command_bytes, supplied.preparation_snapshot,
            supplied.deployment_bytes,
        )
        command = parse_exact_command(rebuilt.command_bytes)
        if (
            rebuilt != supplied or type(command) is not type(supplied_command)
            or command.canonical_bytes != supplied_command.canonical_bytes
            or self._authority.authenticate(rebuilt) is not True
            or rebuilt.deployment != self._deployment
            or rebuilt.client_binding != self._facade.binding
        ):
            raise ValueError("Modal operational binding mismatch")
        return rebuilt, command

    def _provider_boundary(
        self, binding: ModalCommandBinding, volume_ids: tuple[str, str] | None = None,
    ) -> None:
        selection = self._deployment.selection
        if volume_ids is not None:
            profile = parse_canonical_object(
                binding.preparation_snapshot, name="preparation snapshot",
            )["configuration"]["profile"]
            if (
                self._facade.volume_name(volume_ids[0])
                != profile["volumes"]["control_ref"]
                or self._facade.volume_name(volume_ids[1])
                != profile["volumes"]["artifact_ref"]
            ):
                raise ValueError("Modal operational volume mismatch")
        self._facade.bound_scope()
        if self._facade.inspect_deployment(
            app_name=selection.app_name, function_name=selection.function_name,
        ) != selection:
            raise ValueError("Modal operational deployment mismatch")

    def _stage_material(self, value: object, binding: ModalCommandBinding):
        if type(value) is not ModalStageMaterial:
            raise ValueError("exact retained stage material required")
        material = ModalStageMaterial(
            ModalCommandBinding(
                value.binding.command_bytes, value.binding.preparation_snapshot,
                value.binding.deployment_bytes,
            ), value.control_volume_id, value.artifact_volume_id, value.key_ref,
            value.bundle, value.claim, value.claim_tag,
        )
        if material != value or material.binding != binding:
            raise ValueError("retained stage material mismatch")
        if (
            len(material.bundle) > self._bounds.max_bundle_bytes
            or len(material.claim) > self._bounds.max_control_bytes
            or canonical_bytes(_claim_document(material)) != material.claim
        ):
            raise ValueError("retained stage material is invalid")
        try:
            valid = self._stage_verifier.verify(
                "modal-stage-claim/v2", material.claim, material.claim_tag,
                material.key_ref,
            )
        except Exception:
            raise ValueError("stage authentication unavailable") from None
        if valid is not True:
            raise ValueError("stage authentication failed")
        return material

    def execute_once(self, binding: object, command: object) -> ModalEffectOutcome:
        rebuilt, parsed = self._binding(binding, command)
        if type(parsed) is StageCommandV2:
            material = self._stage_material(
                self._stage_source.resolve(parsed.digest), rebuilt,
            )
            self._provider_boundary(
                rebuilt, (material.control_volume_id, material.artifact_volume_id),
            )
            writer = ModalFoundationVolumeWriter(
                self._facade, self._authority, self._stage_verifier,
                bounds=self._bounds,
            )
            try:
                writer.stage_once(material)
            except Exception:
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            return ModalEffectOutcome(
                ObservationDisposition.FOUND, modal_stage_provider_ref(material),
            )
        if type(parsed) is SubmitCommandV2:
            envelope = self._launch_source.resolve(parsed.digest)
            if type(envelope) is not ModalLaunchEnvelope or envelope.submit_binding != rebuilt:
                raise ValueError("retained launch envelope mismatch")
            dispatch = prepare_modal_submit_dispatch(
                envelope,
                foundation_authenticator=self._foundation_authenticator,
                assessment_authenticator=self._assessment_authenticator,
                binding_authority=self._authority,
                stage_verifier=self._stage_verifier,
                launch_verifier=self._launch_verifier,
                recipes=self._recipes, bounds=self._bounds,
            )
            self._provider_boundary(rebuilt, (
                envelope.stage_material.control_volume_id,
                envelope.stage_material.artifact_volume_id,
            ))
            selection = self._deployment.selection
            function = self._facade._function(
                app_name=selection.app_name, function_name=selection.function_name,
            )
            try:
                call = function.spawn(dispatch.canonical_bytes)
                provider_ref = safe_ref(getattr(call, "object_id", None), "provider_job_ref")
            except Exception:
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            return ModalEffectOutcome(ObservationDisposition.FOUND, provider_ref)
        if type(parsed) is CancelCommandV2:
            cancellation = parsed.to_dict()["cancellation"]
            target = safe_ref(cancellation["provider_job_ref"], "provider_job_ref")
            # The reviewed, hash-locked deployment wrapper configures this
            # target as a single-use container with provider retries disabled.
            if self._deployment.selection.wrapper_digest != (
                ModalRuntimeLockV1.packaged().locked_digest("deployment_wrapper")
            ):
                raise ValueError("cancel target is not the locked single-use deployment")
            self._provider_boundary(rebuilt)
            try:
                call = self._facade.sdk.FunctionCall.from_id(
                    target, client=self._facade.client,
                )
                if safe_ref(getattr(call, "object_id", None), "provider_job_ref") != target:
                    raise ValueError("Modal cancel handle identity mismatch")
                call.cancel(terminate_containers=True)
            except Exception:
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            return ModalEffectOutcome(ObservationDisposition.FOUND, target)
        raise ValueError("unsupported exact Modal command")

    def lookup_once(self, binding: object, command: object) -> ModalEffectOutcome:
        # Authentication is repeated, but no provider read can rediscover a lost
        # submit call ID or prove absence from the command alone.
        self._binding(binding, command)
        return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)


__all__: list[str] = []
