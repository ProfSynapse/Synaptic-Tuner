"""Consumer-owned retention before Foundation-native Modal execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tuner.execution.foundation_v2.authority import AuthenticatedGrantV2
from tuner.execution.foundation_v2.canonical import canonical_bytes, safe_ref
from tuner.execution.foundation_v2.commands import (
    CancelCommandV2,
    StageCommandV2,
    SubmitCommandV2,
    parse_exact_command,
)
from tuner.execution.foundation_v2.repository import EffectRecordV2
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
from tuner.training.recipes import RecipeRegistry

from .contracts import BoundsPolicyV1
from .coordinator_binding import ModalCommandBinding
from .coordinator_bundle import ModalCoordinatorBundle
from .coordinator_launch import (
    ModalLaunchEnvelope,
    admit_modal_foundation_launch,
    prepare_modal_foundation_launch,
)
from .coordinator_staging import (
    ModalStageMaterial,
    _claim_document,
    prepare_modal_foundation_stage,
)


class _Catalog(Protocol):
    def resolve(self, key: str) -> object | None: ...
    def publish_if_absent(self, key: str, value: object) -> None: ...


class _RetainedInputSource(Protocol):
    def resolve(self, preparation_digest: str) -> "ModalRetainedPreparation": ...


@dataclass(frozen=True, slots=True)
class ModalRetainedPreparation:
    """Exact consumer-retained inputs needed to reconstruct a stage bundle."""

    preparation_snapshot: bytes
    deployment_bytes: bytes
    material_bytes: bytes
    recipes: RecipeRegistry
    log_terminal_policy: bytes
    worker_closure_manifest: bytes
    control_volume_id: str
    artifact_volume_id: str
    key_ref: str

    def __post_init__(self) -> None:
        if any(type(value) is not bytes for value in (
            self.preparation_snapshot, self.deployment_bytes, self.material_bytes,
            self.log_terminal_policy, self.worker_closure_manifest,
        )):
            raise TypeError("retained Modal inputs require exact immutable bytes")
        if type(self.recipes) is not RecipeRegistry:
            raise TypeError("retained Modal inputs require an exact recipe registry")
        CoordinatorResolvedMaterial.parse(self.material_bytes, self.recipes)
        for name in ("control_volume_id", "artifact_volume_id", "key_ref"):
            safe_ref(getattr(self, name), name)
        if self.control_volume_id == self.artifact_volume_id:
            raise ValueError("retained Modal volumes must differ")

    @property
    def material(self) -> CoordinatorResolvedMaterial:
        return CoordinatorResolvedMaterial.parse(self.material_bytes, self.recipes)


def _binding(command_bytes, retained, authority=None):
    candidate = ModalCommandBinding(
        command_bytes, retained.preparation_snapshot, retained.deployment_bytes,
    )
    if authority is not None and authority.authenticate(candidate) is not True:
        raise ValueError("Modal command binding authentication failed")
    return candidate


def _retained(catalog, key, candidate, expected_type, validate):
    value = catalog.resolve(key)
    if value is None:
        catalog.publish_if_absent(key, candidate)
        value = catalog.resolve(key)
    if type(value) is not expected_type:
        raise ValueError("retained Modal value is unavailable or invalid")
    rebuilt = validate(value)
    if rebuilt != value or rebuilt != candidate:
        raise ValueError("retained Modal publication conflict")
    return rebuilt


class ModalFoundationRetentionDelegate:
    """Retain authenticated Modal facts, then call an unchanged Foundation."""

    def __init__(
        self, foundation, *, foundation_authenticator, assessment_authority,
        binding_authority, stage_authority, launch_authority,
        binding_catalog: _Catalog, stage_catalog: _Catalog,
        launch_catalog: _Catalog, retained_inputs: _RetainedInputSource,
        bounds: BoundsPolicyV1 = BoundsPolicyV1(),
    ) -> None:
        if type(bounds) is not BoundsPolicyV1:
            raise TypeError("exact Modal bounds required")
        self._foundation = foundation
        self._foundation_authenticator = foundation_authenticator
        self._assessments = assessment_authority
        self._bindings, self._stages, self._launches = (
            binding_catalog, stage_catalog, launch_catalog,
        )
        self._inputs = retained_inputs
        self._binding_authority = binding_authority
        self._stage_authority = stage_authority
        self._launch_authority = launch_authority
        self._bounds = bounds

    def get(self, effect_id: str):
        return self._foundation.get(effect_id)

    def assess(self, record):
        """Preserve the composed Foundation's existing assessment surface."""
        return self._foundation.assess(record)

    def authenticate(self, assessment):
        """Preserve the composed Foundation's existing assessment verifier."""
        return self._foundation.authenticate(assessment)

    def reconcile(self, command_bytes, grant, *, now_epoch, continuation=None):
        return self._foundation.reconcile(
            command_bytes, grant, now_epoch=now_epoch, continuation=continuation,
        )

    def recover_orphan(self, effect_id: str, *, now_epoch: int):
        return self._foundation.recover_orphan(effect_id, now_epoch=now_epoch)

    def _stage(self, command, binding, retained):
        bundle = ModalCoordinatorBundle.build(
            binding, retained.material, retained.recipes,
            log_terminal_policy=retained.log_terminal_policy,
            worker_closure_manifest=retained.worker_closure_manifest,
        )
        existing = self._stages.resolve(command.digest)
        candidate = None
        if existing is None:
            candidate = prepare_modal_foundation_stage(
                binding, bundle.transport_bytes, self._binding_authority,
                self._stage_authority,
                control_volume_id=retained.control_volume_id,
                artifact_volume_id=retained.artifact_volume_id,
                key_ref=retained.key_ref, bounds=self._bounds,
            )
            self._stages.publish_if_absent(command.digest, candidate)
            existing = self._stages.resolve(command.digest)
        if type(existing) is not ModalStageMaterial:
            raise ValueError("retained Modal stage is unavailable")
        rebuilt = ModalStageMaterial(
            existing.binding, existing.control_volume_id, existing.artifact_volume_id,
            existing.key_ref, existing.bundle, existing.claim, existing.claim_tag,
        )
        try:
            verified = self._stage_authority.verify(
                "modal-stage-claim/v2", rebuilt.claim, rebuilt.claim_tag,
                rebuilt.key_ref,
            )
        except Exception:
            raise ValueError("retained Modal stage authentication unavailable") from None
        if (
            rebuilt != existing or (candidate is not None and rebuilt != candidate)
            or rebuilt.binding != binding
            or rebuilt.bundle != bundle.transport_bytes
            or rebuilt.control_volume_id != retained.control_volume_id
            or rebuilt.artifact_volume_id != retained.artifact_volume_id
            or rebuilt.key_ref != retained.key_ref
            or canonical_bytes(_claim_document(rebuilt)) != rebuilt.claim
            or verified is not True
        ):
            raise ValueError("retained Modal stage conflict")

    def _validate_submit_stage(self, material, retained):
        if type(material) is not ModalStageMaterial:
            raise ValueError("retained Modal stage material unavailable")
        stage_binding = _binding(
            material.binding.command_bytes, retained, self._binding_authority,
        )
        if type(parse_exact_command(stage_binding.command_bytes)) is not StageCommandV2:
            raise ValueError("retained Modal stage binding is invalid")
        bundle = ModalCoordinatorBundle.build(
            stage_binding, retained.material, retained.recipes,
            log_terminal_policy=retained.log_terminal_policy,
            worker_closure_manifest=retained.worker_closure_manifest,
        )
        rebuilt = ModalStageMaterial(
            stage_binding, material.control_volume_id, material.artifact_volume_id,
            material.key_ref, material.bundle, material.claim, material.claim_tag,
        )
        try:
            verified = self._stage_authority.verify(
                "modal-stage-claim/v2", rebuilt.claim, rebuilt.claim_tag,
                rebuilt.key_ref,
            )
        except Exception:
            raise ValueError("retained Modal stage authentication unavailable") from None
        if (
            rebuilt != material or rebuilt.bundle != bundle.transport_bytes
            or rebuilt.control_volume_id != retained.control_volume_id
            or rebuilt.artifact_volume_id != retained.artifact_volume_id
            or rebuilt.key_ref != retained.key_ref
            or canonical_bytes(_claim_document(rebuilt)) != rebuilt.claim
            or verified is not True
        ):
            raise ValueError("retained Modal stage conflict")
        return rebuilt

    def _submit(self, command, binding, retained):
        stage_record = self._foundation.get(command.stage_predecessor.stage_effect_id)
        if (
            type(stage_record) is not EffectRecordV2
            or stage_record.record_digest != command.stage_predecessor.record_digest
        ):
            raise ValueError("retained Foundation stage record mismatch")
        stage_command = parse_exact_command(stage_record.command_bytes)
        if type(stage_command) is not StageCommandV2:
            raise ValueError("retained Foundation predecessor is not a stage")
        material = self._validate_submit_stage(
            self._stages.resolve(stage_command.digest), retained,
        )

        existing = self._launches.resolve(command.digest)
        candidate = None
        if existing is None:
            assessment = self._assessments.assess(stage_record)
            candidate = prepare_modal_foundation_launch(
                submit_binding=binding, stage_material=material,
                stage_record=stage_record, stage_assessment=assessment,
                foundation_authenticator=self._foundation_authenticator,
                assessment_authenticator=self._assessments,
                binding_authority=self._binding_authority,
                stage_verifier=self._stage_authority,
                signer=self._launch_authority, bounds=self._bounds,
            )
            self._launches.publish_if_absent(command.digest, candidate)
            existing = self._launches.resolve(command.digest)
        if type(existing) is not ModalLaunchEnvelope:
            raise ValueError("retained Modal launch unavailable")
        rebuilt = ModalLaunchEnvelope(
            existing.submit_binding, existing.stage_material, existing.stage_record,
            existing.stage_assessment, existing.claim, existing.claim_tag,
        )
        if (
            rebuilt != existing or rebuilt.submit_binding != binding
            or (candidate is not None and rebuilt != candidate)
            or rebuilt.stage_material != material or rebuilt.stage_record != stage_record
        ):
            raise ValueError("retained Modal launch conflict")
        admit_modal_foundation_launch(
            rebuilt, foundation_authenticator=self._foundation_authenticator,
            assessment_authenticator=self._assessments,
            binding_authority=self._binding_authority,
            stage_verifier=self._stage_authority,
            launch_verifier=self._launch_authority, bounds=self._bounds,
        )

    def execute(
        self, command_bytes: bytes, grant: AuthenticatedGrantV2, *, now_epoch: int,
    ) -> EffectRecordV2:
        if type(command_bytes) is not bytes or type(grant) is not AuthenticatedGrantV2:
            raise ValueError("exact Foundation execution inputs required")
        command = parse_exact_command(command_bytes)
        if self._foundation_authenticator.authenticate_grant(grant, command_bytes) is not True:
            raise ValueError("Foundation grant authentication failed")
        retained = self._inputs.resolve(command.preparation.preparation_digest)
        if type(retained) is not ModalRetainedPreparation:
            raise ValueError("retained Modal preparation unavailable")
        retained = ModalRetainedPreparation(**{
            name: getattr(retained, name) for name in retained.__dataclass_fields__
        })
        # A new binding has no catalog authority until the consumer publishes
        # it. The full-command grant authorizes this bounded retention step;
        # only the exact readback is subsequently trusted as authenticated.
        candidate = _binding(command.canonical_bytes, retained)
        binding = _retained(
            self._bindings, command.digest, candidate, ModalCommandBinding,
            lambda value: _binding(value.command_bytes, retained, self._binding_authority),
        )
        if type(command) is StageCommandV2:
            self._stage(command, binding, retained)
        elif type(command) is SubmitCommandV2:
            self._submit(command, binding, retained)
        elif type(command) is not CancelCommandV2:
            raise ValueError("unsupported Foundation command")
        return self._foundation.execute(command_bytes, grant, now_epoch=now_epoch)


__all__: list[str] = []
