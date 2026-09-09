"""Authenticated Foundation submit launch preparation and admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tuner.execution.coordinator_v1.model import (
    AuthenticatedFoundationRecordAssessmentV1,
    BoundProviderStageRefV1,
    EffectIntentV1,
    FoundationDispositionV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation
from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.foundation_v2.commands import StageCommandV2, SubmitCommandV2, parse_exact_command
from tuner.execution.foundation_v2.repository import EffectRecordV2

from .contracts import BoundsPolicyV1, sha
from .coordinator_binding import ModalCommandBinding
from .coordinator_staging import (
    ModalStageMaterial, _claim_document, modal_stage_provider_ref,
)


class _BindingAuthority(Protocol):
    def authenticate(self, binding: ModalCommandBinding) -> bool: ...


class _Verifier(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


class _Signer(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ModalLaunchEnvelope:
    submit_binding: ModalCommandBinding
    stage_material: ModalStageMaterial
    stage_record: EffectRecordV2
    stage_assessment: AuthenticatedFoundationRecordAssessmentV1
    claim: bytes
    claim_tag: bytes

    def __post_init__(self) -> None:
        if type(self.submit_binding) is not ModalCommandBinding:
            raise TypeError("exact submit binding required")
        if type(self.stage_material) is not ModalStageMaterial:
            raise TypeError("exact stage material required")
        if type(self.stage_record) is not EffectRecordV2:
            raise TypeError("exact Foundation stage record required")
        if type(self.stage_assessment) is not AuthenticatedFoundationRecordAssessmentV1:
            raise TypeError("exact Foundation assessment required")
        if type(self.claim) is not bytes or not self.claim:
            raise ValueError("launch claim must be nonempty bytes")
        if type(self.claim_tag) is not bytes or not self.claim_tag or len(self.claim_tag) > 128:
            raise ValueError("launch claim tag is invalid")


@dataclass(frozen=True, slots=True)
class ModalLaunchAdmission:
    submit_command_bytes: bytes
    submit_command_digest: str
    submit_effect_id: str
    stage_effect_id: str
    control_volume_id: str
    artifact_volume_id: str
    key_ref: str
    bundle: bytes
    stage_claim: bytes
    stage_claim_tag: bytes


def _binding(value: ModalCommandBinding, authority: _BindingAuthority):
    if type(value) is not ModalCommandBinding:
        raise ValueError("exact Modal command binding required")
    rebuilt = ModalCommandBinding(
        value.command_bytes, value.preparation_snapshot, value.deployment_bytes,
    )
    if rebuilt != value or authority.authenticate(rebuilt) is not True:
        raise ValueError("Modal command binding authentication failed")
    return rebuilt, parse_exact_command(rebuilt.command_bytes)


def _stage_proof(envelope, foundation_authenticator, assessment_authenticator,
                 binding_authority, stage_verifier):
    submit_binding, submit = _binding(envelope.submit_binding, binding_authority)
    stage_binding, stage = _binding(
        envelope.stage_material.binding, binding_authority,
    )
    if type(submit) is not SubmitCommandV2 or type(stage) is not StageCommandV2:
        raise ValueError("launch requires exact stage and submit commands")
    material = envelope.stage_material
    if canonical_bytes(_claim_document(material)) != material.claim:
        raise ValueError("stage claim does not bind retained material")
    try:
        stage_valid = stage_verifier.verify(
            "modal-stage-claim/v2", material.claim, material.claim_tag, material.key_ref,
        )
    except Exception:
        raise ValueError("stage claim authentication unavailable") from None
    if stage_valid is not True:
        raise ValueError("stage claim authentication failed")

    intent = EffectIntentV1.from_command_bytes(stage.canonical_bytes)
    derived_binding, outcome, bound = _derive_foundation(
        intent, envelope.stage_record, envelope.stage_assessment,
        foundation_authenticator, assessment_authenticator, None,
    )
    if (outcome.disposition is not FoundationDispositionV1.FOUND
            or type(bound) is not BoundProviderStageRefV1):
        raise ValueError("stage record lacks an authenticated FOUND reference")
    if bound.reference.stage_ref != modal_stage_provider_ref(material):
        raise ValueError("authenticated stage reference does not identify retained material")
    predecessor = submit.stage_predecessor
    prep = submit.preparation
    stage_prep = stage.preparation
    if prep != stage_prep:
        raise ValueError("stage and submit preparations differ")
    expected_predecessor = (
        prep.provider.provider_id, prep.provider.profile_ref,
        prep.scope.account_ref, prep.scope.namespace_ref,
        prep.project_ref, prep.run_id, prep.plan_fingerprint,
        prep.preparation_digest, prep.workload_digest,
        stage.operation.effect.effect_id,
        bound.authenticated_receipt_digest, envelope.stage_record.record_digest,
    )
    actual_predecessor = tuple(
        getattr(predecessor, name) for name in predecessor.__dataclass_fields__
    )
    if actual_predecessor != expected_predecessor:
        raise ValueError("submit stage predecessor is not the authenticated stage result")
    if (stage_binding.preparation_snapshot != submit_binding.preparation_snapshot
            or stage_binding.deployment_bytes != submit_binding.deployment_bytes):
        raise ValueError("stage and submit retained configurations differ")
    return submit_binding, submit, stage_binding, stage, derived_binding, outcome, bound


def _claim(envelope, proof):
    submit_binding, submit, stage_binding, stage, derived_binding, outcome, bound = proof
    material = envelope.stage_material
    snapshot = parse_canonical_object(
        submit_binding.preparation_snapshot, name="preparation snapshot",
    )
    profile = snapshot["configuration"]["profile"]
    return canonical_bytes({
        "schema_version": "synaptic.modal-launch-claim/v1",
        "submit_binding": parse_canonical_object(
            submit_binding.canonical_bytes, name="submit binding",
        ),
        "submit_binding_digest": submit_binding.authenticated_binding_digest,
        "submit_command": submit.to_dict(),
        "submit_command_digest": submit.digest,
        "submit_effect_id": submit.operation.effect.effect_id,
        "submit_invocation_nonce": submit.operation.invocation_nonce,
        "stage_binding_digest": stage_binding.authenticated_binding_digest,
        "stage_command_digest": stage.digest,
        "stage_record_digest": envelope.stage_record.record_digest,
        "stage_assessment_digest": envelope.stage_assessment.authenticated_assessment_digest,
        "stage_foundation_binding_digest": derived_binding.binding_digest,
        "stage_outcome_digest": outcome.outcome_digest,
        "stage_bound_reference_digest": bound.binding_digest,
        "stage_predecessor": submit.stage_predecessor.to_dict(),
        "stage_claim_sha256": sha(material.claim),
        "stage_bundle_sha256": sha(material.bundle),
        "stage_bundle_size": len(material.bundle),
        "control_volume_id": material.control_volume_id,
        "artifact_volume_id": material.artifact_volume_id,
        "configured_control_volume_ref": profile["volumes"]["control_ref"],
        "configured_artifact_volume_ref": profile["volumes"]["artifact_ref"],
        "key_ref": material.key_ref,
    })


def prepare_modal_foundation_launch(
    *, submit_binding: ModalCommandBinding, stage_material: ModalStageMaterial,
    stage_record: EffectRecordV2,
    stage_assessment: AuthenticatedFoundationRecordAssessmentV1,
    foundation_authenticator, assessment_authenticator,
    binding_authority: _BindingAuthority, stage_verifier: _Verifier,
    signer: _Signer, bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalLaunchEnvelope:
    provisional = ModalLaunchEnvelope(
        submit_binding, stage_material, stage_record, stage_assessment, b"{}", b"x",
    )
    proof = _stage_proof(
        provisional, foundation_authenticator, assessment_authenticator,
        binding_authority, stage_verifier,
    )
    claim = _claim(provisional, proof)
    if len(claim) > bounds.max_control_bytes:
        raise ValueError("launch claim exceeds control bound")
    try:
        tag = signer.sign("modal-launch-claim/v1", claim, stage_material.key_ref)
    except Exception:
        raise ValueError("launch claim authentication unavailable") from None
    return ModalLaunchEnvelope(
        proof[0], stage_material, stage_record, stage_assessment, claim, tag,
    )


def admit_modal_foundation_launch(
    envelope: ModalLaunchEnvelope, *, foundation_authenticator,
    assessment_authenticator, binding_authority: _BindingAuthority,
    stage_verifier: _Verifier, launch_verifier: _Verifier,
    bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalLaunchAdmission:
    if type(envelope) is not ModalLaunchEnvelope:
        raise TypeError("exact Modal launch envelope required")
    rebuilt = ModalLaunchEnvelope(
        envelope.submit_binding, envelope.stage_material, envelope.stage_record,
        envelope.stage_assessment, envelope.claim, envelope.claim_tag,
    )
    proof = _stage_proof(
        rebuilt, foundation_authenticator, assessment_authenticator,
        binding_authority, stage_verifier,
    )
    expected = _claim(rebuilt, proof)
    if (len(expected) > bounds.max_control_bytes or expected != rebuilt.claim
            or len(rebuilt.stage_material.bundle) > bounds.max_bundle_bytes):
        raise ValueError("launch claim does not bind retained inputs")
    try:
        valid = launch_verifier.verify(
            "modal-launch-claim/v1", rebuilt.claim, rebuilt.claim_tag,
            rebuilt.stage_material.key_ref,
        )
    except Exception:
        raise ValueError("launch claim authentication unavailable") from None
    if valid is not True:
        raise ValueError("launch claim authentication failed")
    submit = proof[1]
    return ModalLaunchAdmission(
        submit.canonical_bytes, submit.digest, submit.operation.effect.effect_id,
        submit.stage_predecessor.stage_effect_id,
        rebuilt.stage_material.control_volume_id,
        rebuilt.stage_material.artifact_volume_id, rebuilt.stage_material.key_ref,
        rebuilt.stage_material.bundle, rebuilt.stage_material.claim,
        rebuilt.stage_material.claim_tag,
    )


__all__: list[str] = []
