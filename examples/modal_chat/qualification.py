"""One-shot, consumer-owned qualification of an accepted Modal training run."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.coordinator_v1.coordinator import (
    ApplyArtifactVerificationTransitionV1,
    ApplyProviderObservationTransitionV1,
)
from tuner.execution.coordinator_v1.model import (
    AuthenticatedFoundationRecordAssessmentV1,
    ProviderReadPurposeV1,
    WorkflowPhaseV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    apply_artifact_verification,
    apply_provider_observation,
    provider_run_read_request,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
)
from tuner.execution.providers.modal.contracts import ArtifactRole

from .host import ModalChatHost
from .storage import ModalChatStorage


class ModalChatQualificationError(RuntimeError):
    """Closed qualification failure."""


@dataclass(frozen=True, slots=True)
class ModalChatQualification:
    workflow: object
    accepted_evidence: bytes
    observation_evidence: bytes
    verification_evidence: bytes


def _object(raw: bytes, name: str) -> object:
    return parse_canonical_object(raw, name=name)


def _publish(
    storage: ModalChatStorage, catalog_ref: str, item_ref: str, raw: bytes
) -> None:
    catalog = storage.catalog(
        catalog_ref, encode=lambda value: value, decode=lambda value: value
    )
    if catalog.publish_if_absent(item_ref, raw) is not True:
        raise ModalChatQualificationError("modal_chat_qualification_invalid")


def _retain_part(storage: ModalChatStorage, raw: bytes) -> dict[str, str]:
    digest = hashlib.sha256(raw).hexdigest()
    item_ref = "sha256-" + digest
    catalog_ref = "modal-chat-qualification-parts"
    catalog = storage.catalog(
        catalog_ref, encode=lambda value: value, decode=lambda value: value
    )
    catalog.publish_if_absent(item_ref, raw)
    if catalog.resolve(item_ref) != raw:
        raise ModalChatQualificationError("modal_chat_qualification_invalid")
    return {"catalog_ref": catalog_ref, "item_ref": item_ref, "sha256": digest}
    if catalog.resolve(item_ref) != raw:
        raise ModalChatQualificationError("modal_chat_qualification_invalid")


def qualify_modal_chat_run(
    *,
    host: ModalChatHost,
    storage: ModalChatStorage,
    run: TrainingRunRef,
) -> ModalChatQualification:
    """Observe and verify one already-submitted run, exactly once.

    This function never submits, retries, polls, or enables a public provider surface.
    Its permanent attempt claim is intentionally written before provider observation.
    """
    if (
        type(host) is not ModalChatHost
        or type(storage) is not ModalChatStorage
        or type(run) is not TrainingRunRef
    ):
        raise TypeError("exact Modal chat qualification inputs required")
    try:
        store = host.stores.workflow_store
        before = store.get(run)
        if (
            before is None
            or before.phase not in {WorkflowPhaseV1.QUEUED, WorkflowPhaseV1.RUNNING}
            or before.submit is None
            or before.provider_run_ref is None
        ):
            raise ValueError
        record = host.composition.foundation.get(before.submit.effect_id)
        if not before.submit.foundation_bindings:
            raise ValueError
        retained_binding = before.submit.foundation_bindings[-1]
        assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
            retained_binding.canonical_assessment_bytes
        )
        if (
            assessment.canonical_bytes != retained_binding.canonical_assessment_bytes
            or assessment.authenticated_assessment_digest
            != retained_binding.assessment_digest
            or host.foundation_ports.assessment_authority.authenticate(assessment)
            is not True
        ):
            raise ValueError
        request = provider_run_read_request(
            before,
            record,
            assessment,
            host.foundation_ports.foundation_authenticator,
            host.foundation_ports.assessment_authority,
            purpose=ProviderReadPurposeV1.OBSERVE,
        )
        attempt_ref = (
            "qualify-" + hashlib.sha256(canonical_bytes(run.to_dict())).hexdigest()
        )
        accepted_parts = {
            "submit-command": before.submit.canonical_command_bytes,
            "foundation-snapshot": request.foundation_binding.canonical_snapshot_bytes,
            "foundation-assessment": assessment.canonical_bytes,
            "provider-read-request": request.canonical_bytes,
        }
        accepted = canonical_bytes(
            {
                "schema_version": "modal-chat-qualification-accepted/v1",
                "workflow": before.to_dict(),
                "workflow_record_digest": before.record_digest,
                "parts": {
                    name: {
                        "catalog_ref": "modal-chat-qualification-parts",
                        "item_ref": "sha256-" + hashlib.sha256(raw).hexdigest(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                    for name, raw in accepted_parts.items()
                },
            }
        )
        claim = storage.attempts.claim(attempt_ref, accepted)
        if storage.attempts.resolve(attempt_ref) != claim:
            raise ValueError
        for raw in accepted_parts.values():
            _retain_part(storage, raw)

        observation = host.reader.observe(request)
        after_observation = apply_provider_observation(
            before,
            request,
            observation,
            host.observation_authenticator,
        )
        terminal_phases = {
            WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,
            WorkflowPhaseV1.FAILED,
            WorkflowPhaseV1.CANCELLED,
        }
        if after_observation.phase not in terminal_phases:
            raise ValueError
        transition = ApplyProviderObservationTransitionV1(request, observation)
        if (
            store.compare_and_swap(before, after_observation, transition=transition)
            is not True
        ):
            raise ValueError
        if store.get(run) != after_observation:
            raise ValueError
        observed_parts = {
            "pre-workflow": canonical_bytes(before.to_dict()),
            "post-workflow": canonical_bytes(after_observation.to_dict()),
            "foundation-snapshot": request.foundation_binding.canonical_snapshot_bytes,
            "foundation-assessment": assessment.canonical_bytes,
            "observation": observation.canonical_bytes,
        }
        observed = canonical_bytes(
            {
                "schema_version": "modal-chat-qualification-observation/v1",
                "parts": {
                    name: _retain_part(storage, raw)
                    for name, raw in observed_parts.items()
                },
            }
        )
        _publish(
            storage, "modal-chat-qualification-observations", attempt_ref, observed
        )
        if after_observation.phase is not WorkflowPhaseV1.SUCCEEDED_UNVERIFIED:
            raise ModalChatQualificationError("modal_chat_qualification_invalid")

        artifact_request = provider_run_read_request(
            after_observation,
            record,
            assessment,
            host.foundation_ports.foundation_authenticator,
            host.foundation_ports.assessment_authority,
            purpose=ProviderReadPurposeV1.ARTIFACTS,
        )
        binding, provider_ref, inventory, manifest = host.reader.native_artifacts(
            artifact_request
        )
        if provider_ref != after_observation.provider_run_ref.reference:
            raise ValueError
        receipt = host.artifact_verifier.verify(after_observation, manifest)
        verified = apply_artifact_verification(
            after_observation,
            manifest,
            receipt,
            host.artifact_verifier,
        )
        expected_roles = {role.value for role in ArtifactRole}
        if (
            verified.phase is not WorkflowPhaseV1.VERIFIED
            or len(verified.verified_artifacts) != len(expected_roles)
            or {item.role for item in verified.verified_artifacts} != expected_roles
        ):
            raise ValueError
        verify_transition = ApplyArtifactVerificationTransitionV1(manifest, receipt)
        if (
            store.compare_and_swap(
                after_observation, verified, transition=verify_transition
            )
            is not True
        ):
            raise ValueError
        if store.get(run) != verified:
            raise ValueError
        manifest_bytes = canonical_bytes(
            {
                "run": manifest.run.to_dict(),
                "provider_run": manifest.provider_run.to_dict(),
                "artifacts": [item.to_dict() for item in manifest.artifacts],
                "artifact_source_digest": manifest.artifact_source_digest,
                "canonical_evidence": _object(
                    manifest.canonical_evidence, "artifact manifest evidence"
                ),
                "manifest_digest": manifest.manifest_digest,
            }
        )
        final_parts = {
            "pre-workflow": canonical_bytes(before.to_dict()),
            "observed-workflow": canonical_bytes(after_observation.to_dict()),
            "verified-workflow": canonical_bytes(verified.to_dict()),
            "provider-read-request": artifact_request.canonical_bytes,
            "native-inventory": inventory.canonical_evidence,
            "artifact-manifest": manifest_bytes,
            "verification-receipt": receipt.canonical_bytes,
        }
        final = canonical_bytes(
            {
                "schema_version": "modal-chat-native-qualification/v1",
                "parts": {
                    name: _retain_part(storage, raw)
                    for name, raw in final_parts.items()
                },
                "native_command_binding": {
                    "command_digest": binding.command_digest,
                    "command_bytes_sha256": hashlib.sha256(
                        binding.command_bytes
                    ).hexdigest(),
                    "preparation_snapshot_sha256": hashlib.sha256(
                        binding.preparation_snapshot
                    ).hexdigest(),
                    "deployment_bytes_sha256": hashlib.sha256(
                        binding.deployment_bytes
                    ).hexdigest(),
                },
            }
        )
        _publish(storage, "modal-chat-native-qualifications", attempt_ref, final)
        return ModalChatQualification(verified, accepted, observed, final)
    except ModalChatQualificationError:
        raise
    except Exception:
        raise ModalChatQualificationError("modal_chat_qualification_invalid") from None


__all__ = [
    "ModalChatQualification",
    "ModalChatQualificationError",
    "qualify_modal_chat_run",
]
