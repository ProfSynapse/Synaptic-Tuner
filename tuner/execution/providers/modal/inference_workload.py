"""Authenticate retained Modal workload identity without reading model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from inspect import getattr_static
import re

from tuner.execution.coordinator_v1.model import ArtifactManifestV1
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    domain_digest,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import StageCommandV2, parse_exact_command
from tuner.execution.foundation_v2.references import ScopedProviderRunRefV1
from tuner.training.recipes import CompiledWorkload, RecipeRegistry

from .contracts import ArtifactRole, BoundsPolicyV1, sha
from .coordinator_binding import ModalCommandBinding
from .coordinator_bundle import ModalCoordinatorBundle
from .coordinator_launch import ModalLaunchEnvelope
from .coordinator_staging import ModalStageMaterial
from .coordinator_submit_preparation import prepare_modal_submit_dispatch
from .inference_binding import ModalInferenceSourceBinding


_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_TOKEN = object()


class ModalInferenceWorkloadError(RuntimeError):
    """Closed non-secret retained workload admission failure."""


def _source_snapshot(source: ModalInferenceSourceBinding) -> bytes:
    return canonical_bytes(
        {
            "run": source.run.to_dict(),
            "artifacts": [item.to_dict() for item in source.artifacts],
            "read_request_bytes_sha256": sha(source.read_request_bytes),
            "read_request_digest": source.read_request_digest,
            "source_workflow_record_digest": source.source_workflow_record_digest,
            "source_revision": source.source_revision,
            "manifest_digest": source.manifest_digest,
            "artifact_source_digest": source.artifact_source_digest,
            "provider_id": source.provider_id,
            "profile_ref": source.profile_ref,
            "account_ref": source.account_ref,
            "namespace_ref": source.namespace_ref,
            "provider_job_ref": source.provider_job_ref,
            "effect_id": source.effect_id,
            "provider_run_binding_digest": source.provider_run_binding_digest,
            "command_binding_digest": source.command_binding.authenticated_binding_digest,
            "command_bytes_sha256": sha(source.command_binding.command_bytes),
            "preparation_snapshot_sha256": sha(
                source.command_binding.preparation_snapshot
            ),
            "deployment_bytes_sha256": sha(source.command_binding.deployment_bytes),
            "artifact_volume_id": source.artifact_volume_id,
            "native_members": [
                {
                    "role": item.role.value,
                    "path": item.path,
                    "size": item.size,
                    "sha256": item.sha256,
                    "provider_entry_id": item.provider_entry_id,
                }
                for item in source.native_members
            ],
            "native_evidence_sha256": sha(source.native_evidence),
        }
    )


def _envelope_snapshot(envelope: ModalLaunchEnvelope) -> tuple[object, ...]:
    material = envelope.stage_material
    return (
        bytes(envelope.submit_binding.command_bytes),
        bytes(envelope.submit_binding.preparation_snapshot),
        bytes(envelope.submit_binding.deployment_bytes),
        bytes(envelope.claim),
        bytes(envelope.claim_tag),
        bytes(material.binding.command_bytes),
        bytes(material.binding.preparation_snapshot),
        bytes(material.binding.deployment_bytes),
        material.control_volume_id,
        material.artifact_volume_id,
        material.key_ref,
        bytes(material.bundle),
        bytes(material.claim),
        bytes(material.claim_tag),
        envelope.stage_record.record_digest,
        bytes(envelope.stage_assessment.canonical_bytes),
    )


def _model(workload: CompiledWorkload) -> tuple[str, str, str, bool]:
    document = workload.document
    configuration = document.get("configuration")
    identities = document.get("identities")
    if type(configuration) is not dict or type(identities) is not dict:
        raise ValueError("workload model identity is unavailable")
    config_document = configuration.get("document")
    if type(config_document) is not dict:
        raise ValueError("workload configuration is invalid")
    configured = config_document.get("model")
    identified = identities.get("model")
    expected = {"ref", "revision", "tokenizer_revision", "load_in_4bit"}
    if (
        type(configured) is not dict
        or set(configured) != expected
        or configured != identified
    ):
        raise ValueError("workload model identities differ")
    model_ref = configured["ref"]
    revision = configured["revision"]
    tokenizer_revision = configured["tokenizer_revision"]
    load_in_4bit = configured["load_in_4bit"]
    if type(model_ref) is not str:
        raise TypeError("model ref must be an exact string")
    safe_ref(model_ref, "model_ref")
    if (
        type(revision) is not str
        or _REVISION.fullmatch(revision) is None
        or type(tokenizer_revision) is not str
        or _REVISION.fullmatch(tokenizer_revision) is None
        or tokenizer_revision != revision
        or type(load_in_4bit) is not bool
    ):
        raise ValueError("workload model snapshot is invalid")
    return model_ref, revision, tokenizer_revision, load_in_4bit


def _validate_source_projection(
    source: ModalInferenceSourceBinding,
    submit_binding: ModalCommandBinding,
    native: dict[str, object],
) -> None:
    submit = parse_exact_command(submit_binding.command_bytes)
    read_request = parse_canonical_object(
        source.read_request_bytes, name="Modal inference read request"
    )
    expected_request_fields = {
        "schema_version",
        "purpose",
        "source_workflow_record_digest",
        "source_revision",
        "run",
        "provider_run_binding_digest",
        "submit_command_bytes_digest",
        "foundation_record_digest",
        "assessment_digest",
        "foundation_binding_digest",
        "foundation_outcome_digest",
        "found_receipt_digest",
    }
    reference = ScopedProviderRunRefV1(
        source.provider_id,
        source.profile_ref,
        source.account_ref,
        source.namespace_ref,
        source.provider_job_ref,
    )
    native_members = [
        {
            "role": item.role.value,
            "path": item.path,
            "size": item.size,
            "sha256": item.sha256,
            "provider_entry_id": item.provider_entry_id,
        }
        for item in source.native_members
    ]
    native_manifest_evidence = canonical_bytes(
        {
            "completion_artifact_set_digest": native.get("artifact_set_digest"),
            "members": native_members,
        }
    )
    manifest = ArtifactManifestV1.build(
        run=source.run,
        provider_run=reference,
        artifacts=source.artifacts,
        artifact_source_digest=domain_digest(
            "synaptic-modal-artifact-source/v1", source.native_evidence
        ),
        canonical_evidence=native_manifest_evidence,
    )
    preparation = submit.preparation
    if (
        type(read_request) is not dict
        or set(read_request) != expected_request_fields
        or read_request.get("schema_version") != "synaptic-provider-run-read-request/v1"
        or read_request.get("purpose") != "artifacts"
        or read_request.get("run") != source.run.to_dict()
        or read_request.get("source_workflow_record_digest")
        != source.source_workflow_record_digest
        or read_request.get("source_revision") != source.source_revision
        or read_request.get("provider_run_binding_digest")
        != source.provider_run_binding_digest
        or read_request.get("submit_command_bytes_digest")
        != domain_digest(
            "synaptic-foundation-command-bytes/v1", submit_binding.command_bytes
        )
        or source.read_request_digest
        != domain_digest(
            "synaptic-provider-run-read-request/v1", source.read_request_bytes
        )
        or source.manifest_digest != manifest.manifest_digest
        or source.artifact_source_digest != manifest.artifact_source_digest
        or (
            source.run.project_ref,
            source.run.run_id,
            source.provider_id,
            source.profile_ref,
            source.account_ref,
            source.namespace_ref,
            source.effect_id,
        )
        != (
            preparation.project_ref,
            preparation.run_id,
            preparation.provider.provider_id,
            preparation.provider.profile_ref,
            preparation.scope.account_ref,
            preparation.scope.namespace_ref,
            submit.operation.effect.effect_id,
        )
    ):
        raise ValueError("Modal inference source projection is invalid")


@dataclass(frozen=True, slots=True, init=False)
class ModalInferenceWorkloadBinding:
    """Immutable workload projection; this value is not a serving grant.

    ``load_in_4bit`` records training configuration and does not select the
    later inference precision.
    """

    run_id: str
    project_ref: str
    read_request_digest: str
    manifest_digest: str
    artifact_source_digest: str
    native_evidence_sha256: str
    provider_job_ref: str
    submit_effect_id: str
    submit_command_digest: str
    preparation_digest: str
    workload_digest: str
    stage_effect_id: str
    stage_command_digest: str
    stage_binding_digest: str
    control_volume_id: str
    artifact_volume_id: str
    key_ref: str
    bundle_sha256: str
    stage_claim_sha256: str
    launch_claim_sha256: str
    workload_bytes: bytes
    workload_sha256: str
    workload_size: int
    model_ref: str
    model_revision: str
    tokenizer_revision: str
    load_in_4bit: bool
    _token: object

    def __init__(self, *, _token: object = None, **values: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("Modal inference workload bindings are factory issued")
        expected = set(self.__dataclass_fields__) - {"_token"}
        if set(values) != expected:
            raise TypeError("Modal inference workload binding fields are invalid")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "_token", _TOKEN)


def _bind(
    source: ModalInferenceSourceBinding,
    *,
    launch_source,
    foundation_authenticator,
    assessment_authenticator,
    binding_authority,
    stage_verifier,
    launch_verifier,
    recipes: RecipeRegistry,
    bounds: BoundsPolicyV1,
) -> ModalInferenceWorkloadBinding:
    if type(source) is not ModalInferenceSourceBinding:
        raise TypeError("exact Modal inference source binding required")
    if type(recipes) is not RecipeRegistry or type(bounds) is not BoundsPolicyV1:
        raise TypeError("exact Modal workload configuration required")
    initial_source = _source_snapshot(source)
    submit_binding = ModalCommandBinding(
        source.command_binding.command_bytes,
        source.command_binding.preparation_snapshot,
        source.command_binding.deployment_bytes,
    )
    if submit_binding != source.command_binding:
        raise ValueError("Modal inference source binding changed")
    native = parse_canonical_object(
        bytes(source.native_evidence), name="native artifact evidence"
    )
    _validate_source_projection(source, submit_binding, native)
    resolver = getattr_static(launch_source, "resolve", None)
    class_resolver = getattr_static(type(launch_source), "resolve", None)
    if resolver is not class_resolver or not callable(class_resolver):
        raise TypeError("retained Modal launch source is invalid")
    envelope = launch_source.resolve(source.command_digest)
    if (
        type(envelope) is not ModalLaunchEnvelope
        or envelope.submit_binding != submit_binding
    ):
        raise ValueError("retained Modal launch differs from inference source")
    retained_envelope = _envelope_snapshot(envelope)
    stage_material = ModalStageMaterial(
        ModalCommandBinding(
            retained_envelope[5], retained_envelope[6], retained_envelope[7]
        ),
        retained_envelope[8],
        retained_envelope[9],
        retained_envelope[10],
        retained_envelope[11],
        retained_envelope[12],
        retained_envelope[13],
    )
    admitted_envelope = ModalLaunchEnvelope(
        submit_binding,
        stage_material,
        envelope.stage_record,
        envelope.stage_assessment,
        retained_envelope[3],
        retained_envelope[4],
    )
    admitted_snapshot = _envelope_snapshot(admitted_envelope)
    dispatch = prepare_modal_submit_dispatch(
        admitted_envelope,
        foundation_authenticator=foundation_authenticator,
        assessment_authenticator=assessment_authenticator,
        binding_authority=binding_authority,
        stage_verifier=stage_verifier,
        launch_verifier=launch_verifier,
        recipes=recipes,
        bounds=bounds,
    )
    if (
        _envelope_snapshot(envelope) != retained_envelope
        or _envelope_snapshot(admitted_envelope) != admitted_snapshot
    ):
        raise ValueError("retained Modal launch changed during authentication")
    stage_binding = ModalCommandBinding(
        stage_material.binding.command_bytes,
        stage_material.binding.preparation_snapshot,
        stage_material.binding.deployment_bytes,
    )
    stage_command = parse_exact_command(stage_binding.command_bytes)
    if type(stage_command) is not StageCommandV2:
        raise ValueError("retained Modal stage command is invalid")
    bundle = ModalCoordinatorBundle.parse_transport(
        stage_material.bundle,
        binding=stage_binding,
        recipes=recipes,
    )
    if (
        _envelope_snapshot(envelope) != retained_envelope
        or _envelope_snapshot(admitted_envelope) != admitted_snapshot
    ):
        raise ValueError("retained Modal launch changed during bundle admission")
    workload_member = next(
        item for item in bundle.members if item.name == "workload.json"
    )
    workload_bytes = bytes(workload_member.content)
    workload_document = parse_canonical_object(workload_bytes, name="Modal workload")
    workload = CompiledWorkload(
        workload_document.get("method"),
        workload_document.get("schema_version"),
        workload_document.get("entrypoint"),
        workload_bytes,
    )
    model_ref, revision, tokenizer_revision, load_in_4bit = _model(workload)
    matches = tuple(
        item
        for item in source.native_members
        if item.role is ArtifactRole.WORKLOAD_RECORD
    )
    identity = native.get("identity")
    if type(identity) is not dict or len(matches) != 1:
        raise ValueError("native workload identity is unavailable")
    workload_record = matches[0]
    submit = parse_exact_command(submit_binding.command_bytes)
    expectation = dispatch.expectation
    expected_dispatch = (
        submit.canonical_bytes,
        stage_material.control_volume_id,
        stage_material.artifact_volume_id,
        stage_material.key_ref,
        sha(stage_material.claim),
        sha(stage_material.bundle),
    )
    actual_dispatch = (
        expectation.submit_command_bytes,
        expectation.control_volume_id,
        expectation.artifact_volume_id,
        expectation.key_ref,
        expectation.stage_claim_sha256,
        expectation.stage_bundle_sha256,
    )
    client = submit_binding.client_binding
    expected_identity = {
        "account_ref": client.account_ref,
        "workspace_ref": client.workspace_ref,
        "environment_ref": client.environment_ref,
        "client_ref": client.client_ref,
        "sdk_version": client.sdk_version,
        "control_volume_id": stage_material.control_volume_id,
        "artifact_volume_id": stage_material.artifact_volume_id,
        "job_ref": source.provider_job_ref,
        "effect_id": source.effect_id,
        "command_digest": source.command_digest,
        "plan_digest": submit.preparation.plan_fingerprint,
        "deployment_attestation_digest": submit_binding.deployment.attestation_digest,
        "invocation_nonce": submit.operation.invocation_nonce,
    }
    native_members = [
        {
            "role": item.role.value,
            "path": item.path,
            "size": item.size,
            "sha256": item.sha256,
            "provider_entry_id": item.provider_entry_id,
        }
        for item in source.native_members
    ]
    if (
        actual_dispatch != expected_dispatch
        or source.read_request_digest
        != domain_digest(
            "synaptic-provider-run-read-request/v1", source.read_request_bytes
        )
        or source.artifact_volume_id != stage_material.artifact_volume_id
        or any(identity.get(name) != value for name, value in expected_identity.items())
        or identity.get("key_ref") != stage_material.key_ref
        or native.get("members") != native_members
        or source.artifact_source_digest
        != domain_digest("synaptic-modal-artifact-source/v1", source.native_evidence)
        or workload_record.path
        != f"operations/{source.effect_id}/output/workload_record"
        or workload_record.size != len(workload_bytes)
        or workload_record.sha256 != hashlib.sha256(workload_bytes).hexdigest()
        or bundle.material.workload_bytes != workload_bytes
        or source.workload_digest != workload.fingerprint
        or _source_snapshot(source) != initial_source
    ):
        raise ValueError("Modal inference workload lineage mismatch")
    return ModalInferenceWorkloadBinding(
        run_id=source.run.run_id,
        project_ref=source.run.project_ref,
        read_request_digest=source.read_request_digest,
        manifest_digest=source.manifest_digest,
        artifact_source_digest=source.artifact_source_digest,
        native_evidence_sha256=sha(source.native_evidence),
        provider_job_ref=source.provider_job_ref,
        submit_effect_id=source.effect_id,
        submit_command_digest=source.command_digest,
        preparation_digest=source.preparation_digest,
        workload_digest=source.workload_digest,
        stage_effect_id=stage_command.operation.effect.effect_id,
        stage_command_digest=stage_command.digest,
        stage_binding_digest=stage_binding.authenticated_binding_digest,
        control_volume_id=stage_material.control_volume_id,
        artifact_volume_id=stage_material.artifact_volume_id,
        key_ref=stage_material.key_ref,
        bundle_sha256=sha(stage_material.bundle),
        stage_claim_sha256=sha(stage_material.claim),
        launch_claim_sha256=sha(admitted_envelope.claim),
        workload_bytes=workload_bytes,
        workload_sha256=hashlib.sha256(workload_bytes).hexdigest(),
        workload_size=len(workload_bytes),
        model_ref=model_ref,
        model_revision=revision,
        tokenizer_revision=tokenizer_revision,
        load_in_4bit=load_in_4bit,
        _token=_TOKEN,
    )


def bind_modal_inference_workload(
    source: ModalInferenceSourceBinding,
    *,
    launch_source,
    foundation_authenticator,
    assessment_authenticator,
    binding_authority,
    stage_verifier,
    launch_verifier,
    recipes: RecipeRegistry,
    bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalInferenceWorkloadBinding:
    """Reauthenticate retained workload identity for a fresh 2a source result."""

    try:
        return _bind(
            source,
            launch_source=launch_source,
            foundation_authenticator=foundation_authenticator,
            assessment_authenticator=assessment_authenticator,
            binding_authority=binding_authority,
            stage_verifier=stage_verifier,
            launch_verifier=launch_verifier,
            recipes=recipes,
            bounds=bounds,
        )
    except Exception:
        raise ModalInferenceWorkloadError("modal_inference_workload_invalid") from None


__all__: list[str] = []
