"""Bind public verified-run facts to authenticated native Modal artifact placement."""

from __future__ import annotations

from dataclasses import dataclass, fields
from inspect import getattr_static

from synaptic_tuner.api.v1.results import (
    TrainingRunRef,
    TrainingRunState,
    VerifiedArtifact,
)
from synaptic_tuner.api.v1.runs_facade import RunOutcome, RunsAPI

from tuner.execution.coordinator_v1.model import (
    ArtifactManifestV1,
    ProviderReadPurposeV1,
    ProviderRunReadRequestV1,
    WorkflowPhaseV1,
    WorkflowRecordV1,
)
from tuner.execution.coordinator_v1.ports import (
    FoundationEvidenceAuthenticatorPortV1,
    FoundationRecordAssessmentPortV1,
    WorkflowStorePortV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    project_run_outcome,
    provider_run_read_request,
)
from tuner.execution.coordinator_v1.stores import _revalidate_workflow
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import SubmitCommandV2, parse_exact_command
from tuner.execution.foundation_v2.references import ScopedProviderRunRefV1

from .binding import ModalClientBinding
from .contracts import ArtifactMemberV1, provider_entry_identity
from .coordinator_binding import ModalCommandBinding
from .control import CrossPlaneIdentityV1
from .coordinator_reader import ModalArtifactInventory, ModalCoordinatorRunReader
from .manifest import CompletionManifestV1


_ROLES = (
    "final_model",
    "tokenizer",
    "training_lineage",
    "training_metrics",
    "workload_record",
)
_TOKEN = object()


def _method(value: object, name: str) -> None:
    class_member = getattr_static(type(value), name, None)
    instance_member = getattr_static(value, name, None)
    if (
        class_member is None
        or instance_member is not class_member
        or not callable(class_member)
    ):
        raise TypeError(f"{name} collaborator is unavailable")


def _run(value: object) -> TrainingRunRef:
    if type(value) is not TrainingRunRef:
        raise TypeError("run must be an exact TrainingRunRef")
    return TrainingRunRef.from_dict(value.to_dict())


def _artifacts(values: object) -> tuple[VerifiedArtifact, ...]:
    if type(values) is not tuple or any(
        type(item) is not VerifiedArtifact for item in values
    ):
        raise TypeError("artifacts must be an exact tuple of VerifiedArtifact values")
    rebuilt = tuple(VerifiedArtifact.from_dict(item.to_dict()) for item in values)
    if tuple(item.role for item in rebuilt) != _ROLES or any(
        item.size_bytes < 1 for item in rebuilt
    ):
        raise ValueError("Modal inference source requires the canonical five artifacts")
    return rebuilt


def _request(value: object) -> ProviderRunReadRequestV1:
    if type(value) is not ProviderRunReadRequestV1:
        raise TypeError("exact provider read request required")
    rebuilt = ProviderRunReadRequestV1(
        value.purpose,
        value.source_workflow_record_digest,
        value.source_revision,
        _run(value.run),
        value.provider_run,
        bytes(value.submit_command_bytes),
        value.foundation_record,
        value.assessment,
        value.foundation_binding,
        value.foundation_outcome,
        value.found_receipt_digest,
        bytes(value.canonical_bytes),
        value.request_digest,
    )
    if rebuilt != value:
        raise ValueError("provider read request changed during reconstruction")
    return rebuilt


def _inventory(value: object) -> ModalArtifactInventory:
    if type(value) is not ModalArtifactInventory:
        raise TypeError("exact Modal native inventory required")
    source_manifest = value.manifest
    source_identity = value.identity
    if (
        type(source_manifest) is not CompletionManifestV1
        or type(source_identity) is not CrossPlaneIdentityV1
    ):
        raise TypeError("Modal native inventory values are invalid")
    members = tuple(
        ArtifactMemberV1(
            member.role,
            member.path,
            member.size,
            member.sha256,
            member.provider_entry_id,
        )
        for member in source_manifest.members
    )
    manifest = CompletionManifestV1(
        members,
        **{
            field.name: getattr(source_manifest, field.name)
            for field in fields(CompletionManifestV1)
            if field.name != "members"
        },
    )
    client = source_identity.binding
    identity = CrossPlaneIdentityV1(
        ModalClientBinding(
            client.account_ref,
            client.workspace_ref,
            client.environment_ref,
            client.client_ref,
            client.sdk_version,
        ),
        *(
            getattr(source_identity, field.name)
            for field in fields(CrossPlaneIdentityV1)
            if field.name != "binding"
        ),
    )
    rebuilt = ModalArtifactInventory.build(manifest, identity)
    if rebuilt != value or rebuilt.canonical_evidence != value.canonical_evidence:
        raise ValueError("Modal native inventory changed during reconstruction")
    return rebuilt


def _manifest(value: object) -> ArtifactManifestV1:
    if type(value) is not ArtifactManifestV1:
        raise TypeError("exact artifact manifest required")
    reference = value.provider_run
    rebuilt = ArtifactManifestV1(
        _run(value.run),
        ScopedProviderRunRefV1(
            reference.provider_id,
            reference.profile_ref,
            reference.account_ref,
            reference.namespace_ref,
            reference.provider_job_ref,
        ),
        _artifacts(value.artifacts),
        value.artifact_source_digest,
        bytes(value.canonical_evidence),
        value.manifest_digest,
    )
    if rebuilt != value:
        raise ValueError("artifact manifest changed during reconstruction")
    return rebuilt


@dataclass(frozen=True, slots=True, init=False)
class ModalInferenceSourceBinding:
    """Factory-issued correlation of public run facts and native Modal metadata."""

    run: TrainingRunRef
    artifacts: tuple[VerifiedArtifact, ...]
    read_request_bytes: bytes
    read_request_digest: str
    source_workflow_record_digest: str
    source_revision: int
    manifest_digest: str
    artifact_source_digest: str
    provider_id: str
    profile_ref: str
    account_ref: str
    namespace_ref: str
    provider_job_ref: str
    effect_id: str
    provider_run_binding_digest: str
    command_binding: ModalCommandBinding
    artifact_volume_id: str
    native_members: tuple[ArtifactMemberV1, ...]
    native_evidence: bytes
    _token: object

    def __init__(
        self,
        run: TrainingRunRef,
        artifacts: tuple[VerifiedArtifact, ...],
        read_request: ProviderRunReadRequestV1,
        manifest: ArtifactManifestV1,
        command_binding: ModalCommandBinding,
        native_inventory: ModalArtifactInventory,
        *,
        _token: object = None,
    ) -> None:
        if _token is not _TOKEN:
            raise TypeError("Modal inference source bindings are factory issued")
        object.__setattr__(self, "run", _run(run))
        object.__setattr__(self, "artifacts", _artifacts(artifacts))
        request = _request(read_request)
        manifest = _manifest(manifest)
        if type(command_binding) is not ModalCommandBinding:
            raise TypeError("exact Modal command binding required")
        inventory = _inventory(native_inventory)
        reference = request.provider_run.reference
        identity = inventory.identity
        members = tuple(
            ArtifactMemberV1(
                member.role,
                member.path,
                member.size,
                member.sha256,
                member.provider_entry_id,
            )
            for member in sorted(
                inventory.manifest.members, key=lambda item: item.role.value
            )
        )
        for name, value in (
            ("read_request_bytes", bytes(request.canonical_bytes)),
            ("read_request_digest", request.request_digest),
            ("source_workflow_record_digest", request.source_workflow_record_digest),
            ("source_revision", request.source_revision),
            ("manifest_digest", manifest.manifest_digest),
            ("artifact_source_digest", manifest.artifact_source_digest),
            ("provider_id", reference.provider_id),
            ("profile_ref", reference.profile_ref),
            ("account_ref", reference.account_ref),
            ("namespace_ref", reference.namespace_ref),
            ("provider_job_ref", reference.provider_job_ref),
            ("effect_id", request.provider_run.effect_id),
            ("provider_run_binding_digest", request.provider_run.binding_digest),
            ("artifact_volume_id", identity.artifact_volume_id),
            ("native_members", members),
            ("native_evidence", bytes(inventory.canonical_evidence)),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "command_binding",
            ModalCommandBinding(
                command_binding.command_bytes,
                command_binding.preparation_snapshot,
                command_binding.deployment_bytes,
            ),
        )
        object.__setattr__(self, "_token", _TOKEN)
        self._validate(request, manifest, inventory)

    def _validate(
        self,
        request: ProviderRunReadRequestV1,
        manifest: ArtifactManifestV1,
        inventory: ModalArtifactInventory,
    ) -> None:
        command = parse_exact_command(self.command_binding.command_bytes)
        if type(command) is not SubmitCommandV2:
            raise ValueError("Modal inference source command is not submit")
        reference = request.provider_run.reference
        identity = inventory.identity
        members = self.native_members
        expected_manifest_evidence = canonical_bytes(
            {
                "completion_artifact_set_digest": inventory.manifest.artifact_set_digest,
                "members": [
                    {
                        "role": item.role.value,
                        "path": item.path,
                        "size": item.size,
                        "sha256": item.sha256,
                        "provider_entry_id": item.provider_entry_id,
                    }
                    for item in members
                ],
            }
        )
        output_prefix = f"operations/{request.provider_run.effect_id}/output/"
        if (
            request.purpose is not ProviderReadPurposeV1.ARTIFACTS
            or request.run != self.run
            or manifest.run != self.run
            or manifest.artifacts != self.artifacts
            or manifest.provider_run != reference
            or self.command_binding.command_bytes != request.submit_command_bytes
            or identity.job_ref != reference.provider_job_ref
            or identity.effect_id != request.provider_run.effect_id
            or identity.command_digest != command.digest
            or identity.plan_digest != command.preparation.plan_fingerprint
            or identity.invocation_nonce != command.operation.invocation_nonce
            or identity.binding != self.command_binding.client_binding
            or identity.deployment_attestation_digest
            != self.command_binding.deployment.attestation_digest
            or request.provider_run.effect_id != command.operation.effect.effect_id
            or (
                reference.provider_id,
                reference.profile_ref,
                reference.account_ref,
                reference.namespace_ref,
            )
            != (
                command.preparation.provider.provider_id,
                command.preparation.provider.profile_ref,
                command.preparation.scope.account_ref,
                command.preparation.scope.namespace_ref,
            )
            or self.artifact_source_digest
            != domain_digest("synaptic-modal-artifact-source/v1", self.native_evidence)
            or manifest.canonical_evidence != expected_manifest_evidence
            or tuple(item.role.value for item in members) != _ROLES
            or any(
                item.path != output_prefix + item.role.value
                or item.provider_entry_id
                != provider_entry_identity(
                    identity.artifact_volume_id, item.path, item.size
                )
                for item in members
            )
            or tuple((item.size, item.sha256) for item in members)
            != tuple((item.size_bytes, item.sha256) for item in self.artifacts)
        ):
            raise ValueError("Modal inference source lineage mismatch")

    @property
    def command_digest(self) -> str:
        return parse_exact_command(self.command_binding.command_bytes).digest

    @property
    def preparation_digest(self) -> str:
        return parse_exact_command(
            self.command_binding.command_bytes
        ).preparation.preparation_digest

    @property
    def workload_digest(self) -> str:
        return parse_exact_command(
            self.command_binding.command_bytes
        ).preparation.workload_digest


class ModalInferenceSourceBinder:
    """Correlate current public verification with the native authenticated reader."""

    def __init__(
        self,
        *,
        runs: RunsAPI,
        workflows: WorkflowStorePortV1,
        foundation: object,
        foundation_authenticator: FoundationEvidenceAuthenticatorPortV1,
        assessment_authenticator: FoundationRecordAssessmentPortV1,
        reader: ModalCoordinatorRunReader,
    ) -> None:
        if type(runs) is not RunsAPI:
            raise TypeError("runs must be an exact RunsAPI")
        for value, methods in (
            (workflows, ("get",)),
            (foundation, ("get", "assess")),
            (
                foundation_authenticator,
                (
                    "authenticate_grant",
                    "authenticate_receipt",
                    "authenticate_invalid_evidence",
                ),
            ),
            (assessment_authenticator, ("authenticate",)),
        ):
            for method in methods:
                _method(value, method)
        if type(reader) is not ModalCoordinatorRunReader:
            raise TypeError("exact Modal coordinator reader required")
        self._runs = runs
        self._workflows = workflows
        self._foundation = foundation
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._reader = reader

    def _workflow(self, run: TrainingRunRef) -> WorkflowRecordV1:
        value = self._workflows.get(_run(run))
        if type(value) is not WorkflowRecordV1 or value.run != run:
            raise ValueError("retained workflow is unavailable")
        return _revalidate_workflow(value)

    def bind(self, runs: RunsAPI, run: TrainingRunRef) -> ModalInferenceSourceBinding:
        try:
            return self._bind(runs, run)
        except Exception:
            raise ModalInferenceBindingError("modal_inference_source_invalid") from None

    def _bind(self, runs: RunsAPI, run: TrainingRunRef) -> ModalInferenceSourceBinding:
        if runs is not self._runs:
            raise ValueError("runs differs from trusted Modal composition")
        requested = _run(run)
        verification = runs.reverify(requested)
        if verification.run != requested or verification.verified is not True:
            raise ValueError("run reverification failed")
        outcome = runs.outcome(requested)
        if (
            type(outcome) is not RunOutcome
            or outcome.run != requested
            or outcome.state is not TrainingRunState.SUCCEEDED
        ):
            raise ValueError("verified run outcome is unavailable")
        public_artifacts = _artifacts(outcome.artifacts)

        workflow = self._workflow(requested)
        workflow_bytes = canonical_bytes(workflow.to_dict())
        workflow_digest = workflow.record_digest
        if (
            workflow.phase is not WorkflowPhaseV1.VERIFIED
            or workflow.artifact_manifest is None
            or project_run_outcome(workflow) != outcome
            or workflow.verified_artifacts != public_artifacts
        ):
            raise ValueError("public outcome differs from retained workflow")
        submit = workflow.submit
        if submit is None:
            raise ValueError("retained submit is unavailable")
        record = self._foundation.get(submit.effect_id)
        if record is None:
            raise ValueError("retained Foundation record is unavailable")
        assessment = self._foundation.assess(record)
        request = provider_run_read_request(
            workflow,
            record,
            assessment,
            self._foundation_authenticator,
            self._assessment_authenticator,
            purpose=ProviderReadPurposeV1.ARTIFACTS,
        )
        request = _request(request)
        binding, reference, inventory, manifest = self._reader.native_artifacts(request)
        if (
            type(binding) is not ModalCommandBinding
            or reference != request.provider_run.reference
            or type(inventory) is not ModalArtifactInventory
            or type(manifest) is not ArtifactManifestV1
            or manifest != workflow.artifact_manifest
            or manifest.artifacts != public_artifacts
        ):
            raise ValueError("native artifact proof differs from retained workflow")
        current = self._workflow(requested)
        if (
            current != workflow
            or current.record_digest != workflow_digest
            or canonical_bytes(current.to_dict()) != workflow_bytes
            or _run(run) != requested
        ):
            raise ValueError("retained workflow changed during native admission")
        return ModalInferenceSourceBinding(
            requested,
            public_artifacts,
            request,
            manifest,
            binding,
            inventory,
            _token=_TOKEN,
        )


class ModalInferenceBindingError(RuntimeError):
    """Closed non-secret native inference-source admission failure."""


__all__: list[str] = []
