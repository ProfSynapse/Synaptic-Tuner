from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

import tests.execution.coordinator_v1.test_state_machine as foundation_cases
import tests.execution.providers.test_modal_coordinator_reader as reader_cases
from synaptic_tuner.api.v1.results import VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunVerification, RunsAPI
from tuner.execution.coordinator_v1.model import (
    ProviderReadPurposeV1,
    ProviderRunPhaseV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    apply_artifact_verification,
    apply_provider_observation,
    project_run_outcome,
    provider_run_read_request,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.providers.modal.contracts import (
    ArtifactMemberV1,
    ArtifactRole,
    provider_entry_identity,
)
from tuner.execution.providers.modal.control import CrossPlaneIdentityV1
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_reader import ModalCoordinatorRunReader
from tuner.execution.providers.modal.inference_binding import (
    ModalInferenceBindingError,
    ModalInferenceSourceBinder,
    ModalInferenceSourceBinding,
)


class _Workflows:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def get(self, run):
        self.calls += 1
        return self.value


class _Foundation:
    def __init__(self, record, assessment):
        self.record = record
        self.assessment = assessment
        self.calls = []

    def get(self, effect_id):
        self.calls.append("get")
        return self.record

    def assess(self, record):
        self.calls.append("assess")
        assert record is self.record
        return self.assessment


class _Runs:
    def __init__(self, workflow):
        self.workflow = workflow
        self.calls = []

    def reverify(self, run):
        self.calls.append("reverify")
        return RunVerification(run, True, "2026-09-10T00:00:00Z")

    def outcome(self, run):
        self.calls.append("outcome")
        return project_run_outcome(self.workflow)


def _case(monkeypatch):
    workflow, record, assessed, adapter, deployment = reader_cases._foundation(
        monkeypatch
    )
    observation_request, observation = foundation_cases.observation(
        workflow, record, ProviderRunPhaseV1.SUCCEEDED, {"completed": True}
    )
    succeeded = apply_provider_observation(
        workflow, observation_request, observation, foundation_cases.ObservationAuth()
    )
    artifacts = tuple(
        VerifiedArtifact(
            role.value,
            hashlib.sha256(role.value.encode()).hexdigest(),
            len(role.value.encode()),
        )
        for role in ArtifactRole
    )
    request = provider_run_read_request(
        succeeded,
        record,
        assessed,
        foundation_cases.Auth(),
        foundation_cases.AssessmentAuth(),
        purpose=ProviderReadPurposeV1.ARTIFACTS,
    )
    command = foundation_cases.parse_exact_command(request.submit_command_bytes)
    binding = ModalCommandBinding(
        request.submit_command_bytes,
        adapter.snapshot(),
        reader_cases.canonical_bytes(deployment.to_dict()),
    )
    identity = CrossPlaneIdentityV1(
        binding.client_binding,
        "control-volume",
        "artifact-volume",
        "job-a",
        command.operation.effect.effect_id,
        command.digest,
        command.preparation.plan_fingerprint,
        binding.deployment.attestation_digest,
        command.operation.invocation_nonce,
        1,
        "key-a",
    )
    members = []
    for item in artifacts:
        path = f"operations/{request.provider_run.effect_id}/output/{item.role}"
        members.append(
            ArtifactMemberV1(
                ArtifactRole(item.role),
                path,
                item.size_bytes,
                item.sha256,
                provider_entry_identity("artifact-volume", path, item.size_bytes),
            )
        )
    members = tuple(members)
    transport = reader_cases.Transport(identity, members)
    authority = reader_cases.Authority()
    reader = ModalCoordinatorRunReader(
        catalog=reader_cases.Catalog(binding),
        binding_authority=authority,
        foundation_authenticator=foundation_cases.Auth(),
        assessment_authenticator=foundation_cases.AssessmentAuth(),
        evidence_authority=authority,
        transport=transport,
        observed_at="2026-09-10T00:00:00Z",
    )
    _, _, _, manifest = reader.native_artifacts(request)
    receipt = foundation_cases.verification_receipt(
        succeeded, manifest, foundation_cases.VerificationVerdictV1.VERIFIED
    )
    verified = apply_artifact_verification(
        succeeded, manifest, receipt, foundation_cases.Verifier()
    )
    runs_ops = _Runs(verified)
    runs = RunsAPI(runs_ops)
    workflows = _Workflows(verified)
    foundation = _Foundation(record, assessed)
    binder = ModalInferenceSourceBinder(
        runs=runs,
        workflows=workflows,
        foundation=foundation,
        foundation_authenticator=foundation_cases.Auth(),
        assessment_authenticator=foundation_cases.AssessmentAuth(),
        reader=reader,
    )
    transport.calls.clear()
    return binder, runs, runs_ops, workflows, foundation, reader, transport, verified


def test_real_verified_workflow_binds_native_metadata_without_weight_reads(monkeypatch):
    binder, runs, runs_ops, workflows, foundation, _, transport, workflow = _case(
        monkeypatch
    )
    result = binder.bind(runs, workflow.run)
    assert type(result) is ModalInferenceSourceBinding
    assert result.run == workflow.run
    assert result.artifacts == workflow.verified_artifacts
    assert result.manifest_digest == workflow.artifact_manifest.manifest_digest
    assert (
        result.artifact_source_digest
        == workflow.artifact_manifest.artifact_source_digest
    )
    assert result.read_request_bytes
    assert result.command_digest == domain_digest(
        "synaptic-submit-command/v2", result.command_binding.command_bytes
    )
    assert result.native_members == tuple(
        sorted(result.native_members, key=lambda x: x.role.value)
    )
    assert runs_ops.calls == ["reverify", "outcome"]
    assert foundation.calls == ["get", "assess"]
    assert workflows.calls == 2
    assert transport.calls == ["inventory"]
    assert "bytes" not in transport.calls


def test_alternate_runs_identity_fails_before_any_collaborator_call(monkeypatch):
    binder, runs, runs_ops, workflows, foundation, _, transport, workflow = _case(
        monkeypatch
    )
    other = RunsAPI(_Runs(workflow))
    with pytest.raises(
        ModalInferenceBindingError, match="modal_inference_source_invalid"
    ):
        binder.bind(other, workflow.run)
    assert runs_ops.calls == []
    assert workflows.calls == 0 and foundation.calls == [] and transport.calls == []


def test_workflow_change_during_native_read_is_rejected(monkeypatch):
    binder, runs, _, workflows, _, reader, transport, workflow = _case(monkeypatch)
    original = reader.native_artifacts

    def changing(request):
        result = original(request)
        workflows.value = replace(workflow, revision=workflow.revision + 1)
        return result

    monkeypatch.setattr(reader, "native_artifacts", changing)
    with pytest.raises(
        ModalInferenceBindingError, match="modal_inference_source_invalid"
    ):
        binder.bind(runs, workflow.run)
    assert transport.calls == ["inventory"]


def test_public_artifact_substitution_fails_before_native_inventory(monkeypatch):
    binder, runs, runs_ops, workflows, foundation, _, transport, workflow = _case(
        monkeypatch
    )
    original = runs_ops.outcome

    def changed(run):
        outcome = original(run)
        return replace(
            outcome,
            artifacts=(replace(outcome.artifacts[0], size_bytes=0),)
            + outcome.artifacts[1:],
        )

    monkeypatch.setattr(type(runs_ops), "outcome", lambda self, run: changed(run))
    with pytest.raises(
        ModalInferenceBindingError, match="modal_inference_source_invalid"
    ):
        binder.bind(runs, workflow.run)
    assert transport.calls == []


def test_unsorted_native_transport_is_normalized_without_body_reads(monkeypatch):
    binder, runs, _, _, _, _, transport, workflow = _case(monkeypatch)
    transport.members = tuple(reversed(transport.members))
    result = binder.bind(runs, workflow.run)
    assert tuple(item.role.value for item in result.native_members) == tuple(
        sorted(item.role.value for item in result.native_members)
    )
    assert transport.calls == ["inventory"]


def test_in_place_workflow_mutation_during_reader_is_rejected(monkeypatch):
    binder, runs, _, workflows, _, reader, transport, workflow = _case(monkeypatch)
    original = reader.native_artifacts

    def changing(request):
        result = original(request)
        object.__setattr__(workflow, "revision", workflow.revision + 1)
        return result

    monkeypatch.setattr(reader, "native_artifacts", changing)
    with pytest.raises(
        ModalInferenceBindingError, match="modal_inference_source_invalid"
    ):
        binder.bind(runs, workflow.run)
    assert transport.calls == ["inventory"]


def test_other_manifest_evidence_with_same_artifact_hashes_is_rejected(monkeypatch):
    binder, runs, _, _, _, reader, transport, workflow = _case(monkeypatch)
    original = reader.native_artifacts

    def changing(request):
        binding, reference, inventory, manifest = original(request)
        other = type(manifest).build(
            run=manifest.run,
            provider_run=manifest.provider_run,
            artifacts=manifest.artifacts,
            artifact_source_digest=domain_digest(
                "synaptic-modal-artifact-source/v1", canonical_bytes({"other": True})
            ),
            canonical_evidence=canonical_bytes({"other": True}),
        )
        return binding, reference, inventory, other

    monkeypatch.setattr(reader, "native_artifacts", changing)
    with pytest.raises(
        ModalInferenceBindingError, match="modal_inference_source_invalid"
    ):
        binder.bind(runs, workflow.run)
    assert transport.calls == ["inventory"]


def test_false_reverification_stops_before_retained_or_native_reads(monkeypatch):
    binder, runs, runs_ops, workflows, foundation, _, transport, workflow = _case(
        monkeypatch
    )
    monkeypatch.setattr(
        type(runs_ops),
        "reverify",
        lambda self, run: RunVerification(run, False, "2026-09-10T00:00:00Z"),
    )
    with pytest.raises(
        ModalInferenceBindingError, match="modal_inference_source_invalid"
    ):
        binder.bind(runs, workflow.run)
    assert workflows.calls == 0 and foundation.calls == [] and transport.calls == []


@pytest.mark.parametrize(
    "site", ("reverify", "outcome", "workflow", "foundation", "reader", "ownerror")
)
def test_foreign_failures_are_closed_without_secret_text_or_chain(monkeypatch, site):
    binder, runs, runs_ops, workflows, foundation, reader, _, workflow = _case(
        monkeypatch
    )
    message = "secret-token-shape"
    if site in ("reverify", "outcome"):
        monkeypatch.setattr(
            type(runs_ops),
            site,
            lambda *args: (_ for _ in ()).throw(RuntimeError(message)),
        )
    elif site == "workflow":
        monkeypatch.setattr(
            type(workflows),
            "get",
            lambda *args: (_ for _ in ()).throw(RuntimeError(message)),
        )
    elif site == "foundation":
        monkeypatch.setattr(
            type(foundation),
            "get",
            lambda *args: (_ for _ in ()).throw(RuntimeError(message)),
        )
    elif site == "reader":
        monkeypatch.setattr(
            type(reader),
            "native_artifacts",
            lambda *args: (_ for _ in ()).throw(RuntimeError(message)),
        )
    else:
        monkeypatch.setattr(
            type(reader),
            "native_artifacts",
            lambda *args: (_ for _ in ()).throw(ModalInferenceBindingError(message)),
        )
    with pytest.raises(ModalInferenceBindingError) as caught:
        binder.bind(runs, workflow.run)
    assert str(caught.value) == "modal_inference_source_invalid"
    assert caught.value.__cause__ is None


def test_control_interrupt_from_reader_is_preserved(monkeypatch):
    binder, runs, _, _, _, reader, _, workflow = _case(monkeypatch)
    interrupt = KeyboardInterrupt()
    monkeypatch.setattr(
        type(reader),
        "native_artifacts",
        lambda *args: (_ for _ in ()).throw(interrupt),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        binder.bind(runs, workflow.run)
    assert caught.value is interrupt


def test_returned_binding_does_not_alias_mutable_collaborator_outputs(monkeypatch):
    binder, runs, _, _, _, _, transport, workflow = _case(monkeypatch)
    result = binder.bind(runs, workflow.run)
    assert result.artifacts[0] is not workflow.verified_artifacts[0]
    assert all(
        owned is not supplied
        for owned in result.native_members
        for supplied in transport.members
    )
    saved = (
        result.run.to_dict(),
        tuple(item.to_dict() for item in result.artifacts),
        tuple(
            (item.role, item.path, item.size, item.sha256, item.provider_entry_id)
            for item in result.native_members
        ),
        result.native_evidence,
    )
    object.__setattr__(workflow.verified_artifacts[0], "size_bytes", 999)
    object.__setattr__(transport.members[0], "path", "operations/other/output/foreign")
    assert (
        result.run.to_dict(),
        tuple(item.to_dict() for item in result.artifacts),
        tuple(
            (item.role, item.path, item.size, item.sha256, item.provider_entry_id)
            for item in result.native_members
        ),
        result.native_evidence,
    ) == saved


@pytest.mark.parametrize("fault", ("path", "entry"))
def test_native_member_path_and_entry_identity_are_not_hash_only(monkeypatch, fault):
    binder, runs, _, _, _, _, transport, workflow = _case(monkeypatch)
    member = transport.members[0]
    changed = replace(
        member,
        **(
            {"path": member.path + "-other"}
            if fault == "path"
            else {"provider_entry_id": "f" * 64}
        ),
    )
    transport.members = (changed,) + transport.members[1:]
    with pytest.raises(
        ModalInferenceBindingError, match="modal_inference_source_invalid"
    ):
        binder.bind(runs, workflow.run)
