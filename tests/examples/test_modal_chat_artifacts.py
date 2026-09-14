"""Consumer-owned streamed Modal artifact verification tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from examples.modal_chat.artifacts import (
    ModalChatArtifactError,
    ModalChatArtifactVerifier,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.coordinator_v1.model import (
    ProviderReadPurposeV1,
    VerificationVerdictV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    apply_artifact_verification,
    apply_provider_observation,
)
from tuner.execution.providers.modal.contracts import ArtifactMemberV1, ArtifactRole

import tests.execution.coordinator_v1.test_state_machine as foundation_cases
from tests.execution.providers.test_modal_coordinator_reader import _reader


class Foundation:
    def __init__(self, request):
        self.request = request

    def get(self, effect_id):
        return self.request.foundation_record

    def assess(self, record):
        assert record == self.request.foundation_record
        return self.request.assessment


class Clock:
    def now_iso(self):
        return "2026-09-09T00:00:00Z"


def _case(monkeypatch, *, maximum_total_bytes=1024):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.ARTIFACTS)
    contents = {role.value: (role.value + "-bytes").encode() for role in ArtifactRole}
    prefix = f"operations/{request.provider_run.effect_id}/output/"
    transport.members = tuple(
        ArtifactMemberV1(
            role,
            prefix + role.value,
            len(contents[role.value]),
            hashlib.sha256(contents[role.value]).hexdigest(),
            f"entry-{index}",
        )
        for index, role in enumerate(ArtifactRole)
    )
    transport.contents = contents
    verifier = ModalChatArtifactVerifier(
        foundation_authenticator=foundation_cases.Auth(),
        assessment_authenticator=foundation_cases.AssessmentAuth(),
        authority_ref="consumer-artifact-authority",
        key_ref="consumer-artifact-key",
        key=b"a" * 32,
        clock=Clock(),
        maximum_total_bytes=maximum_total_bytes,
    )
    verifier.bind(reader=reader, foundation=Foundation(request))
    return verifier, reader, request, transport


def _workflow(request):
    workflow, _ = foundation_cases.queued_evidence()
    observation_request, observation = foundation_cases.observation(
        workflow,
        request.foundation_record,
        foundation_cases.ProviderRunPhaseV1.SUCCEEDED,
        {"completed": True},
    )
    return apply_provider_observation(
        workflow,
        observation_request,
        observation,
        foundation_cases.ObservationAuth(),
    )


def test_verify_streams_every_artifact_and_authenticates_receipt(monkeypatch):
    verifier, reader, request, transport = _case(monkeypatch)
    workflow = _workflow(request)
    manifest = reader.artifacts(request)
    receipt = verifier.verify(workflow, manifest)

    assert verifier.authenticate(receipt) is True
    assert receipt.content.verdict is VerificationVerdictV1.VERIFIED
    assert receipt.content.run == workflow.run
    assert receipt.content.manifest_artifacts == manifest.artifacts
    assert receipt.content.verified_artifacts == manifest.artifacts
    assert transport.calls.count("bytes") == len(ArtifactRole)


def test_replay_freshly_relists_and_restreams_exact_verified_inventory(monkeypatch):
    verifier, reader, request, transport = _case(monkeypatch)
    workflow = _workflow(request)
    manifest = reader.artifacts(request)
    first = verifier.verify(workflow, manifest)
    verified = apply_artifact_verification(workflow, manifest, first, verifier)
    before = transport.calls.count("bytes")
    replay = verifier.replay(verified, manifest, first)

    assert verifier.authenticate(replay) is True
    assert replay.content.source_revision == verified.revision
    assert replay.content.source_workflow_record_digest == verified.record_digest
    assert transport.calls.count("bytes") == before + len(ArtifactRole)


def test_replay_rejects_changed_artifact_bytes(monkeypatch):
    verifier, reader, request, transport = _case(monkeypatch)
    workflow = _workflow(request)
    manifest = reader.artifacts(request)
    first = verifier.verify(workflow, manifest)
    verified = apply_artifact_verification(workflow, manifest, first, verifier)
    role = ArtifactRole.FINAL_MODEL.value
    transport.contents[role] = transport.contents[role][:-1]
    with pytest.raises(ModalChatArtifactError):
        verifier.replay(verified, manifest, first)


def test_receipt_authentication_is_exact_and_key_bound(monkeypatch):
    verifier, reader, request, _ = _case(monkeypatch)
    workflow = _workflow(request)
    receipt = verifier.verify(workflow, reader.artifacts(request))
    assert verifier.authenticate(replace(receipt, tag="f" * 64)) is False
    other, _, _, _ = _case(monkeypatch)
    other._key = b"b" * 32
    assert other.authenticate(receipt) is False


def test_manifest_run_identity_and_total_bound_are_enforced(monkeypatch):
    verifier, reader, request, _ = _case(monkeypatch, maximum_total_bytes=1)
    workflow = _workflow(request)
    manifest = reader.artifacts(request)
    with pytest.raises(ModalChatArtifactError):
        verifier.verify(workflow, manifest)
    changed = type(manifest).build(
        run=TrainingRunRef("other", workflow.run.project_ref),
        provider_run=manifest.provider_run,
        artifacts=manifest.artifacts,
        artifact_source_digest=manifest.artifact_source_digest,
        canonical_evidence=manifest.canonical_evidence,
    )
    with pytest.raises(ModalChatArtifactError):
        verifier.verify(workflow, changed)


def test_replay_requires_exact_authenticated_latest_predecessor(monkeypatch):
    verifier, reader, request, _ = _case(monkeypatch)
    workflow = _workflow(request)
    manifest = reader.artifacts(request)
    first = verifier.verify(workflow, manifest)
    verified = apply_artifact_verification(workflow, manifest, first, verifier)
    with pytest.raises(ModalChatArtifactError):
        verifier.replay(verified, manifest, replace(first, tag="f" * 64))


def test_verification_fails_before_one_time_reader_binding(monkeypatch):
    _, reader, request, _ = _case(monkeypatch)
    verifier = ModalChatArtifactVerifier(
        foundation_authenticator=foundation_cases.Auth(),
        assessment_authenticator=foundation_cases.AssessmentAuth(),
        authority_ref="consumer-artifact-authority",
        key_ref="consumer-artifact-key",
        key=b"a" * 32,
        clock=Clock(),
    )
    with pytest.raises(ModalChatArtifactError):
        verifier.verify(_workflow(request), reader.artifacts(request))
    verifier.bind(reader=reader, foundation=Foundation(request))
    with pytest.raises(ModalChatArtifactError, match="already_bound"):
        verifier.bind(reader=reader, foundation=Foundation(request))


def test_clock_requires_shared_now_iso_shape() -> None:
    with pytest.raises(TypeError, match="clock"):
        ModalChatArtifactVerifier(
            foundation_authenticator=foundation_cases.Auth(),
            assessment_authenticator=foundation_cases.AssessmentAuth(),
            authority_ref="consumer-artifact-authority",
            key_ref="consumer-artifact-key",
            key=b"a" * 32,
            clock=lambda: "2026-09-09T00:00:00Z",
        )
