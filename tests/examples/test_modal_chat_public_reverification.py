"""Joined public verification and native Modal source-admission regression."""

from __future__ import annotations

from examples.modal_chat.artifacts import ModalChatArtifactVerifier
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tests.examples.test_modal_chat_artifacts import _case, _workflow
import tests.execution.coordinator_v1.test_state_machine as foundation_cases
from tuner.execution.coordinator_v1.operations import RunOperationsV1
from tuner.execution.providers.modal.coordinator_adapter import _descriptor
from tuner.execution.providers.modal.contracts import (
    ArtifactMemberV1,
    provider_entry_identity,
)
from tuner.execution.providers.modal.inference_binding import (
    ModalInferenceSourceBinder,
)


class _Store:
    def __init__(self, value):
        self.value = value

    def get(self, run):
        return self.value if self.value.run == run else None

    def compare_and_swap(self, current, replacement, *, transition):
        if self.value != current:
            return False
        self.value = replacement
        return True

    def is_descendant(self, ancestor, descendant):
        return False


def test_public_verify_reverify_then_native_source_bind_uses_real_artifact_authority(
    monkeypatch,
) -> None:
    verifier, reader, request, transport = _case(monkeypatch)
    assert type(verifier) is ModalChatArtifactVerifier
    transport.members = tuple(
        ArtifactMemberV1(
            member.role,
            member.path,
            member.size,
            member.sha256,
            provider_entry_identity(
                transport.identity.artifact_volume_id, member.path, member.size
            ),
        )
        for member in transport.members
    )
    store = _Store(_workflow(request))
    initial_revision = store.value.revision
    operations = RunOperationsV1(
        planning=None,
        planning_store=None,
        workflow_store=store,
        coordinator=None,
        foundation=verifier._foundation,
        foundation_authenticator=foundation_cases.Auth(),
        assessment_authenticator=foundation_cases.AssessmentAuth(),
        reader=reader,
        observation_authenticator=None,
        log_authenticator=None,
        artifact_verifier=verifier,
        cursor_authority=None,
        clock=verifier._clock,
    )
    monkeypatch.setattr(
        operations,
        "_descriptor",
        lambda workflow: _descriptor(),
    )
    runs = RunsAPI(operations)
    run = store.value.run

    verified = runs.verify(run)
    assert verified.verified is True
    replayed = runs.reverify(run)
    assert replayed.verified is True

    binder = ModalInferenceSourceBinder(
        runs=runs,
        workflows=store,
        foundation=verifier._foundation,
        foundation_authenticator=foundation_cases.Auth(),
        assessment_authenticator=foundation_cases.AssessmentAuth(),
        reader=reader,
    )
    source = binder.bind(runs, run)

    assert source.run == run
    assert source.artifacts == store.value.verified_artifacts
    assert store.value.revision == initial_revision + 3
    assert transport.calls.count("bytes") == 3 * len(source.artifacts)
