"""Qualification uses the real transition kernel and native artifact reader."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from examples.modal_chat.host import ModalChatHost
from examples.modal_chat.qualification import (
    ModalChatQualificationError,
    qualify_modal_chat_run,
)
from examples.modal_chat.storage import ModalChatStorage
from examples.modal_chat.storage import ModalChatStorageError
from tuner.execution.coordinator_v1.foundation import (
    FoundationRecordAssessmentAuthorityV1,
)
from tuner.execution.coordinator_v1.model import (
    ProviderReadPurposeV1,
    ProviderRunPhaseV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    WorkflowTransitionError,
    apply_artifact_verification,
    apply_provider_observation,
    provider_run_read_request,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.receipts import (
    InvalidEvidenceAuthorityV2,
    ReceiptAuthorityV2,
)

import tests.execution.coordinator_v1.test_state_machine as state_cases
from tests.examples.test_modal_chat_artifacts import Foundation, _case as artifact_case


class Store:
    def __init__(self, workflow, *, reject_at=None):
        self.value, self.reject_at, self.calls = workflow, reject_at, 0

    def get(self, run):
        return self.value if run == self.value.run else None

    def compare_and_swap(self, expected, replacement, *, transition):
        self.calls += 1
        if self.calls == self.reject_at or self.value != expected:
            return False
        if self.calls == 1:
            assert replacement == apply_provider_observation(
                expected,
                transition.request,
                transition.observation,
                state_cases.ObservationAuth(),
            )
        else:
            assert replacement == apply_artifact_verification(
                expected,
                transition.manifest,
                transition.receipt,
                self.verifier,
            )
        self.value = replacement
        return True


class Reader:
    def __init__(self, native, observation):
        self.native, self.observation, self.observe_calls = native, observation, 0

    def observe(self, request):
        self.observe_calls += 1
        return self.observation

    def native_artifacts(self, request):
        return self.native.native_artifacts(request)


def _storage(tmp_path):
    tmp_path.chmod(0o700)
    return ModalChatStorage(tmp_path / "qualification.sqlite3", "qualification-tests")


def _fixture(monkeypatch, tmp_path, *, reject_at=None, substituted=False):
    verifier, native_reader, artifact_request, _ = artifact_case(monkeypatch)
    queued, foundation_record = state_cases.queued_evidence()
    request, observation = state_cases.observation(
        queued,
        foundation_record,
        ProviderRunPhaseV1.SUCCEEDED,
        {"completed": True},
    )
    if substituted:
        observation = replace(observation, tag="f" * 64)
    reader = Reader(native_reader, observation)
    store = Store(queued, reject_at=reject_at)
    store.verifier = verifier
    host = ModalChatHost(
        SimpleNamespace(),
        SimpleNamespace(foundation=Foundation(request)),
        reader,
        SimpleNamespace(),
        SimpleNamespace(
            foundation_authenticator=state_cases.Auth(),
            assessment_authority=state_cases.AssessmentAuth(),
        ),
        SimpleNamespace(workflow_store=store),
        verifier,
        SimpleNamespace(),
        state_cases.ObservationAuth(not substituted),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    return host, store, reader, _storage(tmp_path), queued.run


def test_one_shot_qualification_retains_terminal_observation_and_five_artifacts(
    monkeypatch,
    tmp_path,
):
    host, store, reader, storage, run = _fixture(monkeypatch, tmp_path)
    with storage:
        result = qualify_modal_chat_run(host=host, storage=storage, run=run)
        assert result.workflow == store.value
        assert result.workflow.phase.value == "verified"
        assert len(result.workflow.verified_artifacts) == 5
        assert len(result.accepted_evidence) < 16 * 1024
        accepted_parts = storage.catalog(
            "modal-chat-qualification-parts",
            encode=lambda value: value,
            decode=lambda value: value,
        )
        assert (
            accepted_parts.resolve(
                "sha256-"
                + hashlib.sha256(store.value.submit.canonical_command_bytes).hexdigest()
            )
            == store.value.submit.canonical_command_bytes
        )
        assert (
            max(
                map(
                    len,
                    (
                        result.accepted_evidence,
                        result.observation_evidence,
                        result.verification_evidence,
                    ),
                )
            )
            < 16 * 1024
        )
        receipt_bytes = result.workflow.verification_receipts[-1].canonical_bytes
        assert (
            accepted_parts.resolve(
                "sha256-" + hashlib.sha256(receipt_bytes).hexdigest()
            )
            == receipt_bytes
        )
        assert reader.observe_calls == 1
        with pytest.raises(ModalChatQualificationError):
            qualify_modal_chat_run(host=host, storage=storage, run=run)
    assert reader.observe_calls == 1


def test_delayed_qualification_reuses_retained_assessment_without_reissuing(
    monkeypatch, tmp_path
):
    class AdvancedClock:
        value = "2026-08-26T00:00:00Z"

        def now_iso(self):
            return self.value

    clock = AdvancedClock()
    issuer = FoundationRecordAssessmentAuthorityV1(
        "delayed-assessments",
        "delayed-assessment-key",
        b"a" * 32,
        assessor_ref="delayed-assessor",
        assessor_version="1.0.0",
        clock=clock,
        receipt_authority=ReceiptAuthorityV2("receipt-authority", b"r" * 32),
        invalid_evidence_authority=InvalidEvidenceAuthorityV2(
            "invalid-authority", b"i" * 32
        ),
        grant_authority=GrantAuthorityV2("grant-authority", b"g" * 32),
    )
    monkeypatch.setattr(state_cases, "assessment", issuer.assess)
    host, store, reader, storage, run = _fixture(monkeypatch, tmp_path)
    foundation = host.composition.foundation
    retained_at = store.value.submit.foundation_bindings[-1]
    original = issuer.assess(foundation.request.foundation_record)
    assert original.canonical_bytes == retained_at.canonical_assessment_bytes
    host.foundation_ports.assessment_authority = issuer
    clock.value = "2026-08-26T00:00:01Z"
    fresh = issuer.assess(foundation.request.foundation_record)
    assert (
        replace(fresh.content, assessed_at=original.content.assessed_at)
        == original.content
    )
    assert fresh.content.assessed_at == "2026-08-26T00:00:01Z"
    with pytest.raises(WorkflowTransitionError):
        provider_run_read_request(
            store.value,
            foundation.request.foundation_record,
            fresh,
            host.foundation_ports.foundation_authenticator,
            host.foundation_ports.assessment_authority,
            purpose=ProviderReadPurposeV1.OBSERVE,
        )
    foundation.assess = issuer.assess
    with storage:
        result = qualify_modal_chat_run(host=host, storage=storage, run=run)
    assert result.workflow == store.value
    assert result.workflow.phase.value == "verified"
    assert reader.observe_calls == 1


def test_retained_assessment_authentication_denial_fails_before_claim_or_read(
    monkeypatch, tmp_path
):
    host, _, reader, storage, run = _fixture(monkeypatch, tmp_path)
    host.foundation_ports.assessment_authority.allowed = False
    attempt_ref = (
        "qualify-" + hashlib.sha256(canonical_bytes(run.to_dict())).hexdigest()
    )

    with storage:
        with pytest.raises(ModalChatQualificationError):
            qualify_modal_chat_run(host=host, storage=storage, run=run)
        assert storage.attempts.resolve(attempt_ref) is None
    assert reader.observe_calls == 0


def test_tampered_retained_assessment_fails_before_claim_or_provider_read(
    monkeypatch, tmp_path
):
    host, _, reader, storage, run = _fixture(monkeypatch, tmp_path)
    binding = host.stores.workflow_store.value.submit.foundation_bindings[-1]
    object.__setattr__(binding, "canonical_assessment_bytes", b"{}")

    attempt_ref = (
        "qualify-" + hashlib.sha256(canonical_bytes(run.to_dict())).hexdigest()
    )
    with storage:
        with pytest.raises(ModalChatQualificationError):
            qualify_modal_chat_run(host=host, storage=storage, run=run)
        assert storage.attempts.resolve(attempt_ref) is None
    assert reader.observe_calls == 0


def test_nonterminal_observation_is_closed_without_running_fiction(
    monkeypatch, tmp_path
):
    host, store, reader, storage, run = _fixture(monkeypatch, tmp_path)
    queued = store.value
    _, reader.observation = state_cases.observation(
        queued,
        host.composition.foundation.request.foundation_record,
        ProviderRunPhaseV1.RUNNING,
        {"running": True},
    )
    with storage, pytest.raises(ModalChatQualificationError):
        qualify_modal_chat_run(host=host, storage=storage, run=run)
    assert store.value == queued
    assert reader.observe_calls == 1


def test_authenticated_failure_is_cas_retained_before_closed_result(
    monkeypatch, tmp_path
):
    host, store, reader, storage, run = _fixture(monkeypatch, tmp_path)
    queued = store.value
    _, reader.observation = state_cases.observation(
        queued,
        host.composition.foundation.request.foundation_record,
        ProviderRunPhaseV1.FAILED,
        {"failed": True},
        "provider_failed",
    )
    with storage, pytest.raises(ModalChatQualificationError):
        qualify_modal_chat_run(host=host, storage=storage, run=run)
    assert store.value.phase.value == "failed"
    assert store.value.provider_run_observations == (reader.observation,)


def test_substituted_observation_signature_is_rejected(monkeypatch, tmp_path):
    host, store, reader, storage, run = _fixture(
        monkeypatch,
        tmp_path,
        substituted=True,
    )
    with storage, pytest.raises(ModalChatQualificationError):
        qualify_modal_chat_run(host=host, storage=storage, run=run)
    assert store.calls == 0
    assert reader.observe_calls == 1


@pytest.mark.parametrize("reject_at", [1, 2])
def test_cas_failure_is_closed_and_never_reobserves(monkeypatch, tmp_path, reject_at):
    host, _, reader, storage, run = _fixture(
        monkeypatch,
        tmp_path,
        reject_at=reject_at,
    )
    with storage, pytest.raises(ModalChatQualificationError):
        qualify_modal_chat_run(host=host, storage=storage, run=run)
    assert reader.observe_calls == 1


def test_exact_host_storage_and_run_types_are_required(monkeypatch, tmp_path):
    host, _, _, storage, run = _fixture(monkeypatch, tmp_path)
    with storage, pytest.raises(TypeError):
        qualify_modal_chat_run(host=object(), storage=storage, run=run)


def test_content_addressed_part_tamper_is_rejected(monkeypatch, tmp_path):
    host, _, _, storage, run = _fixture(monkeypatch, tmp_path)
    with storage:
        result = qualify_modal_chat_run(host=host, storage=storage, run=run)
        receipt = result.workflow.verification_receipts[-1].canonical_bytes
        item_ref = "sha256-" + hashlib.sha256(receipt).hexdigest()
        storage._connection.execute(
            "UPDATE catalog_items SET payload=? WHERE catalog_ref=? AND item_ref=?",
            (b"{}", "modal-chat-qualification-parts", item_ref),
        )
        catalog = storage.catalog(
            "modal-chat-qualification-parts",
            encode=lambda value: value,
            decode=lambda value: value,
        )
        with pytest.raises(ModalChatStorageError):
            catalog.resolve(item_ref)
