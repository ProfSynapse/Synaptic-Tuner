from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from threading import Event
import time

import pytest

from tuner.training.modal_host_authority import HMACAuthenticator, ReaderEvidenceAuthority, UTCClock
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.coordinator_v1.model import (
    EffectIntentV1, ProviderReadPurposeV1, ProviderRunReadRequestV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.repository import EffectState
from tuner.execution.foundation_v2.references import ScopedProviderRunRefV1
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.packaged_reader import (
    ModalPackagedArtifactMember, ModalPackagedCompletionObservation,
)
from tuner.training.modal_host_reader import (
    ModalPackagedCoordinatorReaderV1, ModalPackagedReadUnavailable,
)
import tuner.training.modal_host_reader as host_reader_module

from tests.execution.providers.test_modal_packaged_dispatch import _case
from tests.execution.coordinator_v1.test_state_machine import (
    Auth as FoundationAuth, AssessmentAuth, assessment, record,
)


class _FakeFacade:
    def __init__(self, state, effect_id, completion_digest):
        self.state = state
        self.client = object()
        facade = self

        class Call:
            is_hydrated = True
            object_id = "fc-1"

            def hydrate(self, client):
                assert client is facade.client
                return self

            def get(self, *, timeout):
                assert timeout == 0
                if facade.state is ModalFunctionCallState.PENDING:
                    raise TimeoutError()
                return {
                    "schema_version": "synaptic-modal-packaged-worker-result/v1",
                    "effect_id": effect_id,
                    "status_code": ("failed" if facade.state == "failed" else "completed"),
                    "completion_sha256": completion_digest,
                }

        class FunctionCall:
            @staticmethod
            def from_id(ref, *, client):
                assert ref == "fc-1" and client is facade.client
                return Call()

        self.sdk = SimpleNamespace(FunctionCall=FunctionCall)


class _FakePackagedReader:
    def __init__(self, completion):
        self.completion = completion

    def observe_completion(self, binding, *, provider_job_ref):
        assert binding.command_digest == self.completion.command_digest
        assert provider_job_ref == self.completion.provider_job_ref
        return self.completion

    def iter_artifact(self, _binding, _observation, *, role, maximum_bytes):
        assert maximum_bytes >= 4 and role == "final_model"
        yield b"data"


class _ProjectedReader(ModalPackagedCoordinatorReaderV1):
    def _binding(self, _request, _purpose):
        return self.binding


def _reader(state):
    binding, _, _, _ = _case()
    ref = ScopedProviderRunRefV1(
        "modal", binding.command.preparation.provider.profile_ref,
        binding.provider_facts.account_ref,
        binding.command.preparation.scope.namespace_ref, "fc-1",
    )
    roles = sorted((
        "workload_record", "training_lineage", "training_metrics",
        "final_model", "tokenizer",
    ))
    members = tuple(ModalPackagedArtifactMember(
        role, f"operations/x/output/{role}", 4, sha256(b"data").hexdigest(),
        f"entry-{role}",
    ) for role in roles)
    completion = ModalPackagedCompletionObservation(
        binding.command.operation.effect.effect_id, binding.command_digest,
        "fc-1", binding.runtime_release.manifest_digest,
        binding.provider_binding.binding_digest,
        binding.execution_binding.binding_digest, "a" * 64, members,
    )
    auth = ReaderEvidenceAuthority(
        "modal-reader", "reader-key",
        HMACAuthenticator(
            {"reader-key": b"x" * 32},
            allowed_purposes=frozenset({
                "provider-run-observation/v1", "provider-log-page/v1",
            }),
        ),
    )
    reader = _ProjectedReader(
        bindings=None, binding_authority=None, foundation_repository=None,
        foundation_authenticator=None, assessment_authenticator=None,
        packaged_reader=_FakePackagedReader(completion),
        facade=_FakeFacade(state, completion.effect_id, completion.completion_digest),
        evidence_authority=auth, clock=UTCClock(),
    )
    reader.binding = binding
    request = SimpleNamespace(
        provider_run=SimpleNamespace(reference=ref, binding_digest="b" * 64),
        request_digest="c" * 64, source_workflow_record_digest="d" * 64,
        source_revision=1, run=TrainingRunRef(binding.execution_binding.run_ref, "project"),
    )
    return reader, request, auth


def test_returned_call_requires_completion_then_projects_five_artifacts():
    reader, request, auth = _reader(ModalFunctionCallState.RETURNED)
    observation = reader.observe(request)
    assert auth.authenticate_observation(observation)
    assert observation.content.phase.value == "succeeded"
    manifest = reader.artifacts(request)
    assert len(manifest.artifacts) == 5
    assert b"".join(reader.iter_artifact_bytes(
        request, manifest, "final_model", maximum_bytes=4,
    )) == b"data"


def test_pending_call_does_not_manufacture_queued_or_running_state():
    reader, request, _ = _reader(ModalFunctionCallState.PENDING)
    with pytest.raises(ModalPackagedReadUnavailable, match="pending"):
        reader.observe(request)
    with pytest.raises(ModalPackagedReadUnavailable, match="logs_unsupported"):
        reader.logs(request, None)


def test_failed_packaged_result_stops_with_fixed_failure_diagnostic():
    reader, request, _ = _reader("failed")
    with pytest.raises(ModalPackagedReadUnavailable, match="call_failed"):
        reader.observe(request)


def test_blocked_sdk_hydration_has_bounded_fixed_read_failure():
    reader, _request, _ = _reader(ModalFunctionCallState.RETURNED)
    release = Event()

    class BlockedCall:
        object_id = "fc-1"
        is_hydrated = False

        def hydrate(self, _client):
            release.wait(5)
            return self

    reader._facade.sdk.FunctionCall = SimpleNamespace(
        from_id=lambda *_args, **_kwargs: BlockedCall(),
    )
    started = time.monotonic()
    try:
        with pytest.raises(ModalPackagedReadUnavailable) as error:
            reader._poll_packaged_call(
                reader.binding, "fc-1", deadline=started + 0.05,
            )
        assert time.monotonic() - started < 1
        assert str(error.value) == "modal_packaged_call_unknown"
    finally:
        release.set()


def test_artifact_stream_sanitizes_provider_iterator_exception():
    reader, request, _ = _reader(ModalFunctionCallState.RETURNED)
    manifest = reader.artifacts(request)

    def broken(*_args, **_kwargs):
        yield b"data"
        raise RuntimeError("private-token-and-volume-path")

    reader._reader.iter_artifact = broken
    with pytest.raises(ModalPackagedReadUnavailable) as error:
        b"".join(reader.iter_artifact_bytes(
            request, manifest, "final_model", maximum_bytes=4,
        ))
    assert str(error.value) == "modal_packaged_artifact_stream_unavailable"
    assert "private-token" not in str(error.value)


def test_blocked_artifact_stream_has_bounded_fixed_read_failure(monkeypatch):
    reader, request, _ = _reader(ModalFunctionCallState.RETURNED)
    manifest = reader.artifacts(request)
    release = Event()
    monkeypatch.setattr(host_reader_module, "_ARTIFACT_READ_STEP_SECONDS", 0.05)

    def blocked(*_args, **_kwargs):
        release.wait(5)
        yield b"data"

    reader._reader.iter_artifact = blocked
    started = time.monotonic()
    try:
        with pytest.raises(ModalPackagedReadUnavailable) as error:
            b"".join(reader.iter_artifact_bytes(
                request, manifest, "final_model", maximum_bytes=4,
            ))
        assert time.monotonic() - started < 1
        assert str(error.value) == "modal_packaged_artifact_stream_unavailable"
    finally:
        release.set()


def test_reader_authenticates_retained_foundation_assessment_after_clock_advance():
    binding, _, _, _ = _case()
    command = binding.command
    intent = EffectIntentV1.from_command_bytes(command.canonical_bytes)
    preparation = command.preparation
    ref = ScopedProviderRunRefV1(
        preparation.provider.provider_id, preparation.provider.profile_ref,
        preparation.scope.account_ref, preparation.scope.namespace_ref, "fc-1",
    )
    current = record(intent, EffectState.FOUND, ref)
    retained = assessment(current)
    derived_binding, outcome, provider_run = _derive_foundation(
        intent, current, retained, FoundationAuth(), AssessmentAuth(), None,
    )
    run = TrainingRunRef(preparation.run_id, preparation.project_ref)
    document = {
        "schema_version": "synaptic-provider-run-read-request/v1",
        "purpose": "observe", "source_workflow_record_digest": "d" * 64,
        "source_revision": 1, "run": run.to_dict(),
        "provider_run_binding_digest": provider_run.binding_digest,
        "submit_command_bytes_digest": domain_digest(
            "synaptic-foundation-command-bytes/v1", command.canonical_bytes,
        ),
        "foundation_record_digest": current.record_digest,
        "assessment_digest": retained.authenticated_assessment_digest,
        "foundation_binding_digest": derived_binding.binding_digest,
        "foundation_outcome_digest": outcome.outcome_digest,
        "found_receipt_digest": provider_run.authenticated_receipt_digest,
    }
    raw = canonical_bytes(document)
    request = ProviderRunReadRequestV1(
        ProviderReadPurposeV1.OBSERVE, "d" * 64, 1, run,
        provider_run, command.canonical_bytes, current, retained,
        derived_binding, outcome, provider_run.authenticated_receipt_digest,
        raw, domain_digest("synaptic-provider-run-read-request/v1", raw),
    )

    class Catalog:
        def resolve(self, digest):
            return binding if digest == binding.command_digest else None

    class Authority:
        def authenticate(self, candidate):
            return candidate == binding

    class Repository:
        def get(self, effect_id):
            return current if effect_id == intent.effect_id else None

    reader = ModalPackagedCoordinatorReaderV1(
        bindings=Catalog(), binding_authority=Authority(),
        foundation_repository=Repository(),
        foundation_authenticator=FoundationAuth(),
        assessment_authenticator=AssessmentAuth(),
        packaged_reader=None, facade=None, evidence_authority=None,
        clock=SimpleNamespace(now=lambda: "2026-10-01T00:00:00Z"),
    )
    assert reader._binding(request, ProviderReadPurposeV1.OBSERVE) == binding
