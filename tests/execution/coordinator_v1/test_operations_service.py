from __future__ import annotations

import hashlib

import pytest

from synaptic_tuner.api.v1.results import TrainingRunState, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest,
    RunListRequest,
    RunLogEntry,
    RunLogLevel,
    RunLogsRequest,
    RunOperationCode,
    RunOperationError,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.coordinator_v1.model import (
    ArtifactManifestV1,
    AuthenticatedProviderLogPageV1,
    AuthenticatedProviderRunObservationV1,
    ProviderLogPageContentV1,
    ProviderRunObservationContentV1,
    ProviderRunPhaseV1,
    VerificationVerdictV1,
    WorkflowPhaseV1,
)
from tuner.execution.coordinator_v1.operations import TrainingOperationsV1
from tuner.execution.coordinator_v1.cursors import (
    HMACCursorAuthorityV1, decode_cursor,
)

from .test_operational_cas import _queued_store, _succeeded_store
from .test_start_reconcile_service import Clock, Planning, PlanningStore
from .test_state_machine import (
    AssessmentAuth,
    Auth,
    D,
    ObservationAuth,
    Verifier,
    assessment,
    verification_receipt,
)


class Foundation:
    def __init__(self, record):
        self.record = record
        self.get_calls = 0

    def get(self, effect_id):
        self.get_calls += 1
        return self.record if self.record.command.operation.effect.effect_id == effect_id else None

    def assess(self, record):
        assert record == self.record
        return assessment(record)

    def authenticate(self, value):
        return AssessmentAuth().authenticate(value)


class LogAuth:
    def __init__(self, allowed=True):
        self.allowed = allowed

    def authenticate(self, value):
        return self.allowed


class Reader:
    def __init__(self, *, data=b"artifact-data"):
        self.data = data
        self.observe_calls = 0
        self.log_calls = 0
        self.artifact_calls = 0
        self.stream_calls = 0

    def observe(self, request):
        self.observe_calls += 1
        ref = request.provider_run.reference
        content = ProviderRunObservationContentV1(
            "synaptic-provider-run-observation-content/v1",
            request.request_digest,
            request.source_workflow_record_digest,
            request.source_revision,
            request.run,
            request.provider_run.binding_digest,
            ref.provider_id,
            ref.profile_ref,
            ref.account_ref,
            ref.namespace_ref,
            ref.provider_job_ref,
            ProviderRunPhaseV1.RUNNING,
            canonical_bytes({"phase": "running"}),
            None,
            "reader-a",
            "1.0.0",
            "2026-08-27T12:00:00Z",
        )
        return AuthenticatedProviderRunObservationV1.parse(
            AuthenticatedProviderRunObservationV1(
                content, "authority-a", "key-a", "b" * 64
            ).canonical_bytes
        )

    def logs(self, request, query):
        self.log_calls += 1
        ref = request.provider_run.reference
        entries = (
            RunLogEntry(
                2 if query.after_sequence is None else query.after_sequence + 2,
                "2026-08-27T12:00:00Z",
                RunLogLevel.INFO,
                "trainer.progress",
                "ok",
                2,
            ),
        )
        content = ProviderLogPageContentV1(
            "synaptic-provider-log-page-content/v1",
            request.request_digest,
            query.log_query_digest,
            request.source_workflow_record_digest,
            request.source_revision,
            request.run,
            request.provider_run.binding_digest,
            ref.provider_id,
            ref.profile_ref,
            ref.account_ref,
            ref.namespace_ref,
            ref.provider_job_ref,
            query.after_sequence,
            entries,
            2,
            True,
            canonical_bytes({"normalized": True}),
            "reader-a",
            "1.0.0",
            "2026-08-27T12:00:00Z",
        )
        return AuthenticatedProviderLogPageV1.parse(
            AuthenticatedProviderLogPageV1(
                content, "authority-a", "key-a", "c" * 64
            ).canonical_bytes
        )

    def artifacts(self, request):
        self.artifact_calls += 1
        artifact = VerifiedArtifact(
            "adapter", hashlib.sha256(self.data).hexdigest(), len(self.data)
        )
        return ArtifactManifestV1.build(
            run=request.run,
            provider_run=request.provider_run.reference,
            artifacts=(artifact,),
            artifact_source_digest=D[5],
            canonical_evidence=canonical_bytes({"inventory": 1}),
        )

    def iter_artifact_bytes(self, request, manifest, role, *, maximum_bytes):
        self.stream_calls += 1
        yield self.data


class ArtifactVerifier(Verifier):
    def verify(self, workflow, manifest):
        return verification_receipt(
            workflow, manifest, VerificationVerdictV1.VERIFIED, "d"
        )

    def replay(self, workflow, manifest, prior_receipt):
        return verification_receipt(
            workflow, manifest, VerificationVerdictV1.VERIFIED, "e"
        )


class CoordinatorStub:
    def __init__(self, workflow):
        self.workflow = workflow
        self.cancel_calls = 0
        self.reconcile_calls = 0

    def cancel(self, run, reason):
        self.cancel_calls += 1
        return self.workflow

    def reconcile(self, run):
        self.reconcile_calls += 1
        return self.workflow


def operations(
    store, workflow, foundation, *, reader=None, verifier=None, log_auth=None,
    cursor_authority=None,
):
    reader = reader or Reader()
    verifier = verifier or ArtifactVerifier()
    return TrainingOperationsV1(
        Planning(),
        PlanningStore(),
        store,
        CoordinatorStub(workflow),
        Foundation(foundation),
        Auth(),
        AssessmentAuth(),
        reader,
        ObservationAuth(),
        log_auth or LogAuth(),
        verifier,
        cursor_authority or HMACCursorAuthorityV1(
            "cursor-authority", {1: b"k" * 32}, active_generation=1
        ),
        Clock(),
    ), reader


def test_list_show_outcome_and_authenticated_logs_use_b1_projections() -> None:
    store, queued, foundation_record = _queued_store()
    service, reader = operations(store, queued, foundation_record)

    page = service.list(RunListRequest(queued.run.project_ref, limit=10))
    assert page.outcomes == (service.show(queued.run),)
    assert page.outcomes[0].state is TrainingRunState.QUEUED
    assert reader.observe_calls == 0

    refreshed = service.outcome(queued.run)
    assert refreshed.state is TrainingRunState.RUNNING
    assert reader.observe_calls == 1

    logs = service.logs(RunLogsRequest(queued.run, limit=10, maximum_bytes=4096))
    assert logs.entries[0].sequence == 2
    assert decode_cursor(logs.next_cursor).content.after_sequence == 2
    assert logs.truncated is True
    assert reader.log_calls == 1


def test_invalid_cursor_and_nonboolean_log_authentication_close_before_output() -> None:
    store, queued, foundation_record = _queued_store()
    service, _ = operations(
        store, queued, foundation_record, log_auth=LogAuth("truthy")
    )
    with pytest.raises(RunOperationError) as invalid:
        service.list(RunListRequest(queued.run.project_ref, "bad.cursor"))
    assert invalid.value.code is RunOperationCode.CURSOR_INVALID
    with pytest.raises(RunOperationError) as denied:
        service.logs(RunLogsRequest(queued.run, limit=10, maximum_bytes=4096))
    assert denied.value.code is RunOperationCode.PROVIDER_READ_INVALID


def test_verify_reverify_and_single_use_artifact_stream() -> None:
    verifier = ArtifactVerifier()
    store, succeeded, foundation_record = _succeeded_store(verifier=verifier)
    reader = Reader()
    service, _ = operations(
        store, succeeded, foundation_record, reader=reader, verifier=verifier
    )

    verification = service.verify(succeeded.run)
    assert verification.verified is True
    assert reader.artifact_calls == 1
    reverified = service.reverify(succeeded.run)
    assert reverified.verified is True
    assert reader.artifact_calls == 1

    stream = service.artifacts(
        RunArtifactRequest(succeeded.run, "adapter", len(reader.data))
    )
    assert tuple(stream.iter_bytes()) == (reader.data,)
    assert reader.stream_calls == 1
    with pytest.raises(RunOperationError) as repeated:
        tuple(stream.iter_bytes())
    assert repeated.value.code is RunOperationCode.ARTIFACT_CONTENT_INVALID


def test_artifact_limit_rejects_before_reader_invocation() -> None:
    verifier = ArtifactVerifier()
    store, succeeded, foundation_record = _succeeded_store(verifier=verifier)
    reader = Reader()
    service, _ = operations(
        store, succeeded, foundation_record, reader=reader, verifier=verifier
    )
    service.verify(succeeded.run)
    with pytest.raises(RunOperationError) as caught:
        service.artifacts(RunArtifactRequest(succeeded.run, "adapter", 1))
    assert caught.value.code is RunOperationCode.ARTIFACT_LIMIT_EXCEEDED
    assert reader.stream_calls == 0


def test_cancel_and_reconcile_delegate_only_after_pinned_capability_checks() -> None:
    store, queued, foundation_record = _queued_store()
    service, _ = operations(store, queued, foundation_record)
    coordinator = service._coordinator
    assert service.cancel(queued.run, "requested").state is TrainingRunState.QUEUED
    assert service.reconcile(queued.run).state is TrainingRunState.QUEUED
    assert coordinator.cancel_calls == coordinator.reconcile_calls == 1


def test_raw_coordinator_failure_is_closed_without_exception_chain() -> None:
    store, queued, foundation_record = _queued_store()
    service, _ = operations(store, queued, foundation_record)

    class RawFailure:
        def cancel(self, run, reason):
            raise RuntimeError("secret-provider-detail")

        def reconcile(self, run):
            raise RuntimeError("secret-provider-detail")

    service._coordinator = RawFailure()
    for operation in (
        lambda: service.cancel(queued.run, "requested"),
        lambda: service.reconcile(queued.run),
    ):
        with pytest.raises(RunOperationError) as caught:
            operation()
        assert caught.value.code is RunOperationCode.STATE_CONFLICT
        assert "secret-provider-detail" not in str(caught.value)
        assert "secret-provider-detail" not in repr(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_zero_byte_artifact_yields_no_chunks_and_never_calls_reader_stream() -> None:
    verifier = ArtifactVerifier()
    store, succeeded, foundation_record = _succeeded_store(verifier=verifier)
    reader = Reader(data=b"")
    service, _ = operations(
        store, succeeded, foundation_record, reader=reader, verifier=verifier
    )
    service.verify(succeeded.run)
    stream = service.artifacts(RunArtifactRequest(succeeded.run, "adapter", 1))
    assert tuple(stream.iter_bytes()) == ()
    assert reader.stream_calls == 0
