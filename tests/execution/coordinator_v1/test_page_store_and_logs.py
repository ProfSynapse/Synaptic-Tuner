from __future__ import annotations

from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, ProviderPlanRef, TrainingPlan
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest,
    RunListRequest,
    RunLogEntry,
    RunLogLevel,
    RunLogPage,
    RunLogsRequest,
    RunOperationCode,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.coordinator_v1.model import (
    AuthenticatedProviderLogPageV1,
    ProviderLogPageContentV1,
    ProviderLogQueryV1,
    WorkflowRecordV1,
    WorkflowStorePageV1,
)
from tuner.execution.coordinator_v1.stores import CoordinatorStoreCode, CoordinatorStoreError

from .test_state_machine import BASIS, CONTEXT, DESC, D, PLAN, PROVIDER
from .test_strong_cas import workflow_store


def _workflow(number: int) -> WorkflowRecordV1:
    basis = replace(BASIS, request_id=f"request-page-{number}")
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1",
        PROVIDER,
        basis.basis_digest,
        DESC.descriptor_digest,
        D[5],
    )
    plan = TrainingPlan(
        "synaptic-training-plan/v2",
        basis,
        ProviderPlanRef(context.provider_context_digest),
    )
    run = TrainingRunRef(f"run-page-{number}", BASIS.project_ref)
    return WorkflowRecordV1.planned(
        run=run,
        plan=plan,
        preflight_digest=D[6],
        context=context,
        provider=PROVIDER,
        descriptor=DESC,
    )


def _run_key(record: WorkflowRecordV1) -> str:
    return domain_digest(
        "synaptic-coordinator-run-key/v1",
        canonical_bytes(record.run.to_dict()),
    )


def _entry(sequence: int, message: str = "ok") -> RunLogEntry:
    return RunLogEntry(
        sequence,
        "2026-08-27T12:00:00Z",
        RunLogLevel.INFO,
        "trainer.progress",
        message,
        len(message.encode("utf-8")),
    )


def _log_content(*, evidence: bytes = canonical_bytes({"source": "reader"})) -> ProviderLogPageContentV1:
    entries = (_entry(3), _entry(8, "done"))
    return ProviderLogPageContentV1(
        schema_version="synaptic-provider-log-page-content/v1",
        read_request_digest=D[0],
        log_query_digest=D[1],
        source_workflow_record_digest=D[2],
        source_revision=7,
        run=TrainingRunRef("run-1", "project-1"),
        provider_run_binding_digest=D[3],
        provider_id="provider-1",
        profile_ref="profile-1",
        account_ref="account-1",
        namespace_ref="namespace-1",
        provider_job_ref="job-1",
        after_sequence=1,
        entries=entries,
        total_bytes=sum(item.size_bytes for item in entries),
        truncated=True,
        canonical_evidence=evidence,
        reader_ref="reader-1",
        reader_version="1.0.0",
        read_at="2026-08-27T12:00:00Z",
    )


def test_workflow_store_page_is_digest_sorted_and_requires_exact_cursor_key() -> None:
    store = workflow_store()
    records = tuple(_workflow(number) for number in range(3))
    for record in reversed(records):
        assert store.create(record) is True
    ordered = tuple(sorted(records, key=_run_key))

    first = store.list_page(BASIS.project_ref, after_run_key=None, limit=2)
    assert first == WorkflowStorePageV1(ordered[:2], True)
    second = store.list_page(
        BASIS.project_ref, after_run_key=_run_key(first.records[-1]), limit=2
    )
    assert second == WorkflowStorePageV1(ordered[2:], False)

    with pytest.raises(CoordinatorStoreError) as caught:
        store.list_page(BASIS.project_ref, after_run_key=D[14], limit=2)
    assert caught.value.code is CoordinatorStoreCode.BINDING_MISMATCH


def test_workflow_store_page_closes_run_key_collisions(monkeypatch) -> None:
    store = workflow_store()
    assert store.create(_workflow(1)) is True
    assert store.create(_workflow(2)) is True
    monkeypatch.setattr(
        "tuner.execution.coordinator_v1.stores.domain_digest", lambda *_: D[0]
    )
    with pytest.raises(CoordinatorStoreError) as caught:
        store.list_page(BASIS.project_ref, after_run_key=None, limit=100)
    assert caught.value.code is CoordinatorStoreCode.INTEGRITY_ERROR


@pytest.mark.parametrize("limit", [False, 0, 101])
def test_workflow_store_page_rejects_noncanonical_limits(limit) -> None:
    with pytest.raises(CoordinatorStoreError):
        workflow_store().list_page(BASIS.project_ref, after_run_key=None, limit=limit)


def test_provider_log_envelope_round_trips_with_gapped_sequences() -> None:
    page = AuthenticatedProviderLogPageV1(_log_content(), "authority-1", "key-1", D[4])
    assert AuthenticatedProviderLogPageV1.parse(page.canonical_bytes) == page
    assert page.content.entries[1].sequence == 8


def test_provider_log_content_rejects_oversized_evidence_before_parsing() -> None:
    with pytest.raises(ValueError, match="65536"):
        _log_content(evidence=b"{" + b"x" * 65535 + b"}")


def test_provider_log_content_accepts_canonical_evidence_above_foundation_limit() -> None:
    evidence = b'{"blob":"' + b"x" * 20000 + b'"}'
    page = AuthenticatedProviderLogPageV1(
        _log_content(evidence=evidence), "authority-1", "key-1", D[4]
    )
    assert AuthenticatedProviderLogPageV1.parse(page.canonical_bytes) == page


def test_provider_log_content_accepts_exact_65536_byte_canonical_evidence() -> None:
    evidence = b'{"blob":"' + b"x" * 65525 + b'"}'
    assert len(evidence) == 65536
    assert _log_content(evidence=evidence).canonical_evidence == evidence


@pytest.mark.parametrize(
    "entries",
    [(_entry(3), _entry(3)), (_entry(8), _entry(3)), (_entry(1),)],
)
def test_provider_log_content_rejects_duplicate_regressed_or_cursor_stale_sequences(entries) -> None:
    with pytest.raises(ValueError):
        replace(
            _log_content(),
            entries=entries,
            total_bytes=sum(item.size_bytes for item in entries),
        )


def test_facade_bounds_and_exact_operation_vocabulary() -> None:
    run = TrainingRunRef("run-1", "project-1")
    assert RunListRequest("project-1", limit=100).limit == 100
    assert RunLogsRequest(run, limit=200, maximum_bytes=262144).maximum_bytes == 262144
    assert RunArtifactRequest(run, "model", 2**63 - 1).maximum_bytes == 2**63 - 1
    assert {item.value for item in RunOperationCode} == {
        "run_missing", "cursor_invalid", "capability_unavailable", "read_ineligible",
        "provider_read_invalid", "log_bounds_invalid", "cancel_ineligible",
        "artifacts_unverified", "artifact_role_missing", "artifact_limit_exceeded",
        "artifact_content_invalid", "state_conflict", "integrity_error",
    }


def test_run_log_page_accepts_gaps_but_enforces_exact_byte_bound() -> None:
    run = TrainingRunRef("run-1", "project-1")
    request = RunLogsRequest(run, limit=2, maximum_bytes=4096)
    entries = (_entry(2), _entry(9, "complete"))
    assert RunLogPage(request, entries, 10).entries == entries
    with pytest.raises(ValueError):
        RunLogPage(request, entries, 9)


def test_provider_log_query_digest_binds_all_bounds() -> None:
    query = ProviderLogQueryV1(after_sequence=4, limit=20, maximum_bytes=8192)
    assert query.log_query_digest == domain_digest(
        "synaptic-provider-log-query/v1", query.canonical_bytes
    )
    with pytest.raises(ValueError):
        ProviderLogQueryV1(after_sequence=4, limit=201, maximum_bytes=8192)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-27Z",
        "2026-08-27",
        "2026-08-27 12:00:00Z",
        "20260827T120000Z",
        "2026-08-27T12:00Z",
        "2026-08-27T12:00:00",
        "2026-08-27T12:00:00z",
        "2026-08-27T12:00:00.Z",
        "2026-08-27T12:00:00+0100",
        "2026-08-27T12:00:00+24:00",
        "2026-02-30T12:00:00Z",
        "2026-08-27T25:00:00Z",
    ],
)
def test_public_and_authenticated_logs_reject_nonexact_rfc3339(timestamp) -> None:
    with pytest.raises(ValueError, match="exact RFC3339"):
        replace(_entry(1), timestamp=timestamp)
    with pytest.raises(ValueError, match="exact RFC3339"):
        replace(_log_content(), read_at=timestamp)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-27T12:00:00Z",
        "2026-08-27T12:00:00.1Z",
        "2026-08-27T12:00:00.123456+05:30",
        "2026-08-27T12:00:00-04:00",
    ],
)
def test_public_and_authenticated_logs_accept_exact_rfc3339(timestamp) -> None:
    assert replace(_entry(1), timestamp=timestamp).timestamp == timestamp
    content = replace(_log_content(), read_at=timestamp)
    page = AuthenticatedProviderLogPageV1(content, "authority-1", "key-1", D[4])
    assert AuthenticatedProviderLogPageV1.parse(page.canonical_bytes) == page
