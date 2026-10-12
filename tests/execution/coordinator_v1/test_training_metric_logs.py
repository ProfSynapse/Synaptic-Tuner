"""Metric projection through existing Modal chunks and authenticated run paging."""

from __future__ import annotations

import hashlib
from dataclasses import replace
import pytest

from synaptic_tuner.api.v1.observations import (
    ObservationKind, TrainingPhaseObservedPayloadV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunLogsRequest
from tuner.execution.coordinator_v1.cursors import decode_cursor
from tuner.execution.coordinator_v1.model import (
    AuthenticatedProviderLogPageV1,
    ProviderLogPageContentV1,
)
from tuner.execution.observation_logs import (
    decode_training_metric_log,
    decode_training_progress_log,
    encode_training_metric_log,
    encode_training_progress_log,
)
from tuner.execution.providers.modal.contracts import canonical_json
from tuner.execution.providers.modal.coordinator_logs import (
    ModalCoordinatorLogChunk,
    validate_modal_log_chain,
)

from ..test_observation_logs import metric_record
from .test_operational_cas import _queued_store
from .test_operations_service import Reader, operations


def _chunk(sequence, previous_digest, entries):
    return ModalCoordinatorLogChunk(
        generation=1,
        sequence=sequence,
        previous_digest=previous_digest,
        payload_digest=hashlib.sha256(
            canonical_json([entry.to_dict() for entry in entries])
        ).hexdigest(),
        job_ref="job-metrics",
        effect_id="submit-metrics",
        plan_digest="a" * 64,
        invocation_nonce="nonce-metrics",
        records=entries,
    )


def test_metric_entries_round_trip_through_actual_modal_chunks_without_renumbering():
    run = TrainingRunRef("run-metric-1", "project-metric-1")
    records = tuple(metric_record(sequence, run=run) for sequence in range(3))
    entries = tuple(encode_training_metric_log(record, expected_run=run) for record in records)
    first = _chunk(0, "0" * 64, entries[:2])
    second = _chunk(1, first.chunk_digest, entries[2:])
    parsed = tuple(ModalCoordinatorLogChunk.parse(chunk.canonical_bytes) for chunk in (first, second))

    assert validate_modal_log_chain(parsed) == second.chunk_digest
    assert tuple(decode_training_metric_log(entry, expected_run=run)
                 for chunk in parsed for entry in chunk.records) == records
    assert records[0].payload.learning_rate == 0.00015

    # The codec preserves source numbers; the chain requires a visible prefix
    # starting at zero, so an omitted metric cannot be silently renumbered.
    gap = _chunk(0, "0" * 64, (entries[1],))
    with pytest.raises(ValueError, match="record sequence"):
        validate_modal_log_chain((gap,))
    gap_after_first = _chunk(1, first.chunk_digest, (
        encode_training_metric_log(metric_record(4, run=run), expected_run=run),
    ))
    with pytest.raises(ValueError, match="record sequence"):
        validate_modal_log_chain((first, gap_after_first))


class MetricReader(Reader):
    def __init__(self, *, with_capture_status=False, tail_visible=True):
        super().__init__()
        self.after_sequences = []
        self.with_capture_status = with_capture_status
        self.tail_visible = tail_visible

    def logs(self, request, query):
        self.log_calls += 1
        self.after_sequences.append(query.after_sequence)
        ref = request.provider_run.reference
        numbers = (0, 1) if query.after_sequence is None else (
            (2,) if self.tail_visible else ()
        )
        entries = tuple(
            encode_training_metric_log(metric_record(number, run=request.run),
                                       expected_run=request.run)
            for number in numbers
        )
        if self.with_capture_status and query.after_sequence is not None \
                and self.tail_visible:
            status = replace(
                metric_record(3, run=request.run),
                kind=ObservationKind.TRAINING_PHASE_OBSERVED,
                payload=TrainingPhaseObservedPayloadV1("metric_capture_exhausted"),
            )
            entries += (encode_training_progress_log(status, expected_run=request.run),)
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
            sum(entry.size_bytes for entry in entries),
            query.after_sequence is None,
            canonical_json({"source": "metric-reader"}),
            "reader-metrics",
            "1.0.0",
            "2026-10-07T10:15:30Z",
        )
        return AuthenticatedProviderLogPageV1.parse(
            AuthenticatedProviderLogPageV1(
                content, "authority-a", "key-a", "c" * 64,
            ).canonical_bytes
        )


def test_authenticated_runs_api_pages_metric_entries_with_bound_cursor():
    store, queued, foundation = _queued_store()
    reader = MetricReader()
    service, _ = operations(store, queued, foundation, reader=reader)

    first = service.logs(RunLogsRequest(queued.run, limit=2, maximum_bytes=4096))
    assert [entry.sequence for entry in first.entries] == [0, 1]
    assert first.truncated is True
    assert decode_cursor(first.next_cursor).content.after_sequence == 1
    assert [decode_training_metric_log(entry, expected_run=queued.run).payload.learning_rate
            for entry in first.entries] == [0.00015, 0.00015]

    second = service.logs(RunLogsRequest(
        queued.run, cursor=first.next_cursor, limit=2, maximum_bytes=4096,
    ))
    assert [entry.sequence for entry in second.entries] == [2]
    assert second.truncated is False
    assert second.next_cursor is None
    assert reader.after_sequences == [None, 1]
    assert decode_training_metric_log(second.entries[0], expected_run=queued.run).sequence == 2


def test_authenticated_runs_api_exports_only_fixed_capture_exhaustion_status():
    store, queued, foundation = _queued_store()
    reader = MetricReader(with_capture_status=True, tail_visible=False)
    service, _ = operations(store, queued, foundation, reader=reader)

    first = service.logs(RunLogsRequest(queued.run, limit=2, maximum_bytes=4096))
    assert [entry.sequence for entry in first.entries] == [0, 1]
    assert first.truncated is True and first.next_cursor is not None
    retained_cursor = first.next_cursor

    pending = service.logs(RunLogsRequest(
        queued.run, cursor=retained_cursor, limit=2, maximum_bytes=4096,
    ))
    assert pending.entries == () and pending.next_cursor is None
    reader.tail_visible = True

    second = service.logs(RunLogsRequest(
        queued.run, cursor=retained_cursor, limit=2, maximum_bytes=4096,
    ))
    assert [entry.sequence for entry in second.entries] == [2, 3]
    assert second.truncated is False and second.next_cursor is None
    assert decode_training_metric_log(second.entries[0], expected_run=queued.run).sequence == 2
    status = decode_training_progress_log(second.entries[1], expected_run=queued.run)
    assert status.kind is ObservationKind.TRAINING_PHASE_OBSERVED
    assert status.payload.phase == "metric_capture_exhausted"
    with pytest.raises(ValueError, match="invalid training metric log projection"):
        decode_training_metric_log(second.entries[1], expected_run=queued.run)

    # A terminal public page has no next cursor. Polling from the retained
    # previous cursor is valid, and sequence-based deduplication is explicit.
    replay = service.logs(RunLogsRequest(
        queued.run, cursor=retained_cursor, limit=2, maximum_bytes=4096,
    ))
    deduped = {entry.sequence: entry for entry in (*first.entries, *second.entries,
                                                  *replay.entries)}
    assert tuple(deduped) == (0, 1, 2, 3)
    assert reader.after_sequences == [None, 1, 1, 1]
