"""Public-page telemetry retention remains separate from training authority."""
from dataclasses import replace
import json
import os

import pytest
from synaptic_tuner.api.v1.observations import ObservationKind, TrainingPhaseObservedPayloadV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunLogPage
from tests.execution.test_observation_logs import metric_record
from tuner.execution.observation_logs import encode_training_progress_log
from tuner.training.run_progress_consumer import RunProgressConsumer, start_progress_consumer

RUN = TrainingRunRef('run-progress', 'project-progress')

class Pages:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.requests = []

    def logs(self, request):
        self.requests.append(request)
        entries, cursor = next(self.pages)
        if entries is None:
            raise ValueError('private provider error')
        return RunLogPage(request, tuple(entries), sum(e.size_bytes for e in entries),
                          cursor, cursor is not None)

def entry(sequence, **changes):
    record = replace(metric_record(sequence, run=RUN), **changes)
    return encode_training_progress_log(record, expected_run=RUN)

@pytest.fixture
def root(tmp_path):
    os.chmod(tmp_path, 0o700)
    return tmp_path

def test_growing_tail_retains_cursor_across_empty_and_failed_reads(root):
    source = Pages([([], None), ([entry(0), entry(1)], 'anchor'),
                    ([], None), (None, None), ([entry(2)], None), ([entry(2), entry(3)], None)])
    consumer = RunProgressConsumer(source, RUN, root)
    for _ in range(6):
        consumer.poll()
    consumer.close()
    assert [r.cursor for r in source.requests] == [None, None, 'anchor', 'anchor', 'anchor', 'anchor']
    assert consumer.status == 'progress_retained'
    assert consumer.result()['retained_records'] == 4
    assert [json.loads(line)['sequence'] for line in consumer.path.read_text().splitlines()] == [0, 1, 2, 3]
    assert consumer.path.stat().st_mode & 0o777 == 0o600
    assert all(r.limit == 100 and r.maximum_bytes == 65536 for r in source.requests)

def test_conflict_rejects_whole_page_preserving_previous_prefix(root):
    changed = replace(metric_record(1, run=RUN), occurred_at='2026-10-08T10:15:30Z')
    source = Pages([([entry(1)], None), ([entry(0), encode_training_progress_log(changed, expected_run=RUN)], 'bad')])
    consumer = RunProgressConsumer(source, RUN, root)
    consumer.poll()
    prior = consumer.path.read_bytes()
    consumer.poll()
    consumer.poll()
    assert consumer.status == 'progress_sequence_conflict'
    assert consumer.disabled and consumer.cursor is None
    assert consumer.path.read_bytes() == prior
    assert len(source.requests) == 2

def test_status_and_null_payload_are_preserved(root):
    metric = metric_record(0, run=RUN)
    metric = replace(metric, payload=replace(metric.payload, loss=None))
    exhausted = entry(1, kind=ObservationKind.TRAINING_PHASE_OBSERVED,
                      payload=TrainingPhaseObservedPayloadV1('metric_capture_exhausted'))
    consumer = RunProgressConsumer(Pages([([encode_training_progress_log(metric, expected_run=RUN), exhausted], None)]), RUN, root)
    consumer.poll()
    consumer.close()
    records = [json.loads(line) for line in consumer.path.read_text().splitlines()]
    assert records[0]['payload']['loss'] is None
    assert records[1]['payload'] == {'phase': 'metric_capture_exhausted'}

@pytest.mark.parametrize('capture,logs', [(False,True),(True,False),(False,False)])
def test_gate_has_no_calls_or_files(root, capture, logs):
    source = Pages([])
    assert start_progress_consumer(source, RUN, root, capture_enabled=capture, logs_enabled=logs) is None
    assert not source.requests and not list(root.iterdir())

@pytest.mark.parametrize('kind', ['existing','symlink','permissions'])
def test_private_create_only_storage_failure_disables_telemetry(root,kind):
    consumer = RunProgressConsumer(Pages([([entry(0)], None)]), RUN, root)
    if kind == 'existing':
        consumer.path.write_bytes(b'prior evidence')
    elif kind == 'symlink':
        consumer.path.symlink_to(root/'target')
    else:
        root.chmod(0o755)
    consumer.poll()
    assert consumer.status == 'progress_storage_unavailable' and consumer.disabled
    assert consumer.result()['retained_records'] == 0
    if kind == 'existing':
        assert consumer.path.read_bytes() == b'prior evidence'

def test_retention_budget_is_closed_and_does_not_accept_page(root):
    consumer = RunProgressConsumer(Pages([([entry(0)], 'tail')]), RUN, root)
    consumer.MAX_RECORDS = 0
    consumer.poll()
    assert consumer.status == 'progress_retention_exhausted'
    assert consumer.cursor is None and not consumer.path.exists()

def test_invalid_public_progress_disables_only_consumer(root):
    invalid = replace(entry(0), message='{}', size_bytes=2)
    consumer = RunProgressConsumer(Pages([([invalid],None)]), RUN, root)
    consumer.poll()
    assert consumer.status == 'progress_record_invalid'
    assert not consumer.path.exists()


def test_partial_write_retains_path_without_repair_or_retry(root, monkeypatch):
    consumer = RunProgressConsumer(Pages([([entry(0)], None)]), RUN, root)
    real_write = os.write
    def partial(fd, data):
        real_write(fd, data[:5])
        raise OSError('private storage error')
    monkeypatch.setattr(os, 'write', partial)
    consumer.poll()
    assert consumer.status == 'progress_storage_unavailable'
    assert consumer.result()['records_path'] == str(consumer.path)
    assert consumer.result()['retained_records'] == 0
    assert len(consumer.path.read_bytes()) == 5
    consumer.poll()
    assert len(consumer.path.read_bytes()) == 5


def test_symlink_directory_is_rejected(root):
    alias = root / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    consumer = RunProgressConsumer(Pages([([entry(0)],None)]), RUN, alias)
    consumer.poll()
    assert consumer.status == 'progress_storage_unavailable'
    assert not consumer.path.exists()
