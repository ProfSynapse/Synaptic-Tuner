"""Lifecycle integration contract independent of a host's database choice."""
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.data_facade import DataOperationError, DataRunState
from synaptic_tuner.api.v1.reference.atomic_storage import atomic_commit_method
from synaptic_tuner.api.v1.reference.data import _plan_key, data_run_ref
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1
from test_reference_data_v1 import FakeLocalClient, Harness


class AtomicMemory:
    """Test-only combined store: reject both writes before either is visible."""
    def __init__(self):
        self.records = InMemoryDurableRecordStoreV1()
        self.streams = InMemoryDurableStreamStoreV1()
        self.reject = False
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.records if hasattr(self.records, name) else self.streams, name)

    def create(self, **kwargs):
        return self.records.create(**kwargs)

    def read(self, **kwargs):
        return self.records.read(**kwargs)

    def compare_and_swap(self, **kwargs):
        return self.records.compare_and_swap(**kwargs)

    def put_if_absent(self, **kwargs):
        return self.records.put_if_absent(**kwargs)

    def list_page(self, **kwargs):
        return self.records.list_page(**kwargs)

    def append(self, **kwargs):
        return self.streams.append(**kwargs)

    def read_page(self, **kwargs):
        return self.streams.read_page(**kwargs)

    def commit_record_and_append(self, **kw):
        self.calls.append(kw)
        record = self.records.read(partition=kw['record_partition'], key=kw['key'])
        page = self.streams.read_page(partition=kw['stream_partition'], stream_key=kw['stream_key'],
                                      after_sequence=None, limit=100)
        revision = None if record is None else record.revision
        tail = page.entries[-1].sequence if page.entries else None
        if self.reject or revision != kw['expected_revision'] or tail != kw['expected_tail']:
            return False
        if revision is None:
            assert self.records.create(partition=kw['record_partition'], key=kw['key'], canonical=kw['record_canonical'])
        else:
            assert self.records.compare_and_swap(partition=kw['record_partition'], key=kw['key'],
                                                 expected_revision=revision, canonical=kw['record_canonical'])
        assert self.streams.append(partition=kw['stream_partition'], stream_key=kw['stream_key'],
                                   sequence=kw['sequence'], canonical=kw['stream_canonical'])
        return True


def test_atomic_rejected_transition_does_not_poison_cold_cancel(tmp_path):
    store = AtomicMemory()
    warm = Harness(tmp_path, FakeLocalClient([]), records=store, streams=store)
    plan = warm.api.plan(warm.request())
    run = data_run_ref(plan)
    result = warm.operations._initial_result(plan, run, DataRunState.PLANNED)
    warm.operations._genesis(plan, result)
    store.reject = True
    with pytest.raises(DataOperationError, match='state_conflict'):
        warm.operations._transition(run, lambda r: replace(r, state=DataRunState.RUNNING))
    assert warm.durable_events(run) == 1
    store.reject = False
    cold = Harness(tmp_path / 'cold', FakeLocalClient([]), records=store, streams=store)
    assert cold.api.cancel(run, 'stop').state is DataRunState.CANCEL_REQUESTED
    assert cold.durable_events(run) == 2


def test_missing_genesis_index_discovers_head_without_client_or_replay(tmp_path):
    store = AtomicMemory()
    client = FakeLocalClient(['text'] * 9)
    warm = Harness(tmp_path, client, records=store, streams=store)
    plan = warm.api.plan(warm.request())
    original = store.records.put_if_absent
    def refuse_index(**kwargs):
        if kwargs['key'] == _plan_key(plan):
            raise OSError('simulated crash boundary')
        return original(**kwargs)
    store.put_if_absent = refuse_index
    with pytest.raises(DataOperationError, match='integrity_error'):
        warm.api.start(plan)
    assert client.chat_calls == 0
    cold = Harness(tmp_path / 'cold', FakeLocalClient([]), records=store, streams=store)
    assert cold.api.start(plan).run == data_run_ref(plan)
    assert cold.api.show(data_run_ref(plan)).state is DataRunState.PLANNED


def test_optional_method_requires_same_object_and_exact_bool():
    store = AtomicMemory()
    assert atomic_commit_method(store, object()) is None
    store.commit_record_and_append = lambda **kwargs: 1
    with pytest.raises(TypeError, match='atomic_store_result_invalid'):
        atomic_commit_method(store, store)()


def test_evaluation_progress_uses_history_tail_not_record_revision(tmp_path):
    from test_reference_evaluation_v1 import ScriptedBackend, _host, _run
    store = AtomicMemory()
    _, api, _, _, _, _ = _host(tmp_path, backends={'local': ScriptedBackend()}, records=store, streams=store)
    plan, _, started = _run(api)
    completed = api.evaluation.show(started.run)
    assert completed.state.value == 'succeeded'
    evaluation_calls = [c for c in store.calls if c['record_partition'] == 'evaluation']
    assert any(c['expected_revision'] > c['expected_tail'] for c in evaluation_calls
               if c['expected_tail'] is not None)
    assert api.evaluation.start(plan) == started


def test_pipeline_genesis_and_transitions_use_atomic_pair(tmp_path):
    from test_reference_pipelines_v1 import ScriptedBackend, _host, _request
    store = AtomicMemory()
    _, api, _, _, _, _ = _host(tmp_path, backends={'local': ScriptedBackend()}, records=store, streams=store)
    plan = api.pipelines.plan(_request())
    started = api.pipelines.start(plan)
    assert api.pipelines.show(started.pipeline).state.value == 'succeeded'
    assert api.pipelines.start(plan) == started
    assert any(c['record_partition'] == 'pipeline' and c['expected_revision'] is None for c in store.calls)


@pytest.mark.parametrize('outcome', [False, 'error', 1])
def test_atomic_refusal_or_error_never_falls_back_to_split_genesis(tmp_path, outcome):
    store = AtomicMemory()
    warm = Harness(tmp_path, FakeLocalClient([]), records=store, streams=store)
    plan = warm.api.plan(warm.request())
    run = data_run_ref(plan)
    def refuse(**kwargs):
        if outcome == 'error':
            raise OSError('transaction uncertain')
        return outcome
    store.commit_record_and_append = refuse
    with pytest.raises(DataOperationError):
        warm.operations._genesis(plan, warm.operations._initial_result(plan, run, DataRunState.PLANNED))
    assert warm.operations._read_head(run) is None
    assert warm.durable_events(run) == 0


def test_capability_detection_does_not_invoke_property():
    class Host:
        @property
        def commit_record_and_append(self):
            pytest.fail('capability detection invoked property')
    host = Host()
    assert atomic_commit_method(host, host) is None


def test_sqlite_close_reopen_missing_index_no_duplicate_effects(tmp_path, monkeypatch):
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    path = private / 'host.sqlite3'
    store = SQLiteDurableStoreV1(path)
    try:
        client = FakeLocalClient(['text'] * 9)
        warm = Harness(tmp_path / 'warm', client, records=store, streams=store)
        plan = warm.api.plan(warm.request())
        original = SQLiteDurableStoreV1.put_if_absent
        def lost_index(self, **kwargs):
            if kwargs['key'] == _plan_key(plan):
                raise OSError('lost index publication')
            return original(self, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(SQLiteDurableStoreV1, 'put_if_absent', lost_index)
            with pytest.raises(DataOperationError, match='integrity_error'):
                warm.api.start(plan)
        assert client.chat_calls == 0
    finally:
        store.close()
    with SQLiteDurableStoreV1(path) as reopened:
        cold_client = FakeLocalClient([])
        cold = Harness(tmp_path / 'cold', cold_client, records=reopened, streams=reopened)
        run = cold.api.start(plan).run
        assert run == data_run_ref(plan) and cold_client.chat_calls == 0
        assert cold.api.cancel(run, 'cancel claimed run').state is DataRunState.CANCEL_REQUESTED
        assert cold.durable_events(run) == 2


def test_sqlite_evaluation_progress_reopen_cancel_uses_exact_tail(tmp_path):
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    from synaptic_tuner.api.v1.reference.evaluation import _RunRecord, _run_id
    from synaptic_tuner.api.v1.evaluation_facade import EvaluationRunRef, EvaluationRunState, EvaluationVerdictCounts
    from test_reference_evaluation_v1 import ScriptedBackend, _host, _request
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    path = private / 'host.sqlite3'
    backend = ScriptedBackend()
    with SQLiteDurableStoreV1(path) as store:
        _, api, _, _, _, _ = _host(tmp_path / 'warm', backends={'local': backend}, records=store, streams=store)
        plan = api.evaluation.plan(_request('alpha.yaml'))
        run = EvaluationRunRef(_run_id(plan), plan.request.project_ref)
        operations = api.evaluation._operations
        record = _RunRecord(run, plan, EvaluationRunState.PLANNED, 0, (), None, (), EvaluationVerdictCounts(0, 0, 0))
        head = operations._create(record)
        head = operations._transition(head, record.replace(state=EvaluationRunState.RUNNING), 'started')
        head = operations._progress(head, head.record)
        assert head.revision > head.sequence
    with SQLiteDurableStoreV1(path) as reopened:
        _, cold, _, _, _, _ = _host(tmp_path / 'cold', backends={'local': backend}, records=reopened, streams=reopened)
        assert cold.evaluation.cancel(run, 'stop').state is EvaluationRunState.CANCEL_REQUESTED
        assert cold.evaluation.start(plan).run == run
        assert not backend.opened


def test_sqlite_pipeline_plan_only_restart_and_existing_head_no_redrive(tmp_path):
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    from synaptic_tuner.api.v1.reference.pipelines import _pipeline_id
    from synaptic_tuner.api.v1.pipelines_facade import PipelineRef
    from test_reference_pipelines_v1 import ScriptedBackend, _host, _request
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    path = private / 'host.sqlite3'
    backend = ScriptedBackend()
    with SQLiteDurableStoreV1(path) as store:
        _, api, family, _, _, _ = _host(tmp_path / 'warm', backends={'local': backend}, records=store, streams=store)
        plan = api.pipelines.plan(_request())
        pipeline = PipelineRef(_pipeline_id(plan), plan.request.project_ref)
        api.pipelines._operations._store_plan(pipeline, plan)
    with SQLiteDurableStoreV1(path) as reopened:
        _, api, _, _, _, _ = _host(tmp_path / 'cold', backends={'local': backend}, family=family,
                                   records=reopened, streams=reopened)
        started = api.pipelines.start(plan)
        assert api.pipelines.show(started.pipeline).state.value == 'succeeded'
        opened = tuple(backend.opened)
    with SQLiteDurableStoreV1(path) as reopened:
        _, cold, _, _, _, _ = _host(tmp_path / 'cold2', backends={'local': backend}, family=family,
                                    records=reopened, streams=reopened)
        assert cold.pipelines.start(plan) == started and tuple(backend.opened) == opened


def test_sqlite_failed_transition_rolls_back_head_and_history_then_reopened_cancel(tmp_path):
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    path = private / 'host.sqlite3'
    with SQLiteDurableStoreV1(path) as store:
        warm = Harness(tmp_path / 'warm', FakeLocalClient([]), records=store, streams=store)
        plan = warm.api.plan(warm.request())
        run = data_run_ref(plan)
        warm.operations._genesis(plan, warm.operations._initial_result(plan, run, DataRunState.PLANNED))
        # Fault exactly between the transaction's record update and stream insert.
        store._connection.execute("CREATE TEMP TRIGGER fail_event BEFORE INSERT ON streams "
                                  "BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with pytest.raises(DataOperationError, match='integrity_error'):
            warm.operations._transition(run, lambda r: replace(r, state=DataRunState.RUNNING))
        assert warm.api.show(run).state is DataRunState.PLANNED
        assert warm.durable_events(run) == 1
    with SQLiteDurableStoreV1(path) as reopened:
        cold = Harness(tmp_path / 'cold', FakeLocalClient([]), records=reopened, streams=reopened)
        assert cold.api.cancel(run, 'stop after rejected transition').state is DataRunState.CANCEL_REQUESTED
        assert cold.durable_events(run) == 2
