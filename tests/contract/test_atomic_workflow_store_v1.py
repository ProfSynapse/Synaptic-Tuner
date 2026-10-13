"""Training workflow atomic writes preserve authenticated coordinator replay."""
import pytest

from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.execution.coordinator_v1.stores import CoordinatorStoreError
from test_reference_composition_v1 import _fixture, _start


@pytest.fixture
def authenticated_transition(monkeypatch):
    composition, api, family, _, _, _ = _fixture()
    workflow = composition.stores.workflow_store
    original = workflow.compare_and_swap
    captured = []
    def capture(expected, replacement, *, transition):
        captured.append((expected, replacement, transition))
        return original(expected, replacement, transition=transition)
    monkeypatch.setattr(workflow, 'compare_and_swap', capture)
    _start(api)
    assert captured
    return captured[0], family


def open_store(tmp_path):
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    return private / 'host.sqlite3'


def test_genesis_fault_retains_index_without_head_and_cannot_retry(tmp_path, authenticated_transition):
    (expected, _, _), family = authenticated_transition
    path = open_store(tmp_path)
    with SQLiteDurableStoreV1(path) as store:
        composition, _, _, _, _, _ = _fixture(records=store, streams=store, family=family)
        workflow = composition.stores.workflow_store
        store._connection.execute("CREATE TEMP TRIGGER reject_genesis BEFORE INSERT ON streams "
                                  "WHEN NEW.partition='workflow' BEGIN SELECT RAISE(ABORT, 'fault'); END")
        with pytest.raises(CoordinatorStoreError):
            workflow.create(expected)
        assert workflow.get(expected.run) is None
        assert not store.list_page(partition='workflow', prefix='run/', after_key=None, limit=10).records
    with SQLiteDurableStoreV1(path) as store:
        cold, _, _, _, _, _ = _fixture(records=store, streams=store, family=family)
        with pytest.raises(CoordinatorStoreError):
            cold.stores.workflow_store.create(expected)
        assert cold.stores.workflow_store.get(expected.run) is None


def test_transition_fault_rolls_back_pair_and_reopened_authenticated_cas_succeeds(tmp_path, authenticated_transition):
    (expected, replacement, transition), family = authenticated_transition
    path = open_store(tmp_path)
    with SQLiteDurableStoreV1(path) as store:
        composition, _, _, _, _, _ = _fixture(records=store, streams=store, family=family)
        workflow = composition.stores.workflow_store
        assert workflow.create(expected)
        store._connection.execute("CREATE TEMP TRIGGER reject_transition BEFORE INSERT ON streams "
                                  "WHEN NEW.partition='workflow' AND NEW.sequence>0 "
                                  "BEGIN SELECT RAISE(ABORT, 'fault'); END")
        with pytest.raises(CoordinatorStoreError):
            workflow.compare_and_swap(expected, replacement, transition=transition)
        assert workflow.get(expected.run) == expected
    with SQLiteDurableStoreV1(path) as store:
        cold, _, _, _, _, _ = _fixture(records=store, streams=store, family=family)
        workflow = cold.stores.workflow_store
        assert workflow.get(expected.run) == expected
        assert workflow.create(expected) is False
        assert workflow.compare_and_swap(expected, replacement, transition=transition)
        assert workflow.get(expected.run) == replacement


def test_two_hosts_same_expected_record_have_one_cas_winner(tmp_path, authenticated_transition):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    (expected, replacement, transition), family = authenticated_transition
    path = open_store(tmp_path)
    with SQLiteDurableStoreV1(path) as store:
        composition, _, _, _, _, _ = _fixture(records=store, streams=store, family=family)
        assert composition.stores.workflow_store.create(expected)
    barrier = Barrier(2)
    def compete():
        # Each host owns its own thread-affine connection and validated cursor.
        with SQLiteDurableStoreV1(path) as store:
            host, _, _, _, _, _ = _fixture(records=store, streams=store, family=family)
            workflow = host.stores.workflow_store
            assert workflow.get(expected.run) == expected
            barrier.wait(timeout=5)
            return workflow.compare_and_swap(expected, replacement, transition=transition)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(compete) for _ in range(2)]
        assert sorted(f.result(timeout=10) for f in futures) == [False, True]
    with SQLiteDurableStoreV1(path) as store:
        cold, _, _, _, _, _ = _fixture(records=store, streams=store, family=family)
        assert cold.stores.workflow_store.get(expected.run) == replacement
        key = cold.stores.workflow_store._head_key(expected.run)
        entries = store.read_page(partition='workflow', stream_key=key, after_sequence=None, limit=10).entries
        assert [e.sequence for e in entries] == [0, 1]


def test_identical_genesis_is_idempotent_but_same_plan_other_run_conflicts(tmp_path, authenticated_transition):
    from dataclasses import replace
    from synaptic_tuner.api.v1.results import TrainingRunRef
    (expected, _, _), family = authenticated_transition
    with SQLiteDurableStoreV1(open_store(tmp_path)) as store:
        host, _, _, _, _, _ = _fixture(records=store, streams=store, family=family)
        workflow = host.stores.workflow_store
        assert workflow.create(expected)
        assert workflow.create(expected) is False
        other = replace(expected, run=TrainingRunRef('other-run', expected.run.project_ref))
        with pytest.raises(CoordinatorStoreError, match='conflict'):
            workflow.create(other)
        assert workflow.get(other.run) is None
        assert workflow.get(expected.run) == expected
