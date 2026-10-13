"""Actual SQLite host reopen; fake provider state is deliberately retained.

These tests qualify stored host facts and new host operations, not provider
restart, live remote recovery, restored credentials or new effect authority.
"""
import pytest

from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.runs_facade import RunListRequest, RunOperationCode, RunOperationError
from tuner.execution.coordinator_v1.coordinator import CoordinatorErrorV1
from tests.contract.test_reference_composition_v1 import (
    RUN, _fixture, _forge, _head, _start, _stores,
)


@pytest.fixture
def database(tmp_path):
    private = tmp_path / 'host'
    private.mkdir(mode=0o700)
    return private / 'workflow.sqlite'


def _effects(fake, kind):
    return fake.trace.snapshot().count(('executor', kind))


def test_new_sqlite_host_retains_plan_grants_effects_and_never_resubmits(database):
    with SQLiteDurableStoreV1(database) as first_store:
        first, api, fake, grants, _, _ = _fixture(records=first_store, streams=first_store)
        plan, _, _ = _start(api)
        assert api.runs.outcome(RUN).state.value == 'running'
        retained = first.stores.workflow_store.get(RUN)
        original = _stores(first_store)
        assert {'workflow', 'plan', 'plan_context', 'preparation', 'execution_grant',
                'authorization', 'effects'} <= set(original)
        assert grants.bound == ['stage', 'submit']
        assert _effects(fake, 'submit') == 1
    # The database connection and all composed host operation objects are new.
    # SAME fake stands for a remote provider whose state survived host closure.
    with SQLiteDurableStoreV1(database) as second_store:
        second, again, _, grants, _, _ = _fixture(records=second_store, streams=second_store, family=fake)
        assert second.stores.workflow_store.get(RUN) == retained
        assert second.stores.workflow_store.get_by_plan(RUN.project_ref, plan.plan_fingerprint) == retained
        assert _stores(second_store) == original
        assert again.runs.show(RUN).state.value == 'running'
        assert tuple(item.run for item in again.runs.list(RunListRequest(RUN.project_ref)).outcomes) == (RUN,)
        with pytest.raises(CoordinatorErrorV1) as refused:
            _start(again)
        assert refused.value.code.value == 'invalid_input'
        assert _stores(second_store) == original
        assert grants.bound == [] and _effects(fake, 'submit') == 1
        assert again.runs.outcome(RUN).state.value == 'succeeded'
        assert _effects(fake, 'submit') == 1


def test_sqlite_reopen_reconcile_and_cancel_use_retained_commitments(database):
    with SQLiteDurableStoreV1(database) as store:
        _, api, fake, _, _, _ = _fixture(records=store, streams=store,
                                        policies={'submit': 'indeterminate_found'})
        _start(api)
        assert api.runs.show(RUN).state.value == 'reconcile_required'
    with SQLiteDurableStoreV1(database) as store:
        _, api, _, grants, _, _ = _fixture(records=store, streams=store, family=fake,
                                          policies={'submit': 'indeterminate_found'})
        assert api.runs.reconcile(RUN).state.value == 'queued'
        assert grants.bound == [] and _effects(fake, 'submit') == 1
        assert api.runs.outcome(RUN).state.value == 'running'
        assert api.runs.cancel(RUN, 'explicit request').state.value == 'cancel_requested'
        assert grants.bound == ['cancel'] and _effects(fake, 'cancel') == 1
        assert sorted(key.split('-')[0] for key in _stores(store)['effects']) == ['cancel', 'stage', 'submit']
    with SQLiteDurableStoreV1(database) as store:
        _, api, _, grants, _, _ = _fixture(records=store, streams=store, family=fake)
        assert api.runs.show(RUN).state.value == 'cancel_requested'
        api.runs.cancel(RUN, 'repeat request')
        assert grants.bound == [] and _effects(fake, 'cancel') == 1


def test_sqlite_cold_host_rejects_tampered_authorization_without_effect(database):
    with SQLiteDurableStoreV1(database) as store:
        _, api, fake, _, _, _ = _fixture(records=store, streams=store,
                                        policies={'submit': 'indeterminate_found'})
        _start(api)
        (key,) = [key for key in _stores(store)['authorization'] if key.startswith('policy/')]
        stored = store.read(partition='authorization', key=key)
        forged = _forge(stored, lambda value: value.__setitem__('expires_at_epoch', 2**40))
        store._connection.execute('UPDATE records SET canonical=? WHERE partition=? AND key=?',
                                  (forged, 'authorization', key))
        trace = fake.trace.snapshot()
    with SQLiteDurableStoreV1(database) as store:
        _, api, _, grants, _, _ = _fixture(records=store, streams=store, family=fake)
        with pytest.raises(RunOperationError) as refused:
            api.runs.reconcile(RUN)
        assert refused.value.code is RunOperationCode.STATE_CONFLICT
        assert api.runs.show(RUN).state.value == 'reconcile_required'
        assert grants.bound == [] and fake.trace.snapshot() == trace


def test_sqlite_cold_host_rejects_tampered_history(database):
    with SQLiteDurableStoreV1(database) as store:
        composition, api, fake, _, _, _ = _fixture(records=store, streams=store)
        _start(api)
        head = _head(store)
        history = store.read_page(partition=StoragePartition.WORKFLOW.value,
                                  stream_key=head.key, after_sequence=None, limit=100).entries
        assert len(history) > 1
        store._connection.execute('UPDATE streams SET canonical=? WHERE partition=? AND stream_key=? AND sequence=?',
            (b'{}', 'workflow', head.key, history[-1].sequence))
        assert composition.stores.workflow_store._validated
    with SQLiteDurableStoreV1(database) as store:
        _, api, _, grants, _, _ = _fixture(records=store, streams=store, family=fake)
        with pytest.raises(RunOperationError):
            api.runs.show(RUN)
        assert grants.bound == [] and _effects(fake, 'submit') == 1


class _TrackingSQLite(SQLiteDurableStoreV1):
    def __init__(self, database):
        self.workflow_pairs = []
        super().__init__(database)

    def commit_record_and_append(self, **kwargs):
        result = super().commit_record_and_append(**kwargs)
        if kwargs['record_partition'] == 'workflow':
            self.workflow_pairs.append((kwargs['expected_revision'], kwargs['expected_tail'],
                                        kwargs['sequence'], result))
        return result


def test_public_sqlite_host_uses_atomic_genesis_and_transitions(database):
    with _TrackingSQLite(database) as store:
        _, api, fake, _, _, _ = _fixture(records=store, streams=store)
        _start(api)
        assert store.workflow_pairs[0] == (None, None, 0, True)
        assert len(store.workflow_pairs) > 1
        assert all(result is True for _, _, _, result in store.workflow_pairs)
        before = len(store.workflow_pairs)
        assert api.runs.outcome(RUN).state.value == 'running'
        assert len(store.workflow_pairs) > before
        assert _effects(fake, 'submit') == 1


def test_reopened_host_atomic_rejection_has_no_orphan_event(database, monkeypatch):
    with SQLiteDurableStoreV1(database) as store:
        _, api, fake, _, _, _ = _fixture(records=store, streams=store)
        _start(api)
    with SQLiteDurableStoreV1(database) as store:
        _, api, _, _, _, _ = _fixture(records=store, streams=store, family=fake)
        head_before = _head(store)
        history_before = store.read_page(partition='workflow', stream_key=head_before.key,
                                         after_sequence=None, limit=100)
        monkeypatch.setattr(store, 'commit_record_and_append', lambda **kwargs: False)
        with pytest.raises(RunOperationError):
            api.runs.outcome(RUN)
        assert _head(store) == head_before
        assert store.read_page(partition='workflow', stream_key=head_before.key,
                               after_sequence=None, limit=100) == history_before
        assert _effects(fake, 'submit') == 1
