from pathlib import Path
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1


@pytest.fixture
def database(tmp_path):
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    return private / 'state.sqlite'


def pair(store, *, revision=None, tail=None, sequence=0, record=b'{}', event=b'{}'):
    return store.commit_record_and_append(record_partition='data', key='run',
        expected_revision=revision, record_canonical=record, stream_partition='data',
        stream_key='history', expected_tail=tail, sequence=sequence, stream_canonical=event)


def test_records_and_reopen(database):
    with SQLiteDurableStoreV1(database) as store:
        assert store.create(partition='data', key='record', canonical=b'{}') is True
        assert store.create(partition='data', key='record', canonical=b'{}') is False
        assert store.put_if_absent(partition='data', key='record', canonical=b'{}') is True
        assert store.put_if_absent(partition='data', key='record', canonical=b'{"new":1}') is False
        assert not store.compare_and_swap(partition='data', key='record', expected_revision=2, canonical=b'{"new":1}')
        assert store.compare_and_swap(partition='data', key='record', expected_revision=1, canonical=b'{"new":1}')
        assert not store.put_if_absent(partition='data', key='record', canonical=b'{"new":1}')
    with SQLiteDurableStoreV1(database) as store:
        record = store.read(partition='data', key='record')
        assert record.revision == 2 and record.canonical == b'{"new":1}'
        assert store.read(partition='data', key='missing') is None


def test_literal_unicode_prefix_and_pagination(database):
    keys = ['a%', 'a%child', 'a_other', 'a[child', 'é', 'é😀', '😀', 'z']
    with SQLiteDurableStoreV1(database) as store:
        for key in reversed(keys):
            assert store.create(partition='data', key=key, canonical=b'{}')
        page = store.list_page(partition='data', prefix='a%', after_key=None, limit=1)
        assert [r.key for r in page.records] == ['a%'] and page.next_cursor == 'a%'
        assert [r.key for r in store.list_page(partition='data', prefix='a%', after_key=page.next_cursor, limit=10).records] == ['a%child']
        assert [r.key for r in store.list_page(partition='data', prefix='', after_key=None, limit=10).records] == sorted(keys)
        assert [r.key for r in store.list_page(partition='data', prefix='é', after_key=None, limit=10).records] == ['é', 'é😀']


def test_stream_semantics(database):
    with SQLiteDurableStoreV1(database) as store:
        for sequence in [0, 3, 10]:
            assert store.append(partition='data', stream_key='s', sequence=sequence, canonical=b'{}') is True
        for sequence in [0, 3, 9, 10]:
            assert store.append(partition='data', stream_key='s', sequence=sequence, canonical=b'{}') is False
        page = store.read_page(partition='data', stream_key='s', after_sequence=None, limit=2)
        assert [e.sequence for e in page.entries] == [0, 3]
        assert page.next_cursor == 3 and page.truncated
        assert [e.sequence for e in store.read_page(partition='data', stream_key='s', after_sequence=3, limit=2).entries] == [10]


def test_atomic_pair_exact_head_tail_and_reopen(database):
    with SQLiteDurableStoreV1(database) as store:
        assert pair(store) is True
        assert pair(store) is False
        for revision, tail in [(1, None), (None, 0), (2, 0), (1, 9)]:
            assert pair(store, revision=revision, tail=tail, sequence=10, record=b'{"bad":1}') is False
            assert store.read(partition='data', key='run').canonical == b'{}'
            assert len(store.read_page(partition='data', stream_key='history', after_sequence=None, limit=10).entries) == 1
        assert pair(store, revision=1, tail=0, sequence=2, record=b'{"new":1}', event=b'{"event":2}')
    with SQLiteDurableStoreV1(database) as store:
        assert store.read(partition='data', key='run').revision == 2
        assert store.read_page(partition='data', stream_key='history', after_sequence=None, limit=10).entries[-1].canonical == b'{"event":2}'


def test_atomic_stream_failure_rolls_back_record(database):
    with SQLiteDurableStoreV1(database) as store:
        store._connection.execute("CREATE TEMP TRIGGER reject_event BEFORE INSERT ON streams BEGIN SELECT RAISE(ABORT,'injected'); END")
        with pytest.raises(sqlite3.IntegrityError):
            pair(store)
        assert store.read(partition='data', key='run') is None
        assert store.read_page(partition='data', stream_key='history', after_sequence=None, limit=10).entries == ()


def _claim_process(path, barrier, results):
    with SQLiteDurableStoreV1(path) as store:
        barrier.wait(timeout=5)
        results.put(pair(store))


def test_cross_process_exactly_one_claim_winner(database):
    with SQLiteDurableStoreV1(database):
        pass
    ctx = multiprocessing.get_context('spawn')
    barrier, results = ctx.Barrier(2), ctx.Queue()
    children = [ctx.Process(target=_claim_process, args=(database, barrier, results)) for _ in range(2)]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0
        assert sorted(results.get(timeout=2) for _ in children) == [False, True]
    finally:
        for child in children:
            if child.pid is not None and child.is_alive():
                child.kill()
                child.join(timeout=5)
        results.close()
        results.join_thread()


def test_process_kill_between_record_and_stream_rolls_back(database):
    with SQLiteDurableStoreV1(database) as store:
        assert pair(store)
    source = str(Path(__file__).resolve().parents[2])
    script = '''
import os, signal, sys
sys.path.insert(0, sys.argv[1])
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
store=SQLiteDurableStoreV1(sys.argv[2])
def kill_between(sql):
    if sql.startswith('INSERT INTO streams'):
        os.kill(os.getpid(), signal.SIGKILL)
store._connection.set_trace_callback(kill_between)
store.commit_record_and_append(record_partition='data',key='run',expected_revision=1,
record_canonical=b'{"changed":1}',stream_partition='data',stream_key='history',
expected_tail=0,sequence=1,stream_canonical=b'{}')
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, source, str(database)], timeout=10)
    assert result.returncode == -9
    with SQLiteDurableStoreV1(database) as store:
        assert store.read(partition='data', key='run').revision == 1
        assert store.read(partition='data', key='run').canonical == b'{}'
        assert len(store.read_page(partition='data', stream_key='history', after_sequence=None, limit=10).entries) == 1


def test_unknown_schema_no_mutation(database):
    with SQLiteDurableStoreV1(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute('PRAGMA user_version=99')
    before = database.read_bytes()
    with pytest.raises(ValueError, match='schema'):
        SQLiteDurableStoreV1(database)
    assert database.read_bytes() == before


def test_unversioned_nonempty_schema_refused(database):
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE unrelated(value)')
    database.chmod(0o600)
    before = database.read_bytes()
    with pytest.raises(ValueError, match='schema'):
        SQLiteDurableStoreV1(database)
    assert database.read_bytes() == before


def test_thread_affinity(database):
    with SQLiteDurableStoreV1(database) as store:
        errors = []
        def other():
            try:
                store.read(partition='data', key='r')
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=other)
        thread.start()
        thread.join(timeout=5)
        assert errors == ['SQLite store is thread-affine']


@pytest.mark.parametrize('value', [True, 0, -1, float('inf'), 61])
def test_invalid_timeout(database, value):
    with pytest.raises(ValueError):
        SQLiteDurableStoreV1(database, timeout=value)
    assert not database.exists()


def test_revision_exhaustion_no_write(database):
    with SQLiteDurableStoreV1(database) as store:
        assert pair(store)
        store._connection.execute('UPDATE records SET revision=?', (2**63-1,))
        with pytest.raises(ValueError, match='exhausted'):
            store.compare_and_swap(partition='data', key='run', expected_revision=2**63-1, canonical=b'{"bad":1}')
        with pytest.raises(ValueError, match='exhausted'):
            pair(store, revision=2**63-1, tail=0, sequence=1)
        assert store.read(partition='data', key='run').revision == 2**63-1
        assert len(store.read_page(partition='data', stream_key='history', after_sequence=None, limit=10).entries) == 1


def test_private_symlink_hardlink_and_permission_rejection(database, tmp_path):
    other = tmp_path / 'other'
    other.write_text('')
    database.symlink_to(other)
    with pytest.raises(OSError):
        SQLiteDurableStoreV1(database)
    database.unlink()
    other.chmod(0o600)
    os.link(other, database)
    with pytest.raises(ValueError):
        SQLiteDurableStoreV1(database)
    database.unlink()
    database.parent.chmod(0o755)
    with pytest.raises(ValueError):
        SQLiteDurableStoreV1(database)


def test_invalid_atomic_values_before_transaction(database):
    with SQLiteDurableStoreV1(database) as store:
        with pytest.raises(ValueError):
            pair(store, tail=0, sequence=0)
        with pytest.raises((ValueError, TypeError)):
            pair(store, revision=True)
        with pytest.raises((ValueError, TypeError)):
            pair(store, event='not-bytes')
        assert not store._connection.in_transaction
        assert store.read(partition='data', key='run') is None


def test_atomic_schema_initialization_rolls_back_failed_second_table(database, monkeypatch):
    from synaptic_tuner.api.v1.reference import sqlite_store
    original = dict(sqlite_store._TABLES)
    monkeypatch.setitem(sqlite_store._TABLES, 'streams', 'INVALID TABLE DECLARATION')
    with pytest.raises(sqlite3.Error):
        SQLiteDurableStoreV1(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 0
        assert connection.execute('PRAGMA application_id').fetchone()[0] == 0
        assert connection.execute('SELECT name FROM sqlite_master').fetchall() == []
    monkeypatch.setattr(sqlite_store, '_TABLES', original)
    with SQLiteDurableStoreV1(database) as store:
        assert pair(store)


def test_unknown_sqlite_like_user_table_cannot_hide(database):
    with SQLiteDurableStoreV1(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE sqliteextra(value)')
    before = database.read_bytes()
    with pytest.raises(ValueError, match='schema'):
        SQLiteDurableStoreV1(database)
    assert database.read_bytes() == before


def test_busy_timeout_rolls_back_no_pair(database):
    with SQLiteDurableStoreV1(database) as first, SQLiteDurableStoreV1(database, timeout=0.02) as second:
        first._connection.execute('BEGIN IMMEDIATE')
        try:
            with pytest.raises(sqlite3.OperationalError):
                pair(second)
            assert not second._connection.in_transaction
        finally:
            first._connection.execute('ROLLBACK')
        assert second.read(partition='data', key='run') is None
        assert second.read_page(partition='data', stream_key='history', after_sequence=None, limit=10).entries == ()


def test_replaced_database_identity_rejects_operation(database):
    with SQLiteDurableStoreV1(database) as store:
        retained = database.with_name('retained.sqlite')
        database.rename(retained)
        database.write_bytes(b'foreign')
        database.chmod(0o600)
        with pytest.raises(ValueError, match='identity'):
            pair(store)
        assert database.read_bytes() == b'foreign'


@pytest.mark.parametrize('key', ['contains\x00nul', 'newline\n'])
def test_contract_forbids_control_keys_before_sql(database, key):
    with SQLiteDurableStoreV1(database) as store:
        with pytest.raises(ValueError):
            store.create(partition='data', key=key, canonical=b'{}')
        assert not store._connection.in_transaction


def test_append_sequence_maximum_and_partition_isolation(database):
    with SQLiteDurableStoreV1(database) as store:
        assert store.append(partition='data', stream_key='same', sequence=2**63-1, canonical=b'{}')
        assert not store.append(partition='data', stream_key='same', sequence=2**63-1, canonical=b'{}')
        assert store.append(partition='evaluation', stream_key='same', sequence=0, canonical=b'{}')
        assert store.read_page(partition='evaluation', stream_key='same', after_sequence=None, limit=1).entries[0].sequence == 0


def _first_open_process(path, barrier, results):
    barrier.wait(timeout=5)
    with SQLiteDurableStoreV1(path) as store:
        results.put(pair(store))


def test_simultaneous_first_open_initialization(database):
    assert not database.exists()
    ctx = multiprocessing.get_context('spawn')
    barrier, results = ctx.Barrier(2), ctx.Queue()
    children = [ctx.Process(target=_first_open_process, args=(database, barrier, results)) for _ in range(2)]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0
        assert sorted(results.get(timeout=2) for _ in children) == [False, True]
    finally:
        for child in children:
            if child.pid is not None and child.is_alive():
                child.kill()
                child.join(timeout=5)
        results.close()
        results.join_thread()
    with SQLiteDurableStoreV1(database) as store:
        assert store.read(partition='data', key='run').revision == 1
        assert len(store.read_page(partition='data', stream_key='history', after_sequence=None, limit=10).entries) == 1


def test_kill_after_successful_commit_preserves_both(database):
    with SQLiteDurableStoreV1(database):
        pass
    source = str(Path(__file__).resolve().parents[2])
    script = '''
import os,signal,sys
sys.path.insert(0,sys.argv[1])
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
store=SQLiteDurableStoreV1(sys.argv[2])
assert store.commit_record_and_append(record_partition='data',key='run',expected_revision=None,
record_canonical=b'{"committed":1}',stream_partition='data',stream_key='history',
expected_tail=None,sequence=0,stream_canonical=b'{"event":0}') is True
os.kill(os.getpid(),signal.SIGKILL)
'''
    result = subprocess.run([sys.executable,'-I','-c',script,source,str(database)],timeout=10)
    assert result.returncode == -9
    with SQLiteDurableStoreV1(database) as store:
        assert store.read(partition='data',key='run').canonical == b'{"committed":1}'
        assert store.read_page(partition='data',stream_key='history',after_sequence=None,limit=10).entries[0].canonical == b'{"event":0}'
