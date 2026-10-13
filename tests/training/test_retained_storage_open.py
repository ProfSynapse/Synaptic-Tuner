"""Existing-only retained state never creates a missing database or its schema."""
import sqlite3

import pytest

from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.training.modal_host_storage import ModalHostStorageV1, ModalHostStorageUnavailable


@pytest.fixture(params=['workflow', 'catalog'])
def store_kind(request):
    return request.param


def open_store(path, kind, **kwargs):
    if kind == 'workflow':
        return SQLiteDurableStoreV1(path, **kwargs)
    return ModalHostStorageV1(path, 'retained-test', **kwargs)


def failures(kind):
    return (OSError, ValueError, sqlite3.Error) if kind == 'workflow' else ModalHostStorageUnavailable


@pytest.fixture
def database(tmp_path):
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    return private / 'state.sqlite3'


def test_missing_database_is_not_created(database, store_kind):
    with pytest.raises(failures(store_kind)):
        open_store(database, store_kind, create=False)
    assert not database.exists()
    assert list(database.parent.iterdir()) == []


@pytest.mark.parametrize('empty_schema', [False, True])
def test_empty_or_unrelated_schema_is_not_initialized(database, store_kind, empty_schema):
    if empty_schema:
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE unrelated(value)')
    else:
        database.touch()
    database.chmod(0o600)
    before = database.read_bytes()
    with pytest.raises(failures(store_kind)):
        open_store(database, store_kind, create=False)
    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        names = connection.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()
        assert names == ([('unrelated',)] if empty_schema else [])


def test_removed_between_validation_and_connect_is_not_recreated(database, store_kind, monkeypatch):
    with open_store(database, store_kind):
        pass
    original = sqlite3.connect
    def remove_then_connect(*args, **kwargs):
        database.unlink()
        return original(*args, **kwargs)
    monkeypatch.setattr(sqlite3, 'connect', remove_then_connect)
    with pytest.raises(failures(store_kind)):
        open_store(database, store_kind, create=False)
    assert not database.exists()


@pytest.mark.parametrize('name', ['state.sqlite3', 'state?#% space.sqlite3'])
def test_default_create_and_existing_reopen_preserve_facts(database, store_kind, name):
    database = database.with_name(name)
    with open_store(database, store_kind) as store:
        if store_kind == 'workflow':
            assert store.create(partition='data', key='run', canonical=b'{}')
        else:
            store.attempts.claim('attempt', b'{}')
    with open_store(database, store_kind, create=False) as store:
        if store_kind == 'workflow':
            assert store.read(partition='data', key='run').canonical == b'{}'
            assert store.compare_and_swap(partition='data', key='run', expected_revision=1, canonical=b'{"read":1}')
        else:
            assert store.attempts.resolve('attempt') == b'{}'
            store.attempts.claim('second', b'{}')
    assert database.exists()


@pytest.mark.parametrize('create', [None, 0, 1, 'false'])
def test_create_requires_exact_bool_before_filesystem_effects(database, store_kind, create):
    with pytest.raises(TypeError, match='exact bool'):
        open_store(database, store_kind, create=create)
    assert list(database.parent.iterdir()) == []
