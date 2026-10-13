"""Optional SQLite implementation of the existing host record and stream ports.

The caller owns an explicit database in an owner-private local directory. There
is no default engine database or authority service. Import this module explicitly;
normal API imports do not load SQLite. Connections are thread-affine. SQLite
coordinates other processes; process descriptors or provider authority are not
restored by reopening stored facts. Private-path checks are trusted-local
protection, not a hostile-filesystem sandbox or backup/restore mechanism.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import sqlite3
import stat
import threading

from ..ports import (StoredPageV1, StoredRecordV1, StoredStreamEntryV1,
    StoredStreamPageV1, require_canonical, require_key, require_page_limit,
    require_partition, require_prefix, require_revision, require_sequence)

_VERSION = 1
_MAX_REVISION = 2**63 - 1
_APPLICATION_ID = 0x53594E31
_TABLES = {
    'records': 'CREATE TABLE records(partition TEXT NOT NULL,key TEXT NOT NULL,revision INTEGER NOT NULL,canonical BLOB NOT NULL,PRIMARY KEY(partition,key))',
    'streams': 'CREATE TABLE streams(partition TEXT NOT NULL,stream_key TEXT NOT NULL,sequence INTEGER NOT NULL,canonical BLOB NOT NULL,PRIMARY KEY(partition,stream_key,sequence))',
}


class SQLiteDurableStoreV1:
    """One object implementing both ports and optional atomic head+history commit.

    Positive finite busy timeout (at most 60 seconds) bounds SQLite lock waits,
    not every OS/filesystem operation. No speculative migration is performed:
    only an empty unversioned database can be initialized, atomically. Unknown
    versions/schema layouts refuse before application writes. Use close/context
    management on the creating thread. Failure never triggers split-write fallback.
    ``create=False`` requires the existing versioned schema and opens with
    SQLite's read/write existing-only mode; it does not initialize empty files.
    """
    def __init__(self, database_path: str | Path, *, timeout: float = 5.0, create: bool = True):
        if type(create) is not bool:
            raise TypeError('create must be an exact bool')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError('finite SQLite timeout required')
        if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY'):
            raise RuntimeError('private SQLite paths require POSIX descriptor facilities')
        path = Path(database_path)
        if not path.is_absolute() or path.name in ('', '.', '..') or path.parent.resolve(strict=True) != path.parent:
            raise ValueError('absolute private database path required')
        self._owner = threading.get_ident()
        self._path = path
        self._parent_fd = None
        self._file_fd = None
        self._connection = None
        try:
            self._parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(self._parent_fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ValueError('private database parent must be owned mode 0700')
            if create:
                try:
                    self._file_fd = os.open(path.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                            0o600, dir_fd=self._parent_fd)
                except FileExistsError:
                    self._file_fd = os.open(path.name, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                                            dir_fd=self._parent_fd)
            else:
                self._file_fd = os.open(path.name, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                                        dir_fd=self._parent_fd)
            self._check_path()
            target = path if create else path.as_uri() + '?mode=rw'
            self._connection = sqlite3.connect(target, timeout=float(timeout), isolation_level=None, uri=not create)
            self._connection.execute('PRAGMA trusted_schema=OFF')
            self._connection.execute('PRAGMA synchronous=FULL')
            self._initialize(create=create)
            self._check_path()
        except BaseException:
            self.close()
            raise

    def _check_path(self):
        parent = os.fstat(self._parent_fd)
        named_parent = self._path.parent.stat()
        file = os.fstat(self._file_fd)
        named_file = os.stat(self._path.name, dir_fd=self._parent_fd, follow_symlinks=False)
        if ((parent.st_dev, parent.st_ino) != (named_parent.st_dev, named_parent.st_ino) or
            (file.st_dev, file.st_ino) != (named_file.st_dev, named_file.st_ino) or
            not stat.S_ISREG(file.st_mode) or file.st_nlink != 1 or file.st_uid != os.getuid() or
            stat.S_IMODE(file.st_mode) != 0o600 or parent.st_uid != os.getuid() or
            stat.S_IMODE(parent.st_mode) != 0o700):
            raise ValueError('private database identity or permissions changed')

    def _check(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('SQLite store is thread-affine')
        if self._connection is None:
            raise RuntimeError('SQLite store is closed')
        self._check_path()

    def _schema(self):
        version = self._connection.execute('PRAGMA user_version').fetchone()[0]
        application = self._connection.execute('PRAGMA application_id').fetchone()[0]
        objects = self._connection.execute("SELECT name,type,sql FROM sqlite_master WHERE substr(name,1,7) <> 'sqlite_' ORDER BY name").fetchall()
        return version, application, objects

    def _validate_schema(self, schema):
        version, application, objects = schema
        if version != _VERSION or application != _APPLICATION_ID or objects != [(name, 'table', _TABLES[name]) for name in sorted(_TABLES)]:
            raise ValueError('unsupported SQLite store schema')

    def _initialize(self, *, create=True):
        # One snapshot/lock: concurrent first-open cannot mix schema generations.
        self._connection.execute('BEGIN IMMEDIATE')
        try:
            current = self._schema()
            if current == (0, 0, []):
                if not create:
                    raise ValueError('existing SQLite store schema required')
                for statement in _TABLES.values():
                    self._connection.execute(statement)
                self._connection.execute(f'PRAGMA application_id={_APPLICATION_ID}')
                self._connection.execute(f'PRAGMA user_version={_VERSION}')
            else:
                self._validate_schema(current)
            self._check_path()
            self._connection.execute('COMMIT')
        except BaseException:
            self._connection.execute('ROLLBACK')
            raise

    @contextmanager
    def _transaction(self):
        self._check()
        self._connection.execute('BEGIN IMMEDIATE')
        try:
            yield self._connection
            self._check_path()
            self._connection.execute('COMMIT')
        except BaseException:
            self._connection.execute('ROLLBACK')
            raise

    def close(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('SQLite store is thread-affine')
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for attribute in ('_file_fd', '_parent_fd'):
            descriptor = getattr(self, attribute)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, attribute, None)

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()

    def create(self, *, partition, key, canonical):
        partition, key, canonical = require_partition(partition), require_key(key), require_canonical(canonical)
        with self._transaction() as connection:
            return connection.execute('INSERT OR IGNORE INTO records VALUES(?,?,1,?)', (partition, key, canonical)).rowcount == 1

    def read(self, *, partition, key):
        partition, key = require_partition(partition), require_key(key)
        self._check()
        row = self._connection.execute('SELECT revision,canonical FROM records WHERE partition=? AND key=?', (partition, key)).fetchone()
        return None if row is None else StoredRecordV1(key, row[0], row[1])

    def compare_and_swap(self, *, partition, key, expected_revision, canonical):
        partition, key = require_partition(partition), require_key(key)
        expected_revision, canonical = require_revision(expected_revision), require_canonical(canonical)
        if expected_revision == _MAX_REVISION:
            raise ValueError('record revision exhausted')
        with self._transaction() as connection:
            return connection.execute('UPDATE records SET revision=revision+1,canonical=? WHERE partition=? AND key=? AND revision=?',
                (canonical, partition, key, expected_revision)).rowcount == 1

    def put_if_absent(self, *, partition, key, canonical):
        partition, key, canonical = require_partition(partition), require_key(key), require_canonical(canonical)
        with self._transaction() as connection:
            if connection.execute('INSERT OR IGNORE INTO records VALUES(?,?,1,?)', (partition, key, canonical)).rowcount == 1:
                return True
            row = connection.execute('SELECT revision,canonical FROM records WHERE partition=? AND key=?', (partition, key)).fetchone()
            return row == (1, canonical)

    def list_page(self, *, partition, prefix, after_key, limit):
        partition, prefix, limit = require_partition(partition), require_prefix(prefix), require_page_limit(limit)
        if after_key is not None:
            after_key = require_key(after_key, 'after_key')
        self._check()
        # Literal prefix, never LIKE wildcards. BINARY ordering matches Python keys.
        rows = self._connection.execute('SELECT key,revision,canonical FROM records WHERE partition=? AND substr(key,1,?)=? AND (? IS NULL OR key>?) ORDER BY key COLLATE BINARY LIMIT ?',
            (partition, len(prefix), prefix, after_key, after_key, limit + 1)).fetchall()
        records = tuple(StoredRecordV1(*row) for row in rows[:limit])
        truncated = len(rows) > limit
        return StoredPageV1(records, records[-1].key if truncated else None, truncated)

    def append(self, *, partition, stream_key, sequence, canonical):
        partition, stream_key = require_partition(partition), require_key(stream_key, 'stream_key')
        sequence, canonical = require_sequence(sequence), require_canonical(canonical)
        with self._transaction() as connection:
            tail = self._tail(connection, partition, stream_key)
            if tail is not None and sequence <= tail:
                return False
            connection.execute('INSERT INTO streams VALUES(?,?,?,?)', (partition, stream_key, sequence, canonical))
            return True

    @staticmethod
    def _tail(connection, partition, stream_key):
        return connection.execute('SELECT MAX(sequence) FROM streams WHERE partition=? AND stream_key=?', (partition, stream_key)).fetchone()[0]

    def read_page(self, *, partition, stream_key, after_sequence, limit):
        partition, stream_key, limit = require_partition(partition), require_key(stream_key, 'stream_key'), require_page_limit(limit)
        if after_sequence is not None:
            after_sequence = require_sequence(after_sequence, 'after_sequence')
        self._check()
        rows = self._connection.execute('SELECT sequence,canonical FROM streams WHERE partition=? AND stream_key=? AND (? IS NULL OR sequence>?) ORDER BY sequence LIMIT ?',
            (partition, stream_key, after_sequence, after_sequence, limit + 1)).fetchall()
        entries = tuple(StoredStreamEntryV1(*row) for row in rows[:limit])
        truncated = len(rows) > limit
        return StoredStreamPageV1(entries, entries[-1].sequence if truncated else None, truncated)

    def commit_record_and_append(self, *, record_partition, key, expected_revision,
                                 record_canonical, stream_partition, stream_key,
                                 expected_tail, sequence, stream_canonical):
        """Atomically compare exact head and tail, write both, or write neither."""
        record_partition, key = require_partition(record_partition), require_key(key)
        stream_partition, stream_key = require_partition(stream_partition), require_key(stream_key, 'stream_key')
        record_canonical, stream_canonical = require_canonical(record_canonical), require_canonical(stream_canonical)
        if expected_revision is not None:
            expected_revision = require_revision(expected_revision)
            if expected_revision == _MAX_REVISION:
                raise ValueError('record revision exhausted')
        if expected_tail is not None:
            expected_tail = require_sequence(expected_tail, 'expected_tail')
        sequence = require_sequence(sequence)
        if expected_tail is not None and sequence <= expected_tail:
            raise ValueError('sequence must advance expected tail')
        with self._transaction() as connection:
            row = connection.execute('SELECT revision FROM records WHERE partition=? AND key=?', (record_partition, key)).fetchone()
            actual_revision = None if row is None else row[0]
            if actual_revision != expected_revision or self._tail(connection, stream_partition, stream_key) != expected_tail:
                return False
            if expected_revision is None:
                connection.execute('INSERT INTO records VALUES(?,?,1,?)', (record_partition, key, record_canonical))
            else:
                connection.execute('UPDATE records SET revision=revision+1,canonical=? WHERE partition=? AND key=?', (record_canonical, record_partition, key))
            connection.execute('INSERT INTO streams VALUES(?,?,?,?)', (stream_partition, stream_key, sequence, stream_canonical))
            return True
