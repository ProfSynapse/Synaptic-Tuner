"""Packaged consumer journal for one-shot claims and exact transport catalogs.

This is not a coordinator or Foundation lifecycle database. The existing
generic stores own that state within one process; this journal only prevents
external effect replay and retains exact packaged transport evidence.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
import stat
from typing import Callable

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, parse_canonical_object, safe_ref,
)


class ModalHostStorageUnavailable(RuntimeError):
    """Fixed local journal diagnostic."""


class ModalHostAttemptAlreadyClaimed(ModalHostStorageUnavailable):
    """An external effect was already authorized; never automatically retry."""


def _ref(value: object) -> str:
    if type(value) is not str:
        raise ModalHostStorageUnavailable("modal_host_storage_invalid")
    try:
        return safe_ref(value, "journal_ref")
    except Exception:
        raise ModalHostStorageUnavailable("modal_host_storage_invalid") from None


def _bytes(value: object) -> bytes:
    if type(value) is not bytes or not value or len(value) > 16 * 1024 * 1024:
        raise ModalHostStorageUnavailable("modal_host_storage_invalid")
    return value


def _private_file(info) -> bool:
    return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and info.st_uid == os.geteuid()
            and not stat.S_IMODE(info.st_mode) & 0o077)


class _Attempts:
    def __init__(self, owner: "ModalHostStorageV1") -> None:
        self._owner = owner

    def claim(self, attempt_ref: str, canonical_evidence: bytes) -> str:
        key = _ref(attempt_ref)
        evidence = _bytes(canonical_evidence)
        try:
            if canonical_bytes(parse_canonical_object(evidence, name="attempt evidence")) != evidence:
                raise ValueError
            digest = hashlib.sha256(evidence).hexdigest()
            self._owner._insert(
                "INSERT INTO attempts(namespace_ref,attempt_ref,digest,evidence) VALUES(?,?,?,?)",
                (self._owner.namespace_ref, key, digest, evidence),
            )
            return digest
        except sqlite3.IntegrityError:
            raise ModalHostAttemptAlreadyClaimed("modal_host_attempt_already_claimed") from None
        except ModalHostStorageUnavailable:
            raise
        except Exception:
            raise ModalHostStorageUnavailable("modal_host_storage_invalid") from None

    def resolve(self, attempt_ref: str) -> bytes | None:
        key = _ref(attempt_ref)
        row = self._owner._one(
            "SELECT digest,evidence FROM attempts WHERE namespace_ref=? AND attempt_ref=?",
            (self._owner.namespace_ref, key),
        )
        if row is None:
            return None
        evidence = _bytes(bytes(row[1]))
        if hashlib.sha256(evidence).hexdigest() != row[0]:
            raise ModalHostStorageUnavailable("modal_host_storage_invalid")
        return evidence


class _Catalog:
    def __init__(self, owner: "ModalHostStorageV1", name: str,
                 encode: Callable[[object], bytes], decode: Callable[[bytes], object]) -> None:
        self._owner, self._name = owner, _ref(name)
        if not callable(encode) or not callable(decode):
            raise TypeError("catalog codec required")
        self._encode, self._decode = encode, decode

    def _owned(self, value: object) -> bytes:
        try:
            raw = _bytes(self._encode(value))
            if _bytes(self._encode(self._decode(raw))) != raw:
                raise ValueError
            return raw
        except Exception:
            raise ModalHostStorageUnavailable("modal_host_storage_invalid") from None

    def resolve(self, item_ref: str) -> object | None:
        key = _ref(item_ref)
        row = self._owner._one(
            "SELECT payload,digest FROM catalogs WHERE namespace_ref=? AND catalog_ref=? AND item_ref=?",
            (self._owner.namespace_ref, self._name, key),
        )
        if row is None:
            return None
        raw = _bytes(bytes(row[0]))
        if hashlib.sha256(raw).hexdigest() != row[1]:
            raise ModalHostStorageUnavailable("modal_host_storage_invalid")
        try:
            value = self._decode(raw)
            if self._owned(value) != raw:
                raise ValueError
            return value
        except Exception:
            raise ModalHostStorageUnavailable("modal_host_storage_invalid") from None

    def publish_if_absent(self, item_ref: str, value: object) -> bool:
        key, raw = _ref(item_ref), self._owned(value)
        try:
            self._owner._insert(
                "INSERT INTO catalogs(namespace_ref,catalog_ref,item_ref,payload,digest) VALUES(?,?,?,?,?)",
                (self._owner.namespace_ref, self._name, key, raw,
                 hashlib.sha256(raw).hexdigest()),
            )
            return True
        except sqlite3.IntegrityError:
            if self.resolve(key) is not None and self._owned(self.resolve(key)) == raw:
                return False
            raise ModalHostStorageUnavailable("modal_host_storage_conflict") from None


class ModalHostStorageV1:
    """Exclusive owner-only POSIX SQLite journal, one process at a time."""

    def __init__(self, database_path: Path, namespace_ref: str) -> None:
        if os.name != "posix" or type(database_path) is not type(Path()) or not database_path.is_absolute():
            raise ModalHostStorageUnavailable("modal_host_storage_posix_required")
        path, self.namespace_ref = database_path, _ref(namespace_ref)
        lock_fd = connection = None
        try:
            parent = path.parent
            info = parent.lstat()
            if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or parent.resolve(strict=True) != parent
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                raise ValueError
            import fcntl

            lock_path = path.with_name(path.name + ".lock")
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            if not _private_file(os.fstat(lock_fd)):
                raise ValueError
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not path.exists():
                created = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                os.close(created)
            if not _private_file(path.lstat()) or path.is_symlink():
                raise ValueError
            connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS attempts(namespace_ref TEXT NOT NULL,attempt_ref TEXT NOT NULL,digest TEXT NOT NULL,evidence BLOB NOT NULL,PRIMARY KEY(namespace_ref,attempt_ref));"
                "CREATE TABLE IF NOT EXISTS catalogs(namespace_ref TEXT NOT NULL,catalog_ref TEXT NOT NULL,item_ref TEXT NOT NULL,payload BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(namespace_ref,catalog_ref,item_ref));"
            )
        except Exception:
            if connection is not None:
                connection.close()
            if lock_fd is not None:
                os.close(lock_fd)
            raise ModalHostStorageUnavailable("modal_host_storage_invalid") from None
        self._connection, self._lock_fd = connection, lock_fd
        self.attempts = _Attempts(self)

    def _one(self, statement: str, values: tuple[object, ...]):
        try:
            return self._connection.execute(statement, values).fetchone()
        except Exception:
            raise ModalHostStorageUnavailable("modal_host_storage_invalid") from None

    def _insert(self, statement: str, values: tuple[object, ...]) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(statement, values)
            self._connection.execute("COMMIT")
        except sqlite3.IntegrityError:
            self._connection.execute("ROLLBACK")
            raise
        except Exception:
            self._connection.execute("ROLLBACK")
            raise ModalHostStorageUnavailable("modal_host_storage_invalid") from None

    def catalog(self, catalog_ref: str, *, encode, decode) -> _Catalog:
        return _Catalog(self, catalog_ref, encode, decode)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            os.close(self._lock_fd)
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
