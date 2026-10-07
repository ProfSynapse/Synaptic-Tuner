"""Consumer-owned durable storage for the minimal Modal chat example.

This is a local, single-process SQLite boundary. It stores authenticated or
canonical bytes supplied by the consumer; it does not store authority keys,
reconstruct Foundation, or make SQLite an engine-owned database. The parent
directory and database are trusted local-owner paths, not hostile mount paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import sys
from typing import Callable

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
    safe_ref,
)

_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


class ModalChatStorageError(RuntimeError):
    """Closed storage failure for the example consumer."""


class AttemptAlreadyClaimed(ModalChatStorageError):
    """The permanent one-shot attempt claim already exists."""


def _ref(value: object, label: str) -> str:
    try:
        if type(value) is not str:
            raise TypeError
        return safe_ref(value, label)
    except Exception:
        raise ModalChatStorageError("modal_chat_storage_invalid") from None


def _private_regular(info: os.stat_result) -> bool:
    return bool(
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and info.st_uid == os.geteuid()
        and not stat.S_IMODE(info.st_mode) & 0o077
    )


def _canonical_evidence(value: bytes) -> bytes:
    if type(value) is not bytes or not value or len(value) > _MAX_PAYLOAD_BYTES:
        raise ModalChatStorageError("modal_chat_storage_invalid")
    try:
        document = parse_canonical_object(value, name="attempt evidence")
        if canonical_bytes(document) != value:
            raise ValueError
    except Exception:
        raise ModalChatStorageError("modal_chat_storage_invalid") from None
    return value


def _payload(value: object) -> bytes:
    if type(value) is not bytes or not value or len(value) > _MAX_PAYLOAD_BYTES:
        raise ModalChatStorageError("modal_chat_storage_invalid")
    return value


def _private_database_path(value: Path) -> Path:
    if not isinstance(value, Path) or os.name != "posix":
        raise ModalChatStorageError("modal_chat_storage_invalid")
    path = value.absolute()
    parent = path.parent
    try:
        info = parent.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or parent.is_symlink()
            or parent.resolve(strict=True) != parent
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if (
                not _private_regular(info)
                or path.is_symlink()
                or path.resolve(strict=True) != path
            ):
                raise ValueError
    except (OSError, ValueError):
        raise ModalChatStorageError("modal_chat_storage_invalid") from None
    return path


@dataclass(frozen=True, slots=True)
class AttemptClaim:
    attempt_ref: str
    commitment_digest: str
    canonical_evidence: bytes

    def __post_init__(self) -> None:
        _ref(self.attempt_ref, "attempt_ref")
        if (
            type(self.commitment_digest) is not str
            or len(self.commitment_digest) != 64
            or any(x not in "0123456789abcdef" for x in self.commitment_digest)
            or hashlib.sha256(self.canonical_evidence).hexdigest()
            != self.commitment_digest
        ):
            raise ModalChatStorageError("modal_chat_storage_invalid")
        _canonical_evidence(self.canonical_evidence)


class SQLitePublishIfAbsentCatalog:
    """Exact-byte implementation of the engine's two-method catalog port."""

    __slots__ = ("_storage", "_catalog_ref", "_encode", "_decode")

    def __init__(
        self,
        storage: "ModalChatStorage",
        catalog_ref: str,
        encode: Callable[[object], bytes],
        decode: Callable[[bytes], object],
    ) -> None:
        self._storage = storage
        self._catalog_ref = _ref(catalog_ref, "catalog_ref")
        if not callable(encode) or not callable(decode):
            raise TypeError("exact catalog codecs required")
        self._encode, self._decode = encode, decode

    def _owned_payload(self, value: object) -> bytes:
        try:
            first = _payload(self._encode(value))
            owned = self._decode(first)
            if _payload(self._encode(owned)) != first:
                raise ValueError
            return first
        except Exception:
            raise ModalChatStorageError("modal_chat_storage_invalid") from None

    def resolve(self, item_ref: str) -> object | None:
        key = _ref(item_ref, "item_ref")
        row = self._storage._one(
            "SELECT payload,payload_sha256 FROM catalog_items WHERE namespace_ref=? AND catalog_ref=? AND item_ref=?",
            (self._storage.namespace_ref, self._catalog_ref, key),
        )
        if row is None:
            return None
        payload = bytes(row[0])
        if (
            len(payload) > _MAX_PAYLOAD_BYTES
            or hashlib.sha256(payload).hexdigest() != row[1]
        ):
            raise ModalChatStorageError("modal_chat_storage_invalid")
        try:
            value = self._decode(payload)
        except Exception:
            raise ModalChatStorageError("modal_chat_storage_invalid") from None
        if self._owned_payload(value) != payload:
            raise ModalChatStorageError("modal_chat_storage_invalid")
        return value

    def publish_if_absent(self, item_ref: str, value: object) -> bool:
        key = _ref(item_ref, "item_ref")
        payload = self._owned_payload(value)
        try:
            self._storage._insert(
                "INSERT INTO catalog_items(namespace_ref,catalog_ref,item_ref,payload,payload_sha256) VALUES(?,?,?,?,?)",
                (
                    self._storage.namespace_ref,
                    self._catalog_ref,
                    key,
                    payload,
                    hashlib.sha256(payload).hexdigest(),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            retained = self.resolve(key)
            if retained is not None and self._owned_payload(retained) == payload:
                return False
            raise ModalChatStorageError("modal_chat_storage_conflict") from None


class SQLiteOneShotAttemptStore:
    __slots__ = ("_storage",)

    def __init__(self, storage: "ModalChatStorage") -> None:
        self._storage = storage

    def claim(self, attempt_ref: str, canonical_evidence: bytes) -> AttemptClaim:
        key = _ref(attempt_ref, "attempt_ref")
        evidence = _canonical_evidence(canonical_evidence)
        claim = AttemptClaim(key, hashlib.sha256(evidence).hexdigest(), evidence)
        try:
            self._storage._insert(
                "INSERT INTO attempt_claims(namespace_ref,attempt_ref,commitment_digest,canonical_evidence) VALUES(?,?,?,?)",
                (self._storage.namespace_ref, key, claim.commitment_digest, evidence),
            )
        except sqlite3.IntegrityError:
            raise AttemptAlreadyClaimed("modal_chat_attempt_already_claimed") from None
        return claim

    def resolve(self, attempt_ref: str) -> AttemptClaim | None:
        key = _ref(attempt_ref, "attempt_ref")
        row = self._storage._one(
            "SELECT commitment_digest,canonical_evidence FROM attempt_claims WHERE namespace_ref=? AND attempt_ref=?",
            (self._storage.namespace_ref, key),
        )
        return None if row is None else AttemptClaim(key, row[0], bytes(row[1]))


class ModalChatStorage:
    """Exclusive local database for exact catalogs and permanent attempts."""

    __slots__ = (
        "database_path",
        "namespace_ref",
        "_lock_fd",
        "_connection",
        "attempts",
    )

    def __init__(self, database_path: Path, namespace_ref: str) -> None:
        path = _private_database_path(database_path)
        namespace = _ref(namespace_ref, "namespace_ref")
        lock_fd = connection = None
        created = not path.exists()
        try:
            lock_path = path.with_name(path.name + ".lock")
            lock_fd = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            lock_info = os.fstat(lock_fd)
            if (
                not _private_regular(lock_info)
                or lock_path.is_symlink()
                or lock_path.lstat().st_ino != lock_info.st_ino
                or lock_path.lstat().st_dev != lock_info.st_dev
            ):
                raise ValueError
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked_info = os.fstat(lock_fd)
            locked_path_info = lock_path.lstat()
            if not _private_regular(locked_info) or (
                locked_path_info.st_dev,
                locked_path_info.st_ino,
            ) != (locked_info.st_dev, locked_info.st_ino):
                raise ValueError
            # Validate again under the process-lifetime lock before SQLite sees
            # the pathname. This is best-effort trusted-local protection.
            if created:
                descriptor = os.open(
                    path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                os.close(descriptor)
            database_info = path.lstat()
            if (
                not _private_regular(database_info)
                or path.is_symlink()
                or path.resolve(strict=True) != path
            ):
                raise ValueError
            connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS catalog_items(namespace_ref TEXT NOT NULL,catalog_ref TEXT NOT NULL,item_ref TEXT NOT NULL,payload BLOB NOT NULL,payload_sha256 TEXT NOT NULL,PRIMARY KEY(namespace_ref,catalog_ref,item_ref));"
                "CREATE TABLE IF NOT EXISTS attempt_claims(namespace_ref TEXT NOT NULL,attempt_ref TEXT NOT NULL,commitment_digest TEXT NOT NULL,canonical_evidence BLOB NOT NULL,PRIMARY KEY(namespace_ref,attempt_ref));"
            )
        except Exception:
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
            raise ModalChatStorageError("modal_chat_storage_invalid") from None
        self.database_path, self.namespace_ref = path, namespace
        self._lock_fd, self._connection = lock_fd, connection
        self.attempts = SQLiteOneShotAttemptStore(self)

    def catalog(
        self,
        catalog_ref: str,
        *,
        encode: Callable[[object], bytes],
        decode: Callable[[bytes], object],
    ) -> SQLitePublishIfAbsentCatalog:
        return SQLitePublishIfAbsentCatalog(self, catalog_ref, encode, decode)

    def _one(self, statement: str, values: tuple[object, ...]):
        if self._connection is None:
            raise ModalChatStorageError("modal_chat_storage_closed")
        try:
            return self._connection.execute(statement, values).fetchone()
        except sqlite3.Error:
            raise ModalChatStorageError("modal_chat_storage_invalid") from None

    def _insert(self, statement: str, values: tuple[object, ...]) -> None:
        if self._connection is None:
            raise ModalChatStorageError("modal_chat_storage_closed")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(statement, values)
            self._connection.execute("COMMIT")
        except sqlite3.IntegrityError:
            try:
                self._connection.execute("ROLLBACK")
            except sqlite3.Error:
                raise ModalChatStorageError("modal_chat_storage_invalid") from None
            raise
        except sqlite3.Error:
            try:
                self._connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise ModalChatStorageError("modal_chat_storage_invalid") from None

    def close(self) -> None:
        if self._connection is None:
            return
        active_failure = sys.exc_info()[0] is not None
        connection, lock_fd = self._connection, self._lock_fd
        self._connection = self._lock_fd = None
        failure = False
        try:
            connection.close()
        except sqlite3.Error:
            failure = True
        finally:
            try:
                os.close(lock_fd)
            except OSError:
                failure = True
        if failure and not active_failure:
            raise ModalChatStorageError("modal_chat_storage_invalid") from None

    def __enter__(self) -> "ModalChatStorage":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "AttemptAlreadyClaimed",
    "AttemptClaim",
    "ModalChatStorage",
    "ModalChatStorageError",
    "SQLiteOneShotAttemptStore",
    "SQLitePublishIfAbsentCatalog",
]
