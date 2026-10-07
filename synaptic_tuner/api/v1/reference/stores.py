"""In-memory reference implementations of the two public storage ports.

Location: ``synaptic_tuner/api/v1/reference/stores.py``.

``InMemoryDurableRecordStoreV1`` satisfies ``DurableRecordStorePort`` and
``InMemoryDurableStreamStoreV1`` satisfies ``DurableStreamStorePort`` from
``synaptic_tuner/api/v1/ports.py``. They are the "reference repositories" the
engine ships: process-local, lock-protected, and free of any file, database
or provider dependency. Nothing here survives the process; a host that needs
durability supplies its own implementations of the same ports.

Both stores validate every argument with the shared helpers in ``ports.py``,
refuse any partition outside the closed vocabulary, and never rewrite: the
record store only advances a revision through compare-and-swap, and the
stream store only appends a strictly greater sequence.

Pure standard library. Imports no ``tuner.*``, no ``sqlite3``, and no
provider SDK, and is registered in both import-closure gates.
"""

from __future__ import annotations

from threading import Lock

from ..ports import (
    StoredPageV1,
    StoredRecordV1,
    StoredStreamEntryV1,
    StoredStreamPageV1,
    require_canonical,
    require_key,
    require_page_limit,
    require_partition,
    require_prefix,
    require_revision,
    require_sequence,
)


class InMemoryDurableRecordStoreV1:
    """Compare-and-swap record store keyed by ``(partition, key)``.

    Verb outcomes when the key already holds a record:

    ==================  =================  ================
    existing record     ``create``         ``put_if_absent``
    ==================  =================  ================
    identical, rev 1    False              True
    different bytes     False              False
    revision > 1        False              False
    ==================  =================  ================

    ``compare_and_swap`` succeeds only when ``expected_revision`` equals the
    stored revision, and then stores the new bytes at ``revision + 1``.
    """

    __slots__ = ("_lock", "_partitions")

    def __init__(self) -> None:
        self._lock = Lock()
        self._partitions: dict[str, dict[str, StoredRecordV1]] = {}

    def _records(self, partition: str) -> dict[str, StoredRecordV1]:
        return self._partitions.setdefault(partition, {})

    def create(self, *, partition: str, key: str, canonical: bytes) -> bool:
        partition = require_partition(partition)
        record = StoredRecordV1(require_key(key), 1, require_canonical(canonical))
        with self._lock:
            records = self._records(partition)
            if record.key in records:
                return False
            records[record.key] = record
            return True

    def read(self, *, partition: str, key: str) -> StoredRecordV1 | None:
        partition = require_partition(partition)
        key = require_key(key)
        with self._lock:
            return self._partitions.get(partition, {}).get(key)

    def compare_and_swap(
        self, *, partition: str, key: str, expected_revision: int, canonical: bytes
    ) -> bool:
        partition = require_partition(partition)
        key = require_key(key)
        expected_revision = require_revision(expected_revision, "expected_revision")
        canonical = require_canonical(canonical)
        with self._lock:
            records = self._records(partition)
            existing = records.get(key)
            if existing is None or existing.revision != expected_revision:
                return False
            records[key] = StoredRecordV1(key, existing.revision + 1, canonical)
            return True

    def put_if_absent(self, *, partition: str, key: str, canonical: bytes) -> bool:
        partition = require_partition(partition)
        record = StoredRecordV1(require_key(key), 1, require_canonical(canonical))
        with self._lock:
            records = self._records(partition)
            existing = records.get(record.key)
            if existing is None:
                records[record.key] = record
                return True
            return existing == record

    def list_page(
        self, *, partition: str, prefix: str, after_key: str | None, limit: int
    ) -> StoredPageV1:
        partition = require_partition(partition)
        prefix = require_prefix(prefix)
        if after_key is not None:
            after_key = require_key(after_key, "after_key")
        limit = require_page_limit(limit)
        with self._lock:
            records = self._partitions.get(partition, {})
            matching = [
                records[key]
                for key in sorted(records)
                if key.startswith(prefix) and (after_key is None or key > after_key)
            ]
        page = tuple(matching[:limit])
        truncated = len(matching) > limit
        return StoredPageV1(page, page[-1].key if truncated else None, truncated)


class InMemoryDurableStreamStoreV1:
    """Append-only stream store keyed by ``(partition, stream_key)``.

    ``append`` succeeds only when ``sequence`` is strictly greater than the
    last appended sequence of that stream (any sequence starts an empty
    stream). Anything else, including a replay of the last sequence, is
    refused with ``False`` and leaves the stream untouched.
    """

    __slots__ = ("_lock", "_streams")

    def __init__(self) -> None:
        self._lock = Lock()
        self._streams: dict[tuple[str, str], list[StoredStreamEntryV1]] = {}

    def append(
        self, *, partition: str, stream_key: str, sequence: int, canonical: bytes
    ) -> bool:
        partition = require_partition(partition)
        stream_key = require_key(stream_key, "stream_key")
        entry = StoredStreamEntryV1(require_sequence(sequence), require_canonical(canonical))
        with self._lock:
            entries = self._streams.setdefault((partition, stream_key), [])
            if entries and entry.sequence <= entries[-1].sequence:
                return False
            entries.append(entry)
            return True

    def read_page(
        self, *, partition: str, stream_key: str, after_sequence: int | None, limit: int
    ) -> StoredStreamPageV1:
        partition = require_partition(partition)
        stream_key = require_key(stream_key, "stream_key")
        if after_sequence is not None:
            after_sequence = require_sequence(after_sequence, "after_sequence")
        limit = require_page_limit(limit)
        with self._lock:
            entries = self._streams.get((partition, stream_key), [])
            matching = [
                entry for entry in entries
                if after_sequence is None or entry.sequence > after_sequence
            ]
        page = tuple(matching[:limit])
        truncated = len(matching) > limit
        return StoredStreamPageV1(page, page[-1].sequence if truncated else None, truncated)


__all__ = ["InMemoryDurableRecordStoreV1", "InMemoryDurableStreamStoreV1"]
