"""Public host ports: the five things a consuming host implements.

Location: ``synaptic_tuner/api/v1/ports.py``.

A host supplies a clock, a secret resolver, a grant authority and two opaque
storage ports. The engine serialises every durable record (workflow, plan,
preparation, grant, publication, evaluation, data, pipeline and chat-session
records) to canonical bytes and hands them over under an engine-chosen key in
a closed partition; observations go to the append-only stream port. The
engine opens no database: ``StoredRecordV1`` is the only record shape a host
sees, and record shapes stay internal behind ``schema_version``.

``DurableRecordStorePort`` is compare-and-swap over an integer revision.
``create`` is first-claim: it succeeds exactly once per key. ``put_if_absent``
is replay-safe: it succeeds when it writes the record or when an identical
record already sits at revision 1, and fails when the key holds anything
else. Neither verb ever rewrites. ``DurableStreamStorePort`` is append-only
with a strictly increasing per-stream sequence and never rewrites either.

Pages reuse the paging discipline of ``runs_facade.py``: ``next_cursor`` and
``truncated`` agree, a truncated page carries at least one item, and the
cursor is the last item's key or sequence.

``synaptic_tuner/api/v1/reference/stores.py`` ships in-memory implementations
of both storage ports. Contract only: this module imports nothing from
``tuner.*`` at runtime and is registered in both import-closure gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from ._contract import exact_integer, required_text
from .execution import AuthorizationRequirement, ExecutionGrant

if TYPE_CHECKING:  # ``api/v1/secrets.py`` re-exports from ``tuner.project``; typing only.
    from .secrets import SecretRef


_MAX_REVISION = 2**63 - 1
_MAX_SEQUENCE = 2**63 - 1
STORE_PAGE_LIMIT_MAXIMUM = 1000


class StoragePartition(str, Enum):
    """Closed, engine-owned partition vocabulary for the storage ports."""

    WORKFLOW = "workflow"
    PLAN = "plan"
    PLAN_CONTEXT = "plan_context"
    PREPARATION = "preparation"
    EXECUTION_GRANT = "execution_grant"
    RECONCILIATION_GRANT = "reconciliation_grant"
    PUBLICATION = "publication"
    EVALUATION = "evaluation"
    DATA = "data"
    PIPELINE = "pipeline"
    CHAT_SESSION = "chat_session"
    OBSERVATION = "observation"


STORAGE_PARTITIONS = frozenset(member.value for member in StoragePartition)


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return required_text(value, name)


def _bounded_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def require_partition(value: object) -> str:
    """Return ``value`` when it names a closed partition; raise otherwise."""

    value = _text(value, "partition")
    if value not in STORAGE_PARTITIONS:
        raise ValueError("unknown storage partition")
    return value


def require_key(value: object, name: str = "key") -> str:
    return _text(value, name)


def require_prefix(value: object) -> str:
    """A prefix may be empty (every key matches) but is otherwise key-shaped."""

    if type(value) is not str:
        raise TypeError("prefix must be an exact string")
    return value if value == "" else required_text(value, "prefix")


def require_canonical(value: object) -> bytes:
    if type(value) is not bytes:
        raise TypeError("canonical must be exact bytes")
    if not value:
        raise ValueError("canonical must not be empty")
    return value


def require_revision(value: object, name: str = "revision") -> int:
    return _bounded_integer(value, name, minimum=1, maximum=_MAX_REVISION)


def require_sequence(value: object, name: str = "sequence") -> int:
    return _bounded_integer(value, name, minimum=0, maximum=_MAX_SEQUENCE)


def require_page_limit(value: object) -> int:
    return _bounded_integer(value, "limit", minimum=1, maximum=STORE_PAGE_LIMIT_MAXIMUM)


@dataclass(frozen=True, slots=True)
class StoredRecordV1:
    """The only record shape a host sees: opaque bytes at a revision."""

    key: str
    revision: int
    canonical: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", require_key(self.key))
        object.__setattr__(self, "revision", require_revision(self.revision))
        object.__setattr__(self, "canonical", require_canonical(self.canonical))


@dataclass(frozen=True, slots=True)
class StoredPageV1:
    records: tuple[StoredRecordV1, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.records) is not tuple or any(type(item) is not StoredRecordV1 for item in self.records):
            raise TypeError("records must be an exact tuple of StoredRecordV1")
        keys = tuple(item.key for item in self.records)
        if any(left >= right for left, right in zip(keys, keys[1:])):
            raise ValueError("record keys must be unique and strictly increasing")
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be an exact boolean")
        if self.next_cursor is not None:
            object.__setattr__(self, "next_cursor", require_key(self.next_cursor, "next_cursor"))
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.records:
            raise ValueError("a truncated page must contain a record")
        if self.truncated and self.next_cursor != keys[-1]:
            raise ValueError("next_cursor must equal the last record key")


@dataclass(frozen=True, slots=True)
class StoredStreamEntryV1:
    sequence: int
    canonical: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", require_sequence(self.sequence))
        object.__setattr__(self, "canonical", require_canonical(self.canonical))


@dataclass(frozen=True, slots=True)
class StoredStreamPageV1:
    entries: tuple[StoredStreamEntryV1, ...]
    next_cursor: int | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or any(type(item) is not StoredStreamEntryV1 for item in self.entries):
            raise TypeError("entries must be an exact tuple of StoredStreamEntryV1")
        sequences = tuple(item.sequence for item in self.entries)
        if any(left >= right for left, right in zip(sequences, sequences[1:])):
            raise ValueError("stream sequences must be unique and strictly increasing")
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be an exact boolean")
        if self.next_cursor is not None:
            require_sequence(self.next_cursor, "next_cursor")
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.entries:
            raise ValueError("a truncated page must contain an entry")
        if self.truncated and self.next_cursor != sequences[-1]:
            raise ValueError("next_cursor must equal the last entry sequence")


class ClockPort(Protocol):
    def now(self) -> str: ...

    def now_epoch(self) -> int: ...


class SecretResolverPort(Protocol):
    """Execution-time resolution of an opaque secret reference; values never persist."""

    def resolve(self, reference: SecretRef) -> str: ...


class GrantAuthorityPort(Protocol):
    """Host authority boundary; returned grants contain no credential values."""

    def authorize(
        self, requirements: tuple[AuthorizationRequirement, ...]
    ) -> ExecutionGrant: ...

    def bind(
        self,
        grant: ExecutionGrant,
        *,
        operation: object,
        requirements: tuple[AuthorizationRequirement, ...],
    ) -> object: ...


class DurableRecordStorePort(Protocol):
    """Opaque canonical bytes under an engine-chosen key, with compare-and-swap."""

    def create(self, *, partition: str, key: str, canonical: bytes) -> bool: ...

    def read(self, *, partition: str, key: str) -> StoredRecordV1 | None: ...

    def compare_and_swap(
        self, *, partition: str, key: str, expected_revision: int, canonical: bytes
    ) -> bool: ...

    def put_if_absent(self, *, partition: str, key: str, canonical: bytes) -> bool: ...

    def list_page(
        self, *, partition: str, prefix: str, after_key: str | None, limit: int
    ) -> StoredPageV1: ...


class DurableStreamStorePort(Protocol):
    """Append-only, monotone per-stream sequence. Never rewrites."""

    def append(
        self, *, partition: str, stream_key: str, sequence: int, canonical: bytes
    ) -> bool: ...

    def read_page(
        self, *, partition: str, stream_key: str, after_sequence: int | None, limit: int
    ) -> StoredStreamPageV1: ...


__all__ = [
    "ClockPort", "DurableRecordStorePort", "DurableStreamStorePort",
    "GrantAuthorityPort", "SecretResolverPort", "StoragePartition",
    "StoredPageV1", "StoredRecordV1", "StoredStreamEntryV1", "StoredStreamPageV1",
]
