"""Artifacts reference composition: publication operations with no destinations.

Location: ``synaptic_tuner/api/v1/reference/artifacts.py``.

``ArtifactsOperations`` (``api/v1/artifacts_facade.py``: destinations,
publications, publish, verify) is implemented by ``PublicationOperationsV1``
(``tuner/execution/coordinator_v1/publication.py``). The reference host
composes it over the engine's strong in-memory publication store, an empty
destination registry, an in-memory spool and a verified-source port that has
no source to describe. ``destinations()`` therefore returns an empty page,
``publications(ref)`` an empty page for any reference, and ``publish``
refuses with ``DESTINATION_MISSING`` before touching any source. A host that
publishes supplies its own destination registry and source port; no
destination adapter ships with the engine.

Consumed by ``synaptic_tuner/api/v1/reference/__init__.py``.
"""

from __future__ import annotations

import hashlib
from threading import RLock

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest
from tuner.execution.coordinator_v1.publication import (
    PublicationOperationsV1,
    StrongInMemoryPublicationStoreV1,
)
from tuner.execution.foundation_v2.canonical import safe_ref

from .authority import ReferenceAuthorityV1


class EmptyDestinationRegistryV1:
    """``ArtifactDestinationRegistryPortV1`` with no configured destinations."""

    def resolve(self, destination_ref: str):
        raise LookupError("no publication destinations are configured")

    def list(self, limit: int):
        return (), True


class UnavailableArtifactSourceV1:
    """``VerifiedArtifactSourcePortV1`` for a host without a verified source port."""

    def describe(self, run: TrainingRunRef):
        raise LookupError("no verified artifact source is configured")

    def open(self, request: RunArtifactRequest):
        raise LookupError("no verified artifact source is configured")


class _MemorySink:
    __slots__ = ("_chunks", "_maximum", "_size", "_finished", "_aborted")

    def __init__(self, maximum_bytes: int) -> None:
        self._chunks: list[bytes] = []
        self._maximum = maximum_bytes
        self._size = 0
        self._finished = False
        self._aborted = False

    def write(self, chunk: bytes) -> None:
        if self._finished or self._aborted or type(chunk) is not bytes:
            raise ValueError("spool sink is not writable")
        if self._size + len(chunk) > self._maximum:
            raise ValueError("spool bound exceeded")
        self._chunks.append(chunk)
        self._size += len(chunk)

    def finish(self) -> str:
        if self._finished or self._aborted:
            raise ValueError("spool sink already closed")
        self._finished = True
        return hashlib.sha256(b"".join(self._chunks)).hexdigest()

    def abort(self) -> None:
        self._aborted = True
        self._chunks.clear()


class InMemoryArtifactSpoolV1:
    """``ArtifactSpoolPortV1`` bounded per sink; nothing survives the process."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._open: dict[tuple[str, str], _MemorySink] = {}

    def open(self, publication_id: str, role: str, maximum_bytes: int):
        safe_ref(publication_id, "publication_id")
        safe_ref(role, "role")
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError("maximum_bytes must be a positive integer")
        sink = _MemorySink(maximum_bytes)
        with self._lock:
            self._open[(publication_id, role)] = sink
        return sink


def compose_reference_artifacts(*, authority: ReferenceAuthorityV1) -> PublicationOperationsV1:
    """The ``ArtifactsOperations`` implementation for a destination-less host."""
    if type(authority) is not ReferenceAuthorityV1:
        raise TypeError("exact ReferenceAuthorityV1 required")
    return PublicationOperationsV1(
        store=StrongInMemoryPublicationStoreV1(),
        destinations=EmptyDestinationRegistryV1(),
        sources=UnavailableArtifactSourceV1(),
        spool=InMemoryArtifactSpoolV1(),
        authority=authority.publication_authority,
        clock=authority.clock.now,
    )


__all__ = [
    "EmptyDestinationRegistryV1",
    "InMemoryArtifactSpoolV1",
    "PublicationOperationsV1",
    "UnavailableArtifactSourceV1",
    "compose_reference_artifacts",
]
