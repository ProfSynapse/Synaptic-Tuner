"""Stable provider-neutral protocols implemented by a consuming host."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from synaptic_tuner.api.v1.planning import TrainingPlan
from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunOutcome


class Clock(Protocol):
    def now(self) -> str: ...


class AuthorizationGrant(Protocol):
    @property
    def grant_ref(self) -> str: ...


class GrantAuthority(Protocol):
    def authorize(self, *, operation: str, plan_fingerprint: str) -> AuthorizationGrant: ...


class EvidenceStore(Protocol):
    def put_if_absent(self, *, evidence_digest: str, canonical_bytes: bytes) -> bool: ...

    def get(self, evidence_digest: str) -> bytes | None: ...


class SourceReader(Protocol):
    def iter_bytes(self, *, source_digest: str, maximum_bytes: int) -> Iterator[bytes]: ...


@runtime_checkable
class OpaqueProviderPreparation(Protocol):
    """Canonical host-persisted bytes; interpretation belongs to the coordinator."""

    @property
    def provider(self) -> ProviderRef: ...

    @property
    def plan_fingerprint(self) -> str: ...

    @property
    def preparation_digest(self) -> str: ...

    @property
    def canonical_bytes(self) -> bytes: ...


class PreparationRepository(Protocol):
    def put_if_absent(self, preparation: OpaqueProviderPreparation) -> bool: ...

    def get(self, plan_fingerprint: str) -> OpaqueProviderPreparation | None: ...


class ProviderSession(Protocol):
    @property
    def descriptor(self) -> ProviderDescriptor: ...


class ProviderSessionFactory(Protocol):
    """Lazy boundary: implementations open authenticated clients only on call."""

    def open(self, provider: ProviderRef) -> ProviderSession: ...


class RunOutcomeRepository(Protocol):
    def get(self, run: TrainingRunRef) -> RunOutcome | None: ...

    def put(self, result: RunOutcome) -> None: ...


class TrainingPlanRepository(Protocol):
    def get(self, plan_fingerprint: str) -> TrainingPlan | None: ...

    def put_if_absent(self, plan: TrainingPlan) -> bool: ...


__all__ = [
    "AuthorizationGrant",
    "Clock",
    "EvidenceStore",
    "GrantAuthority",
    "OpaqueProviderPreparation",
    "PreparationRepository",
    "ProviderSession",
    "ProviderSessionFactory",
    "SourceReader",
    "TrainingPlanRepository",
    "RunOutcomeRepository",
]
