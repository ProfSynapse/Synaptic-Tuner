"""Host-owned ports and dependency composition for the public v1 facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .execution import AuthorizationRequirement, ExecutionGrant
from .persistence import EvidenceReplayRepository, LifecycleRepository
from .runs_facade import RunsAPI, RunsOperations
from .secrets import SecretRef
from .training_facade import TrainingAPI, TrainingOperations


class GrantProvider(Protocol):
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


class SecretProvider(Protocol):
    """Execution-time host boundary for resolving opaque secret references."""

    def resolve(self, reference: SecretRef) -> str: ...


EvidenceReplayStore = EvidenceReplayRepository


class EvidenceAuthenticator(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


class Clock(Protocol):
    def now(self) -> str: ...


class GitRemoteReader(Protocol):
    def read_ref(self, *, canonical_url: str, exact_ref: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class HostPorts:
    runs: RunsOperations
    clock: Clock


class APIHost:
    """Small composition root for host-selected public API implementations."""

    __slots__ = ("ports", "training", "_runs")

    def __init__(self, training: TrainingOperations, ports: HostPorts) -> None:
        self.training = TrainingAPI(training, clock=ports.clock)
        self.ports = ports
        self._runs = RunsAPI(ports.runs)

    @property
    def runs(self) -> RunsAPI:
        return self._runs


__all__ = [
    "APIHost",
    "GrantProvider",
    "EvidenceAuthenticator",
    "EvidenceReplayStore",
    "Clock",
    "GitRemoteReader",
    "HostPorts",
    "LifecycleRepository",
    "SecretProvider",
]
