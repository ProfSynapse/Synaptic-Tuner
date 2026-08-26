"""Host-owned ports and dependency composition for the public v1 facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .execution import AuthorizationRequirement, ExecutionGrant
from .runs import RunsAPI, RunsOperations
from .secrets import SecretRef
from .training import TrainingAPI, TrainingOperations


class LifecycleRepository(Protocol):
    """Host persistence port; concrete database choices remain outside the API.

    This port deliberately contains only atomic lifecycle persistence.  Public
    run verbs are supplied through ``HostPorts.runs`` so a database adapter is
    never forced to become a provider orchestration service.
    """

    def create(self, record: object) -> object: ...

    def load(self, project_ref: str, run_id: str) -> object | None: ...

    def append(
        self,
        project_ref: str,
        run_id: str,
        *,
        expected_revision: int,
        event: object,
    ) -> object: ...

    def compare_and_consume_attempt(
        self,
        project_ref: str,
        run_id: str,
        *,
        expected_revision: int,
        grant_ref: str,
        canonical_command: object,
    ) -> object: ...

    def record_attempt_outcome(
        self,
        project_ref: str,
        run_id: str,
        *,
        expected_revision: int,
        command_digest: str,
        observation: object,
    ) -> object: ...

    def list_runs(
        self,
        project_ref: str,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> object: ...


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


class EvidenceReplayStore(Protocol):
    """Host-durable atomic replay admission; concrete storage stays in the main project."""
    def admit(self, **evidence): ...


class EvidenceAuthenticator(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


class Clock(Protocol):
    def __call__(self) -> str: ...


class GitRemoteReader(Protocol):
    def read_ref(self, *, canonical_url: str, exact_ref: str) -> bytes: ...


class ModalDeploymentReader(Protocol):
    def bound_scope(self): ...
    def capability_proof(self, binding): ...
    def inspect_deployment(self, *, app_name: str, function_name: str): ...


class TrainingResolver(Protocol):
    def resolve(self, request, *, context): ...


@dataclass(frozen=True, slots=True)
class HostPorts:
    lifecycle: LifecycleRepository
    runs: RunsOperations
    grants: GrantProvider
    secrets: SecretProvider
    evidence_replay: EvidenceReplayStore
    authenticator: EvidenceAuthenticator
    clock: Clock
    git_remote: GitRemoteReader
    modal_reads: ModalDeploymentReader
    training_resolver: TrainingResolver


class APIHost:
    """Small composition root for host-selected public API implementations."""

    __slots__ = ("ports", "training", "_runs")

    def __init__(self, training: TrainingOperations, ports: HostPorts) -> None:
        self.training = TrainingAPI(training)
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
    "ModalDeploymentReader",
    "TrainingResolver",
    "HostPorts",
    "LifecycleRepository",
    "SecretProvider",
]
