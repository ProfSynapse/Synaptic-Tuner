"""Host-owned ports and dependency composition for the public v1 facade.

Location: ``synaptic_tuner/api/v1/host.py``.

``HostPorts`` is the eight-field record a host fills with one operations
object per public family plus its clock; ``APIHost`` wraps each filled family
in its public facade and exposes one property per family. A family whose
operations are ``None`` is *uncomposed*: its property raises rather than
returning a facade over nothing. ``training``, ``runs`` and ``clock`` are
always required. ``evaluation`` accepts ``EvaluationOperations`` and is
wrapped in ``EvaluationAPI``. The three families whose contracts have not
landed yet (``chat``, ``data``, ``pipelines``) accept only ``None`` here; each
family's slice replaces its ``object | None`` annotation with its operations
Protocol and adds the facade construction.

Construction sites: ``examples/modal_chat/host.py`` and the contract,
composition and inference tests that build an ``APIHost``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .artifacts_facade import ArtifactsAPI, ArtifactsOperations
from .evaluation_facade import EvaluationAPI, EvaluationOperations
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
    """One operations object per public family, plus the host clock.

    Field order is a public pin (``tests/contract/test_public_training_api_v1.py``).
    Every field is positional and required so a construction site names its
    choice for each family explicitly; ``None`` marks a family the host has not
    composed.
    """

    training: TrainingOperations
    runs: RunsOperations
    artifacts: ArtifactsOperations | None
    evaluation: EvaluationOperations | None
    chat: object | None  # ChatOperations once its contract lands
    data: object | None  # DataOperations once its contract lands
    pipelines: object | None  # PipelinesOperations once its contract lands
    clock: Clock


_REQUIRED_PORTS = ("training", "runs", "clock")
_PENDING_FAMILIES = ("chat", "data", "pipelines")


class APIHost:
    """Composition root exposing one public facade property per family.

    A family left ``None`` in ``HostPorts`` has no facade; reading its property
    raises ``RuntimeError`` so a host never hands out a facade over ``None``.
    """

    __slots__ = ("ports", "_training", "_runs", "_artifacts", "_evaluation")

    def __init__(self, ports: HostPorts) -> None:
        if type(ports) is not HostPorts:
            raise TypeError("ports must be exact HostPorts")
        for name in _REQUIRED_PORTS:
            if getattr(ports, name) is None:
                raise TypeError(f"HostPorts.{name} is required and must not be None")
        for name in _PENDING_FAMILIES:
            if getattr(ports, name) is not None:
                raise TypeError(
                    f"HostPorts.{name} has no public facade in this version; pass None"
                )
        self.ports = ports
        self._training = TrainingAPI(ports.training, clock=ports.clock)
        self._runs = RunsAPI(ports.runs)
        self._artifacts = (
            None if ports.artifacts is None else ArtifactsAPI(ports.artifacts)
        )
        self._evaluation = (
            None if ports.evaluation is None else EvaluationAPI(ports.evaluation)
        )

    @staticmethod
    def _composed(name: str, facade: object | None) -> object:
        if facade is None:
            raise RuntimeError(f"host did not compose the {name!r} family")
        return facade

    @property
    def training(self) -> TrainingAPI:
        return self._training

    @property
    def runs(self) -> RunsAPI:
        return self._runs

    @property
    def artifacts(self) -> ArtifactsAPI:
        return self._composed("artifacts", self._artifacts)

    @property
    def evaluation(self) -> EvaluationAPI:
        return self._composed("evaluation", self._evaluation)

    @property
    def chat(self) -> object:
        return self._composed("chat", None)

    @property
    def data(self) -> object:
        return self._composed("data", None)

    @property
    def pipelines(self) -> object:
        return self._composed("pipelines", None)


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
