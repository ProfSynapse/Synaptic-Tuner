"""Provider-neutral composition for one authenticated run chat context."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from inspect import getattr_static
from typing import Iterator, Protocol, runtime_checkable

from Evaluator.chat_session import ChatSession
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tuner.inference.retrieved_model import ROLES, RetrievedSFTModel


@dataclass(frozen=True, slots=True)
class PreparedModelIdentity:
    model_ref: str
    model_revision: str
    tokenizer_revision: str
    model_kind: str

    def __post_init__(self) -> None:
        if type(self.model_ref) is not str or not self.model_ref:
            raise ValueError("model_ref must be a nonempty string")
        for name in ("model_revision", "tokenizer_revision"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) not in (40, 64)
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be an exact pinned revision")
        if type(self.model_kind) is not str or self.model_kind not in ("full", "lora"):
            raise ValueError("model_kind must be full or lora")


@dataclass(frozen=True, slots=True)
class PreparedRunChat:
    """A session and its neutral model identity; local_model is local-only."""

    session: ChatSession
    run: TrainingRunRef
    artifacts: tuple[VerifiedArtifact, ...]
    model: PreparedModelIdentity
    local_model: RetrievedSFTModel | None = None

    def __post_init__(self) -> None:
        if type(self.session) is not ChatSession:
            raise TypeError("session must be an exact ChatSession")
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be an exact TrainingRunRef")
        if (
            type(self.artifacts) is not tuple
            or len(self.artifacts) != len(ROLES)
            or any(
                type(x) is not VerifiedArtifact or x.size_bytes <= 0
                for x in self.artifacts
            )
            or tuple(x.role for x in self.artifacts) != ROLES
            or tuple(VerifiedArtifact.from_dict(x.to_dict()) for x in self.artifacts)
            != self.artifacts
        ):
            raise ValueError("artifacts must be the exact canonical SFT inventory")
        if type(self.model) is not PreparedModelIdentity:
            raise TypeError("model must be an exact PreparedModelIdentity")
        run = TrainingRunRef.from_dict(self.run.to_dict())
        artifacts = tuple(
            VerifiedArtifact.from_dict(item.to_dict()) for item in self.artifacts
        )
        model = PreparedModelIdentity(
            self.model.model_ref,
            self.model.model_revision,
            self.model.tokenizer_revision,
            self.model.model_kind,
        )
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "model", model)
        if self.local_model is not None:
            if type(self.local_model) is not RetrievedSFTModel:
                raise TypeError(
                    "local_model must be a factory-issued RetrievedSFTModel"
                )
            local = self.local_model
            if (
                local.run != self.run
                or local.artifacts != self.artifacts
                or (
                    local.model_ref,
                    local.model_revision,
                    local.tokenizer_revision,
                    local.model_kind,
                )
                != (
                    self.model.model_ref,
                    self.model.model_revision,
                    self.model.tokenizer_revision,
                    self.model.model_kind,
                )
            ):
                raise ValueError("local model projection differs")


@runtime_checkable
class RunChatRuntime(Protocol):
    """Trusted execution-machine owner for authenticated preparation and chat.

    Implementations must reverify the run and admit its exact artifacts before
    serving effects. Returned metadata is a consistency projection, not proof
    that arbitrary adapter code performed those checks.
    """

    def open(
        self, runs: RunsAPI, run: TrainingRunRef
    ) -> AbstractContextManager[PreparedRunChat]: ...


def _validate_runtime(runtime: object) -> None:
    missing = object()
    member = getattr_static(runtime, "open", missing)
    if member is missing or not callable(member):
        raise TypeError("runtime must provide a callable open method")


def _run(value: TrainingRunRef) -> TrainingRunRef:
    if type(value) is not TrainingRunRef:
        raise TypeError("run must be an exact TrainingRunRef")
    return TrainingRunRef.from_dict(value.to_dict())


@contextmanager
def open_run_chat(
    runs: RunsAPI,
    run: TrainingRunRef,
    *,
    runtime: RunChatRuntime,
) -> Iterator[PreparedRunChat]:
    """Delegate a run to its trusted runtime and check the returned projections.

    Authentication and execution-machine preparation belong to that runtime;
    this helper imposes no local filesystem policy or provider authority.
    """
    if type(runs) is not RunsAPI:
        raise TypeError("runs must be an exact RunsAPI")
    baseline = _run(run)
    presented = _run(baseline)
    _validate_runtime(runtime)
    with runtime.open(runs, presented) as prepared:
        if type(prepared) is not PreparedRunChat:
            raise TypeError("runtime must yield an exact PreparedRunChat")
        snapshot = PreparedRunChat(
            prepared.session,
            prepared.run,
            prepared.artifacts,
            prepared.model,
            prepared.local_model,
        )
        if _run(run) != baseline or _run(presented) != baseline:
            raise ValueError("run changed during runtime acquisition")
        if snapshot.run != baseline:
            raise ValueError("prepared chat does not bind the requested run")
        yield snapshot


__all__: list[str] = []
