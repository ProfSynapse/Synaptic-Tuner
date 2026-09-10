"""Compose authenticated run retrieval with a consumer-supplied chat runtime."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from inspect import getattr_static
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from Evaluator.chat_session import ChatSession
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tuner.inference.retrieved_model import (
    RetrievedSFTModel,
    materialize_verified_sft_model,
)
from tuner.inference.serving_target import (
    PinnedModelPreparer,
    ServingTarget,
    prepare_serving_target,
)


@runtime_checkable
class RunChatRuntime(Protocol):
    """Consumer-selected execution boundary for one prepared serving target."""

    def open(self, target: ServingTarget) -> AbstractContextManager[ChatSession]: ...


@dataclass(frozen=True, slots=True)
class PreparedRunChat:
    """The bounded session and exact verified model values used to open it."""

    session: ChatSession
    retrieved: RetrievedSFTModel
    target: ServingTarget


def _validate_runtime(runtime: object) -> None:
    missing = object()
    member = getattr_static(runtime, "open", missing)
    if member is missing or not callable(member):
        raise TypeError("runtime must provide a callable open method")


@contextmanager
def open_run_chat(
    runs: RunsAPI,
    run: TrainingRunRef,
    *,
    destination: Path,
    runtime: RunChatRuntime,
    preparer: PinnedModelPreparer | None = None,
) -> Iterator[PreparedRunChat]:
    """Retrieve one authenticated run and open one consumer-selected runtime.

    ``runs`` supplies the authentication and retention boundary.  A runtime is
    an execution adapter, not another source of run authenticity.  Retrieved
    files remain in the consumer-owned destination after the context exits.
    """

    if type(runs) is not RunsAPI:
        raise TypeError("runs must be an exact RunsAPI")
    if type(run) is not TrainingRunRef:
        raise TypeError("run must be an exact TrainingRunRef")
    if not isinstance(destination, Path):
        raise TypeError("destination must be a Path")
    _validate_runtime(runtime)
    if preparer is not None and not isinstance(preparer, PinnedModelPreparer):
        raise TypeError("preparer must implement PinnedModelPreparer")

    retrieved = materialize_verified_sft_model(runs, run, destination)
    target = prepare_serving_target(retrieved, preparer)
    with runtime.open(target) as session:
        if type(session) is not ChatSession:
            raise TypeError("runtime must yield an exact ChatSession")
        yield PreparedRunChat(session, retrieved, target)


__all__: list[str] = []
