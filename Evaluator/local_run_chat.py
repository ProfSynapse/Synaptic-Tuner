"""Local vLLM adapter for the provider-neutral verified-run chat workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tuner.inference.retrieved_model import materialize_verified_sft_model
from tuner.inference.run_chat import PreparedModelIdentity, PreparedRunChat
from tuner.inference.serving_target import PinnedModelPreparer, prepare_serving_target

from .chat_session import ChatSessionPolicy
from .verified_vllm_chat import verified_vllm_chat
from .vllm_runtime import (
    VLLMStartupSpec,
    VerifiedLocalVLLMSource,
    _environment,
    _local_environment,
)


@dataclass(frozen=True, slots=True)
class LocalVLLMRunChatRuntime:
    """Own local run preparation and serving without imposing it on other runtimes.

    Startup and generation ranges are enforced by the existing verified-chat
    path before process creation. Construction snapshots only primitive config;
    opening never caches or skips the runtime's fresh target validation.
    """

    policy: ChatSessionPolicy
    cwd: Path
    environment: Mapping[str, str] = field(repr=False)
    destination: Path
    preparer: PinnedModelPreparer | None = field(default=None, repr=False)
    served_model_name: str = "trained"
    startup_options: Mapping[str, object] = field(default_factory=dict)
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    max_request_bytes: int = 1 << 20
    max_response_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        if type(self.policy) is not ChatSessionPolicy:
            raise TypeError("policy must be an exact ChatSessionPolicy")
        self.policy.__post_init__()
        if not isinstance(self.destination, Path) or not self.destination.is_absolute():
            raise TypeError("destination must be an absolute Path")
        if self.preparer is not None and not isinstance(
            self.preparer, PinnedModelPreparer
        ):
            raise TypeError("preparer must implement PinnedModelPreparer")
        if (
            not isinstance(self.cwd, Path)
            or not self.cwd.is_absolute()
            or not self.cwd.is_dir()
        ):
            raise TypeError("cwd must be an existing absolute Path")
        # The verified-local policy rejects credential/proxy/import-injection
        # names now; values are never represented in the adapter's repr.
        environment = _local_environment(_environment(self.environment))
        if type(self.startup_options) is not dict:
            raise TypeError("startup_options must be an explicit dict")
        allowed = {item.name for item in fields(VLLMStartupSpec)} - {
            "source",
            "served_model_name",
        }
        options = dict(self.startup_options)
        if any(type(key) is not str or key not in allowed for key in options):
            raise ValueError("unsupported local startup option")
        if any(
            type(value) not in (str, int, float, bool, type(None))
            for value in options.values()
        ):
            raise TypeError("local startup options must be primitive values")
        if type(self.served_model_name) is not str:
            raise TypeError("served_model_name must be a string")
        if type(self.max_tokens) is not int or type(self.max_response_bytes) is not int:
            raise TypeError("generation limits must be exact integers")
        if (
            type(self.max_request_bytes) is not int
            or not 1 <= self.max_request_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("max_request_bytes is outside its bound")
        if type(self.temperature) not in (int, float) or type(self.top_p) not in (
            int,
            float,
        ):
            raise TypeError("generation parameters must be numbers")
        object.__setattr__(self, "environment", MappingProxyType(environment))
        object.__setattr__(self, "startup_options", MappingProxyType(options))

    @contextmanager
    def open(self, runs: RunsAPI, run: TrainingRunRef) -> Iterator[PreparedRunChat]:
        # The selected adapter owns preparation on its execution machine. The
        # materializer retains the exact RunsAPI reverification and private-root
        # guards; these are not optional just because orchestration moved here.
        retrieved = materialize_verified_sft_model(runs, run, self.destination)
        target = prepare_serving_target(retrieved, self.preparer)
        startup = VLLMStartupSpec(
            source=VerifiedLocalVLLMSource(target),
            served_model_name=self.served_model_name,
            **dict(self.startup_options),
        )
        with verified_vllm_chat(
            startup,
            self.policy,
            cwd=self.cwd,
            environment=dict(self.environment),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            max_request_bytes=self.max_request_bytes,
            max_response_bytes=self.max_response_bytes,
        ) as session:
            yield PreparedRunChat(
                session=session,
                run=retrieved.run,
                artifacts=retrieved.artifacts,
                model=PreparedModelIdentity(
                    model_ref=retrieved.model_ref,
                    model_revision=retrieved.model_revision,
                    tokenizer_revision=retrieved.tokenizer_revision,
                    model_kind=retrieved.model_kind,
                ),
                local_model=retrieved,
            )
