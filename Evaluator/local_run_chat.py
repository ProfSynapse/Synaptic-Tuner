"""Local vLLM adapter for the provider-neutral verified-run chat workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from contextlib import AbstractContextManager
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from tuner.inference.serving_target import ServingTarget

from .chat_session import ChatSession, ChatSessionPolicy
from .verified_vllm_chat import verified_vllm_chat
from .vllm_runtime import (
    VLLMStartupSpec,
    VerifiedLocalVLLMSource,
    _environment,
    _local_environment,
)


@dataclass(frozen=True, slots=True)
class LocalVLLMRunChatRuntime:
    """Retain local runtime policy without coupling generic run access to vLLM.

    Startup and generation ranges are enforced by the existing verified-chat
    path before process creation. Construction snapshots only primitive config;
    opening never caches or skips the runtime's fresh target validation.
    """

    policy: ChatSessionPolicy
    cwd: Path
    environment: Mapping[str, str] = field(repr=False)
    served_model_name: str = "trained"
    startup_options: Mapping[str, object] = field(default_factory=dict)
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    max_response_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        if type(self.policy) is not ChatSessionPolicy:
            raise TypeError("policy must be an exact ChatSessionPolicy")
        self.policy.__post_init__()
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
        if type(self.temperature) not in (int, float) or type(self.top_p) not in (
            int,
            float,
        ):
            raise TypeError("generation parameters must be numbers")
        object.__setattr__(self, "environment", MappingProxyType(environment))
        object.__setattr__(self, "startup_options", MappingProxyType(options))

    def open(self, target: ServingTarget) -> AbstractContextManager[ChatSession]:
        if type(target) is not ServingTarget:
            raise TypeError("local chat requires an exact ServingTarget")
        startup = VLLMStartupSpec(
            source=VerifiedLocalVLLMSource(target),
            served_model_name=self.served_model_name,
            **dict(self.startup_options),
        )
        return verified_vllm_chat(
            startup,
            self.policy,
            cwd=self.cwd,
            environment=dict(self.environment),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            max_response_bytes=self.max_response_bytes,
        )
