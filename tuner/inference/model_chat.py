"""Bounded model-first chat, independent of training runs and authorities.

The caller selects a pinned model (or an existing model directory) and may
attach an existing LoRA directory through VLLMStartupSpec. Paths refer to the
execution machine: a Modal consumer mounts/prepares them there, not locally.
This API does not attest training provenance or provision a cloud resource.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from inspect import getattr_static
import math
from pathlib import Path
import re
import time
from typing import Iterator

from tuner.inference.chat_session import ChatSession, ChatSessionPolicy, _finite_deadline
from Evaluator.config import VLLMSettings
from Evaluator.vllm_client import VLLMClient
from tuner.inference.vllm_runtime import (
    ExplicitNetworkLoRA,
    ExplicitNetworkVLLMSource,
    VLLMStartupSpec,
    _LOCAL_ENV_EXACT,
    start_vllm_runtime,
)

_ENVIRONMENT_NAMES = _LOCAL_ENV_EXACT | {
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_HUB_DISABLE_TELEMETRY",
    "VLLM_NO_USAGE_STATS",
    "DO_NOT_TRACK",
}


class ModelChatFailureCode(str, Enum):
    """Closed vocabulary of model-serving failures, independent of training state."""

    FAILED = "model_chat_failed"
    DEADLINE_EXPIRED = "model_chat_deadline_expired"
    INTERRUPTED = "model_chat_interrupted"


@dataclass(frozen=True, slots=True)
class ModelChatFailure:
    """The typed failure signal: a closed code plus the lease still pending, if any.

    ``cleanup_lease`` is the exact runtime (or lower-level owned-process) lease
    whose teardown did not converge; ``None`` means every owned process family
    is known to be released. Consumers read this field instead of probing an
    attribute attached to an arbitrary exception.
    """

    code: ModelChatFailureCode
    cleanup_lease: object | None = None

    def __post_init__(self) -> None:
        if type(self.code) is not ModelChatFailureCode:
            raise TypeError("code must be an exact ModelChatFailureCode")


class ModelChatError(RuntimeError):
    """Closed model-serving failure carrying an explicit ``ModelChatFailure``.

    ``str(error)`` is the closed code value only; no backend or process detail
    ever reaches the message.
    """

    def __init__(self, failure: ModelChatFailure) -> None:
        if type(failure) is not ModelChatFailure:
            raise TypeError("failure must be an exact ModelChatFailure")
        self.failure = failure
        super().__init__(failure.code.value)


def _pending_lease(error: BaseException, runtime: object | None) -> object | None:
    """The lease still pending after ``error``: our runtime first, then a lower level's."""
    if runtime is not None and runtime.cleanup_pending:
        return runtime
    if isinstance(error, ModelChatError):
        return error.failure.cleanup_lease
    # Lower-level owners (owned_process, vllm_runtime) still signal through an
    # attribute on their own exception; read it statically, never via a property.
    lease = getattr_static(error, "cleanup_lease", None)
    return lease


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None:
        now = time.monotonic()
        if not math.isfinite(now) or now >= deadline:
            raise ModelChatError(ModelChatFailure(ModelChatFailureCode.DEADLINE_EXPIRED))


def _generation(value: object, *, maximum: float, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("generation value must be finite")
    if value < 0 or value > maximum or (positive and value == 0):
        raise ValueError("generation value is outside its bound")
    return float(value)


def _source(startup: VLLMStartupSpec) -> None:
    if type(startup) is not VLLMStartupSpec:
        raise TypeError("exact startup specification required")
    source = startup.source
    if type(source) is not ExplicitNetworkVLLMSource:
        raise TypeError("model-first chat requires an explicit model source")
    if type(source.model_ref) is not str or not source.model_ref:
        raise ValueError("explicit model reference required")
    if source.model_ref.startswith("/"):
        model = Path(source.model_ref)
        if not model.is_dir() or model.resolve(strict=True) != model:
            raise ValueError("model directory must be canonical and present")
        if source.revision is not None:
            raise ValueError("a local model directory has no Hub revision")
    elif (
        re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", source.model_ref
        )
        is None
        or type(source.revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source.revision) is None
    ):
        raise ValueError("Hub model must select an exact repository and commit")
    # A separately selected remote tokenizer would require a second revision.
    if source.tokenizer_ref is not None:
        tokenizer = Path(source.tokenizer_ref)
        if (
            not tokenizer.is_absolute()
            or not tokenizer.is_dir()
            or tokenizer.resolve(strict=True) != tokenizer
        ):
            raise ValueError("a separate tokenizer must be a canonical local directory")
    if source.lora is not None:
        adapter = source.lora
        if (
            type(adapter) is not ExplicitNetworkLoRA
            or adapter.name != startup.served_model_name
        ):
            raise ValueError("adapter must bind the selected served model")
        path = adapter.path
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or not path.is_dir()
            or path.resolve(strict=True) != path
        ):
            raise ValueError("adapter must be a canonical local directory")


@contextmanager
def open_model_chat(
    startup: VLLMStartupSpec,
    policy: ChatSessionPolicy,
    *,
    cwd: Path,
    environment: dict[str, str],
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_request_bytes: int = 1 << 20,
    max_response_bytes: int = 1 << 20,
    deadline: float | None = None,
) -> Iterator[ChatSession]:
    """Open one model directly; never load, submit, or resume a training run.

    Network model preparation uses vLLM's existing Hub cache on this execution
    machine. The explicitly supplied environment is never inherited wholesale.
    The runtime binds loopback only. A Modal adapter must separately enforce a
    provider timeout and retain exact Sandbox cleanup ownership.
    """
    _source(startup)
    if type(environment) is not dict or any(
        type(name) is not str
        or name not in _ENVIRONMENT_NAMES
        or type(value) is not str
        or "\0" in value
        for name, value in environment.items()
    ):
        raise ValueError(
            "model chat environment must contain only allowed runtime settings"
        )
    environment = dict(environment)
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    if type(policy) is not ChatSessionPolicy:
        raise TypeError("exact bounded chat policy required")
    policy.__post_init__()
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise ValueError("token limit is outside its bound")
    for bound in (max_request_bytes, max_response_bytes):
        if type(bound) is not int or not 1 <= bound <= 64 * 1024 * 1024:
            raise ValueError("message byte limit is outside its bound")
    temperature = _generation(temperature, maximum=2.0)
    top_p = _generation(top_p, maximum=1.0, positive=True)
    deadline = _finite_deadline(deadline)
    _check_deadline(deadline)
    runtime = None
    try:
        runtime = start_vllm_runtime(
            startup, cwd=cwd, environment=environment, deadline=deadline
        )
        with runtime:
            _check_deadline(deadline)
            settings = VLLMSettings(
                model=runtime.served_model_name,
                scheme="http",
                host=runtime.host,
                port=runtime.port,
                api_key=None,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model_path=None,
                lora_adapter=None,
            )
            client = VLLMClient(
                settings,
                timeout=policy.request_timeout_seconds,
                retries=0,
                trust_environment=False,
                allow_redirects=False,
                max_request_bytes=max_request_bytes,
                max_response_bytes=max_response_bytes,
            )
            with ChatSession(
                client, runtime, policy, clock=time.monotonic, deadline=deadline
            ) as session:
                _check_deadline(deadline)
                yield session
    except BaseException as error:
        cleanup = _pending_lease(error, runtime)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            # The interrupt itself is re-raised unchanged. A pending lease is
            # signalled by splicing a typed ModelChatError into its context
            # chain, never by attaching an attribute to the interrupt.
            if cleanup is not None:
                pending = ModelChatError(
                    ModelChatFailure(ModelChatFailureCode.INTERRUPTED, cleanup)
                )
                try:
                    pending.__context__ = error.__context__
                    error.__context__ = pending
                except Exception:
                    pass
            raise
        code = (
            error.failure.code
            if isinstance(error, ModelChatError)
            else ModelChatFailureCode.FAILED
        )
        raise ModelChatError(ModelChatFailure(code, cleanup)) from None
