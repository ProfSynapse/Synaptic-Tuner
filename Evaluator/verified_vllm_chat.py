"""Compose verified local model serving with the existing bounded chat client."""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
from typing import Iterator

from .chat_session import ChatSession, ChatSessionPolicy
from .config import VLLMSettings
from .vllm_client import VLLMClient
from .vllm_runtime import (
    VerifiedLocalVLLMSource,
    VLLMStartupSpec,
    start_vllm_runtime,
)


class VerifiedVLLMChatError(RuntimeError):
    """Verified local chat construction failed without exposing backend detail."""


def _number(value: object, *, maximum: float, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("generation parameter must be finite")
    if value < 0 or value > maximum or (positive and value == 0):
        raise ValueError("generation parameter is outside its bound")
    return float(value)


@contextmanager
def verified_vllm_chat(
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
) -> Iterator[ChatSession]:
    """Own one verified-local runtime and conversation for this context.

    Startup verifies the requested loaded model identities. The caller's first
    explicit ``chat`` is the inference check; no hidden prompt is submitted.
    Preparation and authenticated retrieval must already have supplied the
    startup target. This function neither downloads nor contacts a provider.
    """
    if type(startup) is not VLLMStartupSpec:
        raise TypeError("startup must be an exact VLLMStartupSpec")
    if type(startup.source) is not VerifiedLocalVLLMSource:
        raise TypeError("verified chat requires a verified local source")
    if type(policy) is not ChatSessionPolicy:
        raise TypeError("policy must be an exact ChatSessionPolicy")
    # Recheck the immutable policy before acquiring any process ownership.
    policy.__post_init__()
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise ValueError("max_tokens is outside its bound")
    if (
        type(max_request_bytes) is not int
        or not 1 <= max_request_bytes <= 64 * 1024 * 1024
    ):
        raise ValueError("max_request_bytes is outside its bound")
    if (
        type(max_response_bytes) is not int
        or not 1 <= max_response_bytes <= 64 * 1024 * 1024
    ):
        raise ValueError("max_response_bytes is outside its bound")
    temperature = _number(temperature, maximum=2.0)
    top_p = _number(top_p, maximum=1.0, positive=True)

    runtime = start_vllm_runtime(startup, cwd=cwd, environment=environment)
    try:
        # Register process ownership before client/session construction. The
        # lease serializes watchdog and outer-context cleanup and caches only
        # successful cleanup, so this outer guard can retry unresolved cleanup.
        with runtime:
            try:
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
                session = ChatSession(client, runtime, policy)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                raise VerifiedVLLMChatError(
                    "verified local chat setup failed"
                ) from None
            with session:
                yield session
    except BaseException as error:
        if runtime.cleanup_pending:
            # Preserve the exact owned handle for explicit recovery; never
            # replace the user's active exception with a teardown exception.
            error.cleanup_lease = runtime
        raise
