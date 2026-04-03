"""SynthChat LLM Caller - LLM invocation with retry and fallback.

Location: SynthChat/llm/caller.py
Purpose: Call LLM clients (chat and structured_output) with retry logic,
         client-chain fallback, and optional logging.
Usage: Called by SynthChatGenerator methods in generator.py whenever an
       LLM generation is needed.
"""

import random
import time
from typing import Any, Dict, Optional, Sequence

from ..parsing import parse_json_object


def call_llm(
    *,
    prompt: str,
    default_client: Any,
    logger: Any = None,
    randomize: bool = True,
    trace_label: Optional[str] = None,
    max_tokens: Optional[int] = None,
    llm_clients: Optional[Sequence[Any]] = None,
    max_retries: int = 3,
) -> str:
    """Call LLM for generation with retry and client-chain fallback.

    Args:
        prompt: Generation prompt
        default_client: Fallback LLM client
        logger: Optional logger for tracing
        randomize: Whether to randomize temperature
        trace_label: Label for log messages
        max_tokens: Max tokens override
        llm_clients: Ordered client chain to try
        max_retries: Retries per client

    Returns:
        Generated text
    """
    last_error = None
    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = getattr(default_client, "default_max_tokens", None)

    client_chain = list(llm_clients or [default_client])
    for client_index, client in enumerate(client_chain):
        for attempt in range(1, max(1, int(max_retries or 1)) + 1):
            temperature = random.uniform(0.5, 0.9) if randomize else 0.7
            started_at = time.monotonic()
            if logger:
                logger.info(
                    f"LLM chat start [{trace_label or 'unlabeled'}] "
                    f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} temp={temperature:.2f}"
                )
            try:
                chat_kwargs = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                if resolved_max_tokens is not None:
                    chat_kwargs["max_tokens"] = resolved_max_tokens
                response = client.chat(**chat_kwargs)
            except Exception as exc:  # pragma: no cover - provider-specific failures
                last_error = exc
                if logger:
                    logger.warning(
                        f"LLM chat failed [{trace_label or 'unlabeled'}] "
                        f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                        f"elapsed={time.monotonic() - started_at:.1f}s error={exc}"
                    )
                continue

            if isinstance(response, str):
                if response.strip():
                    if logger:
                        logger.info(
                            f"LLM chat success [{trace_label or 'unlabeled'}] "
                            f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                            f"elapsed={time.monotonic() - started_at:.1f}s chars={len(response)}"
                        )
                    return response
            elif response is not None:
                response_text = str(response).strip()
                if response_text:
                    if logger:
                        logger.info(
                            f"LLM chat success [{trace_label or 'unlabeled'}] "
                            f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                            f"elapsed={time.monotonic() - started_at:.1f}s chars={len(response_text)}"
                        )
                    return response_text

            last_error = ValueError("LLM returned an empty response")
            if logger:
                logger.warning(
                    f"LLM chat empty [{trace_label or 'unlabeled'}] "
                    f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                    f"elapsed={time.monotonic() - started_at:.1f}s"
                )

    if last_error is not None:
        raise last_error
    raise ValueError("LLM returned an empty response")


def call_llm_structured(
    *,
    prompt: str,
    schema: Dict[str, Any],
    default_client: Any,
    logger: Any = None,
    randomize: bool = True,
    system_prompt: Optional[str] = None,
    trace_label: Optional[str] = None,
    max_tokens: Optional[int] = None,
    llm_clients: Optional[Sequence[Any]] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call structured output if available, retrying transient empty failures."""
    if not hasattr(default_client, "structured_output"):
        raw = call_llm(
            prompt=f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
            default_client=default_client,
            logger=logger,
            randomize=randomize,
            trace_label=trace_label,
            llm_clients=llm_clients,
            max_retries=max_retries,
        )
        parsed = parse_json_object(raw)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Structured generation requested but provider returned non-JSON output")

    last_error = None
    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = getattr(default_client, "default_max_tokens", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client_chain = list(llm_clients or [default_client])
    for client_index, client in enumerate(client_chain):
        for attempt in range(1, max(1, int(max_retries or 1)) + 1):
            temperature = random.uniform(0.1, 0.4) if randomize else 0.2
            started_at = time.monotonic()
            if logger:
                logger.info(
                    f"LLM structured start [{trace_label or schema.get('name', 'unlabeled')}] "
                    f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} temp={temperature:.2f}"
                )
            try:
                structured_kwargs = {
                    "messages": messages,
                    "schema": schema,
                    "temperature": temperature,
                }
                if resolved_max_tokens is not None:
                    structured_kwargs["max_tokens"] = resolved_max_tokens
                payload = client.structured_output(**structured_kwargs)
            except Exception as exc:  # pragma: no cover - provider-specific failures
                last_error = exc
                if logger:
                    logger.warning(
                        f"LLM structured failed [{trace_label or schema.get('name', 'unlabeled')}] "
                        f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                        f"elapsed={time.monotonic() - started_at:.1f}s error={exc}"
                    )
                continue

            if isinstance(payload, dict) and payload:
                if logger:
                    logger.info(
                        f"LLM structured success [{trace_label or schema.get('name', 'unlabeled')}] "
                        f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                        f"elapsed={time.monotonic() - started_at:.1f}s keys={sorted(payload.keys())}"
                    )
                return payload
            last_error = ValueError("LLM returned an empty structured response")
            if logger:
                logger.warning(
                    f"LLM structured empty [{trace_label or schema.get('name', 'unlabeled')}] "
                    f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                    f"elapsed={time.monotonic() - started_at:.1f}s"
                )

    if last_error is not None:
        raise last_error
    raise ValueError("LLM returned an empty structured response")
