"""A ``BaseLLMClient`` proxy that meters every call.

Location: ``shared/llm/metering.py``.

``MeteredLLMClient`` wraps any ``BaseLLMClient`` so every ``chat`` and
``structured_output`` call feeds a ``UsageAccumulator`` (``shared/llm/usage.py``)
while the proxy behaves exactly like the wrapped client otherwise. The
reference Data implementation (``synaptic_tuner/api/v1/reference/data.py``)
wraps the host-built client in one of these so SynthChat's generation and
improvement calls are all counted toward one per-run usage record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseLLMClient
from .usage import LLMCompletionV1, LLMStructuredV1, UsageAccumulator


class MeteredLLMClient(BaseLLMClient):
    """A ``BaseLLMClient`` proxy that feeds every call's usage to an accumulator.

    Behaves as the wrapped client for every other attribute (``provider_name``,
    ``model_name``, provider-specific settings such as ``default_max_tokens``),
    so consumers that inspect the client see the real one. A call that raises
    counts as unmeasured: the request may have reached the provider, so a paid
    run cannot claim its spend was zero.
    """

    __slots__ = ("_inner", "_meter")

    def __init__(self, inner: object, meter: UsageAccumulator) -> None:
        if not callable(getattr(inner, "chat", None)):
            raise TypeError("wrapped client must expose chat()")
        if type(meter) is not UsageAccumulator:
            raise TypeError("meter must be exact UsageAccumulator")
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_meter", meter)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> LLMCompletionV1:
        try:
            completion = self._inner.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        except BaseException:
            # The request may have reached the provider: an unmeasured call.
            self._meter.add(None)
            raise
        if type(completion) is not LLMCompletionV1:
            raise TypeError("wrapped client must return LLMCompletionV1 from chat()")
        self._meter.add(completion.usage)
        return completion

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMStructuredV1:
        try:
            result = self._inner.structured_output(
                messages, schema, temperature=temperature, max_tokens=max_tokens, **kwargs
            )
        except BaseException:
            self._meter.add(None)
            raise
        if type(result) is not LLMStructuredV1:
            raise TypeError("wrapped client must return LLMStructuredV1 from structured_output()")
        self._meter.add(result.usage)
        return result

    def test_connection(self) -> bool:
        return bool(self._inner.test_connection())

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def list_models(self) -> list:
        return self._inner.list_models()

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)


__all__ = ["MeteredLLMClient"]
