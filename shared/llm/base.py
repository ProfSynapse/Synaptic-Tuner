"""Base LLM client interface.

Location: ``shared/llm/base.py``.

Every provider adapter under ``shared/llm/providers/`` implements
``BaseLLMClient``. ``chat`` returns ``LLMCompletionV1`` and
``structured_output`` returns ``LLMStructuredV1`` (``shared/llm/usage.py``):
the answer plus the provider-reported token usage, or ``usage=None`` when the
provider reported none. Callers read ``.text`` / ``.value`` and never treat
the return value as a bare ``str`` or ``dict``.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .usage import LLMCompletionV1, LLMStructuredV1


class BaseLLMClient(ABC):
    """
    Base interface for LLM clients.

    All providers (OpenRouter, LM Studio, Ollama) implement this interface.
    Makes it trivial to add new providers - just implement these methods.
    """

    @staticmethod
    def _http_controls(timeout_seconds, max_request_bytes=None, max_response_bytes=None, deadline=None):
        """Validate opt-in controls without changing the legacy omitted path."""
        import math
        if max_request_bytes is None and max_response_bytes is None and deadline is None:
            return None
        for value in (max_request_bytes, max_response_bytes):
            if value is not None and (type(value) is not int or not 1 <= value <= 64 * 1024 * 1024):
                raise ValueError("HTTP byte control outside its bound")
        def positive_finite(value):
            try:
                return type(value) in (int, float) and math.isfinite(value) and value > 0
            except (ValueError, OverflowError):
                return False
        if not positive_finite(timeout_seconds):
            raise ValueError("HTTP timeout must be finite and positive")
        if deadline is not None and not positive_finite(deadline):
            raise ValueError("HTTP deadline must be finite and positive")
        return (max_request_bytes or 1024 * 1024, max_response_bytes or 1024 * 1024, deadline)

    @staticmethod
    def _bounded_json_post(url, payload, headers, *, timeout_seconds, controls):
        """Single POST; deadline bounds caller wait, not physical request lifetime.

        The worker owns HTTP resources and never touches journals or meters.
        A timed-out worker may still send/finish later. Its normal finally block
        closes resources; neither Session.close nor this wait proves hard abort.
        Owner-side request serialization measures dispatched bytes; it is not
        a hostile-input allocation bound or a hard serialization time limit.
        """
        import json
        import math
        from queue import Queue, Empty
        from threading import Thread
        import time
        import requests
        from .exceptions import LLMConnectionError
        request_bound, response_bound, admitted_deadline = controls
        effective_deadline = time.monotonic() + timeout_seconds
        if admitted_deadline is not None:
            effective_deadline = min(effective_deadline, admitted_deadline)
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(body) > request_bound:
            raise ValueError("HTTP request exceeds its byte bound")
        header_items = tuple({**headers, "Content-Type": "application/json", "Accept-Encoding": "identity"}.items())
        if time.monotonic() >= effective_deadline:
            raise LLMConnectionError("HTTP caller deadline expired") from None
        results = Queue(maxsize=1)

        def reject_constant(value):
            raise ValueError("nonfinite JSON number")

        def finite_float(value):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("nonfinite JSON number")
            return result

        def request():
            response = session = None
            success, value = False, None
            try:
                if time.monotonic() >= effective_deadline:
                    return
                session = requests.Session()
                session.trust_env = False
                if any(adapter.max_retries.total != 0 for adapter in session.adapters.values()):
                    raise ValueError("HTTP retries unsupported")
                remaining = effective_deadline - time.monotonic()
                if remaining <= 0:
                    return
                response = session.post(url, data=body, headers=dict(header_items),
                    timeout=min(timeout_seconds, remaining), allow_redirects=False, stream=True)
                encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
                if encoding not in ("", "identity"):
                    raise ValueError("HTTP content encoding unsupported")
                content = bytearray()
                while True:
                    if time.monotonic() >= effective_deadline:
                        raise ValueError("HTTP caller deadline expired")
                    chunk = response.raw.read(min(64 * 1024, response_bound + 1 - len(content)),
                                              decode_content=False)
                    if not chunk:
                        break
                    content.extend(chunk[:response_bound + 1 - len(content)])
                    if len(content) > response_bound:
                        raise ValueError("HTTP response exceeds its byte bound")
                if not 200 <= response.status_code < 300 or time.monotonic() >= effective_deadline:
                    raise ValueError("HTTP response unavailable")
                value = json.loads(bytes(content).decode("utf-8"),
                    parse_constant=reject_constant, parse_float=finite_float)
                if time.monotonic() >= effective_deadline:
                    raise ValueError("HTTP caller deadline expired")
                success = True
            except BaseException:
                success, value = False, None
            finally:
                for resource in (response, session):
                    if resource is not None:
                        try:
                            resource.close()
                        except BaseException:
                            success, value = False, None
                results.put_nowait((success, value))

        Thread(target=request, daemon=True, name="bounded-llm-http").start()
        try:
            success, parsed = results.get(timeout=max(0.0, effective_deadline - time.monotonic()))
        except Empty:
            raise LLMConnectionError("HTTP caller deadline expired") from None
        if not success or time.monotonic() >= effective_deadline:
            raise LLMConnectionError("bounded HTTP response unavailable") from None
        return parsed

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMCompletionV1:
        """
        Send chat completion request and return the completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            ``LLMCompletionV1`` with the generated text and, when the provider
            reported prompt/completion token counts, a ``measured`` usage record

        Raises:
            LLMError: If request fails
        """
        pass

    @abstractmethod
    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMStructuredV1:
        """
        Send request and return the parsed structured JSON matching schema.

        Args:
            messages: List of message dicts with 'role' and 'content'
            schema: JSON Schema for structured output
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Optional maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            ``LLMStructuredV1`` with the parsed JSON object and, when the
            provider reported token counts, a ``measured`` usage record

        Raises:
            LLMError: If request fails or response doesn't match schema
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the LLM backend is accessible.

        Returns:
            True if backend is reachable, False otherwise
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openrouter', 'lmstudio', 'ollama')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass

    @abstractmethod
    def list_models(self) -> list:
        """Return a list of available model identifiers for this provider."""
        pass
