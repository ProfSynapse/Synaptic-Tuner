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
