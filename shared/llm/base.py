"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


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
    ) -> str:
        """
        Send chat completion request and return text response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            Generated text response

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
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send request and return structured JSON matching schema.

        Args:
            messages: List of message dicts with 'role' and 'content'
            schema: JSON Schema for structured output
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            Parsed JSON object matching schema

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
