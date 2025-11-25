"""Protocol definitions for the Evaluator module.

This module defines the interfaces (protocols) that enable dependency inversion
and allow for easy testing via mock implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Protocol, Sequence, runtime_checkable


@dataclass
class BackendResponse:
    """Standardized response from any backend client.

    Attributes:
        message: The response content - can be str (ChatML/Mistral) or Dict (OpenAI format)
        raw: The complete raw API response
        latency_s: Response time in seconds
    """
    message: Any  # str or Dict with tool_calls
    raw: Dict[str, Any]
    latency_s: float


@runtime_checkable
class BackendClient(Protocol):
    """Protocol for backend clients that can send chat messages.

    This is the core interface that all backend clients must implement.
    Using a protocol allows for easy mocking in tests and adding new backends
    without modifying existing code.
    """

    def chat(self, messages: Sequence[Mapping[str, str]]) -> BackendResponse:
        """Send a chat conversation to the backend.

        Args:
            messages: Sequence of message dicts with 'role' and 'content' keys

        Returns:
            BackendResponse with the model's response

        Raises:
            BackendError: If the request fails after retries
        """
        ...


@runtime_checkable
class ModelListingClient(Protocol):
    """Protocol for clients that can list available models.

    This is a separate protocol from BackendClient because not all backends
    support model listing (Interface Segregation Principle).
    """

    def list_models(self) -> List[str]:
        """Return list of available model IDs.

        Returns:
            List of model ID strings

        Raises:
            BackendError: If the request fails
        """
        ...


@runtime_checkable
class BackendSettings(Protocol):
    """Protocol for backend configuration settings.

    All backend settings classes should implement these common attributes.
    """

    model: str
    host: str
    port: int
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None

    def base_url(self) -> str:
        """Return the base URL for the backend API."""
        ...


class BackendError(Exception):
    """Base exception for backend errors.

    All backend-specific exceptions should inherit from this class
    to allow for unified error handling.
    """
    pass
