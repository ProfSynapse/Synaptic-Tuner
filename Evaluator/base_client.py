"""Base client implementation with shared functionality.

This module provides the foundation for all backend clients, implementing
common patterns like retry logic and message extraction to avoid code duplication.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Sequence, TypeVar

import requests

from .protocols import BackendError, BackendResponse, BackendSettings

T = TypeVar("T")


class BaseBackendClient(ABC):
    """Abstract base class for backend clients.

    Provides shared functionality:
    - Retry logic with exponential backoff
    - Common error handling
    - Settings management

    Subclasses must implement:
    - _build_payload(): Build request payload for specific API format
    - _get_chat_url(): Return the chat endpoint URL
    - _extract_response(): Extract BackendResponse from API response
    """

    def __init__(
        self,
        settings: BackendSettings,
        timeout: float = 60.0,
        retries: int = 2,
    ) -> None:
        """Initialize the client.

        Args:
            settings: Backend-specific settings (model, host, port, etc.)
            timeout: HTTP request timeout in seconds
            retries: Number of retry attempts on failure
        """
        self.settings = settings
        self.timeout = timeout
        self.retries = max(0, retries)

    def chat(self, messages: Sequence[Mapping[str, str]]) -> BackendResponse:
        """Send a chat conversation to the backend.

        Uses retry logic with exponential backoff on failure.

        Args:
            messages: Sequence of message dicts with 'role' and 'content'

        Returns:
            BackendResponse with model output

        Raises:
            BackendError: If request fails after all retries
        """
        payload = self._build_payload(messages)
        url = self._get_chat_url()

        return self._execute_with_retry(
            operation=lambda: self._make_chat_request(url, payload),
            error_message=f"{self._client_name} chat request failed",
        )

    def _make_chat_request(self, url: str, payload: Dict[str, Any]) -> BackendResponse:
        """Execute a single chat request.

        Args:
            url: The endpoint URL
            payload: Request payload

        Returns:
            BackendResponse with the result
        """
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        latency_s = time.perf_counter() - start
        return self._extract_response(data, latency_s)

    def _execute_with_retry(
        self,
        operation: Callable[[], T],
        error_message: str,
    ) -> T:
        """Execute an operation with retry logic.

        Implements exponential backoff: 1s, 2s, 4s, 5s (capped)

        Args:
            operation: Callable to execute
            error_message: Message prefix for error reporting

        Returns:
            Result of the operation

        Raises:
            BackendError: If all retries are exhausted
        """
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                return operation()
            except (requests.RequestException, ValueError, BackendError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                # Exponential backoff capped at 5 seconds
                time.sleep(min(2 ** attempt, 5))

        raise self._create_error(
            f"{error_message} after {self.retries + 1} attempts: {last_error}"
        )

    @property
    @abstractmethod
    def _client_name(self) -> str:
        """Return the client name for error messages."""
        ...

    @abstractmethod
    def _build_payload(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        """Build the request payload for the specific backend API.

        Args:
            messages: Chat messages to include

        Returns:
            Dict payload ready for JSON serialization
        """
        ...

    @abstractmethod
    def _get_chat_url(self) -> str:
        """Return the chat endpoint URL for this backend."""
        ...

    @abstractmethod
    def _extract_response(self, data: Dict[str, Any], latency_s: float) -> BackendResponse:
        """Extract BackendResponse from raw API response.

        Args:
            data: Raw JSON response from the API
            latency_s: Request latency in seconds

        Returns:
            Standardized BackendResponse
        """
        ...

    @abstractmethod
    def _create_error(self, message: str) -> BackendError:
        """Create a backend-specific error.

        Args:
            message: Error message

        Returns:
            BackendError subclass instance
        """
        ...


def extract_message_content(message: Mapping[str, Any]) -> Any:
    """Extract message content from API response, handling tool calls.

    This is a shared utility for extracting the relevant content from
    a message object, whether it contains tool calls or plain text.

    Supports:
    - OpenAI format: dict with 'tool_calls' array
    - ChatML format: dict with 'content' string
    - Mistral format: string content with [TOOL_CALLS]

    Args:
        message: Message object from API response

    Returns:
        - Dict with tool_calls if present (OpenAI format)
        - String content otherwise (ChatML/Mistral format)

    Raises:
        ValueError: If message format is invalid
    """
    # Check for OpenAI format with non-empty tool_calls
    tool_calls = message.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
        return dict(message)

    # Return content string (ChatML or Mistral format)
    content = message.get("content")
    if content is not None:
        return content if isinstance(content, str) else str(content)

    raise ValueError("Message missing valid content")


def extract_models_from_list(data: Mapping[str, Any]) -> List[str]:
    """Extract model IDs from a /v1/models response.

    Used by backends that support the OpenAI-compatible models endpoint.

    Args:
        data: Raw API response containing model list

    Returns:
        List of model ID strings

    Raises:
        ValueError: If response format is invalid or no models found
    """
    models_data = data.get("data")
    if not isinstance(models_data, list):
        raise ValueError(f"Invalid models response format: {json.dumps(data)[:200]}")

    models: List[str] = []
    for entry in models_data:
        if isinstance(entry, Mapping):
            model_id = entry.get("id")
            if isinstance(model_id, str):
                models.append(model_id)

    if not models:
        raise ValueError("No models found in response")

    return models
