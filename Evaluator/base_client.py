"""Base client implementation with shared functionality.

This module provides the foundation for all backend clients, implementing
common patterns like retry logic and message extraction to avoid code duplication.
"""
from __future__ import annotations

import json
import sys
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
        *,
        trust_environment: bool = True,
        allow_redirects: bool = True,
        max_response_bytes: int | None = None,
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
        if type(trust_environment) is not bool or type(allow_redirects) is not bool:
            raise TypeError("HTTP transport flags must be exact booleans")
        if max_response_bytes is not None and (
            type(max_response_bytes) is not int
            or isinstance(max_response_bytes, bool)
            or not 0 < max_response_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("HTTP response bound must be a positive exact integer")
        self._trust_environment = trust_environment
        self._allow_redirects = allow_redirects
        self._max_response_bytes = max_response_bytes

    @property
    def trust_environment(self) -> bool:
        return self._trust_environment

    @property
    def allow_redirects(self) -> bool:
        return self._allow_redirects

    @property
    def max_response_bytes(self) -> int | None:
        return self._max_response_bytes

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
        data = self._request_json("POST", url, payload=payload)
        latency_s = time.perf_counter() - start
        return self._extract_response(data, latency_s)

    def _request_json(
        self, method: str, url: str, *, payload: Dict[str, Any] | None = None
    ) -> Any:
        if (
            self.trust_environment
            and self.allow_redirects
            and self.max_response_bytes is None
        ):
            function = requests.post if method == "POST" else requests.get
            kwargs = {"timeout": self.timeout, "headers": self._request_headers()}
            if payload is not None:
                kwargs["json"] = payload
            response = function(url, **kwargs)
            response.raise_for_status()
            return response.json()
        session = requests.Session()
        session.trust_env = self.trust_environment
        response = None
        try:
            response = session.request(
                method,
                url,
                json=payload,
                timeout=self.timeout,
                headers=self._request_headers(),
                allow_redirects=self.allow_redirects,
                stream=self.max_response_bytes is not None,
            )
            if not self.allow_redirects and 300 <= response.status_code < 400:
                raise ValueError("HTTP redirect is prohibited")
            response.raise_for_status()
            if self.max_response_bytes is None:
                return response.json()
            content = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                remaining = self.max_response_bytes + 1 - len(content)
                content.extend(chunk[:remaining])
                if len(content) > self.max_response_bytes:
                    raise ValueError("HTTP response exceeds its bound")
            return json.loads(bytes(content))
        finally:
            self._close_transport(response, session)

    def _request_status(self, method: str, url: str, *, timeout: float) -> int:
        if (
            self.trust_environment
            and self.allow_redirects
            and self.max_response_bytes is None
        ):
            function = requests.get if method == "GET" else requests.post
            return function(
                url, timeout=timeout, headers=self._request_headers()
            ).status_code
        session = requests.Session()
        session.trust_env = self.trust_environment
        response = None
        try:
            response = session.request(
                method, url, timeout=timeout, headers=self._request_headers(),
                allow_redirects=self.allow_redirects, stream=True,
            )
            if not self.allow_redirects and 300 <= response.status_code < 400:
                raise ValueError("HTTP redirect is prohibited")
            return response.status_code
        finally:
            self._close_transport(response, session)

    @staticmethod
    def _close_transport(response: Any, session: Any) -> None:
        active_failure = sys.exc_info()[0] is not None
        cleanup_failure: BaseException | None = None
        if response is not None:
            try:
                response.close()
            except BaseException as error:
                cleanup_failure = error
        try:
            session.close()
        except BaseException as error:
            if cleanup_failure is None or isinstance(error, (KeyboardInterrupt, SystemExit)):
                cleanup_failure = error
        if active_failure or cleanup_failure is None:
            return
        if isinstance(cleanup_failure, (KeyboardInterrupt, SystemExit)):
            raise cleanup_failure
        if cleanup_failure is not None:
            raise ValueError("HTTP transport cleanup failed") from None

    def _request_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        api_key = getattr(self.settings, "api_key", None)
        auth_scheme = getattr(self.settings, "auth_scheme", "Bearer")
        if api_key:
            headers["Authorization"] = f"{auth_scheme} {api_key}"
        return headers

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
