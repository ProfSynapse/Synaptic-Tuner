"""Protocol definitions for the Evaluator module.

This module defines the interfaces (protocols) that enable dependency inversion
and allow for easy testing via mock implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Protocol, Sequence, runtime_checkable


@dataclass
class BackendResponse:
    """Standardized response from any backend client.

    Attributes:
        message: The response content - can be str (ChatML/Mistral) or Dict (OpenAI format)
        raw: The complete raw API response
        latency_s: Response time in seconds
        usage: Provider-reported token usage as a ``synaptic_tuner.api.v1.usage``
            ``UsageRecordV1`` (``measured``), or None when the backend reported
            none. Typed loosely so this module keeps its stdlib-only imports.
    """
    message: Any  # str or Dict with tool_calls
    raw: Dict[str, Any]
    latency_s: float
    usage: Any = None


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


class RequestFailureCode(str, Enum):
    """Non-secret request outcomes; these codes are not cause diagnoses."""

    TIMEOUT = "request_timeout"
    CONNECTION = "request_connection"
    HTTP_400 = "http_400"
    HTTP_401 = "http_401"
    HTTP_403 = "http_403"
    HTTP_404 = "http_404"
    HTTP_408 = "http_408"
    HTTP_413 = "http_413"
    HTTP_422 = "http_422"
    HTTP_429 = "http_429"
    HTTP_500 = "http_500"
    HTTP_502 = "http_502"
    HTTP_503 = "http_503"
    HTTP_504 = "http_504"
    HTTP_OTHER = "http_other"
    REQUEST = "request_transport"
    VALIDATION = "request_validation"
    BACKEND = "request_backend"
    UNKNOWN = "request_unknown"


class BackendError(Exception):
    """Base exception for backend errors.

    All backend-specific exceptions should inherit from this class
    to allow for unified error handling.
    """
    def __init__(self, *args: object, request_failure_code: RequestFailureCode | None = None):
        super().__init__(*args)
        try:
            self.request_failure_code = (
                request_failure_code if type(request_failure_code) is RequestFailureCode else None
            )
        except Exception:
            pass  # Optional diagnostics must not break legacy subclass constructors.


def closed_request_failure_code(error: BaseException) -> RequestFailureCode:
    """Accept only typed codes, never arbitrary attributes or error strings."""
    try:
        code = getattr(error, "request_failure_code", None)
    except Exception:
        code = None
    if isinstance(error, BackendError) and type(code) is RequestFailureCode:
        return code
    if isinstance(error, TimeoutError):
        return RequestFailureCode.TIMEOUT
    if isinstance(error, (ValueError, TypeError)):
        return RequestFailureCode.VALIDATION
    if isinstance(error, BackendError):
        return RequestFailureCode.BACKEND
    return RequestFailureCode.UNKNOWN
