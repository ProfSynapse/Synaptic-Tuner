"""HTTP client for interacting with LM Studio server.

This module provides the LMStudioClient for sending chat requests to
LM Studio's OpenAI-compatible API. It inherits from OpenAICompatClient
for shared OpenAI API handling.
"""
from __future__ import annotations

from .config import LMStudioSettings
from .openai_compat_client import OpenAICompatClient, OpenAICompatError
from .protocols import BackendError, BackendResponse


class LMStudioError(OpenAICompatError):
    """Raised when the LM Studio API returns an error or malformatted payload."""
    pass


# Backwards compatibility alias
LMStudioResponse = BackendResponse


class LMStudioClient(OpenAICompatClient):
    """Chat client for LM Studio's OpenAI-compatible /v1/chat/completions endpoint.

    LM Studio implements the OpenAI API format, so this client inherits
    all functionality from OpenAICompatClient.

    Example usage:
        settings = LMStudioSettings(model="local-model")
        client = LMStudioClient(settings=settings)
        response = client.chat([{"role": "user", "content": "Hello"}])

        # List available models
        models = client.list_models()
    """

    settings: LMStudioSettings  # Type narrowing for IDE support

    @property
    def _client_name(self) -> str:
        return "LM Studio"

    def _create_error(self, message: str) -> BackendError:
        """Create an LMStudioError."""
        return LMStudioError(message)
