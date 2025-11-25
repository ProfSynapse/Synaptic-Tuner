"""HTTP client for interacting with an Ollama server.

This module provides the OllamaClient for sending chat requests to Ollama's
/api/chat endpoint. It inherits from BaseBackendClient for shared retry logic.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from .base_client import BaseBackendClient, extract_message_content
from .config import OllamaSettings
from .protocols import BackendError, BackendResponse


class OllamaError(BackendError):
    """Raised when the Ollama API returns an error or malformatted payload."""
    pass


# Backwards compatibility alias
OllamaResponse = BackendResponse


class OllamaClient(BaseBackendClient):
    """Chat client for Ollama's /api/chat endpoint.

    Ollama uses a custom API format that differs from OpenAI:
    - Endpoint: /api/chat
    - Options nested under 'options' key
    - Response message directly in 'message' key

    Example usage:
        settings = OllamaSettings(model="llama2")
        client = OllamaClient(settings=settings)
        response = client.chat([{"role": "user", "content": "Hello"}])
    """

    settings: OllamaSettings  # Type narrowing for IDE support

    @property
    def _client_name(self) -> str:
        return "Ollama"

    def _build_payload(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        """Build Ollama-specific request payload.

        Ollama uses:
        - 'options' dict for generation parameters
        - 'num_predict' instead of 'max_tokens'
        - 'stream': False for non-streaming
        """
        options: Dict[str, Any] = {
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "num_predict": self.settings.max_tokens,
        }
        if self.settings.seed is not None:
            options["seed"] = self.settings.seed

        return {
            "model": self.settings.model,
            "messages": list(messages),
            "stream": False,
            "options": options,
        }

    def _get_chat_url(self) -> str:
        """Return Ollama chat endpoint URL."""
        return f"{self.settings.base_url()}/api/chat"

    def _extract_response(self, data: Dict[str, Any], latency_s: float) -> BackendResponse:
        """Extract response from Ollama's response format.

        Ollama returns:
        {
            "message": {
                "role": "assistant",
                "content": "...",
                "tool_calls": [...]  // optional
            },
            ...
        }
        """
        message = data.get("message")
        if not isinstance(message, Mapping):
            raise OllamaError(
                f"Unexpected Ollama response payload: {json.dumps(data)[:200]}"
            )

        try:
            content = extract_message_content(message)
        except ValueError as exc:
            raise OllamaError(f"Ollama response missing 'message.content': {exc}") from exc

        return BackendResponse(message=content, raw=data, latency_s=latency_s)

    def _create_error(self, message: str) -> BackendError:
        """Create an OllamaError."""
        return OllamaError(message)
