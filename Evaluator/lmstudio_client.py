"""HTTP client for interacting with LM Studio server.

This module provides the LMStudioClient for sending chat requests to
LM Studio's OpenAI-compatible API. It inherits from BaseBackendClient
for shared retry logic.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

import requests

from .base_client import (
    BaseBackendClient,
    extract_message_content,
    extract_models_from_list,
)
from .config import LMStudioSettings
from .protocols import BackendError, BackendResponse


class LMStudioError(BackendError):
    """Raised when the LM Studio API returns an error or malformatted payload."""
    pass


# Backwards compatibility alias
LMStudioResponse = BackendResponse


class LMStudioClient(BaseBackendClient):
    """Chat client for LM Studio's OpenAI-compatible /v1/chat/completions endpoint.

    LM Studio implements the OpenAI API format:
    - Endpoint: /v1/chat/completions
    - Generation params at top level
    - Response in choices[0].message

    Also supports model listing via /v1/models endpoint.

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

    def _build_payload(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        """Build OpenAI-compatible request payload.

        LM Studio uses standard OpenAI format:
        - Generation params at top level
        - 'max_tokens' for output limit
        - Optional 'seed' for reproducibility
        """
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "stream": False,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.seed is not None:
            payload["seed"] = self.settings.seed
        return payload

    def _get_chat_url(self) -> str:
        """Return LM Studio chat endpoint URL."""
        return f"{self.settings.base_url()}/v1/chat/completions"

    def _extract_response(self, data: Dict[str, Any], latency_s: float) -> BackendResponse:
        """Extract response from OpenAI-format response.

        LM Studio returns:
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "...",
                    "tool_calls": [...]  // optional
                }
            }],
            ...
        }
        """
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise LMStudioError(
                f"Unexpected LM Studio response payload: {json.dumps(data)[:200]}"
            )

        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise LMStudioError("LM Studio response missing valid message object")

        try:
            content = extract_message_content(message)
        except ValueError as exc:
            raise LMStudioError(
                f"LM Studio response missing 'choices[0].message.content': {exc}"
            ) from exc

        return BackendResponse(message=content, raw=data, latency_s=latency_s)

    def _create_error(self, message: str) -> BackendError:
        """Create an LMStudioError."""
        return LMStudioError(message)

    def list_models(self) -> List[str]:
        """Return the list of model IDs exposed by the LM Studio server.

        Uses the OpenAI-compatible /v1/models endpoint.

        Returns:
            List of model ID strings

        Raises:
            LMStudioError: If the request fails or no models are returned
        """
        url = f"{self.settings.base_url()}/v1/models"

        def fetch_models() -> List[str]:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return extract_models_from_list(data)

        return self._execute_with_retry(
            operation=fetch_models,
            error_message="Unable to list LM Studio models",
        )
