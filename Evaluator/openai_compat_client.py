"""Shared client for OpenAI-compatible API backends.

This module provides a base client for backends that implement the OpenAI API
format (e.g., LM Studio, vLLM). It consolidates common logic to avoid
duplication between lmstudio_client.py and vllm_client.py.
"""
from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Sequence

import requests

from .base_client import (
    BaseBackendClient,
    extract_message_content,
    extract_models_from_list,
)
from .config import BaseBackendSettings
from .protocols import BackendError, BackendResponse


class OpenAICompatError(BackendError):
    """Raised when an OpenAI-compatible API returns an error."""
    pass


class OpenAICompatClient(BaseBackendClient):
    """Base client for OpenAI-compatible /v1/chat/completions endpoints.

    Both LM Studio and vLLM implement the OpenAI API format:
    - Endpoint: /v1/chat/completions
    - Generation params at top level
    - Response in choices[0].message

    Subclasses only need to override:
    - _client_name: Return client name for error messages
    - _create_error: Create backend-specific error

    Also supports model listing via /v1/models endpoint.
    """

    settings: BaseBackendSettings

    @property
    @abstractmethod
    def _client_name(self) -> str:
        """Return the client name for error messages."""
        ...

    @abstractmethod
    def _create_error(self, message: str) -> BackendError:
        """Create a backend-specific error."""
        ...

    def _build_payload(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        """Build OpenAI-compatible request payload.

        Standard OpenAI format:
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
        """Return OpenAI-compatible chat endpoint URL."""
        return f"{self.settings.base_url()}/v1/chat/completions"

    def _extract_response(self, data: Dict[str, Any], latency_s: float) -> BackendResponse:
        """Extract response from OpenAI-format response.

        Expected format:
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
            raise self._create_error(
                f"Unexpected {self._client_name} response payload: {json.dumps(data)[:200]}"
            )

        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise self._create_error(
                f"{self._client_name} response missing valid message object"
            )

        try:
            content = extract_message_content(message)
        except ValueError as exc:
            raise self._create_error(
                f"{self._client_name} response missing 'choices[0].message.content': {exc}"
            ) from exc

        return BackendResponse(message=content, raw=data, latency_s=latency_s)

    def list_models(self) -> List[str]:
        """Return the list of model IDs exposed by the server.

        Uses the OpenAI-compatible /v1/models endpoint.

        Returns:
            List of model ID strings

        Raises:
            BackendError: If the request fails or no models are returned
        """
        url = f"{self.settings.base_url()}/v1/models"

        def fetch_models() -> List[str]:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return extract_models_from_list(data)

        return self._execute_with_retry(
            operation=fetch_models,
            error_message=f"Unable to list {self._client_name} models",
        )

    def is_server_running(self) -> bool:
        """Check if the server is running and accessible.

        Returns:
            True if server responds to health check, False otherwise
        """
        try:
            url = f"{self.settings.base_url()}/v1/models"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
