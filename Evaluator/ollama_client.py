"""Thin HTTP client for interacting with an Ollama server."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import requests

from .config import OllamaSettings


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns an error or malformatted payload."""


@dataclass
class OllamaResponse:
    message: Any  # Can be str (ChatML) or Dict (OpenAI format with tool_calls)
    raw: Dict[str, Any]
    latency_s: float


class OllamaClient:
    """Simple chat wrapper around Ollama's /api/chat endpoint."""

    def __init__(self, settings: OllamaSettings, timeout: float = 60.0, retries: int = 2) -> None:
        self.settings = settings
        self.timeout = timeout
        self.retries = max(0, retries)

    def chat(self, messages: Sequence[Mapping[str, str]]) -> OllamaResponse:
        """Send a chat conversation to the configured model."""
        payload = self._build_payload(messages)
        url = f"{self.settings.base_url()}/api/chat"

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                message = self._extract_message(data)
                latency_s = time.perf_counter() - start
                return OllamaResponse(message=message, raw=data, latency_s=latency_s)
            except (requests.RequestException, ValueError, OllamaError) as exc:
                last_err = exc
                if attempt == self.retries:
                    break
                # Basic exponential backoff
                time.sleep(min(2 ** attempt, 5))
        raise OllamaError(f"Ollama request failed after {self.retries + 1} attempts: {last_err}")

    def _build_payload(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
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

    @staticmethod
    def _extract_message(payload: Mapping[str, Any]) -> Any:
        """
        Extract message from Ollama response.

        Returns:
            - Dict with tool_calls if present (OpenAI format)
            - String content otherwise (ChatML format)
        """
        message = payload.get("message")
        if not isinstance(message, Mapping):
            raise OllamaError(f"Unexpected Ollama response payload: {json.dumps(payload)[:200]}")

        # Check if this is OpenAI format with tool_calls
        if "tool_calls" in message:
            # Return full message object for OpenAI format
            return dict(message)

        # ChatML format - return content string
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError("Ollama response missing 'message.content'")
        return content
