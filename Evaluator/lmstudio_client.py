"""Thin HTTP client for interacting with LM Studio server."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import requests

from .config import LMStudioSettings


class LMStudioError(RuntimeError):
    """Raised when the LM Studio API returns an error or malformatted payload."""


@dataclass
class LMStudioResponse:
    message: Any  # Can be str (ChatML) or Dict (OpenAI format with tool_calls)
    raw: Dict[str, Any]
    latency_s: float


class LMStudioClient:
    """Simple chat wrapper around LM Studio's OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, settings: LMStudioSettings, timeout: float = 60.0, retries: int = 2) -> None:
        self.settings = settings
        self.timeout = timeout
        self.retries = max(0, retries)

    def chat(self, messages: Sequence[Mapping[str, str]]) -> LMStudioResponse:
        """Send a chat conversation to the configured model."""
        payload = self._build_payload(messages)
        url = f"{self.settings.base_url()}/v1/chat/completions"

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                message = self._extract_message(data)
                latency_s = time.perf_counter() - start
                return LMStudioResponse(message=message, raw=data, latency_s=latency_s)
            except (requests.RequestException, ValueError, LMStudioError) as exc:
                last_err = exc
                if attempt == self.retries:
                    break
                # Basic exponential backoff
                time.sleep(min(2 ** attempt, 5))
        raise LMStudioError(f"LM Studio request failed after {self.retries + 1} attempts: {last_err}")

    def list_models(self) -> List[str]:
        """Return the list of model IDs exposed by the LM Studio server."""
        url = f"{self.settings.base_url()}/v1/models"
        last_err: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return self._extract_models(data)
            except (requests.RequestException, ValueError, LMStudioError) as exc:
                last_err = exc
                if attempt == self.retries:
                    break
                time.sleep(min(2 ** attempt, 5))
        raise LMStudioError(f"Unable to list LM Studio models after {self.retries + 1} attempts: {last_err}")

    def _build_payload(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        payload = {
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

    @staticmethod
    def _extract_message(payload: Mapping[str, Any]) -> Any:
        """
        Extract message from LM Studio response.

        Returns:
            - Dict with tool_calls if present and non-empty (OpenAI format)
            - String content otherwise (ChatML format or [TOOL_CALLS] format)
        """
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise LMStudioError(f"Unexpected LM Studio response payload: {json.dumps(payload)[:200]}")

        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise LMStudioError("LM Studio response missing valid message object")

        # Check if this is OpenAI format with tool_calls (must be non-empty)
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            # Return full message object for OpenAI format
            return dict(message)

        # ChatML format or [TOOL_CALLS] format - return content string
        content = message.get("content")
        if not isinstance(content, str):
            raise LMStudioError("LM Studio response missing 'choices[0].message.content'")
        return content

    @staticmethod
    def _extract_models(payload: Mapping[str, Any]) -> List[str]:
        data = payload.get("data")
        if not isinstance(data, list):
            raise LMStudioError(f"Unexpected LM Studio model list payload: {json.dumps(payload)[:200]}")

        models: List[str] = []
        for entry in data:
            model_id = entry.get("id") if isinstance(entry, Mapping) else None
            if isinstance(model_id, str):
                models.append(model_id)

        if not models:
            raise LMStudioError("LM Studio returned no models")
        return models
