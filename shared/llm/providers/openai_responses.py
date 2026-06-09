"""OpenAI Responses API provider implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import requests

from ..base import BaseLLMClient
from ..exceptions import LLMConnectionError, LLMResponseError


class OpenAIResponsesClient(BaseLLMClient):
    """OpenAI Responses API client for stateless text and structured output."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        store: bool = False,
        structured_output_strict: bool = False,
        thinking_effort: str | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/responses"
        self.timeout_seconds = float(timeout_seconds)
        self.store = bool(store)
        self.structured_output_strict = bool(structured_output_strict)
        self.thinking_effort = _normalize_thinking_effort(thinking_effort)

    @property
    def provider_name(self) -> str:
        return "openai_responses"

    @property
    def model_name(self) -> str:
        return self.model

    def list_models(self) -> List[str]:
        """List models available via OpenAI."""
        headers = self._headers()
        try:
            response = requests.get(f"{self.base_url}/models", headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [str(item["id"]) for item in data.get("data", []) if item.get("id")]
        except Exception as e:
            raise LLMResponseError(f"Failed to list OpenAI models: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> str:
        """Send a stateless Responses request and return text.

        Reasoning effort resolves as a per-call override over the instance
        default: the explicit ``reasoning_effort`` argument wins; otherwise the
        client's ``thinking_effort`` (set at construction, the upstream #98
        mechanism) applies; if neither is set the ``reasoning`` field is omitted
        entirely. When present it is sent as ``reasoning: {effort: ...}``
        (minimal|low|medium|high). gpt-5-family reasoning models default to
        medium effort server-side, which can consume the entire
        max_output_tokens budget and return an empty message item, so callers
        wanting short outputs should pass "minimal".
        """
        _reject_stateful_kwargs(kwargs)
        payload = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "store": self.store,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        effort = reasoning_effort if reasoning_effort is not None else self.thinking_effort
        if effort:
            payload["reasoning"] = {"effort": effort}
        payload.update(kwargs)

        try:
            data = self._make_request(payload)
            content = _extract_output_text(data)
            if not content.strip():
                raise LLMResponseError("Empty response from OpenAI Responses API")
            return content
        except LLMResponseError:
            raise
        except Exception as e:
            raise LLMResponseError(f"OpenAI Responses chat request failed: {e}")

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send request with Responses API JSON Schema structured output.

        Reasoning effort resolves identically to chat(): the per-call
        ``reasoning_effort`` argument overrides the instance ``thinking_effort``
        default; if neither is set the ``reasoning`` field is omitted.
        """
        schema_name = str(schema.get("name") or "response")
        strict = bool(kwargs.pop("strict", self.structured_output_strict))
        _reject_stateful_kwargs(kwargs)
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "store": self.store,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": strict,
                    "schema": schema,
                }
            },
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        effort = reasoning_effort if reasoning_effort is not None else self.thinking_effort
        if effort:
            payload["reasoning"] = {"effort": effort}
        payload.update(kwargs)

        content = ""
        try:
            data = self._make_request(payload)
            content = _extract_output_text(data)
            if not content.strip():
                raise LLMResponseError("Empty response from OpenAI Responses API")
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Failed to parse structured output: {e}\nResponse excerpt: {_truncate_response(content)}",
                raw_response=content,
            )
        except LLMResponseError:
            raise
        except Exception as e:
            raise LLMResponseError(f"OpenAI Responses structured output failed: {e}")

    def test_connection(self) -> bool:
        """Test OpenAI Responses API connection."""
        try:
            self._make_request(
                {
                    "model": self.model,
                    "input": [{"role": "user", "content": "test"}],
                    "max_output_tokens": 10,
                    "store": self.store,
                }
            )
            return True
        except Exception as e:
            print(f"    Connection test error: {e}")
            return False

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to the OpenAI Responses API."""
        try:
            response = requests.post(
                self.api_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Cannot connect to OpenAI Responses API: {e}")
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(f"OpenAI Responses request timed out: {e}")
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json()
                raise LLMResponseError(f"OpenAI Responses HTTP error: {e}\nDetails: {error_detail}")
            except LLMResponseError:
                raise
            except Exception:
                raise LLMResponseError(f"OpenAI Responses HTTP error: {e}")
        except Exception as e:
            raise LLMConnectionError(f"OpenAI Responses request failed: {e}")


def _extract_output_text(data: Dict[str, Any]) -> str:
    """Extract text from Responses API response shapes."""
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: List[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)


def _reject_stateful_kwargs(kwargs: Dict[str, Any]) -> None:
    """Reject Responses API conversation-state fields in this stateless provider."""
    forbidden = {"store", "previous_response_id", "conversation"}
    present = sorted(forbidden.intersection(kwargs))
    if present:
        raise LLMResponseError(
            "OpenAI Responses provider is stateless; unsupported request fields: "
            f"{', '.join(present)}"
        )


def _truncate_response(content: Any, limit: int = 1200) -> str:
    """Return a compact single-line excerpt for malformed structured output."""
    text = str(content or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated>"


def _normalize_thinking_effort(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None
