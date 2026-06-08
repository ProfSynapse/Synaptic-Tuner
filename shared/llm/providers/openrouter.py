"""OpenRouter provider implementation."""

import json
import requests
from typing import Dict, Any, List

from ..base import BaseLLMClient
from ..exceptions import LLMConnectionError, LLMResponseError


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client with structured output and provider routing support."""

    def __init__(
        self,
        api_key: str,
        model: str,
        provider: Dict[str, Any] = None,
        timeout_seconds: float = 60.0,
        thinking_effort: str | None = None,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model name (e.g., 'openai/gpt-5-mini')
            provider: Optional provider routing config:
                - order: List of provider names to prioritize (e.g., ["Groq", "Together"])
                - allow_fallbacks: Whether to fall back to other providers (default: True)
                - require_parameters: Only use providers supporting all params (default: False)
                - data_collection: "allow" or "deny" to filter by data policy
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout_seconds = float(timeout_seconds)
        self.thinking_effort = _normalize_thinking_effort(thinking_effort)

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self.model

    def list_models(self) -> List[str]:
        """List models available via OpenRouter."""
        url = "https://openrouter.ai/api/v1/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = []
            for item in data.get("data", []):
                mid = item.get("id") or item.get("name")
                if mid:
                    models.append(str(mid))
            return models
        except Exception as e:
            raise LLMResponseError(f"Failed to list OpenRouter models: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Send chat completion request to OpenRouter."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add provider routing if configured
        if self.provider:
            payload["provider"] = self.provider
        if self.thinking_effort:
            payload["reasoning"] = {"effort": self.thinking_effort}

        try:
            data = self._make_request(payload)
            message = data["choices"][0]["message"]
            content = message.get("content")

            if content is None:
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    return json.dumps({"content": None, "tool_calls": tool_calls})
                raise LLMResponseError("Empty response from OpenRouter")

            if not isinstance(content, str):
                content = str(content)
            if not content.strip():
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    return json.dumps({"content": None, "tool_calls": tool_calls})
                raise LLMResponseError("Empty response from OpenRouter")

            return content

        except Exception as e:
            raise LLMResponseError(f"OpenRouter chat request failed: {e}")

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send request with JSON schema for structured output."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("name", "response"),
                    "strict": True,
                    "schema": schema
                }
            }
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Add provider routing if configured
        if self.provider:
            payload["provider"] = self.provider
        if self.thinking_effort:
            payload["reasoning"] = {"effort": self.thinking_effort}

        try:
            data = self._make_request(payload)
            content = data["choices"][0]["message"]["content"]

            # Parse JSON response
            if not content or not content.strip():
                raise LLMResponseError("Empty response from OpenRouter")

            return json.loads(content)

        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Failed to parse structured output: {e}\nResponse excerpt: {_truncate_response(content)}",
                raw_response=content,
            )
        except Exception as e:
            raise LLMResponseError(f"OpenRouter structured output failed: {e}")

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ProfSynapse/Toolset-Training",
            "X-Title": "Shared LLM Client"
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Cannot connect to OpenRouter: {e}")
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(f"OpenRouter request timed out: {e}")
        except requests.exceptions.HTTPError as e:
            # Try to get error details from response
            try:
                error_detail = response.json()
                raise LLMResponseError(f"OpenRouter HTTP error: {e}\nDetails: {error_detail}")
            except:
                raise LLMResponseError(f"OpenRouter HTTP error: {e}")
        except Exception as e:
            raise LLMConnectionError(f"OpenRouter request failed: {e}")

    def test_connection(self) -> bool:
        """Test OpenRouter API connection."""
        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10  # Higher for reasoning models
            }
            self._make_request(test_payload)
            return True
        except Exception as e:
            # Print error for debugging
            print(f"    Connection test error: {e}")
            return False


def _truncate_response(content: Any, limit: int = 1200) -> str:
    """Return a compact single-line excerpt for debugging malformed structured output."""
    text = str(content or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated>"


def _normalize_thinking_effort(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None
