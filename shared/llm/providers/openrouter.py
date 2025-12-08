"""OpenRouter provider implementation."""

import json
import requests
from typing import Dict, Any, List

from ..base import BaseLLMClient
from ..exceptions import LLMConnectionError, LLMResponseError


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client with structured output support."""

    def __init__(self, api_key: str, model: str):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model name (e.g., 'openai/gpt-5-mini')
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

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

        try:
            data = self._make_request(payload)
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            raise LLMResponseError(f"OpenRouter chat request failed: {e}")

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """Send request with JSON schema for structured output."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("name", "response"),
                    "strict": True,
                    "schema": schema
                }
            }
        }

        try:
            data = self._make_request(payload)
            content = data["choices"][0]["message"]["content"]

            # Parse JSON response
            if not content or not content.strip():
                raise LLMResponseError("Empty response from OpenRouter")

            return json.loads(content)

        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse structured output: {e}")
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
                timeout=60
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
