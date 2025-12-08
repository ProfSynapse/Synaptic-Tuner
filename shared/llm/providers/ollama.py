"""Ollama provider implementation."""

import json
import requests
from typing import Dict, Any, List

from ..base import BaseLLMClient
from ..exceptions import LLMConnectionError, LLMResponseError


class OllamaClient(BaseLLMClient):
    """Ollama client using native /api/chat endpoint."""

    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "llama2"):
        """
        Initialize Ollama client.

        Args:
            host: Ollama host (default: localhost)
            port: Ollama port (default: 11434)
            model: Model name (e.g., 'llama2', 'mistral')
        """
        self.host = host
        self.port = port
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api/chat"

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    def list_models(self) -> List[str]:
        """List models available in Ollama."""
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            models = []
            for item in data.get("models", []):
                name = item.get("name")
                if name:
                    models.append(str(name))
            return models
        except Exception as e:
            raise LLMResponseError(f"Failed to list Ollama models: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Send chat completion request to Ollama."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens
            }
        }

        try:
            data = self._make_request(payload)
            return data["message"]["content"]

        except Exception as e:
            raise LLMResponseError(f"Ollama chat request failed: {e}")

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Request structured JSON output from Ollama.

        Ollama supports 'format': 'json' to request JSON output.
        We also inject schema instructions for better results.
        """
        # Inject JSON schema instructions into messages
        enhanced_messages = messages.copy()

        # Add JSON instruction to first system message or create one
        json_instruction = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\nRespond ONLY with valid JSON, no other text."

        # Find system message or add one
        system_found = False
        for msg in enhanced_messages:
            if msg.get("role") == "system":
                msg["content"] += json_instruction
                system_found = True
                break

        if not system_found:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": f"You are a helpful assistant.{json_instruction}"
            })

        payload = {
            "model": self.model,
            "messages": enhanced_messages,
            "stream": False,
            "format": "json",  # Ollama's way of requesting JSON
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            data = self._make_request(payload)
            content = data["message"]["content"]

            # Parse JSON response
            if not content or not content.strip():
                raise LLMResponseError("Empty response from Ollama")

            # Clean up markdown if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse JSON from Ollama: {e}\nResponse: {content[:200]}")
        except Exception as e:
            raise LLMResponseError(f"Ollama structured output failed: {e}")

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Ollama API."""
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120  # Longer timeout for local models
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}\n"
                f"Make sure Ollama is running: ollama serve"
            )
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(f"Ollama request timed out: {e}")
        except requests.exceptions.HTTPError as e:
            raise LLMResponseError(f"Ollama HTTP error: {e}")
        except Exception as e:
            raise LLMConnectionError(f"Ollama request failed: {e}")

    def test_connection(self) -> bool:
        """Test Ollama connection."""
        try:
            # Try to get version info
            response = requests.get(
                f"{self.base_url}/api/version",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
