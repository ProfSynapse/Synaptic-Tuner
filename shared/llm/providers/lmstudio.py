"""LM Studio provider implementation."""

import json
import requests
from typing import Dict, Any, List

from ..base import BaseLLMClient
from ..exceptions import LLMConnectionError, LLMResponseError


# WSL troubleshooting message
WSL_HELP = """
WSL Users: If connecting to LM Studio on Windows fails:

1. In LM Studio (Windows):
   - Click "Developer" → "Server"
   - Enable "Serve on Local Network"
   - Note the IP (e.g., 192.168.1.104)

2. Set environment variable:
   export LMSTUDIO_HOST=192.168.1.104

3. Re-run your command
"""


class LMStudioClient(BaseLLMClient):
    """LM Studio client using OpenAI-compatible API."""

    def __init__(self, host: str = "localhost", port: int = 1234, model: str = "local-model"):
        """
        Initialize LM Studio client.

        Args:
            host: LM Studio host (default: localhost)
            port: LM Studio port (default: 1234)
            model: Model name (or 'local-model' for whatever is loaded)
        """
        self.host = host
        self.port = port
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/v1/chat/completions"

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    @property
    def model_name(self) -> str:
        return self.model

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Send chat completion request to LM Studio."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            data = self._make_request(payload)
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            raise LLMResponseError(f"LM Studio chat request failed: {e}")

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Request structured JSON output from LM Studio.

        Note: LM Studio may not support response_format parameter.
        We add JSON instructions to the system prompt instead.
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

        # Regular chat request (no response_format)
        payload = {
            "model": self.model,
            "messages": enhanced_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            data = self._make_request(payload)
            content = data["choices"][0]["message"]["content"]

            # Parse JSON response
            if not content or not content.strip():
                raise LLMResponseError("Empty response from LM Studio")

            # Try to extract JSON if wrapped in markdown or other text
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse JSON from LM Studio: {e}\nResponse: {content[:200]}")
        except Exception as e:
            raise LLMResponseError(f"LM Studio structured output failed: {e}")

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to LM Studio API."""
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
            error_msg = f"Cannot connect to LM Studio at {self.base_url}\n{WSL_HELP}"
            raise LLMConnectionError(error_msg)
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(f"LM Studio request timed out: {e}")
        except requests.exceptions.HTTPError as e:
            raise LLMResponseError(f"LM Studio HTTP error: {e}")
        except Exception as e:
            raise LLMConnectionError(f"LM Studio request failed: {e}")

    def test_connection(self) -> bool:
        """Test LM Studio connection."""
        try:
            # Try to list models endpoint
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
