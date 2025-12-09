"""OpenAI-compatible backend for dataset improvement (LM Studio, Ollama, etc)."""

import json
import requests
from typing import Dict, Any, Optional

from .base_backend import ImprovementBackend
from ...core.exceptions import LLMServiceError


class OpenAICompatibleBackend(ImprovementBackend):
    """Generic OpenAI-compatible backend (LM Studio, Ollama, LocalAI)."""

    def __init__(self, base_url: str, api_key: str = "lm-studio", model: str = "local-model"):
        """
        Initialize OpenAI-compatible backend.

        Args:
            base_url: Base URL for the API (e.g., http://localhost:1234/v1)
            api_key: API key (optional for some local servers)
            model: Model identifier to use
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.chat_endpoint = f"{self.base_url}/chat/completions"

    def improve_thinking_block(
        self,
        thinking_block: Dict[str, Any],
        system_prompt: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Improve thinking block using local LLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Improve this thinking block and respond with ONLY the JSON:\n\n{json.dumps(thinking_block, indent=2)}"
            }
        ]

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._make_request(messages)
                return response

            except Exception as e:
                last_error = e
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                continue
        
        raise LLMServiceError(
            f"Failed to improve thinking block after {max_retries} attempts: {last_error}"
        )

    def _make_request(self, messages: list) -> Dict[str, Any]:
        """Make API request."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Note: Local models might not support response_format="json_object" or schema
        # We'll try to use it if supported, or rely on prompting
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "stream": False
        }

        # Try to enforce JSON mode if possible (works for Ollama and some LM Studio versions)
        # For now, we'll just ask nicely in the prompt and parse the result
        
        try:
            response = requests.post(
                self.chat_endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            
            # Clean up markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content)
            
        except json.JSONDecodeError:
            raise LLMServiceError("Failed to parse JSON response from model")
        except requests.exceptions.RequestException as e:
            raise LLMServiceError(f"API request failed: {e}")

    def test_connection(self) -> bool:
        """Test if backend is reachable."""
        try:
            # Simple models list check or empty chat
            requests.get(f"{self.base_url}/models", timeout=5)
            return True
        except:
            return False
