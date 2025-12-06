"""OpenRouter backend for dataset improvement."""

import json
import requests
from typing import Dict, Any

from .base_backend import ImprovementBackend
from ...core.exceptions import LLMServiceError


class OpenRouterBackend(ImprovementBackend):
    """OpenRouter API backend with structured output support."""

    def __init__(self, api_key: str, model: str = "openai/gpt-5-mini"):
        """
        Initialize OpenRouter backend.

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: openai/gpt-5-mini)
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def improve_thinking_block(
        self,
        thinking_block: Dict[str, Any],
        system_prompt: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Improve thinking block using OpenRouter with structured output."""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Improve this thinking block:\n\n{json.dumps(thinking_block, indent=2)}"
            }
        ]

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._make_request(messages)
                return response

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries - 1} after error: {e}")
                    continue
                else:
                    raise LLMServiceError(
                        f"Failed to improve thinking block after {max_retries} attempts: {e}"
                    )

    def _make_request(self, messages: list) -> Dict[str, Any]:
        """Make API request to OpenRouter with structured output."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ProfSynapse/Toolset-Training",
            "X-Title": "Dataset Improvement Engine"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2048,  # Generous for reasoning models
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "thinking_block",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "Specific, actionable goal (one sentence)"
                            },
                            "memory": {
                                "type": "string",
                                "description": "Mini-compaction maintaining context. Include WHY, WHEN, quantitative details. Min 50 chars."
                            },
                            "requirements": {
                                "type": "array",
                                "description": "Verification checks (VERIFY, CHECK, CONFIRM)",
                                "items": {"type": "string"}
                            },
                            "assessment": {
                                "type": "object",
                                "properties": {
                                    "complex": {
                                        "type": "boolean",
                                        "description": "True if multi-step or coordination needed"
                                    },
                                    "risky": {
                                        "type": "boolean",
                                        "description": "True if modifies/deletes data"
                                    }
                                },
                                "required": ["complex", "risky"],
                                "additionalProperties": False
                            },
                            "confidence": {
                                "type": "number",
                                "description": "0.3-0.5 (risky), 0.6-0.8 (medium), 0.85-0.95 (safe)"
                            },
                            "plan": {
                                "type": "array",
                                "description": "Execution steps (DO, EXECUTE, RUN)",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["goal", "memory", "requirements", "assessment", "confidence", "plan"],
                        "additionalProperties": False
                    }
                }
            }
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            # Extract structured JSON from response
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)

        except requests.exceptions.RequestException as e:
            raise LLMServiceError(f"API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise LLMServiceError(f"Invalid API response format: {e}")
        except json.JSONDecodeError as e:
            raise LLMServiceError(f"Failed to parse structured output: {e}")

    def test_connection(self) -> bool:
        """Test OpenRouter API connection."""
        try:
            # Simple test with minimal tokens
            test_messages = [
                {"role": "user", "content": "test"}
            ]
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": test_messages,
                "max_tokens": 1
            }
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
