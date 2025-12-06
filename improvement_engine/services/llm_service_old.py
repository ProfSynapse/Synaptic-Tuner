"""LLM service for OpenRouter API."""

import json
import requests
from typing import Dict, Any, List
from ..core.exceptions import LLMServiceError
from ..utils.yaml_loader import load_config


class LLMService:
    """Service for interacting with OpenRouter API."""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        """
        Initialize LLM service.

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: openai/gpt-4o-mini)
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.system_prompts = load_config("system_prompts")

    def improve_thinking_block(self, thinking_block: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Improve a single thinking block using LLM with structured output.

        Args:
            thinking_block: Original thinking block
            max_retries: Maximum number of retry attempts

        Returns:
            Improved thinking block

        Raises:
            LLMServiceError: If API call fails after all retries
        """
        system_prompt = self.system_prompts["improvement_prompt"]

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
                # Use structured output - LLM must return valid JSON matching schema
                response = self._make_request(messages, use_structured_output=True)
                return response

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Log retry attempt (but don't fail yet)
                    print(f"  Retry {attempt + 1}/{max_retries - 1} after error: {e}")
                    continue
                else:
                    # Final attempt failed
                    raise LLMServiceError(f"Failed to improve thinking block after {max_retries} attempts: {e}")

    def improve_batch(self, thinking_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Improve a batch of thinking blocks.

        Args:
            thinking_blocks: List of thinking blocks

        Returns:
            List of improved thinking blocks
        """
        improved_blocks = []

        for block in thinking_blocks:
            try:
                improved = self.improve_thinking_block(block)
                improved_blocks.append(improved)
            except LLMServiceError as e:
                # On error, keep original block
                improved_blocks.append(block)
                raise e

        return improved_blocks

    def _make_request(self, messages: List[Dict[str, str]], use_structured_output: bool = False) -> Any:
        """
        Make API request to OpenRouter.

        Args:
            messages: List of messages
            use_structured_output: Use structured JSON output schema

        Returns:
            Response content (str) or structured data (dict)

        Raises:
            LLMServiceError: If request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ProfSynapse/Toolset-Training",
            "X-Title": "Dataset Improvement Engine"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for consistent improvements
            "max_tokens": 2048,  # Generous limit for reasoning models (GPT-5-mini uses reasoning tokens)
        }

        # Add structured output schema if requested
        if use_structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "thinking_block",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "Specific, actionable goal with file paths where relevant"
                            },
                            "memory": {
                                "type": "string",
                                "description": "Mini-compaction of conversation thread maintaining context. Include WHY (rationale), WHEN (timeline/sequence), quantitative details (numbers, counts, dates). Minimum 50 characters. Think of this as maintaining thread continuity even with small context windows."
                            },
                            "requirements": {
                                "type": "array",
                                "description": "Verification checks before execution (VERIFY, CHECK, CONFIRM)",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "assessment": {
                                "type": "object",
                                "properties": {
                                    "complex": {
                                        "type": "boolean",
                                        "description": "True if operation requires multiple steps or coordination"
                                    },
                                    "risky": {
                                        "type": "boolean",
                                        "description": "True if operation modifies or deletes existing data"
                                    }
                                },
                                "required": ["complex", "risky"],
                                "additionalProperties": False
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Risk-calibrated confidence: 0.3-0.5 (risky ops like delete), 0.6-0.8 (medium like update), 0.85-0.95 (safe like read)"
                            },
                            "plan": {
                                "type": "array",
                                "description": "Execution steps (DO, EXECUTE, RUN, CREATE, UPDATE)",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["goal", "memory", "requirements", "assessment", "confidence", "plan"],
                        "additionalProperties": False
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

            if use_structured_output:
                # With structured output, content is already parsed JSON
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                # Regular text response
                return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise LLMServiceError(f"API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise LLMServiceError(f"Invalid API response format: {e}")
        except json.JSONDecodeError as e:
            raise LLMServiceError(f"Failed to parse structured output: {e}")
