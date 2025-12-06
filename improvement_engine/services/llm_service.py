"""LLM service using shared LLM client system."""

from typing import Dict, Any, Optional
from shared.llm import create_client, LLMError
from ..core.exceptions import LLMServiceError
from ..utils.yaml_loader import load_config


class LLMService:
    """Service for improving thinking blocks using shared LLM client."""

    def __init__(self, backend: str = "openrouter", model: Optional[str] = None):
        """
        Initialize LLM service with shared client.

        Args:
            backend: LLM backend to use (openrouter, lmstudio, ollama)
            model: Model name (optional, uses backend defaults if not specified)
        """
        # Create client with specified backend/model
        # API keys/hosts still come from environment variables
        try:
            self.client = create_client(provider=backend, model=model)
        except LLMError as e:
            raise LLMServiceError(f"Failed to initialize LLM client: {e}")

        # Load system prompts
        self.system_prompts = load_config("system_prompts")

        # Schema for structured output
        self.thinking_block_schema = {
            "type": "object",
            "name": "thinking_block",
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

    def improve_thinking_block(self, thinking_block: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Improve a thinking block using the configured LLM client.

        Args:
            thinking_block: Original thinking block
            max_retries: Maximum retry attempts (handled by client)

        Returns:
            Improved thinking block

        Raises:
            LLMServiceError: If improvement fails
        """
        import json

        system_prompt = self.system_prompts["improvement_prompt"]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Improve this thinking block:\n\n{json.dumps(thinking_block, indent=2)}"
            }
        ]

        try:
            # Use structured output - client handles retries internally
            response = self.client.structured_output(
                messages=messages,
                schema=self.thinking_block_schema,
                temperature=0.3,
                max_tokens=2048
            )

            return response

        except LLMError as e:
            raise LLMServiceError(f"Failed to improve thinking block: {e}")
        except Exception as e:
            raise LLMServiceError(f"Unexpected error during improvement: {e}")

    @property
    def provider_name(self) -> str:
        """Get the current provider name."""
        return self.client.provider_name

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self.client.model_name
