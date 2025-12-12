"""LLM client wrapper - thin wrapper around shared.llm.

Single Responsibility: Provide consistent LLM API interface.
"""

from typing import Dict, List, Optional, Any

from ...utils.logger import ImproveLogger


class LLMClient:
    """
    Thin wrapper around LLM client.

    Responsibility: ONLY provide consistent API for LLM calls (SRP).
    Delegates actual calls to underlying client (from shared.llm).
    """

    def __init__(self, client, logger: Optional[ImproveLogger] = None):
        """
        Initialize LLM client wrapper.

        Args:
            client: Underlying LLM client (from shared.llm.create_client)
            logger: Logger instance
        """
        self.client = client
        self.logger = logger or ImproveLogger()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Chat completion (unstructured response).

        Args:
            messages: List of message dicts with role/content
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Dict with 'content' key containing response text
        """
        try:
            response = self.client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Normalize response format
            if isinstance(response, str):
                return {"content": response}
            elif isinstance(response, dict):
                return response
            else:
                return {"content": str(response)}

        except Exception as e:
            self.logger.error(f"LLM chat error: {e}")
            raise

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict,
        temperature: float = 0.0
    ) -> Dict:
        """
        Structured output (JSON response matching schema).

        Args:
            messages: List of message dicts with role/content
            schema: JSON schema for response
            temperature: Sampling temperature

        Returns:
            Parsed JSON dict matching schema
        """
        try:
            return self.client.structured_output(
                messages=messages,
                schema=schema,
                temperature=temperature
            )

        except Exception as e:
            self.logger.error(f"LLM structured output error: {e}")
            raise
