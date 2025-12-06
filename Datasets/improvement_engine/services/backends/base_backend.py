"""Base backend for LLM improvement services."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ImprovementBackend(ABC):
    """Abstract base class for LLM backends used in dataset improvement."""

    @abstractmethod
    def improve_thinking_block(
        self,
        thinking_block: Dict[str, Any],
        system_prompt: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Improve a thinking block using the LLM backend.

        Args:
            thinking_block: Original thinking block
            system_prompt: System prompt with improvement instructions
            max_retries: Maximum number of retry attempts

        Returns:
            Improved thinking block as dictionary

        Raises:
            Exception: If improvement fails after all retries
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the backend is accessible.

        Returns:
            True if backend is reachable, False otherwise
        """
        pass
