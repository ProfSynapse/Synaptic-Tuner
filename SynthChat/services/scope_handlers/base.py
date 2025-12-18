"""Base scope handler - abstract interface.

Single Responsibility: Define the contract for scope handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class ScopeHandler(ABC):
    """
    Abstract base for scope handlers (Strategy Pattern).

    Each handler encapsulates ALL logic for ONE scope:
    - How to apply improved content
    - How to build prompt variables
    - Any scope-specific validation

    This eliminates hardcoded if/elif chains and enables:
    - Adding new scopes without modifying existing code (OCP)
    - Each scope has focused, single-responsibility class (SRP)
    """

    def __init__(self, scope_config, scope_extractor, logger=None):
        """
        Initialize handler.

        Args:
            scope_config: ScopeConfig instance (for markers/patterns)
            scope_extractor: ScopeExtractor instance (for parsing)
            logger: Logger instance
        """
        self.scope_config = scope_config
        self.scope_extractor = scope_extractor
        self.logger = logger

    @abstractmethod
    def apply_improvement(
        self,
        example: Dict,
        improved_content: str,
        output_format: Optional[Dict] = None
    ) -> Dict:
        """
        Apply improved content to example.

        Args:
            example: Original example dict
            improved_content: New content from LLM
            output_format: Optional output format config from rubric
                          {type: "assistant_message" | "content_only" | "tool_calls_only"}
                          - assistant_message: Parse as JSON with content + tool_calls
                          - content_only: Apply as text content (default)
                          - tool_calls_only: Parse as tool_calls, set content to null

        Returns:
            Updated example (should deep copy)
        """
        pass

    @abstractmethod
    def build_prompt_variables(
        self,
        example: Dict,
        judgment: Dict
    ) -> Dict:
        """
        Build scope-specific variables for prompt template.

        Args:
            example: Current example
            judgment: Judge result

        Returns:
            Dict of variables for template rendering

        Example:
            {
                "current_content": "...",
                "feedback": "...",
                "additional_context": "..."
            }
        """
        pass
