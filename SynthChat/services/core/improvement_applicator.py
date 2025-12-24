"""Improvement applicator - applies improvements to examples only.

Single Responsibility: Delegate to scope handlers (Strategy Pattern).
"""

from typing import Dict, Optional

from ..parsing import ScopeExtractor
from ..scope_handlers import get_handler
from ...config import ScopeConfig
from ...utils.logger import ImproveLogger


class ImprovementApplicator:
    """
    Apply improved content to examples.

    Responsibility: ONLY coordinate delegation to handlers (SRP).
    NO scope-specific logic - that's in handlers.
    """

    def __init__(
        self,
        scope_config: ScopeConfig,
        scope_extractor: ScopeExtractor,
        logger: Optional[ImproveLogger] = None
    ):
        """
        Initialize improvement applicator.

        Args:
            scope_config: Scope configuration
            scope_extractor: Scope extractor
            logger: Logger instance
        """
        self.scope_config = scope_config
        self.scope_extractor = scope_extractor
        self.logger = logger or ImproveLogger()

        # Cache of handler instances (one per scope)
        self._handler_cache = {}

    def apply(
        self,
        example: Dict,
        scope_name: str,
        improved_content: str,
        output_format: Optional[Dict] = None
    ) -> Dict:
        """
        Apply improved content for a scope.

        Delegates to appropriate ScopeHandler (Strategy Pattern).
        NO hardcoded scope names!

        Args:
            example: Original example
            scope_name: Scope that was improved
            improved_content: New content from LLM
            output_format: Optional output format config from rubric
                          {type: "assistant_message" | "content_only" | "tool_calls_only"}

        Returns:
            Updated example (deep copy)
        """
        # Get handler for this scope
        handler = self._get_handler(scope_name)
        if not handler:
            self.logger.warning(f"No handler for scope: {scope_name}")
            return example

        # Delegate to handler with output_format
        return handler.apply_improvement(example, improved_content, output_format)

    def _get_handler(self, scope_name: str):
        """Get cached handler instance for scope."""
        if scope_name not in self._handler_cache:
            try:
                handler = get_handler(
                    scope_name,
                    self.scope_config,
                    self.scope_extractor,
                    self.logger
                )
                self._handler_cache[scope_name] = handler
            except ValueError as e:
                self.logger.warning(f"Handler not found: {e}")
                return None

        return self._handler_cache.get(scope_name)
