"""Scope handlers - Strategy pattern for scope-specific logic.

Each handler encapsulates ALL logic for one scope:
- Applying improvements
- Building prompt variables
- Scope-specific validation
"""

from .base import ScopeHandler
from .system_prompt_handler import SystemPromptHandler
from .user_handler import UserHandler
from .thinking_handler import ThinkingHandler
from .response_handler import ResponseHandler

# Registry of all handlers
SCOPE_HANDLERS = {
    "system_prompt": SystemPromptHandler,
    "user": UserHandler,
    "thinking": ThinkingHandler,
    "response": ResponseHandler,
}


def get_handler(scope_name: str, scope_config, scope_extractor, logger=None) -> ScopeHandler:
    """
    Get handler instance for a scope.

    Args:
        scope_name: Name of scope (e.g., "thinking", "response")
        scope_config: ScopeConfig instance
        scope_extractor: ScopeExtractor instance
        logger: Logger instance

    Returns:
        ScopeHandler instance

    Raises:
        ValueError: If scope not found
    """
    handler_class = SCOPE_HANDLERS.get(scope_name)
    if not handler_class:
        raise ValueError(f"No handler registered for scope: {scope_name}")

    return handler_class(scope_config, scope_extractor, logger)


__all__ = [
    "ScopeHandler",
    "SystemPromptHandler",
    "UserHandler",
    "ThinkingHandler",
    "ResponseHandler",
    "SCOPE_HANDLERS",
    "get_handler",
]
