"""Base classes and protocols for content validators."""

import re
from typing import Protocol, Tuple, List, Dict, Any, Optional


class ContentValidatorProtocol(Protocol):
    """Protocol for content validators."""

    def validate(
        self,
        content: str,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate content against config.

        Args:
            content: Content string to validate
            config: Validator-specific config from YAML
            context: Shared context (extracted values, etc.)

        Returns:
            (is_valid, list_of_errors)
        """
        ...


class BaseContentValidator:
    """Base class for content validators with shared utilities."""

    def __init__(self, logger=None):
        self.logger = logger

    def _get_tag_content(self, content: str, tag: str) -> Optional[str]:
        """Extract content from an XML tag."""
        match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', content, re.DOTALL)
        return match.group(1) if match else None

    def _interpolate(self, text: str, context: Dict[str, Any]) -> str:
        """Replace {var} with context values."""
        if not context:
            return text
        for key, val in context.items():
            text = text.replace(f"{{{key}}}", str(val))
        return text

    def _log_warning(self, message: str) -> None:
        """Log a warning if logger is available."""
        if self.logger:
            self.logger.warning(message)
