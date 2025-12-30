"""XML structure validator."""

from typing import Dict, List, Tuple, Any, Optional

from ..base import BaseContentValidator


class XmlContentValidator(BaseContentValidator):
    """Validates XML structure in content."""

    def validate(
        self,
        content: str,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate XML structure.

        Config:
            xml:
              required_tags:
                - '<session_context>'
                - '<vault_structure>'

        Args:
            content: Content string to validate
            config: XML validation config
            context: Shared context (unused for XML)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        required_tags = config.get("required_tags", [])

        for tag in required_tags:
            if tag not in content:
                errors.append(f"Missing required XML tag: {tag}")

        return (len(errors) == 0, errors)
