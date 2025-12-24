"""Regex pattern validator with interpolation support."""

import re
from typing import Dict, List, Tuple, Any, Optional

from ..base import BaseContentValidator


class RegexContentValidator(BaseContentValidator):
    """Validates content against regex patterns."""

    def validate(
        self,
        content: str,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate content against regex patterns with {var} interpolation.

        Config:
            regex:
              patterns:
                - pattern: '\\btool_call:\\s*\\w+'
                  description: 'Must contain tool_call: format'
                - pattern: '{workspace_root}'
                  in_tag: 'vault_structure'
                  description: 'Workspace rootFolder must exist in vault_structure'

        Args:
            content: Content string to validate
            config: Regex validation config
            context: Shared context for interpolation

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        context = context if context is not None else {}

        patterns = config.get("patterns", [])
        for pattern_spec in patterns:
            pattern = pattern_spec["pattern"]
            description = pattern_spec.get("description", pattern)
            in_tag = pattern_spec.get("in_tag")

            # Interpolate {var} with context values
            interpolated_pattern = self._interpolate(pattern, context)

            # Skip if pattern still has unresolved {var}
            if '{' in interpolated_pattern and '}' in interpolated_pattern:
                continue

            # Determine search content (scoped to tag or full content)
            if in_tag:
                search_content = self._get_tag_content(content, in_tag)
                if search_content is None:
                    errors.append(f"Tag <{in_tag}> not found for pattern: {description}")
                    continue
            else:
                search_content = content

            # Check if this is a regex pattern or interpolated literal
            if interpolated_pattern == pattern_spec["pattern"]:
                # Original pattern (no interpolation) - use as regex
                try:
                    if not re.search(interpolated_pattern, search_content, re.DOTALL):
                        errors.append(f"Pattern not found: {description}")
                except re.error as e:
                    self._log_warning(f"Invalid regex pattern: {e}")
                    errors.append(f"Invalid regex pattern: {description}")
            else:
                # Interpolated value - do literal string search
                # Normalize paths by stripping leading slashes for comparison
                normalized_pattern = interpolated_pattern.lstrip('/')
                if interpolated_pattern not in search_content and normalized_pattern not in search_content:
                    errors.append(f"{description}: '{interpolated_pattern}' not found in <{in_tag}>")

        return (len(errors) == 0, errors)
