"""Prompt renderer - template rendering only.

Single Responsibility: Render prompt templates with variables.
"""

from typing import Dict, Any


class PromptRenderer:
    """
    Render prompt templates.

    Responsibility: ONLY template rendering (SRP).
    Does NOT call LLM or handle responses.
    """

    def render(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Render template with variables.

        Args:
            template: Template string with {placeholders}
            variables: Dict of variable name -> value

        Returns:
            Rendered prompt string
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    def render_safe(self, template: str, variables: Dict[str, Any], default: str = "") -> str:
        """
        Render template with variables, using defaults for missing vars.

        Args:
            template: Template string with {placeholders}
            variables: Dict of variable name -> value
            default: Default value for missing variables

        Returns:
            Rendered prompt string
        """
        # Create a defaultdict-like behavior
        class SafeDict(dict):
            def __missing__(self, key):
                return default

        safe_vars = SafeDict(variables)
        return template.format_map(safe_vars)
