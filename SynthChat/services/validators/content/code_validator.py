"""Code syntax validator."""

from typing import Dict, List, Tuple, Any, Optional

from ..base import BaseContentValidator


class CodeContentValidator(BaseContentValidator):
    """Validates code syntax."""

    def validate(
        self,
        content: str,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate code syntax.

        Config:
            code:
              language: python  # or javascript, etc.
              check_syntax: true

        Args:
            content: Content string to validate
            config: Code validation config
            context: Shared context (unused for code)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        language = config.get("language", "python")
        check_syntax = config.get("check_syntax", True)

        if check_syntax:
            if language == "python":
                try:
                    compile(content, '<string>', 'exec')
                except SyntaxError as e:
                    errors.append(f"Python syntax error: {str(e)}")

            elif language == "javascript":
                # Could use a JS parser if needed
                # For now, basic check
                if "syntax error" in content.lower():
                    errors.append("JavaScript may have syntax errors")

            # Add more languages as needed

        return (len(errors) == 0, errors)
