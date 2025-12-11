"""Validation service - coordinates schema validation only.

Single Responsibility: Coordinate validation across scopes.
"""

from typing import Dict, List, Tuple, Optional

from ..parsing import ScopeExtractor
from ..schema_validator import SchemaValidator
from ...config import ScopeConfig
from ...utils.logger import ImproveLogger


class ValidationService:
    """
    Coordinate schema validation.

    Responsibility: ONLY coordinate validation (SRP).
    Delegates actual validation to SchemaValidator.
    """

    def __init__(
        self,
        scope_extractor: ScopeExtractor,
        scope_config: ScopeConfig,
        logger: Optional[ImproveLogger] = None
    ):
        """
        Initialize validation service.

        Args:
            scope_extractor: Scope extractor for content extraction
            scope_config: Scope configuration
            logger: Logger instance
        """
        self.scope_extractor = scope_extractor
        self.scope_config = scope_config
        self.logger = logger or ImproveLogger()

    def validate_example(
        self,
        example: Dict,
        rubrics: List[Dict]
    ) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Validate example against all rubrics.

        Args:
            example: Example to validate
            rubrics: List of rubric dicts

        Returns:
            Dict mapping rubric_key to (is_valid, errors)
        """
        results = {}

        for rubric in rubrics:
            rubric_key = rubric.get("name", "unknown")
            scope = rubric.get("scope", "response")

            # Only validate scopes with structured content
            if scope not in ["thinking", "system_prompt"]:
                continue

            try:
                # Extract content for this scope
                content = self.scope_extractor.extract(example, scope)

                if content is None:
                    results[rubric_key] = (False, [f"No {scope} content found"])
                    continue

                # Create validator for this rubric
                validator = SchemaValidator(rubric_key, logger=self.logger)

                # Validate based on scope type
                if scope == "system_prompt":
                    system_prompt_text = content if isinstance(content, str) else content.get("system_prompt", "")
                    is_valid, errors = validator.validate_content(system_prompt_text)
                    results[rubric_key] = (is_valid, errors)

                elif scope == "thinking":
                    # content should already be a dict from ScopeExtractor
                    if isinstance(content, dict):
                        is_valid, errors = validator.validate(content)
                        results[rubric_key] = (is_valid, errors)
                    else:
                        results[rubric_key] = (False, ["Thinking content is not a dict"])

            except Exception as e:
                self.logger.warning(f"Validation error for {rubric_key}: {e}")
                results[rubric_key] = (False, [f"Validation error: {str(e)}"])

        return results
