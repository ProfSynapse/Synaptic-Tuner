"""Validation service - coordinates schema validation only.

Single Responsibility: Coordinate validation across scopes.
Delegates all validation logic to StructureValidator.
"""

from typing import Dict, List, Tuple, Optional

from ..parsing import ScopeExtractor
from ..validators import StructureValidator
from ...config import ScopeConfig
from ...utils.logger import ImproveLogger


class ValidationService:
    """
    Coordinate schema validation.

    Responsibility: ONLY coordinate validation (SRP).
    Delegates actual validation to StructureValidator.
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
        self.structure_validator = StructureValidator(logger=self.logger)

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
            rubric_key = rubric.get("key", rubric.get("name", "unknown"))
            scope = rubric.get("scope", "response")
            validations = rubric.get("validations", [])

            # Skip if no validations defined
            if not validations:
                continue

            try:
                # Build data dict for validation
                data = self._extract_validation_data(example, scope)

                # Get raw content for pattern matching
                raw_content = self._get_raw_content(example, scope)

                # Run validation through StructureValidator
                is_valid, errors = self.structure_validator.validate(
                    data=data,
                    validations=validations,
                    raw_content=raw_content
                )

                results[rubric_key] = (is_valid, errors)

            except Exception as e:
                self.logger.warning(f"Validation error for {rubric_key}: {e}")
                results[rubric_key] = (False, [f"Validation error: {str(e)}"])

        return results

    def _extract_validation_data(self, example: Dict, scope: str) -> Dict:
        """
        Extract data for validation based on scope.

        For response scope, includes tool_calls from assistant message.
        For other scopes, extracts relevant content.

        Args:
            example: Example dict
            scope: Scope name

        Returns:
            Data dict for validation
        """
        data = {}

        # Extract scope-specific content
        content = self.scope_extractor.extract(example, scope)
        if content:
            if isinstance(content, dict):
                data.update(content)
            else:
                data["content"] = content

        # For response scope, also extract tool_calls
        if scope == "response":
            conversations = example.get("conversations", [])
            for conv in conversations:
                if conv.get("role") == "assistant":
                    tool_calls = conv.get("tool_calls", [])
                    if tool_calls:
                        data["tool_calls"] = tool_calls
                    break

        return data

    def _get_raw_content(self, example: Dict, scope: str) -> Optional[str]:
        """
        Get raw text content for pattern matching.

        Args:
            example: Example dict
            scope: Scope name

        Returns:
            Raw content string or None
        """
        conversations = example.get("conversations", [])

        if scope == "system_prompt":
            for conv in conversations:
                if conv.get("role") == "system":
                    return conv.get("content", "")

        elif scope == "response":
            for conv in conversations:
                if conv.get("role") == "assistant":
                    return conv.get("content") or ""

        elif scope == "user":
            for conv in conversations:
                if conv.get("role") == "user":
                    return conv.get("content", "")

        return None
