"""Validation service - coordinates schema validation only.

Single Responsibility: Coordinate validation across scopes.
"""

from typing import Dict, List, Tuple, Optional

from ..parsing import ScopeExtractor
from ..schema_validator import SchemaValidator
from ..validators.tool_call_validator import ToolCallValidator
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

        # Initialize tool call validator
        self.tool_call_validator = ToolCallValidator(logger=self.logger)

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

            # Check if this rubric has tool call validation
            validation_config = rubric.get("validation", {})
            has_tool_validation = "tool_calls" in validation_config and validation_config["tool_calls"].get("enabled", False)

            # Skip validation if not applicable
            if scope not in ["thinking", "system_prompt"] and not has_tool_validation:
                continue

            try:
                # Handle tool call validation (for response scope)
                if has_tool_validation and scope == "response":
                    is_valid, errors = self._validate_tool_calls(example)
                    results[f"{rubric_key}_tool_calls"] = (is_valid, errors)
                    if not is_valid:
                        # Also store under main rubric key
                        results[rubric_key] = (is_valid, errors)
                    continue

                # Handle schema validation for structured content
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
                        # Use validate_thinking_content() which checks against content_schema
                        is_valid, errors = validator.validate_thinking_content(content)
                        results[rubric_key] = (is_valid, errors)
                    else:
                        results[rubric_key] = (False, ["Thinking content is not a dict"])

            except Exception as e:
                self.logger.warning(f"Validation error for {rubric_key}: {e}")
                results[rubric_key] = (False, [f"Validation error: {str(e)}"])

        return results

    def _validate_tool_calls(self, example: Dict) -> Tuple[bool, List[str]]:
        """
        Validate tool calls in assistant response.

        Args:
            example: Example dict with conversations

        Returns:
            Tuple of (is_valid, error_messages)
        """
        conversations = example.get("conversations", [])

        # Extract system prompt, user request, and tool calls
        system_prompt = ""
        user_request = ""
        tool_calls = []

        for conv in conversations:
            role = conv.get("role")
            if role == "system":
                system_prompt = conv.get("content", "")
            elif role == "user":
                user_request = conv.get("content", "")
            elif role == "assistant":
                tool_calls = conv.get("tool_calls", [])

        # If no tool calls, validation passes
        if not tool_calls:
            return (True, [])

        # Validate using tool call validator
        return self.tool_call_validator.validate_tool_calls(
            tool_calls,
            system_prompt,
            user_request
        )
