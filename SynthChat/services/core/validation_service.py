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

            # Check if rubric has cross_scope validation
            schema_validation = rubric.get("schema_validation", {})
            has_cross_scope = "cross_scope" in schema_validation.get("types", [])

            # Skip validation if not applicable
            if scope not in ["thinking", "system_prompt", "response"] and not has_tool_validation:
                continue

            # For response scope, only process if has tool validation OR cross_scope validation
            if scope == "response" and not has_tool_validation and not has_cross_scope:
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
                    is_valid, errors = validator.validate_system_prompt(system_prompt_text)
                    results[rubric_key] = (is_valid, errors)

                elif scope == "thinking":
                    # content should already be a dict from ScopeExtractor
                    if isinstance(content, dict):
                        # Use validate_thinking_content() which checks against content_schema
                        is_valid, errors = validator.validate_thinking_content(content)

                        # Check for cross_scope validation
                        if validator.has_cross_scope_validation():
                            target_scope = validator.get_cross_scope_target()
                            if target_scope:
                                # Handle multiple target scopes (e.g., [system_prompt, user])
                                target_scopes = target_scope if isinstance(target_scope, list) else [target_scope]
                                combined_target = []

                                for ts in target_scopes:
                                    # Extract target scope content from original example
                                    target_content = self.scope_extractor.extract(example, ts)
                                    if target_content:
                                        # If target is system_prompt, it returns a dict with 'system_prompt' key
                                        if isinstance(target_content, dict) and "system_prompt" in target_content:
                                            combined_target.append(target_content["system_prompt"])
                                        elif isinstance(target_content, str):
                                            combined_target.append(target_content)
                                        else:
                                            combined_target.append(str(target_content))

                                if combined_target:
                                    # Join all target content for validation
                                    target_str = "\n\n".join(combined_target)

                                    # Run cross-scope validation
                                    cross_valid, cross_errors = validator.validate_cross_scope(
                                        source_content=content,
                                        target_content=target_str
                                    )

                                    if not cross_valid:
                                        is_valid = False
                                        errors.extend(cross_errors)

                        results[rubric_key] = (is_valid, errors)
                    else:
                        results[rubric_key] = (False, ["Thinking content is not a dict"])

                elif scope == "response" and has_cross_scope:
                    # Handle cross-scope validation for response scope
                    # Content is the assistant response (string)
                    response_text = content if isinstance(content, str) else content.get("content", "")

                    # Create validator and run cross-scope
                    validator = SchemaValidator(rubric_key, logger=self.logger)
                    is_valid = True
                    errors = []

                    if validator.has_cross_scope_validation():
                        target_scope = validator.get_cross_scope_target()
                        if target_scope:
                            # Handle multiple target scopes (e.g., [system_prompt, user])
                            target_scopes = target_scope if isinstance(target_scope, list) else [target_scope]
                            combined_target = []

                            for ts in target_scopes:
                                target_content = self.scope_extractor.extract(example, ts)
                                if target_content:
                                    if isinstance(target_content, dict) and "system_prompt" in target_content:
                                        combined_target.append(target_content["system_prompt"])
                                    elif isinstance(target_content, str):
                                        combined_target.append(target_content)
                                    else:
                                        combined_target.append(str(target_content))

                            if combined_target:
                                # Join all target content for validation
                                target_str = "\n\n".join(combined_target)

                                # For response, we need to wrap it in a dict for validate_cross_scope
                                cross_valid, cross_errors = validator.validate_cross_scope(
                                    source_content={"content": response_text},
                                    target_content=target_str
                                )

                                if not cross_valid:
                                    is_valid = False
                                    errors.extend(cross_errors)

                    results[rubric_key] = (is_valid, errors)

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
