"""Generic schema validation for any rubric scope."""

import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..utils.yaml_loader import load_yaml
from ..utils.logger import ImproveLogger


class SchemaValidator:
    """Validates content against rubric-defined schemas."""

    def __init__(self, rubric_name: str, logger: Optional[ImproveLogger] = None):
        """
        Initialize schema validator.

        Args:
            rubric_name: Name of rubric (YAML file stem in rubrics/)
            logger: Logger instance
        """
        self.rubric_name = rubric_name
        self.logger = logger or ImproveLogger()

        # Load schema from rubric YAML
        rubric_file = Path(__file__).parent.parent / "rubrics" / f"{rubric_name}.yaml"
        if not rubric_file.exists():
            raise FileNotFoundError(f"Rubric not found: {rubric_file}")

        rubric_data = load_yaml(rubric_file)
        self.schema = rubric_data.get("output_schema", {})
        self.scope = rubric_data.get("scope", "response")

        if not self.schema:
            raise ValueError(f"Rubric {rubric_name} has no output_schema defined")

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data against loaded schema.

        Args:
            data: Data to validate (thinking block, system prompt, etc.)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        required = self.schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                continue

            # Validate field type and constraints
            field_schema = self.schema.get("properties", {}).get(field, {})
            value = data[field]

            # Type validation
            expected_type = field_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"{field}: Expected string, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"{field}: Expected number, got {type(value).__name__}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"{field}: Expected array, got {type(value).__name__}")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"{field}: Expected object, got {type(value).__name__}")

            # String length constraints
            if expected_type == "string" and isinstance(value, str):
                min_length = field_schema.get("minLength")
                if min_length and len(value) < min_length:
                    errors.append(f"{field}: Too short ({len(value)} chars, min {min_length})")

            # Array length constraints
            if expected_type == "array" and isinstance(value, list):
                min_items = field_schema.get("minItems")
                if min_items and len(value) < min_items:
                    errors.append(f"{field}: Too few items ({len(value)}, min {min_items})")

            # Number range constraints
            if expected_type == "number" and isinstance(value, (int, float)):
                minimum = field_schema.get("minimum")
                maximum = field_schema.get("maximum")
                if minimum is not None and value < minimum:
                    errors.append(f"{field}: Below minimum ({value} < {minimum})")
                if maximum is not None and value > maximum:
                    errors.append(f"{field}: Above maximum ({value} > {maximum})")

            # Object required properties
            if expected_type == "object" and isinstance(value, dict):
                required_props = field_schema.get("required", [])
                for prop in required_props:
                    if prop not in value:
                        errors.append(f"{field}.{prop}: Missing required property")

        is_valid = len(errors) == 0
        return is_valid, errors

    def extract_content(self, example: Dict) -> Optional[Dict]:
        """
        Extract content from example based on rubric scope.

        Args:
            example: Full example with conversations

        Returns:
            Extracted content dict, or None if not found
        """
        if self.scope == "thinking":
            return self._extract_thinking_block(example)
        elif self.scope == "system_prompt":
            return self._extract_system_prompt(example)
        elif self.scope == "response":
            return self._extract_assistant_response(example)
        else:
            self.logger.warning(f"Unknown scope: {self.scope}")
            return None

    def _extract_thinking_block(self, example: Dict) -> Optional[Dict]:
        """Extract and parse thinking block from assistant response."""
        for conv in example.get("conversations", []):
            if conv.get("role") == "assistant":
                content = conv.get("content", "")

                # Extract thinking block
                match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
                if not match:
                    return None

                thinking_str = match.group(1).strip()

                # Try to parse as JSON
                try:
                    thinking_block = json.loads(thinking_str)
                    return thinking_block
                except json.JSONDecodeError:
                    self.logger.warning("Thinking block is not valid JSON")
                    return None

        return None

    def _extract_system_prompt(self, example: Dict) -> Optional[Dict]:
        """Extract system prompt from conversations."""
        for conv in example.get("conversations", []):
            if conv.get("role") == "system":
                return {"system_prompt": conv.get("content", "")}

        # If no system role, check if first message has system-like content
        conversations = example.get("conversations", [])
        if conversations and conversations[0].get("role") == "user":
            # Some formats include system prompt in user message
            # For now, return None - system prompt rubric will handle extraction
            return None

        return None

    def _extract_assistant_response(self, example: Dict) -> Optional[Dict]:
        """Extract full assistant response."""
        for conv in example.get("conversations", []):
            if conv.get("role") == "assistant":
                return {
                    "content": conv.get("content", ""),
                    "tool_calls": conv.get("tool_calls", [])
                }

        return None

    def get_schema_description(self) -> str:
        """Get human-readable schema description."""
        lines = []
        lines.append(f"Schema for {self.rubric_name} (scope: {self.scope}):")
        lines.append("")

        required = self.schema.get("required", [])
        properties = self.schema.get("properties", {})

        for field in required:
            field_schema = properties.get(field, {})
            field_type = field_schema.get("type", "unknown")
            description = field_schema.get("description", "")

            lines.append(f"- {field} ({field_type}): {description}")

        return "\n".join(lines)

    def enforce_schema_on_dict(self, data: Dict, original_data: Dict) -> Dict:
        """
        Enforce schema by filling in missing fields from original.

        Args:
            data: Possibly incomplete data
            original_data: Original data to use as fallback

        Returns:
            Complete data with all required fields
        """
        enforced = data.copy()
        properties = self.schema.get("properties", {})

        # Ensure all required fields exist
        for field in self.schema.get("required", []):
            if field not in enforced or not enforced[field]:
                # Use original value if available, otherwise use default
                if field in original_data:
                    enforced[field] = original_data[field]
                    self.logger.warning(f"Schema enforcement: Restored '{field}' from original")
                else:
                    # Provide sensible defaults based on type
                    field_type = properties[field].get("type")
                    if field_type == "string":
                        enforced[field] = "Not specified"
                    elif field_type == "array":
                        enforced[field] = []
                    elif field_type == "object":
                        enforced[field] = {}
                    elif field_type == "number":
                        enforced[field] = 0.5
                    self.logger.warning(f"Schema enforcement: Added default for '{field}'")

        return enforced
