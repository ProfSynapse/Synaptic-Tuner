"""Schema validation for thinking blocks."""

import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..utils.yaml_loader import load_yaml
from ..utils.logger import ImproveLogger


class ThinkingSchemaValidator:
    """Validates thinking blocks against required schema."""

    def __init__(self, logger: Optional[ImproveLogger] = None):
        """Initialize schema validator."""
        self.logger = logger or ImproveLogger()

        # Load schema definition
        schema_file = Path(__file__).parent.parent / "config" / "thinking_schema.yaml"
        self.schema_config = load_yaml(schema_file)
        self.schema = self.schema_config["schema"]
        self.field_descriptions = self.schema_config.get("field_descriptions", {})

    def validate(self, thinking_block: Dict) -> Tuple[bool, List[str]]:
        """
        Validate thinking block against schema.

        Args:
            thinking_block: Parsed thinking JSON

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        required = self.schema.get("required", [])
        for field in required:
            if field not in thinking_block:
                errors.append(f"Missing required field: {field}")
                continue

            # Validate field type and constraints
            field_schema = self.schema["properties"].get(field, {})
            value = thinking_block[field]

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

    def get_schema_description(self) -> str:
        """Get human-readable schema description."""
        lines = []
        lines.append("Required Thinking Block Schema:")
        lines.append("")

        for field in self.schema.get("required", []):
            desc = self.field_descriptions.get(field, "")
            field_schema = self.schema["properties"].get(field, {})
            field_type = field_schema.get("type", "unknown")

            lines.append(f"- {field} ({field_type}): {desc}")

        return "\n".join(lines)

    def enforce_schema_on_dict(self, thinking_block: Dict, original_block: Dict) -> Dict:
        """
        Enforce schema by filling in missing fields from original.

        Args:
            thinking_block: Possibly incomplete thinking block
            original_block: Original thinking block to use as fallback

        Returns:
            Complete thinking block with all required fields
        """
        enforced = thinking_block.copy()

        # Ensure all required fields exist
        for field in self.schema.get("required", []):
            if field not in enforced or not enforced[field]:
                # Use original value if available, otherwise use default
                if field in original_block:
                    enforced[field] = original_block[field]
                    self.logger.warning(f"Schema enforcement: Restored '{field}' from original")
                else:
                    # Provide sensible defaults
                    field_type = self.schema["properties"][field].get("type")
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
