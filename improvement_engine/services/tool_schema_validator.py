"""Tool call schema validation against tool_schemas.json."""

import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..utils.logger import ImproveLogger


class ToolSchemaValidator:
    """Validates tool calls against tool schema definitions."""

    def __init__(self, logger: Optional[ImproveLogger] = None):
        """Initialize tool schema validator."""
        self.logger = logger or ImproveLogger()

        # Load tool schemas
        schema_file = Path(__file__).parent.parent.parent / "Tools" / "tool_schemas.json"
        with open(schema_file, 'r', encoding='utf-8') as f:
            self.tool_schemas = json.load(f)

    def validate_tool_call(self, tool_name: str, arguments: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a tool call against its schema.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments dict for the tool call

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check if tool exists
        if tool_name not in self.tool_schemas:
            errors.append(f"Unknown tool: {tool_name}")
            return False, errors

        schema = self.tool_schemas[tool_name]

        # Check required parameters
        required_params = schema.get("required_params", [])
        for param in required_params:
            if param not in arguments:
                errors.append(f"{tool_name}: Missing required parameter '{param}'")

        # Validate context object if present
        if "context" in arguments:
            context = arguments["context"]
            if not isinstance(context, dict):
                errors.append(f"{tool_name}: 'context' must be an object")
            else:
                context_schema = schema.get("context_schema", {})
                context_fields = context_schema.get("fields", [])

                for field_def in context_fields:
                    field_name = field_def["name"]
                    is_optional = field_def.get("optional", False)

                    if not is_optional and field_name not in context:
                        errors.append(f"{tool_name}.context: Missing required field '{field_name}'")

        # Validate parameter types (basic validation)
        params_schema = schema.get("parameters", [])
        for param_def in params_schema:
            param_name = param_def["name"]
            expected_type = param_def.get("type")

            if param_name in arguments:
                value = arguments[param_name]

                # Basic type checking
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"{tool_name}.{param_name}: Expected string, got {type(value).__name__}")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"{tool_name}.{param_name}: Expected number, got {type(value).__name__}")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"{tool_name}.{param_name}: Expected boolean, got {type(value).__name__}")
                elif expected_type == "array" and not isinstance(value, list):
                    errors.append(f"{tool_name}.{param_name}: Expected array, got {type(value).__name__}")
                elif expected_type == "object" and not isinstance(value, dict):
                    errors.append(f"{tool_name}.{param_name}: Expected object, got {type(value).__name__}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_tool_schema_summary(self, tool_name: str) -> str:
        """Get human-readable summary of tool schema."""
        if tool_name not in self.tool_schemas:
            return f"Unknown tool: {tool_name}"

        schema = self.tool_schemas[tool_name]
        lines = []

        lines.append(f"Tool: {tool_name}")
        lines.append(f"Required parameters: {', '.join(schema.get('required_params', []))}")

        # Context schema
        context_schema = schema.get("context_schema", {})
        if context_schema:
            context_fields = [f["name"] for f in context_schema.get("fields", []) if not f.get("optional", False)]
            if context_fields:
                lines.append(f"Required context fields: {', '.join(context_fields)}")

        return "\n".join(lines)

    def get_all_vaultmanager_tools(self) -> List[str]:
        """Get list of all vaultManager tools."""
        return [
            tool_name for tool_name, schema in self.tool_schemas.items()
            if schema.get("agent") == "vaultManager"
        ]
