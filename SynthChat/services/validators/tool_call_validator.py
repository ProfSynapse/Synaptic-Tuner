"""Simple tool call validator - checks existence and params only.

Validates tool calls against tool_schemas.yaml configuration.
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ...utils.yaml_loader import load_yaml
from ...utils.logger import ImproveLogger


class ToolCallValidator:
    """
    Validates tool calls against tool_schemas.yaml.

    Checks:
    1. Tool exists in schema
    2. Required parameters are present
    """

    def __init__(self, schema_path: Optional[Path] = None, logger: Optional[ImproveLogger] = None):
        """
        Initialize validator with tool schema.

        Args:
            schema_path: Path to tool_schemas.yaml (default: config/tool_schemas.yaml)
            logger: Logger instance
        """
        self.logger = logger or ImproveLogger()

        # Load schema from YAML
        if schema_path is None:
            schema_path = Path(__file__).parent.parent.parent / "config" / "tool_schemas.yaml"

        if not schema_path.exists():
            self.logger.warning(f"Tool schema not found: {schema_path}")
            self.tool_schemas = {}
        else:
            self.tool_schemas = load_yaml(schema_path)
            self.logger.info(f"Loaded {len(self.tool_schemas)} tool definitions from {schema_path.name}")

    def validate_tool_calls(
        self,
        tool_calls: List[Dict],
        system_prompt: str,
        user_request: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate tool calls against schema and context.

        Args:
            tool_calls: List of tool call dicts from assistant message
            system_prompt: System prompt content
            user_request: User request content

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not tool_calls:
            return (True, [])  # No tool calls = valid

        errors = []

        for tool_call in tool_calls:
            # Extract tool name and arguments
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")
            arguments = function.get("arguments", "{}")

            if not tool_name:
                errors.append("Tool call missing 'function.name' field")
                continue

            # Parse arguments JSON
            try:
                if isinstance(arguments, str):
                    args = json.loads(arguments)
                else:
                    args = arguments
            except json.JSONDecodeError as e:
                errors.append(f"Tool '{tool_name}': Invalid JSON arguments - {e}")
                continue

            # Validate this tool call
            tool_errors = self._validate_single_tool(
                tool_name,
                args,
                system_prompt,
                user_request
            )
            errors.extend(tool_errors)

        return (len(errors) == 0, errors)

    def _validate_single_tool(
        self,
        tool_name: str,
        args: Dict,
        system_prompt: str,
        user_request: str
    ) -> List[str]:
        """
        Validate a single tool call.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check 1: Tool exists in schema
        if tool_name not in self.tool_schemas:
            errors.append(f"Tool '{tool_name}' not found in schema")
            return errors  # Can't validate further without schema

        schema = self.tool_schemas[tool_name]

        # Check 2: Required parameters are present
        required_params = schema.get("required_params", [])
        for param in required_params:
            if param not in args:
                errors.append(
                    f"Tool '{tool_name}': Missing required parameter '{param}'"
                )

        return errors

    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """
        Get schema information for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema dict or None if not found
        """
        return self.tool_schemas.get(tool_name)

    def list_tools(self) -> List[str]:
        """
        Get list of all available tool names.

        Returns:
            Sorted list of tool names
        """
        return sorted(self.tool_schemas.keys())
