"""Structure validator with config-driven validation rules.

Validates data against a flat list of validation rules:
- Field validation: `field` key for parsed dict fields with dot notation
- Pattern validation: `match` key for text pattern matching
- Tools validation: `tools` key for validating tool calls against a manifest
- Cross-scope: `cross_scope` key (delegated to CrossScopeValidator)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class StructureValidator:
    """Validates data against a flat list of validation rules."""

    def __init__(self, logger=None):
        self.logger = logger

    def validate(
        self,
        data: Dict,
        validations: List[Dict],
        raw_content: str = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate data against validations list.

        Args:
            data: Parsed data dict for field validation
            validations: List of validation rules:
                Field validation (field key):
                    - field: Field path (supports dot notation for nesting)
                    - type: Expected type (string, number, array, object, boolean)
                    - min: Optional minimum (length for string/array, value for number)
                    - max: Optional maximum (length for string/array, value for number)
                    - error: User-defined error message
                Pattern validation (match key):
                    - match: Pattern to find in text
                    - type: Optional "regex" for regex matching (default: contains)
                    - error: User-defined error message
                Cross-scope (cross_scope key):
                    - Handled separately by CrossScopeValidator
            raw_content: Raw text content for pattern validation

        Returns:
            (is_valid, error_messages)
        """
        if not validations:
            return (True, [])

        errors = []

        for validation in validations:
            # Route to appropriate validator based on key
            if "field" in validation:
                error = self._validate_field(data, validation)
                if error:
                    errors.append(error)
            elif "match" in validation:
                error = self._validate_pattern(raw_content, validation)
                if error:
                    errors.append(error)
            elif "tools" in validation:
                tool_errors = self._validate_tools(data, validation)
                if tool_errors:
                    errors.extend(tool_errors)
            elif "cross_scope" in validation:
                # Skip - handled by CrossScopeValidator
                continue
            else:
                # Unknown validation type
                if self.logger:
                    self.logger.warning(f"Unknown validation type: {validation}")
                continue

        return (len(errors) == 0, errors)

    def _validate_field(self, data: Dict, validation: Dict) -> Optional[str]:
        """Validate a field in the parsed data dict."""
        field = validation.get("field")
        expected_type = validation.get("type")
        min_val = validation.get("min")
        max_val = validation.get("max")
        error_template = validation.get("error", f"Field '{field}' failed validation")

        # Get value using dot notation
        value = self._get_nested_value(data, field)

        # Check 1: Field exists (auto-required)
        if value is None:
            return f"Field '{field}': {self._format_error(error_template, validation, actual='missing')}"

        # Check 2: Type matches
        if not self._check_type(value, expected_type):
            return f"Field '{field}': {self._format_error(error_template, validation, actual=type(value).__name__)}"

        # Check 3: Min constraint
        if min_val is not None:
            actual = self._get_size(value, expected_type)
            if actual < min_val:
                return f"Field '{field}': {self._format_error(error_template, validation, actual=actual)}"

        # Check 4: Max constraint
        if max_val is not None:
            actual = self._get_size(value, expected_type)
            if actual > max_val:
                return f"Field '{field}': {self._format_error(error_template, validation, actual=actual)}"

        return None

    def _validate_pattern(self, content: str, validation: Dict) -> Optional[str]:
        """
        Validate text content against a pattern.

        Supported types:
        - xml: Check both <tag> and </tag> exist
        - json: Check JSON field exists (supports dot notation for nested)
        - regex: Regex pattern matching
        - contains (default): Simple text contains
        """
        pattern = validation.get("match")
        val_type = validation.get("type", "contains")
        in_tag = validation.get("in_tag")
        error_template = validation.get("error", f"Pattern '{pattern}' not found")

        if content is None:
            return "Pattern validation requires raw_content but none provided"

        # Scope content to tag if specified
        search_content = content
        if in_tag:
            search_content = self._extract_tag_content(content, in_tag)
            if search_content is None:
                return f"Tag '<{in_tag}>' not found for validation"

        # Validate based on type
        if val_type == "xml":
            return self._validate_xml_tag(content, pattern, error_template)
        elif val_type == "json":
            return self._validate_json_field(search_content, pattern, error_template)
        elif val_type == "regex":
            if not re.search(pattern, search_content):
                return error_template
        else:  # contains
            if pattern not in search_content:
                return error_template

        return None

    def _validate_xml_tag(self, content: str, tag: str, error_template: str) -> Optional[str]:
        """Validate XML tag has both opening and closing."""
        # Check opening tag: <tag> or <tag attr="...">
        has_opening = f"<{tag}>" in content or f"<{tag} " in content
        # Check closing tag
        has_closing = f"</{tag}>" in content

        if not has_opening or not has_closing:
            return error_template
        return None

    def _validate_json_field(self, content: str, field_path: str, error_template: str) -> Optional[str]:
        """Validate JSON field exists. Supports dot notation for nested fields."""
        import json

        if "." in field_path:
            # Nested field - parse JSON and navigate
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return error_template
            try:
                data = json.loads(json_match.group())
                if self._get_nested_value(data, field_path) is None:
                    return error_template
            except json.JSONDecodeError:
                return f"Invalid JSON when checking field '{field_path}'"
        else:
            # Top-level field - look for "field": pattern
            if not re.search(rf'"{re.escape(field_path)}"\s*:', content):
                return error_template

        return None

    def _extract_tag_content(self, content: str, tag: str) -> Optional[str]:
        """Extract content between opening and closing XML tags."""
        # Match <tag> or <tag attr="..."> and capture content until </tag>
        pattern = rf'<{re.escape(tag)}[^>]*>([\s\S]*?)</{re.escape(tag)}>'
        match = re.search(pattern, content)
        return match.group(1) if match else None

    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """Get value from nested dict using dot notation."""
        if not data or not field_path:
            return None

        parts = field_path.split(".")
        value = data

        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]

        return value

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "array": list,
            "object": dict,
            "boolean": bool,
        }
        return isinstance(value, type_map.get(expected_type, object))

    def _get_size(self, value: Any, value_type: str) -> float:
        """Get size/length/value based on type."""
        if value_type in ("string", "array"):
            return len(value)
        elif value_type == "number":
            return value
        return 0

    def _format_error(self, template: str, validation: Dict, actual: Any = None) -> str:
        """Format error message with interpolation variables."""
        format_vars = {**validation}
        if actual is not None:
            format_vars["actual"] = actual
        try:
            return template.format(**format_vars)
        except KeyError:
            return template

    def _validate_tools(self, data: Dict, validation: Dict) -> List[str]:
        """
        Validate tool calls against a manifest of tool schemas.

        Args:
            data: Dict containing 'tool_calls' key with list of tool calls
            validation: Validation config with 'tools' manifest:
                tools:
                  toolName:
                    field1: type
                    field2: type
                    nested:
                      subfield: type
                error: Optional error template

        Returns:
            List of error messages (empty if valid)
        """
        tools_manifest = validation.get("tools", {})
        error_template = validation.get("error", "Tool '{tool_name}': {details}")

        # Get tool calls from data
        tool_calls = data.get("tool_calls", [])

        if not tool_calls:
            return []  # No tool calls to validate

        errors = []

        for tool_call in tool_calls:
            # Extract tool name and arguments
            function_data = tool_call.get("function", {})
            tool_name = function_data.get("name", "")
            args_str = function_data.get("arguments", "{}")

            # Parse arguments
            try:
                if isinstance(args_str, str):
                    args = json.loads(args_str)
                else:
                    args = args_str
            except json.JSONDecodeError:
                errors.append(f"Tool '{tool_name}': Invalid JSON in arguments")
                continue

            # Check if tool exists in manifest
            if tool_name not in tools_manifest:
                errors.append(f"Unknown tool: '{tool_name}' not in manifest")
                continue

            # Get schema for this tool
            tool_schema = tools_manifest[tool_name]

            # Validate arguments against schema
            schema_errors = self._validate_against_schema(args, tool_schema, tool_name)
            errors.extend(schema_errors)

        return errors

    def _validate_against_schema(
        self,
        data: Dict,
        schema: Dict,
        tool_name: str,
        path: str = "",
        subtools: Optional[Dict] = None
    ) -> List[str]:
        """
        Recursively validate data against a schema definition.

        Schema format:
            field: type        -> field must exist and be of type
            field:             -> field must exist and be an object
              subfield: type   -> nested field validation
            _additionalProperties: false  -> no extra fields allowed
            _required: [field1, field2]   -> only these fields are required
            _item_schema: {...}           -> schema for array items
            _subtools:                    -> subtool manifests for validating calls
              agentName:
                toolName:
                  _required: [param1]
                  param1: type

        Supported types: string, number, boolean, array, object

        Args:
            data: Data to validate
            schema: Schema definition
            tool_name: Tool name for error messages
            path: Current path for nested fields
            subtools: Subtool manifests passed down for calls validation

        Returns:
            List of error messages
        """
        errors = []

        # Check for additionalProperties constraint
        allow_additional = schema.get("_additionalProperties", True)

        # Get required fields list (if specified, only these are required)
        required_fields = schema.get("_required")

        # Get item schema for arrays
        item_schema = schema.get("_item_schema")

        # Get subtools manifest (for validating calls array)
        schema_subtools = schema.get("_subtools", subtools)

        # Get defined fields (excluding special keys starting with _)
        defined_fields = {k for k in schema.keys() if not k.startswith("_")}

        for field, expected in schema.items():
            # Skip special config keys
            if field.startswith("_"):
                continue

            field_path = f"{path}.{field}" if path else field

            # Get value from data
            value = data.get(field) if isinstance(data, dict) else None

            # Check field exists (respect _required list if specified)
            if value is None:
                if required_fields is None or field in required_fields:
                    errors.append(f"Tool '{tool_name}': Missing required field '{field_path}'")
                continue

            # If expected is a dict with _item_schema, it's an array with item validation
            if isinstance(expected, dict) and "_item_schema" in expected:
                if not isinstance(value, list):
                    errors.append(f"Tool '{tool_name}': Field '{field_path}' must be an array")
                else:
                    # Validate each item in the array
                    nested_item_schema = expected.get("_item_schema", {})
                    nested_subtools = expected.get("_subtools", schema_subtools)

                    for i, item in enumerate(value):
                        item_path = f"{field_path}[{i}]"
                        item_errors = self._validate_against_schema(
                            item, nested_item_schema, tool_name, item_path, nested_subtools
                        )
                        errors.extend(item_errors)

                        # If this is a calls array, validate params against subtool schema
                        if nested_subtools and isinstance(item, dict):
                            subtool_errors = self._validate_subtool_params(
                                item, nested_subtools, tool_name, item_path
                            )
                            errors.extend(subtool_errors)

            # If expected is a dict, it's a nested object schema
            elif isinstance(expected, dict):
                if not isinstance(value, dict):
                    errors.append(f"Tool '{tool_name}': Field '{field_path}' must be an object")
                else:
                    # Recurse into nested schema
                    nested_errors = self._validate_against_schema(
                        value, expected, tool_name, field_path, schema_subtools
                    )
                    errors.extend(nested_errors)

            # If expected is a string, it's a type name
            elif isinstance(expected, str):
                if not self._check_type(value, expected):
                    errors.append(
                        f"Tool '{tool_name}': Field '{field_path}' must be {expected}, "
                        f"got {type(value).__name__}"
                    )

        # Check for extra fields if additionalProperties is false
        if not allow_additional and isinstance(data, dict):
            extra_fields = set(data.keys()) - defined_fields
            for extra in extra_fields:
                extra_path = f"{path}.{extra}" if path else extra
                errors.append(f"Tool '{tool_name}': Unexpected field '{extra_path}' (additionalProperties: false)")

        return errors

    def _validate_subtool_params(
        self,
        call_item: Dict,
        subtools: Dict,
        tool_name: str,
        path: str
    ) -> List[str]:
        """
        Validate params for a specific agent/tool combination.

        Args:
            call_item: A single call item with {agent, tool, params}
            subtools: Subtool manifests {agentName: {toolName: {schema}}}
            tool_name: Parent tool name for error messages
            path: Current path for error messages

        Returns:
            List of error messages
        """
        errors = []

        agent = call_item.get("agent", "")
        subtool = call_item.get("tool", "")
        params = call_item.get("params", {})

        # Look up schema for this agent/tool
        agent_manifest = subtools.get(agent)
        if not agent_manifest:
            # Agent not in manifest - could be valid if manifest is partial
            return errors

        subtool_schema = agent_manifest.get(subtool)
        if not subtool_schema:
            # Tool not in agent manifest - report as unknown
            valid_tools = list(agent_manifest.keys())
            errors.append(
                f"Tool '{tool_name}': Unknown subtool '{agent}.{subtool}'. "
                f"Valid tools for {agent}: {valid_tools}"
            )
            return errors

        # Get required params
        required_params = subtool_schema.get("_required", [])

        # Validate required params exist
        for param in required_params:
            if param not in params:
                errors.append(
                    f"Tool '{tool_name}': {path} - '{agent}.{subtool}' "
                    f"missing required param '{param}'"
                )

        # Validate param types
        for param, expected_type in subtool_schema.items():
            if param.startswith("_"):
                continue

            if param in params:
                value = params[param]
                if isinstance(expected_type, str) and not self._check_type(value, expected_type):
                    errors.append(
                        f"Tool '{tool_name}': {path} - '{agent}.{subtool}' "
                        f"param '{param}' must be {expected_type}, got {type(value).__name__}"
                    )

        return errors
