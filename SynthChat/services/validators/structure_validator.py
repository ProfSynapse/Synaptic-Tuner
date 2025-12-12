"""Structure validator with config-driven validation rules.

Validates data against a flat list of validation rules:
- Field validation: `field` key for parsed dict fields with dot notation
- Pattern validation: `match` key for text pattern matching
- Cross-scope: `cross_scope` key (delegated to CrossScopeValidator)
"""

import re
from typing import Dict, List, Tuple, Any, Optional


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
            elif "match" in validation:
                error = self._validate_pattern(raw_content, validation)
            elif "cross_scope" in validation:
                # Skip - handled by CrossScopeValidator
                continue
            else:
                # Unknown validation type
                if self.logger:
                    self.logger.warning(f"Unknown validation type: {validation}")
                continue

            if error:
                errors.append(error)

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
