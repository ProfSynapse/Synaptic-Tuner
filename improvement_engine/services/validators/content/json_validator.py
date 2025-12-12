"""JSON structure validator with extraction support."""

import json
import re
from typing import Dict, List, Tuple, Any, Optional

from ..base import BaseContentValidator


class JsonContentValidator(BaseContentValidator):
    """Validates JSON structure in content."""

    def validate(
        self,
        content: str,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        r"""
        Validate JSON structure and optionally extract values for cross-reference.

        Config options:

        1. Standalone JSON:
           json:
             required_fields: [field1, field2]

        2. JSON in XML sections:
           json:
             sections:
               - tag: 'selected_workspace'
                 extract_pattern: '\{[\s\S]*\}'
                 required_fields: [context, workspaceStructure]
                 extract:
                   - path: 'context.rootFolder'
                     as: 'workspace_root'
                     skip_if: '/'

        Args:
            content: Content string to validate
            config: JSON validation config
            context: Shared context for extracted values (mutated)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        context = context if context is not None else {}

        # Standalone JSON validation
        if "required_fields" in config:
            standalone_errors = self._validate_standalone(content, config)
            errors.extend(standalone_errors)

        # JSON in sections
        for section in config.get("sections", []):
            section_errors = self._validate_section(content, section, context)
            errors.extend(section_errors)

        return (len(errors) == 0, errors)

    def _validate_standalone(self, content: str, config: Dict) -> List[str]:
        """Validate standalone JSON content."""
        errors = []
        try:
            data = json.loads(content)
            for field in config.get("required_fields", []):
                if field not in data:
                    errors.append(f"Missing required JSON field: {field}")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
        return errors

    def _validate_section(
        self,
        content: str,
        section_config: Dict,
        context: Dict
    ) -> List[str]:
        """Validate JSON in an XML section."""
        errors = []
        tag = section_config["tag"]
        extract_pattern = section_config.get("extract_pattern", r'\{.*\}')
        required_fields = section_config.get("required_fields", [])

        # Extract section content
        match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', content, re.DOTALL)
        if not match:
            return errors  # Tag not found - skip

        section_content = match.group(1).strip()

        # Extract JSON using pattern
        json_match = re.search(extract_pattern, section_content, re.DOTALL)
        if not json_match:
            errors.append(f"No JSON found in <{tag}>")
            return errors

        try:
            json_data = json.loads(json_match.group(0))

            # Extract values for cross-reference if configured
            extract_config = section_config.get("extract", [])
            if extract_config:
                self._extract_and_store(json_data, extract_config, context)

            # Check required fields
            for field_spec in required_fields:
                field_errors = self._validate_field(tag, json_data, field_spec)
                errors.extend(field_errors)

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in <{tag}>: {str(e)}")

        return errors

    def _validate_field(
        self,
        tag: str,
        json_data: Dict,
        field_spec: Any
    ) -> List[str]:
        """Validate a single field against its spec."""
        errors = []

        # Support both simple string and dict format
        if isinstance(field_spec, str):
            field_name = field_spec
            field_constraints = {}
        else:
            field_name = field_spec.get("name", str(field_spec))
            field_constraints = field_spec

        # Check field exists
        if field_name not in json_data:
            errors.append(f"Missing field in <{tag}>: {field_name}")
            return errors

        field_value = json_data[field_name]
        field_type = field_constraints.get("type")

        # Array validation
        if field_type == "array":
            if not isinstance(field_value, list):
                errors.append(
                    f"Field <{tag}>.{field_name} must be array, "
                    f"got {type(field_value).__name__}"
                )
            else:
                min_items = field_constraints.get("min_items")
                if min_items and len(field_value) < min_items:
                    errors.append(
                        f"Field <{tag}>.{field_name} must have at least "
                        f"{min_items} items, has {len(field_value)}"
                    )

        # Object validation
        elif field_type == "object":
            if not isinstance(field_value, dict):
                errors.append(
                    f"Field <{tag}>.{field_name} must be object, "
                    f"got {type(field_value).__name__}"
                )
            else:
                min_properties = field_constraints.get("min_properties")
                if min_properties and len(field_value) < min_properties:
                    errors.append(
                        f"Field <{tag}>.{field_name} must have at least "
                        f"{min_properties} properties, has {len(field_value)}"
                    )

        # String validation
        elif field_type == "string":
            if not isinstance(field_value, str):
                errors.append(
                    f"Field <{tag}>.{field_name} must be string, "
                    f"got {type(field_value).__name__}"
                )
            else:
                min_length = field_constraints.get("min_length")
                if min_length and len(field_value) < min_length:
                    errors.append(
                        f"Field <{tag}>.{field_name} must have at least "
                        f"{min_length} characters, has {len(field_value)}"
                    )

        return errors

    def _extract_and_store(
        self,
        data: Dict,
        extract_config: List[Dict],
        context: Dict
    ) -> None:
        """Extract values from data and store in context."""
        for item in extract_config:
            path = item.get("path")
            var_name = item.get("as")
            skip_if = item.get("skip_if")

            if not path or not var_name:
                continue

            value = self._get_value_at_path(data, path)

            if value is None:
                continue

            if skip_if and value == skip_if:
                continue

            context[var_name] = value

    def _get_value_at_path(self, data: Dict, path: str) -> Optional[str]:
        """Navigate to value using dot notation path."""
        value = data
        for key in path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value if isinstance(value, str) else str(value) if value is not None else None
