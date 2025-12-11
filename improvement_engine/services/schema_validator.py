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
        self.scope = rubric_data.get("scope", "response")

        # output_schema: What the JUDGE should return (scores, feedback)
        self.output_schema = rubric_data.get("output_schema", {})

        # content_schema: What the IMPROVED CONTENT should look like
        self.content_schema = rubric_data.get("content_schema", {})

        # Load content validation rules (composable types for xml, json, yaml, etc.)
        self.content_validation = rubric_data.get("schema_validation", {})
        self.validation_types = self.content_validation.get("types", [])

        # Shared context for cross-reference validation (extract → reference)
        self.extracted_values = {}

    def validate_thinking_content(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate thinking block content against content_schema.

        Args:
            data: Thinking block dict (goal, memory, requirements, plan, etc.)

        Returns:
            (is_valid, list_of_errors)
        """
        if not self.content_schema:
            # No content_schema defined, skip validation
            return (True, [])

        return self._validate_against_schema(data, self.content_schema)

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data against output_schema (for judge response validation).

        Args:
            data: Data to validate (judge response)

        Returns:
            (is_valid, list_of_errors)
        """
        return self._validate_against_schema(data, self.output_schema)

    def _validate_against_schema(self, data: Dict, schema: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data against a given schema.

        Args:
            data: Data to validate
            schema: JSON schema to validate against

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        if not schema:
            return (True, [])

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                continue

            # Validate field type and constraints
            field_schema = schema.get("properties", {}).get(field, {})
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

    def get_schema_description(self, schema_type: str = "content") -> str:
        """Get human-readable schema description."""
        schema = self.content_schema if schema_type == "content" else self.output_schema
        if not schema:
            return f"No {schema_type} schema defined for {self.rubric_name}"

        lines = []
        lines.append(f"Schema for {self.rubric_name} (scope: {self.scope}):")
        lines.append("")

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            field_schema = properties.get(field, {})
            field_type = field_schema.get("type", "unknown")
            description = field_schema.get("description", "")

            lines.append(f"- {field} ({field_type}): {description}")

        return "\n".join(lines)

    def enforce_schema_on_dict(self, data: Dict, original_data: Dict) -> Dict:
        """
        Enforce content_schema by filling in missing fields from original.

        Args:
            data: Possibly incomplete data
            original_data: Original data to use as fallback

        Returns:
            Complete data with all required fields
        """
        schema = self.content_schema
        if not schema:
            return data

        enforced = data.copy()
        properties = schema.get("properties", {})

        # Ensure all required fields exist
        for field in schema.get("required", []):
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

    def validate_content(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate content against schema_validation rules from YAML.

        Supports composable validation types:
        - xml: Check for required XML tags
        - json: Validate JSON structure (standalone or in sections)
        - regex: Pattern matching with {var} interpolation
        - yaml: YAML structure validation
        - code: Code syntax validation (Python, JS, etc.)

        Cross-reference validation:
        - Any type can extract values using 'extract:' config
        - Any type can reference extracted values using {var_name}
        - Types run in order specified by 'types:' list

        Args:
            content: Content string to validate

        Returns:
            (is_valid, list_of_error_messages)
        """
        all_errors = []

        # Reset extracted values for each validation run
        self.extracted_values = {}

        # Run each validation type in sequence
        for validation_type in self.validation_types:
            if validation_type == "xml":
                is_valid, errors = self._validate_xml(content)
            elif validation_type == "json":
                is_valid, errors = self._validate_json(content)
            elif validation_type == "regex":
                is_valid, errors = self._validate_regex(content)
            elif validation_type == "yaml":
                is_valid, errors = self._validate_yaml(content)
            elif validation_type == "code":
                is_valid, errors = self._validate_code(content)
            else:
                self.logger.warning(f"Unknown validation type: {validation_type}")
                continue

            # Collect errors from all validators
            if not is_valid:
                all_errors.extend(errors)

        return (len(all_errors) == 0, all_errors)

    def _validate_xml(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate XML structure.

        YAML config:
          xml:
            required_tags:
              - '<session_context>'
              - '<vault_structure>'
        """
        errors = []
        xml_config = self.content_validation.get("xml", {})

        # Check required tags
        required_tags = xml_config.get("required_tags", [])
        for tag in required_tags:
            if tag not in content:
                errors.append(f"Missing required XML tag: {tag}")

        return (len(errors) == 0, errors)

    def _validate_json(self, content: str) -> Tuple[bool, List[str]]:
        r"""
        Validate JSON structure and optionally extract values for cross-reference.

        YAML config options:

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
        """
        errors = []
        json_config = self.content_validation.get("json", {})

        # Check if content is standalone JSON
        if "required_fields" in json_config:
            try:
                data = json.loads(content)
                for field in json_config.get("required_fields", []):
                    if field not in data:
                        errors.append(f"Missing required JSON field: {field}")
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON: {str(e)}")

        # Check JSON in XML sections
        sections = json_config.get("sections", [])
        for section in sections:
            tag = section["tag"]
            extract_pattern = section.get("extract_pattern", r'\{.*\}')
            required_fields = section.get("required_fields", [])

            # Extract section content
            match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', content, re.DOTALL)
            if match:
                section_content = match.group(1).strip()

                # Extract JSON using pattern
                json_match = re.search(extract_pattern, section_content, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group(0))

                        # Extract values for cross-reference if configured
                        extract_config = section.get("extract", [])
                        if extract_config:
                            self._extract_and_store(json_data, extract_config)

                        # Check required fields
                        for field_spec in required_fields:
                            # Support both simple string and dict format
                            if isinstance(field_spec, str):
                                field_name = field_spec
                                field_constraints = {}
                            else:
                                field_name = field_spec.get("name", field_spec)
                                field_constraints = field_spec

                            # Check field exists
                            if field_name not in json_data:
                                errors.append(f"Missing field in <{tag}>: {field_name}")
                                continue

                            field_value = json_data[field_name]

                            # Type and constraint validation
                            field_type = field_constraints.get("type")

                            # Array validation
                            if field_type == "array":
                                if not isinstance(field_value, list):
                                    errors.append(f"Field <{tag}>.{field_name} must be array, got {type(field_value).__name__}")
                                else:
                                    min_items = field_constraints.get("min_items")
                                    if min_items and len(field_value) < min_items:
                                        errors.append(f"Field <{tag}>.{field_name} must have at least {min_items} items, has {len(field_value)}")

                            # Object validation
                            elif field_type == "object":
                                if not isinstance(field_value, dict):
                                    errors.append(f"Field <{tag}>.{field_name} must be object, got {type(field_value).__name__}")
                                else:
                                    min_properties = field_constraints.get("min_properties")
                                    if min_properties and len(field_value) < min_properties:
                                        errors.append(f"Field <{tag}>.{field_name} must have at least {min_properties} properties, has {len(field_value)}")

                            # String validation
                            elif field_type == "string":
                                if not isinstance(field_value, str):
                                    errors.append(f"Field <{tag}>.{field_name} must be string, got {type(field_value).__name__}")
                                else:
                                    min_length = field_constraints.get("min_length")
                                    if min_length and len(field_value) < min_length:
                                        errors.append(f"Field <{tag}>.{field_name} must have at least {min_length} characters, has {len(field_value)}")
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON in <{tag}>: {str(e)}")
                else:
                    errors.append(f"No JSON found in <{tag}>")

        return (len(errors) == 0, errors)

    def _validate_regex(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate content against regex patterns with {var} interpolation.

        YAML config:
          regex:
            patterns:
              - pattern: '\\btool_call:\\s*\\w+'
                description: 'Must contain tool_call: format'
              - pattern: '{workspace_root}'
                in_tag: 'vault_structure'
                description: 'Workspace rootFolder must exist in vault_structure'
        """
        errors = []
        regex_config = self.content_validation.get("regex", {})

        patterns = regex_config.get("patterns", [])
        for pattern_spec in patterns:
            pattern = pattern_spec["pattern"]
            description = pattern_spec.get("description", pattern)
            in_tag = pattern_spec.get("in_tag")

            # Interpolate {var} with extracted values
            interpolated_pattern = self._interpolate(pattern)

            # Skip if pattern still has unresolved {var} (value was skipped or not found)
            if '{' in interpolated_pattern and '}' in interpolated_pattern:
                continue

            # Determine search content (scoped to tag or full content)
            if in_tag:
                search_content = self._get_tag_content(content, in_tag)
                if search_content is None:
                    errors.append(f"Tag <{in_tag}> not found for pattern: {description}")
                    continue
            else:
                search_content = content

            # Use re.escape for literal string matching when pattern is a simple extracted value
            if interpolated_pattern == pattern_spec["pattern"]:
                # Original pattern (no interpolation) - use as regex
                if not re.search(interpolated_pattern, search_content, re.DOTALL):
                    errors.append(f"Pattern not found: {description}")
            else:
                # Interpolated value - do literal string search
                if interpolated_pattern not in search_content:
                    errors.append(f"{description}: '{interpolated_pattern}' not found in <{in_tag}>")

        return (len(errors) == 0, errors)

    def _validate_yaml(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate YAML structure.

        YAML config options:

        1. Standalone YAML:
           yaml:
             required_keys: [name, version, dependencies]

        2. YAML in XML sections:
           yaml:
             sections:
               - tag: 'available_agents'
                 min_items: 2
                 item_fields: [name, description]
        """
        errors = []
        yaml_config = self.content_validation.get("yaml", {})

        # Standalone YAML validation
        if "required_keys" in yaml_config:
            try:
                import yaml
                data = yaml.safe_load(content)

                # Check required keys
                required_keys = yaml_config.get("required_keys", [])
                for key in required_keys:
                    if key not in data:
                        errors.append(f"Missing required YAML key: {key}")

            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML: {str(e)}")

        # YAML in XML sections
        sections = yaml_config.get("sections", [])
        for section in sections:
            tag = section["tag"]

            # Extract section content
            match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', content, re.DOTALL)
            if match:
                section_content = match.group(1).strip()

                try:
                    import yaml
                    data = yaml.safe_load(section_content)

                    # Check if it's a list
                    if isinstance(data, list):
                        min_items = section.get("min_items")
                        if min_items and len(data) < min_items:
                            errors.append(f"<{tag}> must have at least {min_items} items, has {len(data)}")

                        # Check each item has required fields
                        item_fields = section.get("item_fields", [])
                        for idx, item in enumerate(data):
                            if isinstance(item, dict):
                                for field in item_fields:
                                    if field not in item:
                                        errors.append(f"<{tag}> item {idx + 1} missing field: {field}")
                            else:
                                errors.append(f"<{tag}> item {idx + 1} must be an object")
                    else:
                        errors.append(f"<{tag}> content must be a YAML list")

                except yaml.YAMLError as e:
                    errors.append(f"Invalid YAML in <{tag}>: {str(e)}")

        return (len(errors) == 0, errors)

    def _validate_code(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate code syntax.

        YAML config:
          code:
            language: python  # or javascript, etc.
            check_syntax: true
        """
        errors = []
        code_config = self.content_validation.get("code", {})

        language = code_config.get("language", "python")
        check_syntax = code_config.get("check_syntax", True)

        if check_syntax:
            if language == "python":
                try:
                    compile(content, '<string>', 'exec')
                except SyntaxError as e:
                    errors.append(f"Python syntax error: {str(e)}")

            elif language == "javascript":
                # Could use a JS parser if needed
                # For now, basic check
                if "syntax error" in content.lower():
                    errors.append("JavaScript may have syntax errors")

            # Add more languages as needed

        return (len(errors) == 0, errors)

    # ==================== Cross-Reference Helpers ====================

    def _interpolate(self, text: str) -> str:
        """Replace {var_name} with extracted values."""
        for key, val in self.extracted_values.items():
            text = text.replace(f"{{{key}}}", str(val))
        return text

    def _get_value_at_path(self, data: Dict, path: str) -> Optional[str]:
        """
        Navigate to value using dot notation path.

        Args:
            data: Dict to navigate
            path: Dot notation path (e.g., 'context.rootFolder')

        Returns:
            Value at path, or None if not found
        """
        value = data
        for key in path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value if isinstance(value, str) else str(value) if value is not None else None

    def _extract_and_store(self, data: Dict, extract_config: List[Dict]) -> None:
        """
        Extract values from data and store in shared context.

        Args:
            data: Parsed data (JSON or YAML)
            extract_config: List of extraction specs from YAML config
        """
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

            self.extracted_values[var_name] = value

    def _get_tag_content(self, content: str, tag: str) -> Optional[str]:
        """Extract content from an XML tag."""
        match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', content, re.DOTALL)
        return match.group(1) if match else None
