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

        # Load content validation rules (composable types)
        self.content_validation = rubric_data.get("schema_validation", {})
        self.validation_types = self.content_validation.get("types", [])

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

    def validate_content(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate content against schema_validation rules from YAML.

        Supports composable validation types:
        - xml: Check for required XML tags
        - json: Validate JSON structure (standalone or in sections)
        - regex: Pattern matching
        - yaml: YAML structure validation
        - code: Code syntax validation (Python, JS, etc.)

        YAML can specify multiple types: types: [xml, json]

        Args:
            content: Content string to validate

        Returns:
            (is_valid, list_of_error_messages)
        """
        all_errors = []

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
        Validate JSON structure.

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

                        # Check required fields
                        for field in required_fields:
                            if field not in json_data:
                                errors.append(f"Missing field in <{tag}>: {field}")
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON in <{tag}>: {str(e)}")
                else:
                    errors.append(f"No JSON found in <{tag}>")

        return (len(errors) == 0, errors)

    def _validate_regex(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate content against regex patterns.

        YAML config:
          regex:
            patterns:
              - pattern: '\\btool_call:\\s*\\w+'
                description: 'Must contain tool_call: format'
              - pattern: 'arguments:\\s*\\{'
                description: 'Must have arguments: JSON'
        """
        errors = []
        regex_config = self.content_validation.get("regex", {})

        patterns = regex_config.get("patterns", [])
        for pattern_spec in patterns:
            pattern = pattern_spec["pattern"]
            description = pattern_spec.get("description", pattern)

            if not re.search(pattern, content, re.DOTALL):
                errors.append(f"Pattern not found: {description}")

        return (len(errors) == 0, errors)

    def _validate_yaml(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate YAML structure.

        YAML config:
          yaml:
            required_keys: [name, version, dependencies]
        """
        errors = []
        yaml_config = self.content_validation.get("yaml", {})

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
