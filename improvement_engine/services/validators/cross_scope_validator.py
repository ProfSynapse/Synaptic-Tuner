"""Cross-scope validator for validating content from one scope against another."""

import json
import re
from typing import Dict, List, Tuple, Set, Any, Optional


class CrossScopeValidator:
    """Validates content from one scope against another."""

    def __init__(self, logger=None):
        self.logger = logger

    def validate(
        self,
        source_content: Dict,
        target_content: str,
        config: Dict,
        error_template: str = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate content from source scope against target scope.

        Config format (from validations list):
            cross_scope:
              from: thinking
              to: system_prompt
              extract:
                fields: [goal, memory, plan]
                pattern: '...'
              skip_if:
                - pattern: '(?:create|new).*{value}'
              validate_in: [vault_structure, selected_workspace]
            error: "..."

        Args:
            source_content: Dict from source scope (e.g., thinking block)
            target_content: String from target scope (e.g., system_prompt)
            config: Cross-scope validation config (cross_scope dict)
            error_template: Error message template (from validation.error)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        if not config:
            return (True, [])

        extract_config = config.get("extract", {})

        # STEP 1: Extract values from source content
        extracted = self._extract_from_source(source_content, extract_config)

        if not extracted:
            return (True, [])

        # STEP 2: Filter out skipped values
        to_validate = self._filter_skipped(
            extracted,
            config.get("skip_if", []),
            source_content
        )

        if not to_validate:
            return (True, [])

        # STEP 3: Extract valid values from target locations
        validate_in = config.get("validate_in", [])
        valid_values, searched_locations = self._extract_from_target(
            target_content,
            validate_in,
            extract_config.get("pattern", "")
        )

        # STEP 4: Compare and report errors
        err_template = error_template or "'{value}' not found"
        unique_values = list(set(to_validate))

        for value in unique_values:
            if value not in valid_values:
                error = err_template.format(
                    value=value,
                    locations=", ".join(searched_locations)
                )
                errors.append(error)

        return (len(errors) == 0, errors)

    def _extract_from_source(
        self,
        content: Dict,
        config: Dict
    ) -> List[str]:
        """Extract values from source fields using pattern."""
        extracted = []
        fields = config.get("fields", [])
        source_pattern = config.get("pattern", "")

        if not source_pattern:
            return extracted

        for field in fields:
            field_value = self._get_field_value(content, field)
            if field_value:
                field_str = self._stringify_value(field_value)
                try:
                    matches = re.findall(source_pattern, field_str)
                    extracted.extend(matches)
                except re.error as e:
                    self._log_warning(f"Invalid source pattern: {e}")

        return extracted

    def _filter_skipped(
        self,
        extracted: List[str],
        skip_configs: List[Dict],
        source_content: Dict
    ) -> List[str]:
        """Filter values matching skip patterns."""
        values_to_validate = []

        # Stringify entire source content for context matching
        full_content = self._stringify_value(source_content)

        for value in extracted:
            should_skip = False

            for skip_config in skip_configs:
                skip_pattern = skip_config.get("pattern", "")
                if not skip_pattern:
                    continue

                # Interpolate {value} in skip pattern
                interpolated_pattern = skip_pattern.replace("{value}", re.escape(value))

                try:
                    if re.search(interpolated_pattern, full_content, re.IGNORECASE):
                        should_skip = True
                        break
                except re.error as e:
                    self._log_warning(f"Invalid skip pattern: {e}")

            if not should_skip:
                values_to_validate.append(value)

        return values_to_validate

    def _extract_from_target(
        self,
        content: str,
        tag_names: List[str],
        source_pattern: str
    ) -> Tuple[Set[str], List[str]]:
        """Extract valid values from target tags using same pattern."""
        valid_values: Set[str] = set()
        searched_locations: List[str] = []

        for tag in tag_names:
            tag_content = self._get_tag_content(content, tag)
            if tag_content is None:
                continue

            searched_locations.append(tag)

            # Use the same pattern from source to find valid values
            try:
                matches = re.findall(source_pattern, tag_content)
                valid_values.update(matches)
            except re.error as e:
                self._log_warning(f"Invalid pattern for {tag}: {e}")

        return valid_values, searched_locations

    def _get_field_value(self, data: Dict, field: str) -> Optional[Any]:
        """Get value from dict, supporting dot notation for nested paths."""
        if not isinstance(data, dict):
            return None

        if '.' in field:
            parts = field.split('.', 1)
            if parts[0] in data:
                return self._get_field_value(data[parts[0]], parts[1])
            return None

        return data.get(field)

    def _stringify_value(self, value: Any) -> str:
        """Convert a value to string for regex matching."""
        if isinstance(value, str):
            return value
        elif isinstance(value, list):
            return " ".join(self._stringify_value(item) for item in value)
        elif isinstance(value, dict):
            return json.dumps(value)
        else:
            return str(value)

    def _get_tag_content(self, content: str, tag: str) -> Optional[str]:
        """Extract content from an XML tag."""
        match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', content, re.DOTALL)
        return match.group(1) if match else None

    def _log_warning(self, message: str) -> None:
        """Log a warning if logger is available."""
        if self.logger:
            self.logger.warning(f"cross_scope: {message}")
