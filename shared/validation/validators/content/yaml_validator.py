"""YAML structure validator."""

import re
from typing import Dict, List, Tuple, Any, Optional

from ..base import BaseContentValidator


class YamlContentValidator(BaseContentValidator):
    """Validates YAML structure in content."""

    def validate(
        self,
        content: str,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate YAML structure.

        Config options:

        1. Standalone YAML:
           yaml:
             required_keys: [name, version, dependencies]

        2. YAML in XML sections:
           yaml:
             sections:
               - tag: 'available_prompts'
                 min_items: 2
                 item_fields: [name, description]

        Args:
            content: Content string to validate
            config: YAML validation config
            context: Shared context (unused for YAML)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Standalone YAML validation
        if "required_keys" in config:
            standalone_errors = self._validate_standalone(content, config)
            errors.extend(standalone_errors)

        # YAML in sections
        for section in config.get("sections", []):
            section_errors = self._validate_section(content, section)
            errors.extend(section_errors)

        return (len(errors) == 0, errors)

    def _validate_standalone(self, content: str, config: Dict) -> List[str]:
        """Validate standalone YAML content."""
        errors = []
        try:
            import yaml
            data = yaml.safe_load(content)

            # Check required keys
            required_keys = config.get("required_keys", [])
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required YAML key: {key}")

        except Exception as e:  # yaml.YAMLError
            errors.append(f"Invalid YAML: {str(e)}")

        return errors

    def _validate_section(self, content: str, section_config: Dict) -> List[str]:
        """Validate YAML in an XML section."""
        errors = []
        tag = section_config["tag"]

        # Extract section content
        match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', content, re.DOTALL)
        if not match:
            return errors  # Tag not found - skip

        section_content = match.group(1).strip()

        try:
            import yaml
            data = yaml.safe_load(section_content)

            # Check if it's a list
            if isinstance(data, list):
                min_items = section_config.get("min_items")
                if min_items and len(data) < min_items:
                    errors.append(
                        f"<{tag}> must have at least {min_items} items, "
                        f"has {len(data)}"
                    )

                # Check each item has required fields
                item_fields = section_config.get("item_fields", [])
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        for field in item_fields:
                            if field not in item:
                                errors.append(
                                    f"<{tag}> item {idx + 1} missing field: {field}"
                                )
                    else:
                        errors.append(f"<{tag}> item {idx + 1} must be an object")
            else:
                errors.append(f"<{tag}> content must be a YAML list")

        except Exception as e:  # yaml.YAMLError
            errors.append(f"Invalid YAML in <{tag}>: {str(e)}")

        return errors
