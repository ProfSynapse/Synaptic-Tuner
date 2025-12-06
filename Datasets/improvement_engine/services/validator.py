"""Validation service for examples and thinking blocks."""

import json
from typing import Dict, Any, List, Tuple
from ..core.exceptions import ValidationError
from ..core.models import ThinkingBlock, Example
from ..utils.yaml_loader import load_config


class Validator:
    """Validates examples and thinking blocks against schema rules."""

    def __init__(self):
        """Initialize validator with validation rules."""
        self.rules = load_config("validation_rules")

    def validate_thinking_block(self, thinking_block: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a thinking block against schema rules.

        Args:
            thinking_block: Thinking block to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        schema = self.rules["thinking_block_schema"]

        # Check required fields
        for field in schema["required_fields"]:
            if field not in thinking_block:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Validate field types
        field_types = schema["field_types"]
        for field, expected_type in field_types.items():
            if field not in thinking_block:
                continue

            value = thinking_block[field]

            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Field '{field}' must be string, got {type(value).__name__}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"Field '{field}' must be array, got {type(value).__name__}")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"Field '{field}' must be object, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field '{field}' must be number, got {type(value).__name__}")

        if errors:
            return False, errors

        # Validate constraints
        constraints = schema["constraints"]

        # Goal constraints
        goal = thinking_block.get("goal", "")
        if len(goal) < constraints["goal"]["min_length"]:
            errors.append(f"Goal too short (min {constraints['goal']['min_length']} chars)")
        if len(goal) > constraints["goal"]["max_length"]:
            errors.append(f"Goal too long (max {constraints['goal']['max_length']} chars)")

        # Memory constraints
        memory = thinking_block.get("memory", "")
        if len(memory) < constraints["memory"]["min_length"]:
            errors.append(f"Memory too short (min {constraints['memory']['min_length']} chars)")

        # Requirements constraints
        requirements = thinking_block.get("requirements", [])
        if len(requirements) < constraints["requirements"]["min_items"]:
            errors.append(f"Too few requirements (min {constraints['requirements']['min_items']})")

        # Assessment constraints
        assessment = thinking_block.get("assessment", {})
        for key in constraints["assessment"]["required_keys"]:
            if key not in assessment:
                errors.append(f"Assessment missing required key: {key}")
            elif not isinstance(assessment[key], bool):
                errors.append(f"Assessment '{key}' must be boolean")

        # Confidence constraints
        confidence = thinking_block.get("confidence", 0)
        if not (constraints["confidence"]["min"] <= confidence <= constraints["confidence"]["max"]):
            errors.append(
                f"Confidence out of range "
                f"({constraints['confidence']['min']}-{constraints['confidence']['max']})"
            )

        # Plan constraints
        plan = thinking_block.get("plan", [])
        if len(plan) < constraints["plan"]["min_items"]:
            errors.append(f"Too few plan items (min {constraints['plan']['min_items']})")

        return len(errors) == 0, errors

    def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a complete example.

        Args:
            example: Example to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        schema = self.rules["example_structure"]

        # Check required fields
        for field in schema["required_fields"]:
            if field not in example:
                errors.append(f"Missing required field: {field}")
                return False, errors

        # Validate conversations
        conversations = example.get("conversations", [])
        if len(conversations) < schema["conversations"]["min_items"]:
            errors.append(f"Too few conversations (min {schema['conversations']['min_items']})")

        for i, conv in enumerate(conversations):
            for req_field in schema["conversations"]["item_structure"]["required"]:
                if req_field not in conv:
                    errors.append(f"Conversation {i} missing field: {req_field}")

            role = conv.get("role")
            if role and role not in schema["conversations"]["item_structure"]["roles"]:
                errors.append(f"Invalid role in conversation {i}: {role}")

        return len(errors) == 0, errors

    def validate_json(self, json_string: str) -> Tuple[bool, List[str]]:
        """
        Validate that a string is valid JSON.

        Args:
            json_string: JSON string to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            json.loads(json_string)
            return True, []
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]

    def is_valid_thinking_block(self, thinking_block: Dict[str, Any]) -> bool:
        """
        Check if thinking block is valid (simple boolean check).

        Args:
            thinking_block: Thinking block to check

        Returns:
            True if valid
        """
        is_valid, _ = self.validate_thinking_block(thinking_block)
        return is_valid

    def is_valid_example(self, example: Dict[str, Any]) -> bool:
        """
        Check if example is valid (simple boolean check).

        Args:
            example: Example to check

        Returns:
            True if valid
        """
        is_valid, _ = self.validate_example(example)
        return is_valid
