"""Schema validator facade - thin coordinator for all validators."""

from typing import Dict, List, Tuple, Optional
from pathlib import Path

from shared.validation.validators import StructureValidator, CrossScopeValidator
from ...utils.yaml_loader import load_yaml


class SchemaValidator:
    """
    Facade for schema validation.

    Coordinates:
    - StructureValidator (field, match, xml, json validations)
    - CrossScopeValidator (cross-scope validation)
    """

    def __init__(self, rubric_name: str, logger=None):
        """
        Initialize schema validator.

        Args:
            rubric_name: Name of rubric (YAML file stem in rubrics/)
            logger: Logger instance
        """
        self.rubric_name = rubric_name
        self.logger = logger

        # Load rubric config
        self._load_rubric(rubric_name)

        # Initialize validators
        self.structure_validator = StructureValidator(logger)
        self.cross_scope_validator = CrossScopeValidator(logger)

    def _load_rubric(self, rubric_name: str) -> None:
        """Load rubric YAML and extract schemas."""
        rubric_file = Path(__file__).parent.parent.parent / "rubrics" / f"{rubric_name}.yaml"
        if not rubric_file.exists():
            raise FileNotFoundError(f"Rubric not found: {rubric_file}")

        rubric_data = load_yaml(rubric_file)

        self.scope = rubric_data.get("scope", "response")
        self.output_schema = rubric_data.get("output_schema", {})
        self.validations = rubric_data.get("validations", [])

    # === Structure Validation ===

    def validate_thinking_content(self, data: Dict, raw_content: str = None) -> Tuple[bool, List[str]]:
        """Validate thinking block against validations list."""
        if self.validations:
            return self.structure_validator.validate(data, self.validations, raw_content)
        return (True, [])

    def validate_system_prompt(self, raw_content: str) -> Tuple[bool, List[str]]:
        """Validate system prompt against pattern validations."""
        if self.validations:
            return self.structure_validator.validate({}, self.validations, raw_content)
        return (True, [])

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """Validate judge response against output_schema."""
        if not self.output_schema:
            return (True, [])

        errors = []
        required = self.output_schema.get("required", [])
        properties = self.output_schema.get("properties", {})

        for field in required:
            if field not in data:
                errors.append(f"Missing required field '{field}' in judge output")
                continue

            field_schema = properties.get(field, {})
            expected_type = field_schema.get("type")
            value = data[field]

            type_map = {"string": str, "number": (int, float), "array": list, "object": dict, "boolean": bool}
            if expected_type and not isinstance(value, type_map.get(expected_type, object)):
                errors.append(f"Field '{field}' must be {expected_type}")
                continue

            # Check min/max for numbers
            if expected_type == "number" and isinstance(value, (int, float)):
                min_val = field_schema.get("min")
                max_val = field_schema.get("max")
                if min_val is not None and value < min_val:
                    errors.append(f"Field '{field}' must be >= {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"Field '{field}' must be <= {max_val}")

        return (len(errors) == 0, errors)

    # === Cross-Scope Validation ===

    def has_cross_scope_validation(self) -> bool:
        """Check if this rubric has cross_scope validation configured."""
        for v in self.validations:
            if "cross_scope" in v:
                return True
        return False

    def get_cross_scope_target(self) -> Optional[str]:
        """Get the target scope for cross-scope validation."""
        for v in self.validations:
            if "cross_scope" in v:
                return v["cross_scope"].get("to")
        return None

    def get_cross_scope_validations(self) -> List[Dict]:
        """Get all cross_scope validations from the validations list."""
        return [v for v in self.validations if "cross_scope" in v]

    def validate_cross_scope(
        self,
        source_content: Dict,
        target_content: str
    ) -> Tuple[bool, List[str]]:
        """Validate content from source scope against target scope."""
        all_errors = []

        for validation in self.get_cross_scope_validations():
            config = validation.get("cross_scope", {})
            error_template = validation.get("error")
            is_valid, errors = self.cross_scope_validator.validate(
                source_content, target_content, config, error_template
            )
            all_errors.extend(errors)

        return (len(all_errors) == 0, all_errors)

    # === Utilities ===

    def get_schema_description(self) -> str:
        """Get human-readable schema description."""
        if not self.validations:
            return f"No validations defined for {self.rubric_name}"

        lines = [f"Validations for {self.rubric_name} (scope: {self.scope}):", ""]

        for v in self.validations:
            if "field" in v:
                field = v["field"]
                vtype = v.get("type", "any")
                lines.append(f"- field: {field} ({vtype})")
            elif "match" in v:
                match = v["match"]
                mtype = v.get("type", "contains")
                lines.append(f"- match: {match} ({mtype})")
            elif "cross_scope" in v:
                cs = v["cross_scope"]
                lines.append(f"- cross_scope: {cs.get('from')} -> {cs.get('to')}")

        return "\n".join(lines)
