"""Rubric-based validation for model responses.

Integrates SynthChat rubrics/schemas into the Evaluator for consistent
validation between training data generation and model evaluation.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Import StructureValidator directly (avoid SynthChat's complex init)
synthchat_validators = Path(__file__).parent.parent / 'SynthChat' / 'services' / 'validators'
if str(synthchat_validators.parent.parent) not in sys.path:
    sys.path.insert(0, str(synthchat_validators.parent.parent))

# Direct import to avoid circular dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "structure_validator",
    synthchat_validators / "structure_validator.py"
)
structure_validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(structure_validator_module)
StructureValidator = structure_validator_module.StructureValidator


@dataclass
class RubricValidationResult:
    """Result of validating against a rubric."""
    rubric_name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rubric_name": self.rubric_name,
            "passed": self.passed,
            "errors": self.errors,
            "score": self.score
        }


@dataclass
class FullValidationResult:
    """Result of validating against multiple rubrics."""
    passed: bool
    results: List[RubricValidationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results]
        }


class RubricValidator:
    """
    Validates model responses against SynthChat rubrics.

    Uses the same rubric YAML files as the improvement engine for
    consistent validation across training and evaluation.
    """

    def __init__(self, rubrics_dir: Optional[Path] = None):
        """
        Initialize the validator.

        Args:
            rubrics_dir: Path to rubrics directory. Defaults to SynthChat/rubrics/
        """
        if rubrics_dir is None:
            rubrics_dir = Path(__file__).parent.parent / 'SynthChat' / 'rubrics'

        self.rubrics_dir = Path(rubrics_dir)
        self.structure_validator = StructureValidator()
        self._rubric_cache: Dict[str, Dict] = {}

    def list_rubrics(self) -> List[str]:
        """List available rubric keys."""
        rubrics = []
        for yaml_file in self.rubrics_dir.glob("*.yaml"):
            rubrics.append(yaml_file.stem)
        return sorted(rubrics)

    def load_rubric(self, rubric_key: str) -> Dict:
        """Load a rubric by key."""
        if rubric_key in self._rubric_cache:
            return self._rubric_cache[rubric_key]

        yaml_path = self.rubrics_dir / f"{rubric_key}.yaml"
        if not yaml_path.exists():
            raise ValueError(f"Rubric not found: {rubric_key}")

        with open(yaml_path, 'r') as f:
            rubric = yaml.safe_load(f)

        self._rubric_cache[rubric_key] = rubric
        return rubric

    def validate_response(
        self,
        response: Dict[str, Any],
        rubric_key: str,
        system_prompt: Optional[str] = None
    ) -> RubricValidationResult:
        """
        Validate a response against a single rubric.

        Args:
            response: Assistant response dict with 'content' and/or 'tool_calls'
            rubric_key: Rubric to validate against (e.g., 'vaultManager_tools')
            system_prompt: Optional system prompt for cross-scope validation

        Returns:
            RubricValidationResult
        """
        rubric = self.load_rubric(rubric_key)
        validations = rubric.get("validations", [])

        if not validations:
            return RubricValidationResult(
                rubric_name=rubric.get("name", rubric_key),
                passed=True,
                errors=[],
                score=1.0
            )

        # Determine scope
        scope = rubric.get("scope", "response")

        # Get appropriate content for validation
        if scope == "response":
            # Validate tool_calls
            is_valid, errors = self.structure_validator.validate(
                data=response,
                validations=validations,
                raw_content=response.get("content", "")
            )
        elif scope == "system_prompt" and system_prompt:
            # Validate system prompt content
            is_valid, errors = self.structure_validator.validate(
                data={},
                validations=validations,
                raw_content=system_prompt
            )
        else:
            is_valid = True
            errors = []

        # Calculate score (binary for now, could weight by error severity)
        score = 1.0 if is_valid else 0.0
        threshold = rubric.get("pass_threshold", 0.8)

        return RubricValidationResult(
            rubric_name=rubric.get("name", rubric_key),
            passed=score >= threshold,
            errors=errors,
            score=score
        )

    def validate_tool_calls(
        self,
        tool_calls: List[Dict],
        agent: str = "vaultManager"
    ) -> RubricValidationResult:
        """
        Validate tool calls against an agent's rubric.

        Convenience method for validating just tool calls.

        Args:
            tool_calls: List of tool call dicts
            agent: Agent name to determine rubric (default: vaultManager)

        Returns:
            RubricValidationResult
        """
        rubric_key = f"{agent}_tools"
        response = {"tool_calls": tool_calls}
        return self.validate_response(response, rubric_key)

    def validate_full(
        self,
        response: Dict[str, Any],
        rubric_keys: List[str],
        system_prompt: Optional[str] = None
    ) -> FullValidationResult:
        """
        Validate a response against multiple rubrics.

        Args:
            response: Assistant response dict
            rubric_keys: List of rubrics to validate against
            system_prompt: Optional system prompt for system_prompt scope rubrics

        Returns:
            FullValidationResult with all individual results
        """
        results = []
        all_passed = True

        for rubric_key in rubric_keys:
            try:
                result = self.validate_response(response, rubric_key, system_prompt)
                results.append(result)
                if not result.passed:
                    all_passed = False
            except ValueError as e:
                results.append(RubricValidationResult(
                    rubric_name=rubric_key,
                    passed=False,
                    errors=[str(e)],
                    score=0.0
                ))
                all_passed = False

        return FullValidationResult(passed=all_passed, results=results)

    def extract_and_validate(
        self,
        raw_response: str,
        agent: str = "vaultManager"
    ) -> Tuple[RubricValidationResult, Optional[Dict]]:
        """
        Parse raw response text and validate.

        Handles both:
        - OpenAI format: {"tool_calls": [...]}
        - Plain JSON with tool calls embedded

        Args:
            raw_response: Raw response text from model
            agent: Agent for rubric selection

        Returns:
            (validation_result, parsed_tool_calls or None)
        """
        # Try to parse as JSON
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                if "tool_calls" in parsed:
                    # OpenAI format
                    result = self.validate_tool_calls(parsed["tool_calls"], agent)
                    return result, parsed
                elif "function" in parsed:
                    # Legacy single-function format
                    result = self.validate_tool_calls([parsed], agent)
                    return result, {"tool_calls": [parsed]}
        except json.JSONDecodeError:
            pass

        # Not valid JSON - return error
        return RubricValidationResult(
            rubric_name=f"{agent}_tools",
            passed=False,
            errors=["Response is not valid JSON"],
            score=0.0
        ), None


# Convenience function for quick validation
def validate_response(
    response: Dict[str, Any],
    rubric_keys: List[str],
    system_prompt: Optional[str] = None
) -> FullValidationResult:
    """
    Quick validation function.

    Args:
        response: Assistant response dict
        rubric_keys: List of rubric keys to validate against
        system_prompt: Optional system prompt

    Returns:
        FullValidationResult
    """
    validator = RubricValidator()
    return validator.validate_full(response, rubric_keys, system_prompt)
