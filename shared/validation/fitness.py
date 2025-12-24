"""Fitness evaluation for evolutionary fine-tuning.

This module provides config-driven fitness evaluation by wrapping
the unified parsing and validation layers from Phase 1.

The FitnessEvaluator converts model outputs to fitness scores (0.0-1.0)
that can be used during training to select the best gradient updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from shared.utilities import load_yaml
from .parsing import parse_response, ParsedResponse
from .validators import StructureValidator


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""

    score: float
    """Fitness score from 0.0 (worst) to 1.0 (best)."""

    is_valid: bool
    """Whether the output passed all validations."""

    errors: List[str] = field(default_factory=list)
    """List of validation errors."""

    parsed_response: Optional[ParsedResponse] = None
    """The parsed response (if parsing succeeded)."""

    scoring_method: str = "error_count"
    """The scoring method that was used."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "scoring_method": self.scoring_method,
            "has_tool_calls": self.parsed_response.has_tool_calls if self.parsed_response else False,
        }


class FitnessEvaluator:
    """
    Config-driven fitness evaluation for evolutionary training.

    Wraps the unified parsing + validation layers to provide
    a simple score (0.0-1.0) for model outputs.

    Usage:
        evaluator = FitnessEvaluator(config_path="configs/fitness/tool_calling.yaml")
        score = evaluator.evaluate(model_output)

        # Or with inline config
        evaluator = FitnessEvaluator(config={
            "validations": [...],
            "scoring": {"method": "error_count", "params": {"max_errors": 5}}
        })
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize fitness evaluator.

        Args:
            config_path: Path to YAML config file (same format as rubrics)
            config: Inline config dict (alternative to config_path)
        """
        if config_path:
            self.config = load_yaml(config_path)
        elif config:
            self.config = config
        else:
            # Default minimal config
            self.config = {
                "validations": [],
                "scoring": {"method": "binary"},
            }

        self.validations = self.config.get("validations", [])
        self.scoring = self.config.get("scoring", {})

        # Use shared validators
        self.validator = StructureValidator()

    def evaluate(
        self,
        model_output: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> FitnessResult:
        """
        Evaluate model output and return fitness score.

        Args:
            model_output: Raw model output (string or dict)
            context: Optional context for cross-scope validation
                     (e.g., system prompt, expected tool, etc.)

        Returns:
            FitnessResult with score and details
        """
        # Layer 1: Parse (format-agnostic)
        parsed = parse_response(model_output)

        # Check if we got any tool calls
        if not parsed.has_tool_calls:
            # No tool calls - could be text-only response
            # Score based on config
            no_tool_score = self.scoring.get("no_tool_call_score", 0.0)
            return FitnessResult(
                score=no_tool_score,
                is_valid=False,
                errors=["No tool calls found in response"],
                parsed_response=parsed,
                scoring_method="no_tool_call",
            )

        # Layer 2: Validate (config-driven)
        # Build data structure for validation
        data = self._build_validation_data(parsed)

        # Run validation
        is_valid, errors = self.validator.validate(
            data=data,
            validations=self.validations,
            raw_content=parsed.raw_response if isinstance(parsed.raw_response, str) else None,
        )

        # Layer 3: Score
        score = self._compute_score(is_valid, errors)

        return FitnessResult(
            score=score,
            is_valid=is_valid,
            errors=errors,
            parsed_response=parsed,
            scoring_method=self.scoring.get("method", "error_count"),
        )

    def evaluate_batch(
        self,
        outputs: List[Union[str, Dict[str, Any]]],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[FitnessResult]:
        """
        Evaluate a batch of model outputs.

        Args:
            outputs: List of model outputs
            contexts: Optional list of contexts (same length as outputs)

        Returns:
            List of FitnessResults
        """
        contexts = contexts or [None] * len(outputs)
        return [
            self.evaluate(output, context)
            for output, context in zip(outputs, contexts)
        ]

    def _build_validation_data(self, parsed: ParsedResponse) -> Dict[str, Any]:
        """Build the data structure for validation."""
        # Convert ParsedToolCall objects to OpenAI-style dicts
        tool_calls = []
        for tc in parsed.tool_calls:
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
            })

        return {
            "tool_calls": tool_calls,
            "text_content": parsed.text_content,
            "thinking": parsed.thinking,
            "format_detected": str(parsed.format_detected),
        }

    def _compute_score(self, is_valid: bool, errors: List[str]) -> float:
        """
        Compute fitness score based on validation results.

        Scoring methods:
        - binary: 1.0 if valid, 0.0 if not
        - error_count: Linear decrease based on error count
        - weighted: Custom weights per validation type (TODO)
        """
        method = self.scoring.get("method", "error_count")
        params = self.scoring.get("params", {})

        if method == "binary":
            return 1.0 if is_valid else 0.0

        elif method == "error_count":
            if is_valid:
                return 1.0

            max_errors = params.get("max_errors_before_zero", 5)
            # Linear decrease: 1 error = 0.8, 2 errors = 0.6, etc.
            score = max(0.0, 1.0 - len(errors) / max_errors)
            return score

        elif method == "error_penalty":
            # Each error deducts a fixed amount
            penalty = params.get("penalty_per_error", 0.1)
            score = max(0.0, 1.0 - len(errors) * penalty)
            return score

        else:
            # Unknown method, default to binary
            return 1.0 if is_valid else 0.0


def create_fitness_evaluator(
    config_path: Optional[Union[str, Path]] = None,
    validations: Optional[List[Dict]] = None,
    scoring_method: str = "error_count",
    max_errors: int = 5,
) -> FitnessEvaluator:
    """
    Factory function to create a FitnessEvaluator.

    Args:
        config_path: Path to config YAML (takes precedence)
        validations: Inline validation rules
        scoring_method: How to compute score ("binary", "error_count", "error_penalty")
        max_errors: For error_count method, errors that result in 0 score

    Returns:
        Configured FitnessEvaluator
    """
    if config_path:
        return FitnessEvaluator(config_path=config_path)

    config = {
        "validations": validations or [],
        "scoring": {
            "method": scoring_method,
            "params": {
                "max_errors_before_zero": max_errors,
            },
        },
    }

    return FitnessEvaluator(config=config)
