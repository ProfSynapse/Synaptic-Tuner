"""Evaluation orchestration logic.

This module provides the core evaluation loop that runs prompt cases
against a backend client and collects validation results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .behavior_validator import BehaviorIssue, BehaviorValidationResult, validate_behavior
from .prompt_sets import PromptCase
from .protocols import BackendClient
from .schema_validator import ValidationResult, ValidatorIssue, validate_assistant_response


@dataclass
class EvaluationRecord:
    """Result of evaluating a single prompt case.

    Attributes:
        case: The prompt case that was evaluated
        response_text: Model's response content (string or dict)
        validator: Schema validation result
        latency_s: Response time in seconds
        raw_response: Complete API response
        error: Error message if request/validation failed
        behavior: Behavior validation result (if expectations defined)
    """

    case: PromptCase
    response_text: Optional[Any]
    validator: Optional[ValidationResult]
    latency_s: Optional[float]
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    behavior: Optional[BehaviorValidationResult] = None

    @property
    def passed(self) -> bool:
        """Check if evaluation passed all validations."""
        if self.error is not None:
            return False
        if self.validator is None or not self.validator.passed:
            return False
        # If behavior expectations exist, behavior validation must also pass
        if self.behavior is not None and not self.behavior.passed:
            return False
        return True

    @property
    def schema_passed(self) -> bool:
        """Check if schema validation passed (ignoring behavior)."""
        return self.error is None and self.validator is not None and self.validator.passed

    @property
    def behavior_passed(self) -> bool:
        """Check if behavior validation passed (or not applicable)."""
        return self.behavior is None or self.behavior.passed


def evaluate_cases(
    cases: Sequence[PromptCase],
    client: BackendClient,
    dry_run: bool = False,
    on_record: Callable[[EvaluationRecord], None] | None = None,
    validate_context: bool = False,
) -> List[EvaluationRecord]:
    """Run evaluation for the provided prompts.

    Args:
        cases: Prompt cases to evaluate
        client: Backend client implementing BackendClient protocol
        dry_run: Skip backend calls (for testing)
        on_record: Optional callback for each completed record
        validate_context: If True, validate that model uses IDs from
                         expected_context in prompt metadata

    Returns:
        List of evaluation records
    """
    records: List[EvaluationRecord] = []

    for case in cases:
        record = _evaluate_single_case(case, client, dry_run, validate_context)
        records.append(record)
        if on_record:
            on_record(record)

    return records


def _evaluate_single_case(
    case: PromptCase,
    client: BackendClient,
    dry_run: bool,
    validate_context: bool = False,
) -> EvaluationRecord:
    """Evaluate a single prompt case.

    Args:
        case: The prompt case to evaluate
        client: Backend client
        dry_run: Skip backend calls
        validate_context: If True, validate IDs against expected_context in metadata

    Returns:
        EvaluationRecord with results
    """
    # Handle dry run
    if dry_run:
        return EvaluationRecord(
            case=case,
            response_text=None,
            validator=None,
            latency_s=None,
            raw_response=None,
            error=None,
        )

    # Get expected_context from prompt metadata (if validation enabled)
    # The system prompt is already in case.metadata["system"] and will be
    # included via case.chat_messages()
    eval_context = None
    if validate_context:
        eval_context = case.metadata.get("expected_context")

    # Make request to backend
    try:
        response = client.chat(case.chat_messages())
    except Exception as exc:
        return EvaluationRecord(
            case=case,
            response_text=None,
            validator=None,
            latency_s=None,
            raw_response=None,
            error=str(exc),
        )

    # Run schema validation (with optional context validation)
    try:
        validator_result = validate_assistant_response(response.message, eval_context)
    except Exception as exc:
        return EvaluationRecord(
            case=case,
            response_text=response.message,
            validator=None,
            latency_s=response.latency_s,
            raw_response=response.raw,
            error=f"Validation error: {exc}",
        )

    # Check expected tools
    _check_expected_tools(case, validator_result)

    # Run behavior validation
    behavior_result = _run_behavior_validation(case, response.message)

    return EvaluationRecord(
        case=case,
        response_text=response.message,
        validator=validator_result,
        latency_s=response.latency_s,
        raw_response=response.raw,
        error=None,
        behavior=behavior_result,
    )


def _check_expected_tools(case: PromptCase, validator_result: ValidationResult) -> None:
    """Check if expected tools were called, updating validator_result in place."""
    if not case.expected_tools:
        return

    called_tool_names = {tc.name for tc in validator_result.tool_calls}
    missing_tools = set(case.expected_tools) - called_tool_names

    if missing_tools:
        validator_result.passed = False
        validator_result.issues.append(
            ValidatorIssue(
                level="error",
                message=f"Expected tools not called: {', '.join(sorted(missing_tools))}"
            )
        )


def _run_behavior_validation(
    case: PromptCase,
    response: Any,
) -> Optional[BehaviorValidationResult]:
    """Run behavior validation if expectations are defined.

    Args:
        case: The prompt case with potential behavior expectations
        response: Model's response

    Returns:
        BehaviorValidationResult or None if no expectations defined
    """
    behavior_expectations = case.metadata.get("behavior_expectations")
    expected_response_type = case.metadata.get("expected_response_type")
    anti_patterns = case.metadata.get("anti_patterns_to_avoid")

    # Skip if no behavior expectations defined
    if not (behavior_expectations or expected_response_type or anti_patterns):
        return None

    try:
        return validate_behavior(
            response=response,
            behavior_expectations=behavior_expectations,
            expected_response_type=expected_response_type,
            anti_patterns=anti_patterns,
        )
    except Exception as exc:
        # Return error result instead of failing completely
        return BehaviorValidationResult(
            passed=False,
            issues=[BehaviorIssue(
                check="validation_error",
                expected="successful validation",
                actual=str(exc),
                passed=False,
                message=f"Behavior validation error: {exc}"
            )]
        )
