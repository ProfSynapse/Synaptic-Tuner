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
    def status(self) -> str:
        """Get evaluation status: 'pass', 'warn', or 'fail'.

        - pass: Tool correct + behavior expectations met (or no behavior expectations)
        - warn: Tool correct + behavior expectations NOT met
        - fail: Tool incorrect OR error
        """
        if self.error is not None:
            return "fail"
        if self.validator is None or not self.validator.passed:
            return "fail"
        # Schema passed - check behavior
        if self.behavior is not None and not self.behavior.passed:
            return "warn"
        return "pass"

    @property
    def passed(self) -> bool:
        """Check if evaluation passed all validations (pass or warn)."""
        return self.status in ("pass", "warn")

    @property
    def failed(self) -> bool:
        """Check if evaluation hard-failed (tool incorrect or error)."""
        return self.status == "fail"

    @property
    def warned(self) -> bool:
        """Check if evaluation warned (tool correct but behavior failed)."""
        return self.status == "warn"

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

    # Run behavior validation - pass extracted tool names from schema validation
    extracted_tool_names = [tc.name for tc in validator_result.tool_calls]
    behavior_result = _run_behavior_validation(case, response.message, extracted_tool_names)

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
    """Check if expected tools were called, updating validator_result in place.

    Supports two modes:
    - expected_tools: AND logic - ALL listed tools must be called (primary expectation)
    - acceptable_tools: OR logic - ANY listed tool is valid as an alternative (includes TEXT_ONLY)

    When both are defined, expected_tools are ALSO acceptable (they're what we want!).
    acceptable_tools provides additional alternatives beyond the expected ones.
    """
    called_tool_names = {tc.name for tc in validator_result.tool_calls}
    has_tool_calls = len(called_tool_names) > 0

    # Build the full set of acceptable tools
    # expected_tools are always acceptable (they're what we want)
    # acceptable_tools provides additional alternatives
    all_acceptable = set()
    if case.expected_tools:
        all_acceptable.update(case.expected_tools)
    if case.acceptable_tools:
        all_acceptable.update(case.acceptable_tools)

    # If we have acceptable alternatives (OR logic)
    if all_acceptable:
        # TEXT_ONLY is a pseudo-tool meaning no tool call is acceptable
        text_only_acceptable = "TEXT_ONLY" in all_acceptable
        tool_options = [t for t in all_acceptable if t != "TEXT_ONLY"]

        # Check if any acceptable tool was called OR text-only is valid
        any_acceptable_called = bool(called_tool_names & set(tool_options))
        text_only_response = not has_tool_calls and text_only_acceptable

        if not any_acceptable_called and not text_only_response:
            validator_result.passed = False
            options_str = ", ".join(sorted(all_acceptable))
            validator_result.issues.append(
                ValidatorIssue(
                    level="error",
                    message=f"No acceptable tool called. Valid options: {options_str}"
                )
            )


def _run_behavior_validation(
    case: PromptCase,
    response: Any,
    extracted_tool_names: List[str] = None,
) -> Optional[BehaviorValidationResult]:
    """Run behavior validation if expectations are defined.

    Args:
        case: The prompt case with potential behavior expectations
        response: Model's response
        extracted_tool_names: Tool names already extracted by schema validation
                             (with useTools expansion applied)

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
            extracted_tool_names=extracted_tool_names,
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
