"""Evaluation orchestration logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .lmstudio_client import LMStudioClient
from .ollama_client import OllamaClient
from .prompt_sets import PromptCase
from .schema_validator import ValidationResult, validate_assistant_response
from .behavior_validator import BehaviorValidationResult, validate_behavior


@dataclass
class EvaluationRecord:
    case: PromptCase
    response_text: Optional[str]
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
    client: Union[OllamaClient, LMStudioClient],
    dry_run: bool = False,
    on_record: Callable[[EvaluationRecord], None] | None = None,
) -> List[EvaluationRecord]:
    """Run evaluation for the provided prompts."""
    records: List[EvaluationRecord] = []
    for case in cases:
        if dry_run:
            records.append(
                record := EvaluationRecord(
                    case=case,
                    response_text=None,
                    validator=None,
                    latency_s=None,
                    raw_response=None,
                    error=None,
                )
            )
            if on_record:
                on_record(record)
            continue

        try:
            response = client.chat(case.chat_messages())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            records.append(
                record := EvaluationRecord(
                    case=case,
                    response_text=None,
                    validator=None,
                    latency_s=None,
                    raw_response=None,
                    error=str(exc),
                )
            )
            if on_record:
                on_record(record)
            continue

        try:
            validator_result = validate_assistant_response(response.message)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Validation failed - record the error but continue evaluation
            records.append(
                record := EvaluationRecord(
                    case=case,
                    response_text=response.message,
                    validator=None,
                    latency_s=response.latency_s,
                    raw_response=response.raw,
                    error=f"Validation error: {exc}",
                )
            )
            if on_record:
                on_record(record)
            continue

        # Check if expected tools were actually called
        if case.expected_tools:
            called_tool_names = {tc.name for tc in validator_result.tool_calls}
            missing_tools = set(case.expected_tools) - called_tool_names

            if missing_tools:
                from .schema_validator import ValidatorIssue
                validator_result.passed = False
                validator_result.issues.append(
                    ValidatorIssue(
                        level="error",
                        message=f"Expected tools not called: {', '.join(sorted(missing_tools))}"
                    )
                )

        # Run behavior validation if expectations are defined
        behavior_result: Optional[BehaviorValidationResult] = None
        behavior_expectations = case.metadata.get("behavior_expectations")
        expected_response_type = case.metadata.get("expected_response_type")
        anti_patterns = case.metadata.get("anti_patterns_to_avoid")

        if behavior_expectations or expected_response_type or anti_patterns:
            try:
                behavior_result = validate_behavior(
                    response=response.message,
                    behavior_expectations=behavior_expectations,
                    expected_response_type=expected_response_type,
                    anti_patterns=anti_patterns,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Behavior validation error - don't fail the whole evaluation
                from .behavior_validator import BehaviorValidationResult, BehaviorIssue
                behavior_result = BehaviorValidationResult(
                    passed=False,
                    issues=[BehaviorIssue(
                        check="validation_error",
                        expected="successful validation",
                        actual=str(exc),
                        passed=False,
                        message=f"Behavior validation error: {exc}"
                    )]
                )

        records.append(
            record := EvaluationRecord(
                case=case,
                response_text=response.message,
                validator=validator_result,
                latency_s=response.latency_s,
                raw_response=response.raw,
                error=None,
                behavior=behavior_result,
            )
        )
        if on_record:
            on_record(record)
    return records
