"""Evaluation orchestration logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .ollama_client import OllamaClient, OllamaResponse
from .prompt_sets import PromptCase
from .schema_validator import ValidationResult, validate_assistant_response


@dataclass
class EvaluationRecord:
    case: PromptCase
    response_text: Optional[str]
    validator: Optional[ValidationResult]
    latency_s: Optional[float]
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.validator is not None and self.validator.passed


def evaluate_cases(
    cases: Sequence[PromptCase],
    client: OllamaClient,
    dry_run: bool = False,
) -> List[EvaluationRecord]:
    """Run evaluation for the provided prompts."""
    records: List[EvaluationRecord] = []
    for case in cases:
        if dry_run:
            records.append(
                EvaluationRecord(
                    case=case,
                    response_text=None,
                    validator=None,
                    latency_s=None,
                    raw_response=None,
                    error=None,
                )
            )
            continue

        try:
            response = client.chat(case.chat_messages())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            records.append(
                EvaluationRecord(
                    case=case,
                    response_text=None,
                    validator=None,
                    latency_s=None,
                    raw_response=None,
                    error=str(exc),
                )
            )
            continue

        validator_result = validate_assistant_response(response.message)

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

        records.append(
            EvaluationRecord(
                case=case,
                response_text=response.message,
                validator=validator_result,
                latency_s=response.latency_s,
                raw_response=response.raw,
                error=None,
            )
        )
    return records
