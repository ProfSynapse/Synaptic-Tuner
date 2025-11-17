"""Wrapper utilities around the dataset validator for single responses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from tools import validate_syngen as dataset_validator


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ValidatorIssue:
    level: str
    message: str


@dataclass
class ValidationResult:
    passed: bool
    issues: List[ValidatorIssue] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.__dict__ for issue in self.issues],
            "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls],
        }


def validate_assistant_response(content: str) -> ValidationResult:
    """Validate a single assistant response string."""
    report = dataset_validator.ExampleReport(index=0, label=True)
    dataset_validator.validate_assistant_content(content, report)
    issues = [ValidatorIssue(level=issue.level, message=issue.message) for issue in report.issues]
    # Extract tool calls even if validation failed to help debugging.
    tool_calls: List[ToolCall] = []
    if "tool_call:" in content:
        try:
            for name, args in dataset_validator.extract_tool_calls(content):
                tool_calls.append(ToolCall(name=name, arguments=args))
        except Exception:
            # extractor raises ValueError for broken JSON; already surfaced as issue
            pass
    return ValidationResult(passed=report.is_valid, issues=issues, tool_calls=tool_calls)
