"""Wrapper utilities around the dataset validator for single responses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import sys
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).parent.parent / 'tools'
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import validate_syngen as dataset_validator


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


def validate_assistant_response(content: Union[str, Dict[str, Any]]) -> ValidationResult:
    """
    Validate a single assistant response.

    Supports both formats:
    - ChatML: string content with "tool_call:" markers
    - OpenAI: dict with "tool_calls" array
    """
    report = dataset_validator.ExampleReport(index=0, label=True)
    tool_calls: List[ToolCall] = []

    # Detect format and validate accordingly
    if isinstance(content, dict):
        # OpenAI format with structured tool_calls
        if "tool_calls" in content:
            dataset_validator.validate_assistant_message_openai(content, report)
            # Extract tool calls
            try:
                tool_calls_array = content.get("tool_calls", [])
                for name, args in dataset_validator.extract_tool_calls_openai(tool_calls_array):
                    tool_calls.append(ToolCall(name=name, arguments=args))
            except Exception:
                # Extraction errors already surfaced as validation issues
                pass
        else:
            # Dict without tool_calls - invalid
            report.add("ERROR", "Assistant response dict must contain 'tool_calls' field")
    elif isinstance(content, str):
        # ChatML or Mistral format with content string
        dataset_validator.validate_assistant_content(content, report)
        # Extract tool calls even if validation failed to help debugging.
        # Check for Mistral format first (more specific marker)
        if "[TOOL_CALLS]" in content:
            try:
                for name, args in dataset_validator.extract_tool_calls_mistral(content):
                    tool_calls.append(ToolCall(name=name, arguments=args))
            except Exception:
                # extractor raises ValueError for broken JSON; already surfaced as issue
                pass
        elif "tool_call:" in content:
            try:
                for name, args in dataset_validator.extract_tool_calls(content):
                    tool_calls.append(ToolCall(name=name, arguments=args))
            except Exception:
                # extractor raises ValueError for broken JSON; already surfaced as issue
                pass
    else:
        report.add("ERROR", f"Assistant response must be string or dict, got {type(content).__name__}")

    issues = [ValidatorIssue(level=issue.level, message=issue.message) for issue in report.issues]
    return ValidationResult(passed=report.is_valid, issues=issues, tool_calls=tool_calls)
