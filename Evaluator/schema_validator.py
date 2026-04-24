"""Structural validation utilities for single assistant responses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import sys
from pathlib import Path

# Add canonical skill validator path to sys.path.
validator_dir = Path(__file__).parent.parent / ".skills" / "synethetic-data-generation" / "scripts"
if str(validator_dir) not in sys.path:
    sys.path.insert(0, str(validator_dir))

import validate_syngen as dataset_validator
from shared.validation.parsing import parse_qwen_tool_calls
from shared.validation.parsing.tool_call_parser import parse_gemma_tool_calls, is_gemma_tool_call


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
    context_validation: Optional[Dict[str, Any]] = None  # ID match results

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "passed": self.passed,
            "issues": [issue.__dict__ for issue in self.issues],
            "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls],
        }
        if self.context_validation:
            result["context_validation"] = self.context_validation
        return result


def validate_assistant_response(
    content: Union[str, Dict[str, Any]],
    eval_context: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """
    Validate a single assistant response.

    Supports both formats:
    - ChatML: string content with "tool_call:" markers
    - OpenAI: dict with "tool_calls" array

    Args:
        content: Assistant response (string or dict with tool_calls)
        eval_context: Optional context from context_injection for ID validation
                     Expected keys: session_id, workspace_id, workspace_ids, agent_ids
    """
    report = dataset_validator.ExampleReport(index=0, label=True)
    tool_calls: List[ToolCall] = []

    def _convert_qwen_to_openai(msg: str) -> Optional[Dict[str, Any]]:
        """Convert Qwen <tool_call> content into OpenAI-style tool_calls."""
        calls = parse_qwen_tool_calls(msg)
        if not calls:
            return None
        return {"role": "assistant", "content": msg, "tool_calls": calls}

    def _convert_gemma_to_openai(msg: str) -> Optional[Dict[str, Any]]:
        """Convert Gemma <|tool_call>call:name{...}<tool_call|> content into OpenAI-style tool_calls."""
        calls = parse_gemma_tool_calls(msg)
        if not calls:
            return None
        return {"role": "assistant", "content": msg, "tool_calls": calls}

    # Detect format and validate accordingly
    if isinstance(content, dict):
        message = dict(content)
        if message.get("tool_calls", "__missing__") is None and isinstance(message.get("content"), str):
            message.pop("tool_calls", None)
        # Models sometimes embed tool calls in content without tool_calls array
        if (not message.get("tool_calls")) and isinstance(message.get("content"), str):
            converted = (
                _convert_gemma_to_openai(message["content"])
                or _convert_qwen_to_openai(message["content"])
            )
            if converted:
                message = converted

        # OpenAI format with structured tool_calls
        if "tool_calls" in message:
            dataset_validator.validate_assistant_message_openai(message, report)
            # Extract tool calls
            try:
                tool_calls_array = message.get("tool_calls", [])
                for name, args in dataset_validator.extract_tool_calls_openai(tool_calls_array):
                    tool_calls.append(ToolCall(name=name, arguments=args))
            except Exception:
                # Extraction errors already surfaced as validation issues
                pass
        elif isinstance(message.get("content"), str):
            dataset_validator.validate_assistant_content(message["content"], report)
        else:
            # Dict without tool_calls - invalid
            report.add("ERROR", "Assistant response dict must contain 'tool_calls' field")
    elif isinstance(content, str):
        if is_gemma_tool_call(content):
            converted = _convert_gemma_to_openai(content)
            if converted:
                dataset_validator.validate_assistant_message_openai(converted, report)
                try:
                    for name, args in dataset_validator.extract_tool_calls_openai(converted["tool_calls"]):
                        tool_calls.append(ToolCall(name=name, arguments=args))
                except Exception:
                    pass
            else:
                report.add("ERROR", "Assistant response contains <|tool_call> markers but could not be parsed")
        elif "<tool_call>" in content:
            converted = _convert_qwen_to_openai(content)
            if converted:
                dataset_validator.validate_assistant_message_openai(converted, report)
                recovered_calls = [
                    tc for tc in converted.get("tool_calls", [])
                    if isinstance(tc, dict) and tc.get("recovered")
                ]
                if recovered_calls:
                    report.add(
                        "ERROR",
                        "Assistant response contains malformed <tool_call> JSON recovered heuristically",
                    )
                try:
                    for name, args in dataset_validator.extract_tool_calls_openai(converted["tool_calls"]):
                        tool_calls.append(ToolCall(name=name, arguments=args))
                except Exception:
                    pass
            else:
                report.add("ERROR", "Assistant response contains <tool_call> markers but could not be parsed")
        else:
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

    # Validate IDs against eval context if provided
    context_validation = None
    if eval_context and tool_calls:
        context_validation = _validate_ids_against_context(tool_calls, eval_context, issues)

    return ValidationResult(
        passed=report.is_valid and (context_validation is None or context_validation.get("all_match", True)),
        issues=issues,
        tool_calls=tool_calls,
        context_validation=context_validation,
    )


def _validate_ids_against_context(
    tool_calls: List[ToolCall],
    eval_context: Dict[str, Any],
    issues: List[ValidatorIssue],
) -> Dict[str, Any]:
    """Validate that tool call IDs match the evaluation context.

    Args:
        tool_calls: Extracted tool calls
        eval_context: Context with expected IDs
        issues: List to append validation issues to

    Returns:
        Dict with validation results
    """
    id_expectations: Dict[str, List[Any]] = {}
    for key, value in eval_context.items():
        if not key.endswith("_id"):
            continue
        field_name = _snake_to_camel(key)
        candidates: List[Any] = []
        if value is not None:
            candidates.append(value)
        plural_key = f"{key[:-3]}_ids"
        extra_values = eval_context.get(plural_key, [])
        if isinstance(extra_values, list):
            candidates.extend(item for item in extra_values if item is not None)
        if candidates:
            id_expectations[field_name] = candidates

    results = {"all_match": True}
    for field_name in id_expectations:
        results[f"{field_name}_matches"] = []

    for idx, tc in enumerate(tool_calls, 1):
        for field_name, valid_values in id_expectations.items():
            actual_value = tc.arguments.get(field_name)
            if actual_value is None and isinstance(tc.arguments.get("context"), dict):
                actual_value = tc.arguments["context"].get(field_name)
            if actual_value is None:
                continue
            matches = actual_value in valid_values
            results[f"{field_name}_matches"].append({
                "tool_call": idx,
                "expected": valid_values,
                "actual": actual_value,
                "matches": matches,
            })
            if not matches:
                results["all_match"] = False
                issues.append(ValidatorIssue(
                    level="ERROR",
                    message=f"Tool call #{idx}: {field_name} '{actual_value}' not in context {valid_values}"
                ))
    return results


def _snake_to_camel(value: str) -> str:
    parts = [part for part in str(value).split("_") if part]
    if not parts:
        return str(value)
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])
