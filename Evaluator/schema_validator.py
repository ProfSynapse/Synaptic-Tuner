"""Wrapper utilities around the dataset validator for single responses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import sys
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).parent.parent / 'tools'
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import validate_syngen as dataset_validator
from shared.validation.parsing import parse_qwen_tool_calls


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


def _looks_like_wrapper_call(tool_call: ToolCall) -> bool:
    """Return True when a tool call wraps delegated calls inside arguments."""
    if not isinstance(tool_call.arguments, dict):
        return False
    calls = tool_call.arguments.get("calls")
    if not isinstance(calls, list) or not calls:
        return False
    return any(
        isinstance(call, dict) and (call.get("tool") or call.get("name"))
        for call in calls
    )


def _expand_wrapper_calls(tool_calls: List[ToolCall]) -> List[ToolCall]:
    """Expand delegated wrapper calls into individual concrete tool calls."""
    expanded = []
    for tc in tool_calls:
        if _looks_like_wrapper_call(tc) and isinstance(tc.arguments, dict):
            wrapper_context = tc.arguments.get("context")
            calls = tc.arguments.get("calls", [])
            added = 0
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        agent = call.get("agent", "")
                        tool = call.get("tool", "")
                        params = call.get("params", {})
                        if agent and tool:
                            merged_args = dict(params) if isinstance(params, dict) else {}
                            if isinstance(wrapper_context, dict) and "context" not in merged_args:
                                merged_args["context"] = wrapper_context
                            # Construct full tool name: agent_tool
                            full_name = f"{agent}_{tool}"
                            expanded.append(ToolCall(name=full_name, arguments=merged_args))
                            added += 1
            # If no valid delegated calls found, keep the original wrapper call.
            if added == 0:
                expanded.append(tc)
        else:
            # Not a delegated wrapper, keep as-is.
            expanded.append(tc)
    return expanded


def _filter_wrapper_schema_warnings(
    issues: List["ValidatorIssue"],
    raw_tool_calls: List[ToolCall],
    expanded_tool_calls: List[ToolCall],
) -> List["ValidatorIssue"]:
    """Remove wrapper-level schema warnings once delegated calls expand cleanly.

    The lower-level dataset validator validates raw function names before wrapper
    expansion. For delegated-call wrappers, that can emit a generic "No schema
    found" warning even when the evaluator successfully expands the response
    into concrete tool/action names. Keep the concrete scoring signal and drop
    the wrapper-only warning.
    """
    suppress_messages = set()
    for index, tool_call in enumerate(raw_tool_calls, start=1):
        if not _looks_like_wrapper_call(tool_call):
            continue
        expanded_from_call = _expand_wrapper_calls([tool_call])
        if not expanded_from_call:
            continue
        if len(expanded_from_call) == 1 and expanded_from_call[0].name == tool_call.name:
            continue
        suppress_messages.add(f"Tool call #{index} ({tool_call.name}): No schema found for this tool")

    if not suppress_messages:
        return issues

    return [
        issue for issue in issues
        if not (issue.level == "WARN" and issue.message in suppress_messages)
    ]


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

    # Detect format and validate accordingly
    if isinstance(content, dict):
        message = dict(content)
        # Qwen sometimes embeds tool calls in content without tool_calls array
        if (not message.get("tool_calls")) and isinstance(message.get("content"), str):
            converted = _convert_qwen_to_openai(message["content"])
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
        else:
            # Dict without tool_calls - invalid
            report.add("ERROR", "Assistant response dict must contain 'tool_calls' field")
    elif isinstance(content, str):
        if "<tool_call>" in content:
            converted = _convert_qwen_to_openai(content)
            if converted:
                dataset_validator.validate_assistant_message_openai(converted, report)
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

    raw_tool_calls = list(tool_calls)
    issues = [ValidatorIssue(level=issue.level, message=issue.message) for issue in report.issues]

    # Expand wrapper calls into concrete tool names before downstream checks.
    tool_calls = _expand_wrapper_calls(tool_calls)
    issues = _filter_wrapper_schema_warnings(issues, raw_tool_calls, tool_calls)

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
    expected_session_id = eval_context.get("session_id")
    expected_workspace_id = eval_context.get("workspace_id")
    valid_workspace_ids = [expected_workspace_id] + eval_context.get("workspace_ids", [])
    valid_agent_ids = eval_context.get("agent_ids", [])

    results = {
        "session_id_matches": [],
        "workspace_id_matches": [],
        "agent_id_matches": [],
        "all_match": True,
    }

    for idx, tc in enumerate(tool_calls, 1):
        context = tc.arguments.get("context", {})

        # Check sessionId
        tool_session_id = context.get("sessionId")
        if tool_session_id:
            matches = tool_session_id == expected_session_id
            results["session_id_matches"].append({
                "tool_call": idx,
                "expected": expected_session_id,
                "actual": tool_session_id,
                "matches": matches,
            })
            if not matches:
                results["all_match"] = False
                issues.append(ValidatorIssue(
                    level="ERROR",
                    message=f"Tool call #{idx}: sessionId '{tool_session_id}' does not match context '{expected_session_id}'"
                ))

        # Check workspaceId
        tool_workspace_id = context.get("workspaceId")
        if tool_workspace_id:
            matches = tool_workspace_id in valid_workspace_ids
            results["workspace_id_matches"].append({
                "tool_call": idx,
                "expected": valid_workspace_ids,
                "actual": tool_workspace_id,
                "matches": matches,
            })
            if not matches:
                results["all_match"] = False
                issues.append(ValidatorIssue(
                    level="ERROR",
                    message=f"Tool call #{idx}: workspaceId '{tool_workspace_id}' not in context {valid_workspace_ids}"
                ))

        # Check agent IDs for promptManager tools
        if tc.name.startswith("promptManager_") and valid_agent_ids:
            agent_id = tc.arguments.get("id") or tc.arguments.get("agent")
            if agent_id and agent_id.startswith("agent_"):
                matches = agent_id in valid_agent_ids
                results["agent_id_matches"].append({
                    "tool_call": idx,
                    "expected": valid_agent_ids,
                    "actual": agent_id,
                    "matches": matches,
                })
                if not matches:
                    # Warn but don't fail for agent IDs (model might create new ones)
                    issues.append(ValidatorIssue(
                        level="WARN",
                        message=f"Tool call #{idx}: agentId '{agent_id}' not in context {valid_agent_ids}"
                    ))

    return results
