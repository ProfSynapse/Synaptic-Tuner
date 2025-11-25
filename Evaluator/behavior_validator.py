"""Automated behavior validation for evaluation responses."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class BehaviorIssue:
    """Single behavior validation issue."""
    check: str
    expected: Any
    actual: Any
    passed: bool
    message: str


@dataclass
class BehaviorValidationResult:
    """Result of behavior validation checks."""
    passed: bool
    issues: List[BehaviorIssue] = field(default_factory=list)
    response_type_detected: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "response_type_detected": self.response_type_detected,
            "issues": [
                {
                    "check": i.check,
                    "expected": i.expected,
                    "actual": i.actual,
                    "passed": i.passed,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }


def detect_response_type(response: Union[str, Dict[str, Any]]) -> str:
    """
    Detect the response type from model output.

    Supports multiple formats:
    - OpenAI format: dict with "tool_calls" array
    - ChatML format: string with "tool_call:" markers
    - Mistral format: string with "[TOOL_CALLS]" marker

    Returns:
        "text_only" - Response has text content but no tool calls
        "tool_only" - Response has tool calls but no/minimal text
        "tool_text" - Response has both meaningful text and tool calls
        "empty" - No meaningful content
    """
    has_text = False
    has_tool = False
    text_content = ""

    if isinstance(response, dict):
        # OpenAI format
        text_content = response.get("content") or ""
        tool_calls = response.get("tool_calls") or []
        has_tool = len(tool_calls) > 0
    elif isinstance(response, str):
        # Check for Mistral format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
        if "[TOOL_CALLS]" in response:
            has_tool = True
            # Extract text before [TOOL_CALLS]
            parts = response.split("[TOOL_CALLS]", 1)
            text_content = parts[0].strip()
        # Check for ChatML format: tool_call: toolName
        elif "tool_call:" in response:
            has_tool = True
            # Extract text before first tool_call
            parts = response.split("tool_call:", 1)
            text_content = parts[0].strip()
        else:
            text_content = response.strip()

    # Meaningful text is more than just whitespace or very short filler
    # Consider text meaningful if > 20 chars after stripping
    has_text = len(text_content.strip()) > 20

    if has_text and has_tool:
        return "tool_text"
    elif has_tool and not has_text:
        return "tool_only"
    elif has_text and not has_tool:
        return "text_only"
    else:
        return "empty"


def extract_context_from_response(response: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract context object from tool call arguments.

    Supports multiple formats:
    - OpenAI format: dict with "tool_calls" array
    - ChatML format: string with "tool_call:" and "arguments:" markers
    - Mistral format: string with "[TOOL_CALLS]" marker followed by JSON array
    """
    try:
        if isinstance(response, dict):
            # OpenAI format
            tool_calls = response.get("tool_calls") or []
            if tool_calls:
                args = tool_calls[0].get("function", {}).get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)
                return args.get("context")
        elif isinstance(response, str):
            # Check for Mistral format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
            if "[TOOL_CALLS]" in response:
                parts = response.split("[TOOL_CALLS]", 1)
                if len(parts) > 1:
                    json_part = parts[1].strip()
                    # Find the JSON array
                    if json_part.startswith("["):
                        # Extract the JSON array
                        bracket_count = 0
                        end_idx = 0
                        for i, char in enumerate(json_part):
                            if char == "[":
                                bracket_count += 1
                            elif char == "]":
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i + 1
                                    break
                        if end_idx > 0:
                            tool_calls_json = json_part[:end_idx]
                            tool_calls = json.loads(tool_calls_json)
                            if tool_calls and isinstance(tool_calls, list):
                                first_call = tool_calls[0]
                                args = first_call.get("arguments", {})
                                if isinstance(args, str):
                                    args = json.loads(args)
                                return args.get("context") if isinstance(args, dict) else None
            # ChatML format: tool_call: toolName\narguments: {...}
            elif "tool_call:" in response:
                # Extract arguments from ChatML format - use more robust extraction
                match = re.search(r'arguments:\s*(\{.*?\})\s*(?:\n\n|Result:|$)', response, re.DOTALL)
                if match:
                    args = json.loads(match.group(1))
                    return args.get("context")
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        pass
    return None


def validate_behavior(
    response: Union[str, Dict[str, Any]],
    behavior_expectations: Optional[Dict[str, Any]] = None,
    expected_response_type: Optional[str] = None,
    anti_patterns: Optional[Dict[str, bool]] = None,
) -> BehaviorValidationResult:
    """
    Validate model response against behavior expectations.

    Args:
        response: Model response (string or dict)
        behavior_expectations: Dict of expected behaviors (from prompt case)
        expected_response_type: Expected response type (text_only, tool_only, tool_text)
        anti_patterns: Dict of anti-patterns that should NOT be present

    Returns:
        BehaviorValidationResult with pass/fail and detailed issues
    """
    issues: List[BehaviorIssue] = []
    all_passed = True

    # Detect actual response type
    actual_type = detect_response_type(response)

    # Check response type if specified
    if expected_response_type:
        type_match = actual_type == expected_response_type
        issues.append(BehaviorIssue(
            check="response_type",
            expected=expected_response_type,
            actual=actual_type,
            passed=type_match,
            message=f"Response type {'matches' if type_match else 'mismatch'}: expected {expected_response_type}, got {actual_type}"
        ))
        if not type_match:
            all_passed = False

    # Extract context for context-based checks
    context = extract_context_from_response(response)

    # Validate behavior expectations
    if behavior_expectations:
        for expectation, value in behavior_expectations.items():
            if expectation == "reason":
                # Skip reason field - it's documentation
                continue

            issue = _check_expectation(expectation, value, response, context, actual_type)
            if issue:
                issues.append(issue)
                if not issue.passed:
                    all_passed = False

    # Check anti-patterns (things that should NOT happen)
    if anti_patterns:
        for pattern, should_avoid in anti_patterns.items():
            if should_avoid:
                issue = _check_anti_pattern(pattern, response, context, actual_type)
                if issue:
                    issues.append(issue)
                    if not issue.passed:
                        all_passed = False

    return BehaviorValidationResult(
        passed=all_passed,
        issues=issues,
        response_type_detected=actual_type,
    )


def _check_expectation(
    expectation: str,
    value: Any,
    response: Union[str, Dict[str, Any]],
    context: Optional[Dict[str, Any]],
    response_type: str,
) -> Optional[BehaviorIssue]:
    """Check a single behavior expectation."""

    # Response type expectations
    if expectation == "does_not_call_tool":
        has_tool = response_type in ("tool_only", "tool_text")
        passed = not has_tool if value else has_tool
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=not has_tool,
            passed=passed,
            message=f"does_not_call_tool: {'PASS' if passed else 'FAIL'} - tool call {'not present' if not has_tool else 'present'}"
        )

    if expectation == "calls_tool_directly":
        has_tool = response_type in ("tool_only", "tool_text")
        passed = has_tool if value else not has_tool
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=has_tool,
            passed=passed,
            message=f"calls_tool_directly: {'PASS' if passed else 'FAIL'} - tool call {'present' if has_tool else 'not present'}"
        )

    if expectation == "minimal_or_no_explanation":
        # For tool_only, text should be minimal (< 50 chars) or empty
        text_len = _get_text_length(response)
        is_minimal = text_len < 50
        passed = is_minimal if value else not is_minimal
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=f"{text_len} chars",
            passed=passed,
            message=f"minimal_or_no_explanation: {'PASS' if passed else 'FAIL'} - text length {text_len} chars"
        )

    if expectation in ("explains_choice", "reasons_about_selection", "then_calls_tool",
                       "explains_folder_priority", "reasons_about_active_vs_archived",
                       "explains_name_based_choice", "reasons_about_naming"):
        # For tool_text, should have meaningful explanation (> 30 chars) AND tool call
        text_len = _get_text_length(response)
        has_text = text_len > 30
        has_tool = response_type in ("tool_only", "tool_text")
        passed = has_text and has_tool if value else True
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=f"text={text_len}chars, tool={has_tool}",
            passed=passed,
            message=f"{expectation}: {'PASS' if passed else 'FAIL'} - has explanation: {has_text}, has tool: {has_tool}"
        )

    # Text content expectations
    if expectation in ("presents_results_clearly", "asks_for_user_input", "offers_alternatives",
                       "asks_user_preference", "asks_for_direction", "asks_if_more_needed",
                       "acknowledges_no_results", "explains_error", "suggests_alternatives",
                       "confirms_completion"):
        # These require text response with certain patterns
        text = _get_text_content(response)
        has_meaningful_text = len(text) > 30

        # Check for question patterns if asking user
        if "asks" in expectation or "offers" in expectation:
            has_question = _check_asks_for_input(text)
            passed = has_meaningful_text and has_question if value else True
            return BehaviorIssue(
                check=expectation,
                expected=value,
                actual=f"has_text={has_meaningful_text}, has_question={has_question}",
                passed=passed,
                message=f"{expectation}: {'PASS' if passed else 'FAIL'}"
            )
        else:
            passed = has_meaningful_text if value else True
            return BehaviorIssue(
                check=expectation,
                expected=value,
                actual=has_meaningful_text,
                passed=passed,
                message=f"{expectation}: {'PASS' if passed else 'FAIL'} - meaningful text present: {has_meaningful_text}"
            )

    # Context quality expectations
    if expectation == "sessionMemory_min_chars" and context:
        session_memory = context.get("sessionMemory", "")
        actual_len = len(session_memory)
        passed = actual_len >= value
        return BehaviorIssue(
            check=expectation,
            expected=f">= {value} chars",
            actual=f"{actual_len} chars",
            passed=passed,
            message=f"sessionMemory length: {'PASS' if passed else 'FAIL'} - {actual_len} chars (min: {value})"
        )

    if expectation == "toolContext_explains_why" and context:
        tool_context = context.get("toolContext", "")
        # Should be > 50 chars and explain reasoning
        has_explanation = len(tool_context) > 50
        passed = has_explanation if value else True
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=f"{len(tool_context)} chars",
            passed=passed,
            message=f"toolContext explains why: {'PASS' if passed else 'FAIL'} - {len(tool_context)} chars"
        )

    # Workflow continuation expectations
    if expectation in ("continues_workflow_silently", "creates_next_subfolder", "appends_content"):
        has_tool = response_type in ("tool_only", "tool_text")
        passed = has_tool if value else True
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=has_tool,
            passed=passed,
            message=f"{expectation}: {'PASS' if passed else 'FAIL'}"
        )

    # Default: return None for unhandled expectations (don't fail on unknown)
    return None


def _check_anti_pattern(
    pattern: str,
    response: Union[str, Dict[str, Any]],
    context: Optional[Dict[str, Any]],
    response_type: str,
) -> Optional[BehaviorIssue]:
    """Check that an anti-pattern is NOT present."""

    if pattern in ("immediate_tool_call", "assumes_user_choice"):
        # Should NOT have tool call for text_only scenarios
        has_tool = response_type in ("tool_only", "tool_text")
        passed = not has_tool  # Pass if NO tool call
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="should NOT occur",
            actual="occurred" if has_tool else "not present",
            passed=passed,
            message=f"Anti-pattern {pattern}: {'PASS (avoided)' if passed else 'FAIL (occurred)'}"
        )

    if pattern in ("auto_creates_file", "auto_creates_content", "auto_broadens_search",
                   "auto_continues_cleanup", "retries_without_asking"):
        # Should NOT have tool call when user input is needed
        has_tool = response_type in ("tool_only", "tool_text")
        passed = not has_tool
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="should NOT occur",
            actual="tool called" if has_tool else "no auto-action",
            passed=passed,
            message=f"Anti-pattern {pattern}: {'PASS (avoided)' if passed else 'FAIL (auto-acted)'}"
        )

    if pattern in ("excessive_explanation", "asks_confirmation_for_obvious"):
        # Should NOT have excessive text for tool_only scenarios
        text_len = _get_text_length(response)
        is_excessive = text_len > 100
        passed = not is_excessive
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="should NOT occur",
            actual=f"{text_len} chars",
            passed=passed,
            message=f"Anti-pattern {pattern}: {'PASS' if passed else 'FAIL'} - text {text_len} chars"
        )

    if pattern in ("silent_choice", "no_reasoning_for_selection", "arbitrary_selection",
                   "no_naming_reasoning", "silent_folder_choice"):
        # Should have explanation before tool call
        text_len = _get_text_length(response)
        has_explanation = text_len > 30
        passed = has_explanation
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="should have explanation",
            actual=f"{text_len} chars explanation",
            passed=passed,
            message=f"Anti-pattern {pattern}: {'PASS (explained)' if passed else 'FAIL (no explanation)'}"
        )

    if pattern in ("searches_for_more_to_delete", "reads_file_again", "asks_where_to_append",
                   "explains_each_step", "asks_before_each_subfolder", "reads_archived_without_reason"):
        # These are context-specific, check for tool call presence as proxy
        has_tool = response_type in ("tool_only", "tool_text")
        # For these patterns, having a tool call might be the anti-pattern
        # But we need more context to determine - default to pass with warning
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="context-dependent",
            actual=f"tool_call={has_tool}",
            passed=True,  # Don't fail on ambiguous patterns
            message=f"Anti-pattern {pattern}: needs manual review (tool_call={has_tool})"
        )

    return None


def _get_text_content(response: Union[str, Dict[str, Any]]) -> str:
    """Extract text content from response (excluding tool call markers)."""
    if isinstance(response, dict):
        return response.get("content") or ""
    elif isinstance(response, str):
        # Check for Mistral format first
        if "[TOOL_CALLS]" in response:
            return response.split("[TOOL_CALLS]", 1)[0].strip()
        # Then ChatML format
        elif "tool_call:" in response:
            return response.split("tool_call:", 1)[0].strip()
        return response.strip()
    return ""


def _get_text_length(response: Union[str, Dict[str, Any]]) -> int:
    """Get length of text content in response."""
    return len(_get_text_content(response))


# Comprehensive keywords for detecting "asks for user input" patterns
_USER_INPUT_KEYWORDS = [
    # Direct questions - modal verbs
    "would you like",
    "would you prefer",
    "would you want",
    "would you rather",
    "do you want",
    "do you need",
    "do you prefer",
    "should i",
    "shall i",
    "can i",
    "may i",
    "could i",
    "could you",
    "can you",
    "will you",

    # Question starters
    "which one",
    "which file",
    "which folder",
    "which option",
    "which would",
    "which do you",
    "which should",
    "what would you",
    "what do you",
    "what should",
    "where would you",
    "where should",
    "where do you",
    "how would you",
    "how should",
    "how do you",
    "when should",
    "when would",

    # Request phrases
    "let me know",
    "please let me know",
    "please specify",
    "please confirm",
    "please tell me",
    "please indicate",
    "please select",
    "please choose",
    "please clarify",
    "please provide",
    "kindly specify",
    "kindly confirm",
    "kindly let me know",

    # Choice/selection language
    "your choice",
    "your preference",
    "your decision",
    "you choose",
    "you select",
    "you pick",
    "you decide",
    "you prefer",
    "prefer to",
    "choice between",
    "choose between",
    "select from",
    "pick from",
    "decide between",
    "option to",
    "options are",
    "options include",
    "alternatives are",
    "alternatives include",

    # Confirmation requests
    "confirm that",
    "confirm if",
    "confirm whether",
    "verify that",
    "verify if",
    "clarify what",
    "clarify which",
    "clarify if",
    "specify which",
    "specify what",
    "specify the",
    "indicate which",
    "indicate what",
    "indicate your",

    # Conditional offers
    "if you'd like",
    "if you would like",
    "if you want",
    "if you prefer",
    "if you need",
    "if you wish",
    "if that works",
    "if that's okay",
    "if that sounds good",

    # Alternative suggestions
    "alternatively",
    "or would you",
    "or should i",
    "or i could",
    "or i can",
    "or do you",
    "or perhaps",
    "otherwise",
    "instead",
    "on the other hand",

    # Offers of assistance
    "i can also",
    "i could also",
    "i'm able to",
    "i am able to",
    "i'd be happy to",
    "i would be happy to",
    "happy to help",
    "glad to help",

    # Waiting for input
    "waiting for",
    "awaiting your",
    "ready when you",
    "whenever you're ready",
    "when you're ready",
    "at your convenience",
    "up to you",
    "your call",
    "you decide",

    # Numbered/listed options indicators
    "option 1",
    "option 2",
    "choice 1",
    "choice 2",
    "1.",
    "2.",
    "a)",
    "b)",
    "(1)",
    "(2)",
    "first option",
    "second option",
    "either",
    "or",

    # Direct ask patterns
    "what next",
    "what now",
    "what else",
    "anything else",
    "something else",
    "any other",
    "more help",
    "further assistance",
    "need anything",
    "want me to",
    "like me to",
    "need me to",

    # Uncertainty acknowledgment + ask
    "not sure which",
    "unclear which",
    "ambiguous",
    "multiple options",
    "several options",
    "few options",
    "different options",
    "various options",
]


def _check_asks_for_input(text: str) -> bool:
    """
    Check if text contains patterns indicating the model is asking for user input.

    Returns True if:
    - Text contains a question mark AND meaningful content, OR
    - Text contains any of the comprehensive user input keywords
    """
    if not text:
        return False

    text_lower = text.lower()

    # Check for question mark with meaningful content
    if "?" in text and len(text) > 20:
        return True

    # Check for any user input keywords
    for keyword in _USER_INPUT_KEYWORDS:
        if keyword in text_lower:
            return True

    return False
