"""Automated behavior validation for evaluation responses.

This module validates model responses against expected behaviors,
checking for proper tool usage patterns, context quality, and
avoiding anti-patterns.

Uses response_parser.py for all format parsing (DRY principle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .enums import ResponseType
from .response_parser import (
    ParsedResponse,
    parse_response,
    get_text_content,
    get_text_length,
)


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


# ---------------------------------------------------------------------------
# Public API - Backwards Compatibility Wrappers
# ---------------------------------------------------------------------------

def detect_response_type(response: Union[str, Dict[str, Any]]) -> str:
    """Detect the response type from model output.

    Returns:
        "text_only", "tool_only", "tool_text", or "empty"
    """
    parsed = parse_response(response)
    return str(parsed.response_type)


def extract_arguments_from_response(
    response: Union[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Extract full arguments dict from tool call."""
    parsed = parse_response(response)
    if parsed.first_tool_call:
        return parsed.first_tool_call.arguments
    return None


def extract_context_from_response(
    response: Union[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Extract context object from tool call arguments."""
    parsed = parse_response(response)
    return parsed.context


# ---------------------------------------------------------------------------
# Main Validation Function
# ---------------------------------------------------------------------------

def validate_behavior(
    response: Union[str, Dict[str, Any]],
    behavior_expectations: Optional[Dict[str, Any]] = None,
    expected_response_type: Optional[str] = None,
    anti_patterns: Optional[Dict[str, bool]] = None,
) -> BehaviorValidationResult:
    """Validate model response against behavior expectations.

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

    # Parse response once (single source of truth)
    parsed = parse_response(response)
    actual_type = str(parsed.response_type)

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

    # Validate behavior expectations
    if behavior_expectations:
        for expectation, value in behavior_expectations.items():
            if expectation == "reason":
                continue  # Skip reason field - it's documentation

            issue = _check_expectation(expectation, value, parsed)
            if issue:
                issues.append(issue)
                if not issue.passed:
                    all_passed = False

    # Check anti-patterns (things that should NOT happen)
    if anti_patterns:
        for pattern, should_avoid in anti_patterns.items():
            if should_avoid:
                issue = _check_anti_pattern(pattern, parsed)
                if issue:
                    issues.append(issue)
                    if not issue.passed:
                        all_passed = False

    return BehaviorValidationResult(
        passed=all_passed,
        issues=issues,
        response_type_detected=actual_type,
    )


# ---------------------------------------------------------------------------
# Expectation Checkers
# ---------------------------------------------------------------------------

def _check_expectation(
    expectation: str,
    value: Any,
    parsed: ParsedResponse,
) -> Optional[BehaviorIssue]:
    """Check a single behavior expectation."""
    context = parsed.context
    has_tool = parsed.has_tool_calls
    text_len = len(parsed.text_content)

    # Response type expectations
    if expectation == "does_not_call_tool":
        passed = not has_tool if value else has_tool
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=not has_tool,
            passed=passed,
            message=f"does_not_call_tool: {'PASS' if passed else 'FAIL'} - tool call {'not present' if not has_tool else 'present'}"
        )

    if expectation == "calls_tool_directly":
        passed = has_tool if value else not has_tool
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=has_tool,
            passed=passed,
            message=f"calls_tool_directly: {'PASS' if passed else 'FAIL'} - tool call {'present' if has_tool else 'not present'}"
        )

    if expectation == "minimal_or_no_explanation":
        is_minimal = text_len < 50
        passed = is_minimal if value else not is_minimal
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=f"{text_len} chars",
            passed=passed,
            message=f"minimal_or_no_explanation: {'PASS' if passed else 'FAIL'} - text length {text_len} chars"
        )

    # Explanation expectations
    if expectation in ("explains_choice", "reasons_about_selection", "then_calls_tool",
                       "explains_folder_priority", "reasons_about_active_vs_archived",
                       "explains_name_based_choice", "reasons_about_naming"):
        has_text = text_len > 30
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
        has_meaningful_text = text_len > 30

        if "asks" in expectation or "offers" in expectation:
            has_question = _check_asks_for_input(parsed.text_content)
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
        passed = has_tool if value else True
        return BehaviorIssue(
            check=expectation,
            expected=value,
            actual=has_tool,
            passed=passed,
            message=f"{expectation}: {'PASS' if passed else 'FAIL'}"
        )

    # Context efficiency expectations (limit usage)
    if expectation == "uses_appropriate_limit":
        limit_value = _get_limit_value(parsed)
        is_appropriate = limit_value is not None and 10 <= limit_value <= 100
        passed = is_appropriate if value else True
        return BehaviorIssue(
            check=expectation,
            expected="limit between 10-100",
            actual=f"limit={limit_value}" if limit_value else "no limit found",
            passed=passed,
            message=f"uses_appropriate_limit: {'PASS' if passed else 'FAIL'} - limit={limit_value}"
        )

    if expectation in ("explains_limit_reasoning", "sessionMemory_explains_limit_choice",
                       "sessionMemory_explains_temporal_limit", "limit_matches_expected_count"):
        if context:
            session_memory = context.get("sessionMemory", "")
            reasoning_keywords = ["limit", "typically", "usually", "expect", "approximate", "estimate", "reasonable"]
            has_limit_reasoning = any(word in session_memory.lower() for word in reasoning_keywords)
            passed = has_limit_reasoning if value else True
            return BehaviorIssue(
                check=expectation,
                expected="sessionMemory explains limit choice",
                actual=f"has_reasoning={has_limit_reasoning}",
                passed=passed,
                message=f"{expectation}: {'PASS' if passed else 'FAIL'}"
            )
        return None

    # Execute prompt usage expectations (delegation)
    if expectation in ("delegates_complex_task", "uses_execute_prompt"):
        tool_name = parsed.first_tool_call.name if parsed.first_tool_call else None
        uses_execute_prompt = tool_name == "agentManager_executePrompt"
        passed = uses_execute_prompt if value else True
        return BehaviorIssue(
            check=expectation,
            expected="agentManager_executePrompt",
            actual=tool_name or "no tool",
            passed=passed,
            message=f"{expectation}: {'PASS' if passed else 'FAIL'} - tool={tool_name}"
        )

    # Default: return None for unhandled expectations
    return None


# ---------------------------------------------------------------------------
# Anti-Pattern Checkers
# ---------------------------------------------------------------------------

def _check_anti_pattern(
    pattern: str,
    parsed: ParsedResponse,
) -> Optional[BehaviorIssue]:
    """Check that an anti-pattern is NOT present."""
    has_tool = parsed.has_tool_calls
    text_len = len(parsed.text_content)

    if pattern in ("immediate_tool_call", "assumes_user_choice"):
        passed = not has_tool
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="should NOT occur",
            actual="occurred" if has_tool else "not present",
            passed=passed,
            message=f"Anti-pattern {pattern}: {'PASS (avoided)' if passed else 'FAIL (occurred)'}"
        )

    if pattern in ("auto_creates_file", "auto_creates_content", "auto_broadens_search",
                   "auto_continues_cleanup", "retries_without_asking"):
        passed = not has_tool
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="should NOT occur",
            actual="tool called" if has_tool else "no auto-action",
            passed=passed,
            message=f"Anti-pattern {pattern}: {'PASS (avoided)' if passed else 'FAIL (auto-acted)'}"
        )

    if pattern in ("excessive_explanation", "asks_confirmation_for_obvious"):
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
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="context-dependent",
            actual=f"tool_call={has_tool}",
            passed=True,
            message=f"Anti-pattern {pattern}: needs manual review (tool_call={has_tool})"
        )

    # Context efficiency anti-patterns
    if pattern in ("excessive_limit", "no_limit_reasoning", "loads_unnecessary_content",
                   "no_efficiency_reasoning", "ignores_temporal_scope"):
        limit_value = _get_limit_value(parsed)
        is_excessive = limit_value is not None and limit_value > 200
        passed = not is_excessive
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="limit <= 200",
            actual=f"limit={limit_value}" if limit_value else "no limit",
            passed=passed,
            message=f"Anti-pattern {pattern}: {'PASS (avoided)' if passed else 'FAIL'} - limit={limit_value}"
        )

    # Execute prompt usage anti-patterns
    if pattern in ("creates_shallow_content_directly", "handles_complex_task_without_delegation",
                   "gives_legal_advice_without_delegation", "superficial_marketing_plan",
                   "superficial_comparison", "oversimplified_license_advice"):
        tool_name = parsed.first_tool_call.name if parsed.first_tool_call else None
        uses_shallow_tool = tool_name in ("contentManager_createContent", "contentManager_writeContent")
        passed = not uses_shallow_tool
        return BehaviorIssue(
            check=f"anti:{pattern}",
            expected="should use agentManager_executePrompt",
            actual=f"tool={tool_name}",
            passed=passed,
            message=f"Anti-pattern {pattern}: {'PASS (avoided)' if passed else 'FAIL'} - tool={tool_name}"
        )

    return None


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _get_limit_value(parsed: ParsedResponse) -> Optional[int]:
    """Extract limit parameter value from parsed response."""
    limit = parsed.get_argument("limit")
    if limit is not None:
        try:
            return int(limit)
        except (ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# User Input Detection
# ---------------------------------------------------------------------------

# Comprehensive keywords for detecting "asks for user input" patterns
_USER_INPUT_KEYWORDS = [
    # Direct questions - modal verbs
    "would you like", "would you prefer", "would you want", "would you rather",
    "do you want", "do you need", "do you prefer",
    "should i", "shall i", "can i", "may i", "could i", "could you", "can you", "will you",

    # Question starters
    "which one", "which file", "which folder", "which option", "which would", "which do you", "which should",
    "what would you", "what do you", "what should",
    "where would you", "where should", "where do you",
    "how would you", "how should", "how do you",
    "when should", "when would",

    # Request phrases
    "let me know", "please let me know", "please specify", "please confirm", "please tell me",
    "please indicate", "please select", "please choose", "please clarify", "please provide",
    "kindly specify", "kindly confirm", "kindly let me know",

    # Choice/selection language
    "your choice", "your preference", "your decision",
    "you choose", "you select", "you pick", "you decide", "you prefer",
    "prefer to", "choice between", "choose between", "select from", "pick from", "decide between",
    "option to", "options are", "options include", "alternatives are", "alternatives include",

    # Confirmation requests
    "confirm that", "confirm if", "confirm whether",
    "verify that", "verify if",
    "clarify what", "clarify which", "clarify if",
    "specify which", "specify what", "specify the",
    "indicate which", "indicate what", "indicate your",

    # Conditional offers
    "if you'd like", "if you would like", "if you want", "if you prefer",
    "if you need", "if you wish", "if that works", "if that's okay", "if that sounds good",

    # Alternative suggestions
    "alternatively", "or would you", "or should i", "or i could", "or i can",
    "or do you", "or perhaps", "otherwise", "instead", "on the other hand",

    # Offers of assistance
    "i can also", "i could also", "i'm able to", "i am able to",
    "i'd be happy to", "i would be happy to", "happy to help", "glad to help",

    # Waiting for input
    "waiting for", "awaiting your", "ready when you", "whenever you're ready",
    "when you're ready", "at your convenience", "up to you", "your call", "you decide",

    # Numbered/listed options indicators
    "option 1", "option 2", "choice 1", "choice 2",
    "1.", "2.", "a)", "b)", "(1)", "(2)",
    "first option", "second option", "either", "or",

    # Direct ask patterns
    "what next", "what now", "what else", "anything else", "something else", "any other",
    "more help", "further assistance", "need anything", "want me to", "like me to", "need me to",

    # Uncertainty acknowledgment + ask
    "not sure which", "unclear which", "ambiguous",
    "multiple options", "several options", "few options", "different options", "various options",
]


def _check_asks_for_input(text: str) -> bool:
    """Check if text contains patterns indicating the model is asking for user input.

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
