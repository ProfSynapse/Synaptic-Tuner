"""Config-driven validation system.

This module provides a generic validator that reads all validation rules
from YAML config files. No tool names, behaviors, or response types are
hardcoded in Python.

Usage:
    validator = ConfigDrivenValidator(config_dir="Evaluator/config")
    result = validator.validate(response, test_case)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import yaml


@dataclass
class ValidationIssue:
    """A single validation issue."""
    level: str  # "error", "warn", "info"
    check: str  # Which check produced this
    message: str
    expected: Any = None
    actual: Any = None


@dataclass
class ParsedToolCall:
    """A parsed tool call from the response."""
    name: str  # Full tool name (e.g., "storageManager_move")
    agent: Optional[str] = None  # Agent part (e.g., "storageManager")
    tool: Optional[str] = None  # Tool part (e.g., "move")
    params: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedResponse:
    """A parsed response with extracted components."""
    raw: str
    text_content: str  # Non-tool-call text
    tool_calls: List[ParsedToolCall] = field(default_factory=list)
    wrapper_name: Optional[str] = None  # e.g., "useTools"
    batch_strategy: Optional[str] = None  # "serial" or "parallel"


@dataclass
class ValidationResult:
    """Result of validating a response against a test case."""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    parsed: Optional[ParsedResponse] = None
    checks_run: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [
                {
                    "level": i.level,
                    "check": i.check,
                    "message": i.message,
                    "expected": i.expected,
                    "actual": i.actual,
                }
                for i in self.issues
            ],
            "checks_run": self.checks_run,
        }


class ConfigDrivenValidator:
    """Validates responses based on YAML configuration files."""

    def __init__(self, config_dir: Union[str, Path]):
        """Initialize validator with config directory.

        Args:
            config_dir: Path to config directory containing YAML files
        """
        self.config_dir = Path(config_dir)
        self._load_configs()
        self._register_checks()

    def _load_configs(self) -> None:
        """Load all configuration files."""
        self.tool_schema = self._load_yaml("tool_schema.yaml")
        self.response_types = self._load_yaml("response_types.yaml")
        self.behaviors = self._load_yaml("behaviors.yaml")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML config file."""
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _register_checks(self) -> None:
        """Register all check functions by name."""
        self.check_functions: Dict[str, Callable] = {
            "tool_called": self._check_tool_called,
            "any_tool_called": self._check_any_tool_called,
            "all_tools_called": self._check_all_tools_called,
            "tool_not_called": self._check_tool_not_called,
            "tool_count": self._check_tool_count,
            "tool_params_present": self._check_tool_params_present,
            "tool_sequence": self._check_tool_sequence,
            "fields_present": self._check_fields_present,
            "fields_min_length": self._check_fields_min_length,
            "field_equals": self._check_field_equals,
            "text_contains": self._check_text_contains,
            "text_contains_any": self._check_text_contains_any,
            "text_not_contains": self._check_text_not_contains,
            "text_matches": self._check_text_matches,
            "text_min_length": self._check_text_min_length,
            "text_max_length": self._check_text_max_length,
            "batch_structure": self._check_batch_structure,
            "batch_strategy": self._check_batch_strategy,
        }

    # =========================================================
    # MAIN VALIDATION
    # =========================================================

    def validate(
        self,
        response: Union[str, Dict[str, Any]],
        test_case: Dict[str, Any],
    ) -> ValidationResult:
        """Validate a response against a test case.

        Args:
            response: Model response (string or dict with tool_calls)
            test_case: Test case with expectations

        Returns:
            ValidationResult with pass/fail and issues
        """
        issues: List[ValidationIssue] = []
        checks_run: List[str] = []

        # Parse the response
        parsed = self.parse_response(response)

        # Get expectations
        expect = test_case.get("expect", {})

        # Check expected tool
        if "tool" in expect:
            checks_run.append("expected_tool")
            self._validate_expected_tool(parsed, expect["tool"], issues)

        # Check acceptable tools (OR logic)
        if "acceptable" in expect:
            checks_run.append("acceptable_tools")
            self._validate_acceptable(parsed, expect["acceptable"], issues)

        # Check params
        if "params_include" in expect:
            checks_run.append("params_include")
            self._validate_params_include(parsed, expect["params_include"], issues)

        # Check response type
        if "response_type" in expect:
            checks_run.append("response_type")
            self._validate_response_type(parsed, expect["response_type"], issues)

        # Check tool sequence
        if "first_tool" in expect or "first_tool_any_of" in expect:
            checks_run.append("tool_sequence")
            self._validate_tool_sequence(parsed, expect, issues)

        # Check behaviors
        for behavior in test_case.get("behaviors", []):
            checks_run.append(f"behavior:{behavior}")
            self._validate_behavior(parsed, behavior, issues)

        # Determine overall pass/fail
        has_errors = any(i.level == "error" for i in issues)

        return ValidationResult(
            passed=not has_errors,
            issues=issues,
            parsed=parsed,
            checks_run=checks_run,
        )

    # =========================================================
    # RESPONSE PARSING
    # =========================================================

    def parse_response(self, response: Union[str, Dict[str, Any]]) -> ParsedResponse:
        """Parse a response into structured components.

        Handles multiple formats:
        - String with tool_call: markers
        - String with [TOOL_CALLS] markers (Mistral)
        - String with <tool_call> tags (Qwen)
        - Dict with tool_calls array (OpenAI)
        """
        if isinstance(response, dict):
            return self._parse_dict_response(response)
        return self._parse_string_response(str(response))

    def _parse_dict_response(self, response: Dict[str, Any]) -> ParsedResponse:
        """Parse OpenAI-style dict response."""
        content = response.get("content", "")
        tool_calls_raw = response.get("tool_calls", [])

        tool_calls = []
        for tc in tool_calls_raw:
            if isinstance(tc, dict):
                func = tc.get("function", tc)
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                parsed_tc = self._expand_tool_call(name, args)
                tool_calls.extend(parsed_tc)

        return ParsedResponse(
            raw=str(response),
            text_content=content,
            tool_calls=tool_calls,
        )

    def _parse_string_response(self, response: str) -> ParsedResponse:
        """Parse string response with various tool call formats."""
        tool_calls = []
        text_content = response

        # Try to extract tool calls based on configured wrapper
        wrapper = self.tool_schema.get("tool_format", {}).get("wrapper")

        if wrapper:
            # Look for wrapper-based tool calls
            pattern = rf'{wrapper}\s*\((.*?)\)'
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    args = json.loads(match)
                    parsed = self._expand_tool_call(wrapper, args)
                    tool_calls.extend(parsed)
                    # Remove from text content
                    text_content = re.sub(
                        rf'{wrapper}\s*\({re.escape(match)}\)',
                        '',
                        text_content
                    ).strip()
                except json.JSONDecodeError:
                    pass

        # Try ChatML format: tool_call: name\narguments: {...}
        if not tool_calls and "tool_call:" in response:
            pattern = r'tool_call:\s*(\w+)\s*\narguments:\s*({.*?})'
            for match in re.finditer(pattern, response, re.DOTALL):
                name = match.group(1)
                try:
                    args = json.loads(match.group(2))
                    tool_calls.append(ParsedToolCall(
                        name=name,
                        params=args,
                    ))
                except json.JSONDecodeError:
                    pass

        # Try Qwen format: <tool_call>...</tool_call>
        if not tool_calls and "<tool_call>" in response:
            pattern = r'<tool_call>(.*?)</tool_call>'
            for match in re.finditer(pattern, response, re.DOTALL):
                try:
                    data = json.loads(match.group(1))
                    name = data.get("name", "")
                    args = data.get("arguments", {})
                    parsed = self._expand_tool_call(name, args)
                    tool_calls.extend(parsed)
                except json.JSONDecodeError:
                    pass

        # Try Mistral format: [TOOL_CALLS][...]
        if not tool_calls and "[TOOL_CALLS]" in response:
            pattern = r'\[TOOL_CALLS\]\s*(\[.*?\])'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    calls = json.loads(match.group(1))
                    for call in calls:
                        name = call.get("name", "")
                        args = call.get("arguments", {})
                        parsed = self._expand_tool_call(name, args)
                        tool_calls.extend(parsed)
                except json.JSONDecodeError:
                    pass

        return ParsedResponse(
            raw=response,
            text_content=text_content,
            tool_calls=tool_calls,
        )

    def _expand_tool_call(
        self,
        name: str,
        args: Dict[str, Any]
    ) -> List[ParsedToolCall]:
        """Expand a tool call, handling wrapper format.

        If using a wrapper like useTools, expands the calls array
        into individual ParsedToolCall objects.
        """
        wrapper = self.tool_schema.get("tool_format", {}).get("wrapper")

        if name == wrapper and "calls" in args:
            # Expand wrapper into individual calls
            result = []
            context = args.get("context", {})
            calls = args.get("calls", [])

            for call in calls:
                agent = call.get("agent", "")
                tool = call.get("tool", "")
                params = call.get("params", {})

                if agent and tool:
                    full_name = f"{agent}_{tool}"
                    result.append(ParsedToolCall(
                        name=full_name,
                        agent=agent,
                        tool=tool,
                        params=params,
                        context=context,
                    ))

            return result if result else [ParsedToolCall(name=name, params=args)]

        # Non-wrapper tool call
        return [ParsedToolCall(name=name, params=args)]

    # =========================================================
    # VALIDATION METHODS
    # =========================================================

    def _validate_expected_tool(
        self,
        parsed: ParsedResponse,
        expected_tool: str,
        issues: List[ValidationIssue],
    ) -> None:
        """Validate that the expected tool was called."""
        called_tools = {tc.name for tc in parsed.tool_calls}

        if expected_tool not in called_tools:
            issues.append(ValidationIssue(
                level="error",
                check="expected_tool",
                message=f"Expected tool '{expected_tool}' was not called",
                expected=expected_tool,
                actual=list(called_tools) if called_tools else "no tools called",
            ))

    def _validate_acceptable(
        self,
        parsed: ParsedResponse,
        acceptable: List[Dict[str, Any]],
        issues: List[ValidationIssue],
    ) -> None:
        """Validate that one of the acceptable options was met."""
        called_tools = {tc.name for tc in parsed.tool_calls}

        for option in acceptable:
            # Check if this option matches
            if "tool" in option:
                if option["tool"] in called_tools:
                    return  # Passed

            if "response_type" in option:
                if self._matches_response_type(parsed, option["response_type"]):
                    return  # Passed

            if "pseudo_tool" in option and option["pseudo_tool"] == "TEXT_ONLY":
                if not called_tools:
                    return  # Text-only is acceptable

        # None matched
        issues.append(ValidationIssue(
            level="error",
            check="acceptable_tools",
            message="No acceptable option was satisfied",
            expected=acceptable,
            actual={
                "tools_called": list(called_tools),
                "has_text": bool(parsed.text_content.strip()),
            },
        ))

    def _validate_params_include(
        self,
        parsed: ParsedResponse,
        expected_params: Dict[str, Any],
        issues: List[ValidationIssue],
    ) -> None:
        """Validate that tool calls include expected parameters."""
        for tc in parsed.tool_calls:
            for key, expected_value in expected_params.items():
                actual_value = tc.params.get(key)
                if actual_value is None:
                    issues.append(ValidationIssue(
                        level="error",
                        check="params_include",
                        message=f"Missing parameter '{key}' in {tc.name}",
                        expected=expected_value,
                        actual=None,
                    ))
                elif actual_value != expected_value:
                    # Check if it's a partial match (e.g., path contains expected)
                    if not self._values_match(expected_value, actual_value):
                        issues.append(ValidationIssue(
                            level="warn",
                            check="params_include",
                            message=f"Parameter '{key}' value mismatch in {tc.name}",
                            expected=expected_value,
                            actual=actual_value,
                        ))

    def _values_match(self, expected: Any, actual: Any) -> bool:
        """Check if values match (with some flexibility)."""
        if expected == actual:
            return True
        # String containment
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.lower() in actual.lower()
        return False

    def _validate_response_type(
        self,
        parsed: ParsedResponse,
        expected_type: str,
        issues: List[ValidationIssue],
    ) -> None:
        """Validate response matches expected type."""
        if not self._matches_response_type(parsed, expected_type):
            issues.append(ValidationIssue(
                level="error",
                check="response_type",
                message=f"Response does not match type '{expected_type}'",
                expected=expected_type,
                actual=self._describe_response(parsed),
            ))

    def _matches_response_type(self, parsed: ParsedResponse, type_name: str) -> bool:
        """Check if response matches a response type definition."""
        type_def = self.response_types.get("response_types", {}).get(type_name, {})
        if not type_def:
            # Check aliases
            type_name = self.response_types.get("aliases", {}).get(type_name, type_name)
            type_def = self.response_types.get("response_types", {}).get(type_name, {})

        if not type_def:
            return False

        reqs = type_def.get("requirements", {})

        # Check has_tool_calls
        if "has_tool_calls" in reqs:
            has_calls = len(parsed.tool_calls) > 0
            if reqs["has_tool_calls"] != has_calls:
                return False

        # Check text_content requirements
        text_reqs = reqs.get("text_content", {})
        text_len = len(parsed.text_content.strip())

        if "min_length" in text_reqs:
            if text_len < text_reqs["min_length"]:
                return False

        if "max_length" in text_reqs:
            if text_len > text_reqs["max_length"]:
                return False

        if "contains_any" in text_reqs:
            text_lower = parsed.text_content.lower()
            if not any(p.lower() in text_lower for p in text_reqs["contains_any"]):
                return False

        # Check min_tool_calls
        if "min_tool_calls" in reqs:
            if len(parsed.tool_calls) < reqs["min_tool_calls"]:
                return False

        return True

    def _describe_response(self, parsed: ParsedResponse) -> str:
        """Create a description of the response for error messages."""
        parts = []
        if parsed.tool_calls:
            tools = [tc.name for tc in parsed.tool_calls]
            parts.append(f"tools: {tools}")
        text_len = len(parsed.text_content.strip())
        parts.append(f"text_length: {text_len}")
        return ", ".join(parts)

    def _validate_tool_sequence(
        self,
        parsed: ParsedResponse,
        expect: Dict[str, Any],
        issues: List[ValidationIssue],
    ) -> None:
        """Validate tool call ordering."""
        if not parsed.tool_calls:
            issues.append(ValidationIssue(
                level="error",
                check="tool_sequence",
                message="No tool calls to check sequence",
                expected="at least one tool call",
                actual="no tool calls",
            ))
            return

        first_tool = parsed.tool_calls[0].name

        # Check first_tool
        if "first_tool" in expect:
            if first_tool != expect["first_tool"]:
                issues.append(ValidationIssue(
                    level="error",
                    check="tool_sequence",
                    message=f"First tool should be '{expect['first_tool']}'",
                    expected=expect["first_tool"],
                    actual=first_tool,
                ))

        # Check first_tool_any_of
        if "first_tool_any_of" in expect:
            valid_first = expect["first_tool_any_of"]
            if first_tool not in valid_first:
                issues.append(ValidationIssue(
                    level="error",
                    check="tool_sequence",
                    message=f"First tool should be one of {valid_first}",
                    expected=valid_first,
                    actual=first_tool,
                ))

        # Check not_first
        if "not_first" in expect:
            forbidden_first = expect["not_first"]
            if first_tool in forbidden_first:
                issues.append(ValidationIssue(
                    level="error",
                    check="tool_sequence",
                    message=f"Tool '{first_tool}' should not be called first",
                    expected=f"not in {forbidden_first}",
                    actual=first_tool,
                ))

    def _validate_behavior(
        self,
        parsed: ParsedResponse,
        behavior: str,
        issues: List[ValidationIssue],
    ) -> None:
        """Validate a behavior check from config."""
        # Handle OR syntax: "behavior1 OR behavior2"
        if " OR " in behavior:
            sub_behaviors = [b.strip() for b in behavior.split(" OR ")]
            sub_issues: List[List[ValidationIssue]] = []

            for sub_behavior in sub_behaviors:
                sub_issue_list: List[ValidationIssue] = []
                self._validate_single_behavior(parsed, sub_behavior, sub_issue_list)
                sub_issues.append(sub_issue_list)

            # If any sub-behavior passed (no errors), the OR passes
            if any(
                not any(i.level == "error" for i in si)
                for si in sub_issues
            ):
                return  # At least one passed

            # All failed - report
            issues.append(ValidationIssue(
                level="error",
                check=f"behavior:{behavior}",
                message=f"None of the behaviors passed: {sub_behaviors}",
            ))
            return

        self._validate_single_behavior(parsed, behavior, issues)

    def _validate_single_behavior(
        self,
        parsed: ParsedResponse,
        behavior: str,
        issues: List[ValidationIssue],
    ) -> None:
        """Validate a single behavior (no OR)."""
        # Handle parameterized behaviors: uses_tool(storageManager_move)
        param_match = re.match(r'(\w+)\(([^)]+)\)', behavior)
        if param_match:
            behavior_name = param_match.group(1)
            param_value = param_match.group(2)
        else:
            behavior_name = behavior
            param_value = None

        # Look up behavior definition
        behavior_def = self.behaviors.get("behaviors", {}).get(behavior_name)
        if not behavior_def:
            issues.append(ValidationIssue(
                level="warn",
                check=f"behavior:{behavior}",
                message=f"Unknown behavior '{behavior_name}'",
            ))
            return

        # Get check type and params
        check_type = behavior_def.get("check")
        params = dict(behavior_def.get("params", {}))

        # Substitute parameterized value
        if param_value:
            for key, val in params.items():
                if isinstance(val, str) and "{" in val:
                    params[key] = param_value

        # Run the check
        check_func = self.check_functions.get(check_type)
        if check_func:
            check_func(parsed, params, issues, behavior_name)
        else:
            issues.append(ValidationIssue(
                level="warn",
                check=f"behavior:{behavior}",
                message=f"Unknown check type '{check_type}'",
            ))

    # =========================================================
    # CHECK IMPLEMENTATIONS
    # =========================================================

    def _check_tool_called(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check if a specific tool was called."""
        tool = params.get("tool", "")
        called = {tc.name for tc in parsed.tool_calls}

        if tool not in called:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Tool '{tool}' was not called",
                expected=tool,
                actual=list(called),
            ))

    def _check_any_tool_called(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check if any of specified tools was called."""
        tools = params.get("tools", [])
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(",")]

        called = {tc.name for tc in parsed.tool_calls}

        if not called.intersection(set(tools)):
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"None of tools {tools} were called",
                expected=tools,
                actual=list(called),
            ))

    def _check_all_tools_called(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check if all specified tools were called."""
        tools = params.get("tools", [])
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(",")]

        called = {tc.name for tc in parsed.tool_calls}
        missing = set(tools) - called

        if missing:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Tools not called: {list(missing)}",
                expected=tools,
                actual=list(called),
            ))

    def _check_tool_not_called(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check that a tool was NOT called."""
        tool = params.get("tool", "")
        called = {tc.name for tc in parsed.tool_calls}

        if tool in called:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Tool '{tool}' should not be called",
                expected=f"not {tool}",
                actual=list(called),
            ))

    def _check_tool_count(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check number of tool calls."""
        count = len(parsed.tool_calls)
        min_count = params.get("min")
        max_count = params.get("max")

        if min_count is not None and count < min_count:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Too few tool calls: {count} < {min_count}",
                expected=f">= {min_count}",
                actual=count,
            ))

        if max_count is not None and count > max_count:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Too many tool calls: {count} > {max_count}",
                expected=f"<= {max_count}",
                actual=count,
            ))

    def _check_tool_params_present(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check tool has required params."""
        tool = params.get("tool", "")
        required = params.get("required_params", [])

        for tc in parsed.tool_calls:
            if tc.name == tool:
                missing = [p for p in required if p not in tc.params]
                if missing:
                    issues.append(ValidationIssue(
                        level="error",
                        check=check_name,
                        message=f"Missing params in {tool}: {missing}",
                        expected=required,
                        actual=list(tc.params.keys()),
                    ))
                return

    def _check_tool_sequence(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check tool call ordering."""
        if not parsed.tool_calls:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message="No tool calls",
            ))
            return

        first_tool = parsed.tool_calls[0].name
        first_any_of = params.get("first_any_of", [])
        not_first = params.get("not_first", [])

        if first_any_of and first_tool not in first_any_of:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"First tool should be one of {first_any_of}",
                expected=first_any_of,
                actual=first_tool,
            ))

        if not_first and first_tool in not_first:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Tool '{first_tool}' should not be first",
                expected=f"not in {not_first}",
                actual=first_tool,
            ))

    def _check_fields_present(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check fields exist in object."""
        location = params.get("in", "context")
        fields = params.get("fields", [])

        for tc in parsed.tool_calls:
            if location == "context":
                obj = tc.context
            elif location == "params":
                obj = tc.params
            else:
                obj = {}

            missing = [f for f in fields if f not in obj]
            if missing:
                issues.append(ValidationIssue(
                    level="error",
                    check=check_name,
                    message=f"Missing {location} fields: {missing}",
                    expected=fields,
                    actual=list(obj.keys()),
                ))

    def _check_fields_min_length(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check field values have min length."""
        location = params.get("in", "context")
        field_mins = params.get("fields", {})

        for tc in parsed.tool_calls:
            obj = tc.context if location == "context" else tc.params

            for field, min_len in field_mins.items():
                value = obj.get(field, "")
                if len(str(value)) < min_len:
                    issues.append(ValidationIssue(
                        level="error",
                        check=check_name,
                        message=f"Field '{field}' too short: {len(str(value))} < {min_len}",
                        expected=f">= {min_len}",
                        actual=len(str(value)),
                    ))

    def _check_field_equals(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check field equals expected value."""
        location = params.get("in", "context")
        field = params.get("field", "")
        expected = params.get("value", "")

        for tc in parsed.tool_calls:
            obj = tc.context if location == "context" else tc.params
            actual = obj.get(field)

            if actual != expected:
                issues.append(ValidationIssue(
                    level="error",
                    check=check_name,
                    message=f"Field '{field}' mismatch",
                    expected=expected,
                    actual=actual,
                ))

    def _check_text_contains(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check text contains substring."""
        text = params.get("text", "")
        if text.lower() not in parsed.text_content.lower():
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Text does not contain '{text}'",
                expected=text,
                actual=parsed.text_content[:100] + "...",
            ))

    def _check_text_contains_any(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check text contains any of patterns."""
        patterns = params.get("patterns", [])
        text_lower = parsed.text_content.lower()

        if not any(p.lower() in text_lower for p in patterns):
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Text does not contain any of {patterns}",
                expected=f"any of {patterns}",
                actual=parsed.text_content[:100] + "...",
            ))

    def _check_text_not_contains(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check text does NOT contain patterns."""
        patterns = params.get("patterns", [])
        text_lower = parsed.text_content.lower()

        found = [p for p in patterns if p.lower() in text_lower]
        if found:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Text contains forbidden patterns: {found}",
                expected=f"none of {patterns}",
                actual=found,
            ))

    def _check_text_matches(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check text matches regex."""
        pattern = params.get("pattern", "")
        if not re.search(pattern, parsed.text_content):
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Text does not match pattern '{pattern}'",
                expected=pattern,
                actual=parsed.text_content[:100] + "...",
            ))

    def _check_text_min_length(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check text meets min length."""
        min_len = params.get("min", 0)
        actual_len = len(parsed.text_content.strip())

        if actual_len < min_len:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Text too short: {actual_len} < {min_len}",
                expected=f">= {min_len}",
                actual=actual_len,
            ))

    def _check_text_max_length(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check text under max length."""
        max_len = params.get("max", float('inf'))
        actual_len = len(parsed.text_content.strip())

        if actual_len > max_len:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Text too long: {actual_len} > {max_len}",
                expected=f"<= {max_len}",
                actual=actual_len,
            ))

    def _check_batch_structure(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check batch operation structure."""
        min_calls = params.get("min_calls", 2)

        if len(parsed.tool_calls) < min_calls:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Batch should have >= {min_calls} calls",
                expected=f">= {min_calls}",
                actual=len(parsed.tool_calls),
            ))

    def _check_batch_strategy(
        self,
        parsed: ParsedResponse,
        params: Dict[str, Any],
        issues: List[ValidationIssue],
        check_name: str,
    ) -> None:
        """Check batch uses specific strategy."""
        expected_strategy = params.get("strategy", "serial")

        if parsed.batch_strategy and parsed.batch_strategy != expected_strategy:
            issues.append(ValidationIssue(
                level="error",
                check=check_name,
                message=f"Batch strategy mismatch",
                expected=expected_strategy,
                actual=parsed.batch_strategy,
            ))


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_all_valid_tools(config_dir: Union[str, Path]) -> Set[str]:
    """Get all valid tool names from config."""
    config_path = Path(config_dir) / "tool_schema.yaml"
    if not config_path.exists():
        return set()

    with open(config_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f) or {}

    tools = set()
    for agent, agent_tools in schema.get("tools", {}).items():
        for tool_def in agent_tools:
            tool_name = tool_def.get("name", "")
            if tool_name:
                tools.add(f"{agent}_{tool_name}")

    return tools
