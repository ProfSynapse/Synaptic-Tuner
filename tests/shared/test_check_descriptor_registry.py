"""
Tests for the CheckDescriptor registry in Evaluator/config_validator.py.

Verifies:
- All 18 check descriptors are registered
- Each runner function produces correct results for valid and invalid inputs
- tool_sequence composite logic (first_any_of + not_first)
- ConfigDrivenValidator._register_checks() populates check_functions from registry
- Unknown check types fall through gracefully
- Edge cases: empty tool calls, empty text, None params
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from Evaluator.config_validator import (
    CHECK_REGISTRY,
    CheckDescriptor,
    ConfigDrivenValidator,
    ParsedResponse,
    ParsedToolCall,
    ValidationIssue,
)


# =========================================================
# Helpers
# =========================================================

def _parsed(
    text: str = "",
    tool_calls: list | None = None,
    batch_strategy: str | None = None,
) -> ParsedResponse:
    """Build a ParsedResponse for testing."""
    return ParsedResponse(
        raw=text,
        text_content=text,
        tool_calls=tool_calls or [],
        batch_strategy=batch_strategy,
    )


def _tool_call(name: str, params: dict | None = None, context: dict | None = None) -> ParsedToolCall:
    return ParsedToolCall(
        name=name,
        params=params or {},
        context=context or {},
    )


def _run_check(check_name: str, parsed: ParsedResponse, params: dict | None = None) -> List[ValidationIssue]:
    """Run a single check from the registry and return issues."""
    desc = CHECK_REGISTRY[check_name]
    issues: List[ValidationIssue] = []
    desc.run(parsed, params or {}, issues, check_name)
    return issues


# =========================================================
# Registry Structure
# =========================================================

class TestRegistryStructure:
    EXPECTED_CHECKS = {
        "tool_called", "any_tool_called", "all_tools_called", "tool_not_called",
        "tool_count", "tool_params_present", "tool_sequence",
        "fields_present", "fields_min_length", "field_equals",
        "text_contains", "text_contains_any", "text_not_contains",
        "text_matches", "text_min_length", "text_max_length",
        "batch_structure", "batch_strategy",
    }

    def test_registry_has_all_18_checks(self):
        assert set(CHECK_REGISTRY.keys()) == self.EXPECTED_CHECKS

    def test_registry_count(self):
        assert len(CHECK_REGISTRY) == 18

    def test_all_descriptors_have_callable_run(self):
        for name, desc in CHECK_REGISTRY.items():
            assert isinstance(desc, CheckDescriptor), f"{name} is not a CheckDescriptor"
            assert callable(desc.run), f"{name}.run is not callable"

    def test_descriptor_names_match_keys(self):
        for key, desc in CHECK_REGISTRY.items():
            assert desc.name == key, f"Key '{key}' != desc.name '{desc.name}'"


# =========================================================
# ConfigDrivenValidator Integration
# =========================================================

class TestValidatorRegistration:
    def test_register_checks_populates_from_registry(self, tmp_path):
        """_register_checks should populate check_functions from CHECK_REGISTRY."""
        # Create minimal config dir
        (tmp_path / "tool_schema.yaml").write_text("{}")
        (tmp_path / "response_types.yaml").write_text("{}")
        (tmp_path / "behaviors.yaml").write_text("{}")

        validator = ConfigDrivenValidator(tmp_path)
        assert len(validator.check_functions) == len(CHECK_REGISTRY)
        for name in CHECK_REGISTRY:
            assert name in validator.check_functions

    def test_unknown_check_type_emits_warning(self, tmp_path):
        """When a behavior references an unknown check type, it should warn."""
        (tmp_path / "tool_schema.yaml").write_text("{}")
        (tmp_path / "response_types.yaml").write_text("{}")
        import yaml
        behaviors = {
            "behaviors": {
                "test_unknown": {
                    "check": "nonexistent_check_type",
                    "params": {},
                }
            }
        }
        (tmp_path / "behaviors.yaml").write_text(yaml.dump(behaviors))

        validator = ConfigDrivenValidator(tmp_path)
        result = validator.validate(
            "some response text",
            {"behaviors": ["test_unknown"]},
        )
        # Should have a warning about unknown check type
        warns = [i for i in result.issues if i.level == "warn"]
        assert any("nonexistent_check_type" in i.message for i in warns)


# =========================================================
# Tool Check Runners
# =========================================================

class TestToolCalledCheck:
    def test_passes_when_tool_is_called(self):
        parsed = _parsed(tool_calls=[_tool_call("search")])
        issues = _run_check("tool_called", parsed, {"tool": "search"})
        assert len(issues) == 0

    def test_fails_when_tool_not_called(self):
        parsed = _parsed(tool_calls=[_tool_call("other")])
        issues = _run_check("tool_called", parsed, {"tool": "search"})
        assert len(issues) == 1
        assert "search" in issues[0].message

    def test_fails_when_no_tool_calls(self):
        parsed = _parsed(tool_calls=[])
        issues = _run_check("tool_called", parsed, {"tool": "search"})
        assert len(issues) == 1


class TestAnyToolCalledCheck:
    def test_passes_when_one_of_tools_called(self):
        parsed = _parsed(tool_calls=[_tool_call("search")])
        issues = _run_check("any_tool_called", parsed, {"tools": ["search", "browse"]})
        assert len(issues) == 0

    def test_fails_when_none_called(self):
        parsed = _parsed(tool_calls=[_tool_call("other")])
        issues = _run_check("any_tool_called", parsed, {"tools": ["search", "browse"]})
        assert len(issues) == 1

    def test_handles_comma_separated_string(self):
        parsed = _parsed(tool_calls=[_tool_call("browse")])
        issues = _run_check("any_tool_called", parsed, {"tools": "search, browse"})
        assert len(issues) == 0


class TestAllToolsCalledCheck:
    def test_passes_when_all_tools_called(self):
        parsed = _parsed(tool_calls=[_tool_call("search"), _tool_call("browse")])
        issues = _run_check("all_tools_called", parsed, {"tools": ["search", "browse"]})
        assert len(issues) == 0

    def test_fails_when_one_missing(self):
        parsed = _parsed(tool_calls=[_tool_call("search")])
        issues = _run_check("all_tools_called", parsed, {"tools": ["search", "browse"]})
        assert len(issues) == 1
        assert "browse" in issues[0].message

    def test_handles_comma_separated_string(self):
        parsed = _parsed(tool_calls=[_tool_call("a"), _tool_call("b")])
        issues = _run_check("all_tools_called", parsed, {"tools": "a, b"})
        assert len(issues) == 0


class TestToolNotCalledCheck:
    def test_passes_when_tool_absent(self):
        parsed = _parsed(tool_calls=[_tool_call("other")])
        issues = _run_check("tool_not_called", parsed, {"tool": "forbidden"})
        assert len(issues) == 0

    def test_fails_when_tool_present(self):
        parsed = _parsed(tool_calls=[_tool_call("forbidden")])
        issues = _run_check("tool_not_called", parsed, {"tool": "forbidden"})
        assert len(issues) == 1


class TestToolCountCheck:
    def test_passes_within_bounds(self):
        parsed = _parsed(tool_calls=[_tool_call("a"), _tool_call("b")])
        issues = _run_check("tool_count", parsed, {"min": 1, "max": 3})
        assert len(issues) == 0

    def test_fails_below_min(self):
        parsed = _parsed(tool_calls=[])
        issues = _run_check("tool_count", parsed, {"min": 1})
        assert len(issues) == 1
        assert "Too few" in issues[0].message

    def test_fails_above_max(self):
        parsed = _parsed(tool_calls=[_tool_call("a")] * 5)
        issues = _run_check("tool_count", parsed, {"max": 3})
        assert len(issues) == 1
        assert "Too many" in issues[0].message

    def test_no_bounds_always_passes(self):
        parsed = _parsed(tool_calls=[_tool_call("a")] * 10)
        issues = _run_check("tool_count", parsed, {})
        assert len(issues) == 0


class TestToolParamsPresentCheck:
    def test_passes_with_all_params(self):
        parsed = _parsed(tool_calls=[_tool_call("search", params={"query": "test", "limit": 10})])
        issues = _run_check("tool_params_present", parsed, {"tool": "search", "required_params": ["query", "limit"]})
        assert len(issues) == 0

    def test_fails_with_missing_params(self):
        parsed = _parsed(tool_calls=[_tool_call("search", params={"query": "test"})])
        issues = _run_check("tool_params_present", parsed, {"tool": "search", "required_params": ["query", "limit"]})
        assert len(issues) == 1
        assert "limit" in issues[0].message

    def test_ignores_wrong_tool(self):
        """If the named tool isn't in tool_calls, no issue emitted (tool might not exist)."""
        parsed = _parsed(tool_calls=[_tool_call("other", params={})])
        issues = _run_check("tool_params_present", parsed, {"tool": "search", "required_params": ["query"]})
        assert len(issues) == 0


# =========================================================
# Tool Sequence (Composite Check - HIGH priority)
# =========================================================

class TestToolSequenceCheck:
    def test_no_tool_calls_emits_error(self):
        parsed = _parsed(tool_calls=[])
        issues = _run_check("tool_sequence", parsed, {"first_any_of": ["search"]})
        assert len(issues) == 1
        assert "No tool calls" in issues[0].message

    def test_first_any_of_passes(self):
        parsed = _parsed(tool_calls=[_tool_call("search"), _tool_call("browse")])
        issues = _run_check("tool_sequence", parsed, {"first_any_of": ["search", "init"]})
        assert len(issues) == 0

    def test_first_any_of_fails(self):
        parsed = _parsed(tool_calls=[_tool_call("browse"), _tool_call("search")])
        issues = _run_check("tool_sequence", parsed, {"first_any_of": ["search", "init"]})
        assert len(issues) == 1
        assert "First tool" in issues[0].message

    def test_not_first_passes(self):
        parsed = _parsed(tool_calls=[_tool_call("search"), _tool_call("forbidden")])
        issues = _run_check("tool_sequence", parsed, {"not_first": ["forbidden"]})
        assert len(issues) == 0

    def test_not_first_fails(self):
        parsed = _parsed(tool_calls=[_tool_call("forbidden"), _tool_call("search")])
        issues = _run_check("tool_sequence", parsed, {"not_first": ["forbidden"]})
        assert len(issues) == 1
        assert "should not be first" in issues[0].message

    def test_both_constraints_pass(self):
        """Both first_any_of and not_first can pass simultaneously."""
        parsed = _parsed(tool_calls=[_tool_call("search"), _tool_call("browse")])
        issues = _run_check("tool_sequence", parsed, {
            "first_any_of": ["search"],
            "not_first": ["browse"],
        })
        assert len(issues) == 0

    def test_both_constraints_can_fail_independently(self):
        """If first tool is not in first_any_of AND is in not_first, both fire."""
        parsed = _parsed(tool_calls=[_tool_call("forbidden")])
        issues = _run_check("tool_sequence", parsed, {
            "first_any_of": ["search"],
            "not_first": ["forbidden"],
        })
        assert len(issues) == 2

    def test_empty_first_any_of_skips_check(self):
        """Empty list for first_any_of should not trigger an issue."""
        parsed = _parsed(tool_calls=[_tool_call("anything")])
        issues = _run_check("tool_sequence", parsed, {"first_any_of": []})
        assert len(issues) == 0

    def test_empty_not_first_skips_check(self):
        parsed = _parsed(tool_calls=[_tool_call("anything")])
        issues = _run_check("tool_sequence", parsed, {"not_first": []})
        assert len(issues) == 0


# =========================================================
# Field Check Runners
# =========================================================

class TestFieldsPresentCheck:
    def test_passes_with_context_fields(self):
        parsed = _parsed(tool_calls=[_tool_call("t", context={"key1": "v", "key2": "v"})])
        issues = _run_check("fields_present", parsed, {"in": "context", "fields": ["key1", "key2"]})
        assert len(issues) == 0

    def test_fails_with_missing_context_fields(self):
        parsed = _parsed(tool_calls=[_tool_call("t", context={"key1": "v"})])
        issues = _run_check("fields_present", parsed, {"in": "context", "fields": ["key1", "key2"]})
        assert len(issues) == 1
        assert "key2" in issues[0].message

    def test_checks_params_location(self):
        parsed = _parsed(tool_calls=[_tool_call("t", params={"p1": "v"})])
        issues = _run_check("fields_present", parsed, {"in": "params", "fields": ["p1"]})
        assert len(issues) == 0

    def test_defaults_to_context(self):
        parsed = _parsed(tool_calls=[_tool_call("t", context={"f": "v"})])
        issues = _run_check("fields_present", parsed, {"fields": ["f"]})
        assert len(issues) == 0


class TestFieldsMinLengthCheck:
    def test_passes_long_enough(self):
        parsed = _parsed(tool_calls=[_tool_call("t", context={"note": "hello world"})])
        issues = _run_check("fields_min_length", parsed, {"in": "context", "fields": {"note": 5}})
        assert len(issues) == 0

    def test_fails_too_short(self):
        parsed = _parsed(tool_calls=[_tool_call("t", context={"note": "hi"})])
        issues = _run_check("fields_min_length", parsed, {"in": "context", "fields": {"note": 5}})
        assert len(issues) == 1
        assert "too short" in issues[0].message.lower()


class TestFieldEqualsCheck:
    def test_passes_on_match(self):
        parsed = _parsed(tool_calls=[_tool_call("t", context={"status": "active"})])
        issues = _run_check("field_equals", parsed, {"in": "context", "field": "status", "value": "active"})
        assert len(issues) == 0

    def test_fails_on_mismatch(self):
        parsed = _parsed(tool_calls=[_tool_call("t", context={"status": "inactive"})])
        issues = _run_check("field_equals", parsed, {"in": "context", "field": "status", "value": "active"})
        assert len(issues) == 1


# =========================================================
# Text Check Runners
# =========================================================

class TestTextContainsCheck:
    def test_passes_with_match(self):
        parsed = _parsed(text="Hello World, welcome to the test")
        issues = _run_check("text_contains", parsed, {"text": "hello world"})
        assert len(issues) == 0  # case-insensitive

    def test_fails_without_match(self):
        parsed = _parsed(text="Hello World")
        issues = _run_check("text_contains", parsed, {"text": "goodbye"})
        assert len(issues) == 1


class TestTextContainsAnyCheck:
    def test_passes_with_one_match(self):
        parsed = _parsed(text="The weather is sunny today")
        issues = _run_check("text_contains_any", parsed, {"patterns": ["rainy", "sunny"]})
        assert len(issues) == 0

    def test_fails_with_no_match(self):
        parsed = _parsed(text="The weather is cloudy")
        issues = _run_check("text_contains_any", parsed, {"patterns": ["rainy", "sunny"]})
        assert len(issues) == 1


class TestTextNotContainsCheck:
    def test_passes_with_no_forbidden(self):
        parsed = _parsed(text="Safe text here")
        issues = _run_check("text_not_contains", parsed, {"patterns": ["forbidden", "banned"]})
        assert len(issues) == 0

    def test_fails_with_forbidden_pattern(self):
        parsed = _parsed(text="This has forbidden content")
        issues = _run_check("text_not_contains", parsed, {"patterns": ["forbidden", "banned"]})
        assert len(issues) == 1


class TestTextMatchesCheck:
    def test_passes_with_regex_match(self):
        parsed = _parsed(text="Order #12345 confirmed")
        issues = _run_check("text_matches", parsed, {"pattern": r"#\d+"})
        assert len(issues) == 0

    def test_fails_without_match(self):
        parsed = _parsed(text="No order number here")
        issues = _run_check("text_matches", parsed, {"pattern": r"#\d+"})
        assert len(issues) == 1


class TestTextMinLengthCheck:
    def test_passes_long_enough(self):
        parsed = _parsed(text="  This is a response with enough text  ")
        issues = _run_check("text_min_length", parsed, {"min": 10})
        assert len(issues) == 0

    def test_fails_too_short(self):
        parsed = _parsed(text="Hi")
        issues = _run_check("text_min_length", parsed, {"min": 10})
        assert len(issues) == 1

    def test_strips_whitespace_before_measuring(self):
        parsed = _parsed(text="     ")
        issues = _run_check("text_min_length", parsed, {"min": 1})
        assert len(issues) == 1  # stripped length is 0


class TestTextMaxLengthCheck:
    def test_passes_short_enough(self):
        parsed = _parsed(text="Short")
        issues = _run_check("text_max_length", parsed, {"max": 100})
        assert len(issues) == 0

    def test_fails_too_long(self):
        parsed = _parsed(text="A" * 200)
        issues = _run_check("text_max_length", parsed, {"max": 100})
        assert len(issues) == 1


# =========================================================
# Batch Check Runners
# =========================================================

class TestBatchStructureCheck:
    def test_passes_with_enough_calls(self):
        parsed = _parsed(tool_calls=[_tool_call("a"), _tool_call("b")])
        issues = _run_check("batch_structure", parsed, {"min_calls": 2})
        assert len(issues) == 0

    def test_fails_with_too_few(self):
        parsed = _parsed(tool_calls=[_tool_call("a")])
        issues = _run_check("batch_structure", parsed, {"min_calls": 2})
        assert len(issues) == 1

    def test_default_min_calls_is_2(self):
        parsed = _parsed(tool_calls=[_tool_call("a")])
        issues = _run_check("batch_structure", parsed, {})
        assert len(issues) == 1  # default min_calls=2


class TestBatchStrategyCheck:
    def test_passes_with_matching_strategy(self):
        parsed = _parsed(batch_strategy="parallel")
        issues = _run_check("batch_strategy", parsed, {"strategy": "parallel"})
        assert len(issues) == 0

    def test_fails_with_wrong_strategy(self):
        parsed = _parsed(batch_strategy="serial")
        issues = _run_check("batch_strategy", parsed, {"strategy": "parallel"})
        assert len(issues) == 1

    def test_no_issue_when_no_strategy_detected(self):
        """If parsed.batch_strategy is None, no comparison is made."""
        parsed = _parsed(batch_strategy=None)
        issues = _run_check("batch_strategy", parsed, {"strategy": "parallel"})
        assert len(issues) == 0


# =========================================================
# Edge Cases / Adversarial
# =========================================================

class TestEdgeCases:
    def test_empty_params_dict(self):
        """All checks should handle empty params without crashing."""
        parsed = _parsed(text="test", tool_calls=[_tool_call("t")])
        for name in CHECK_REGISTRY:
            issues: List[ValidationIssue] = []
            try:
                CHECK_REGISTRY[name].run(parsed, {}, issues, name)
            except Exception as e:
                pytest.fail(f"Check '{name}' crashed with empty params: {e}")

    def test_empty_tool_calls_list(self):
        """Checks that reference tool_calls should not crash on empty list."""
        parsed = _parsed(text="hello", tool_calls=[])
        # These checks reference tool_calls
        tool_checks = [
            "tool_called", "any_tool_called", "all_tools_called",
            "tool_not_called", "tool_count", "tool_params_present",
            "tool_sequence", "fields_present", "fields_min_length",
            "field_equals",
        ]
        for name in tool_checks:
            issues: List[ValidationIssue] = []
            try:
                CHECK_REGISTRY[name].run(parsed, {}, issues, name)
            except Exception as e:
                pytest.fail(f"Check '{name}' crashed with empty tool_calls: {e}")

    def test_empty_text_content(self):
        """Text checks should handle empty text gracefully."""
        parsed = _parsed(text="")
        text_checks = [
            "text_contains", "text_contains_any", "text_not_contains",
            "text_matches", "text_min_length", "text_max_length",
        ]
        for name in text_checks:
            issues: List[ValidationIssue] = []
            try:
                CHECK_REGISTRY[name].run(parsed, {}, issues, name)
            except Exception as e:
                pytest.fail(f"Check '{name}' crashed with empty text: {e}")
