"""Tests for SynthChat.template_utils — pure utility functions."""
from __future__ import annotations

import pytest
from SynthChat.template_utils import (
    _clean_path,
    _deep_merge_dicts,
    _make_json_safe,
    _render_template_object,
    _task_context_template_vars,
    _user_generation_style_instructions,
)


# ---- _deep_merge_dicts ----

class TestDeepMergeDicts:
    def test_merge_flat_dicts(self):
        result = _deep_merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_takes_precedence(self):
        result = _deep_merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge_dicts(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_none_base(self):
        result = _deep_merge_dicts(None, {"a": 1})
        assert result == {"a": 1}

    def test_none_override(self):
        result = _deep_merge_dicts({"a": 1}, None)
        assert result == {"a": 1}

    def test_both_none(self):
        result = _deep_merge_dicts(None, None)
        assert result is None

    def test_override_replaces_non_dict_with_dict(self):
        result = _deep_merge_dicts({"a": "string"}, {"a": {"nested": True}})
        assert result == {"a": {"nested": True}}

    def test_result_is_deep_copy(self):
        base = {"a": {"b": [1, 2]}}
        result = _deep_merge_dicts(base, {})
        result["a"]["b"].append(3)
        assert base["a"]["b"] == [1, 2]


# ---- _make_json_safe ----

class TestMakeJsonSafe:
    def test_passthrough_primitives(self):
        assert _make_json_safe(None) is None
        assert _make_json_safe("hello") == "hello"
        assert _make_json_safe(42) == 42
        assert _make_json_safe(True) is True

    def test_dict_keys_stringified(self):
        result = _make_json_safe({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}

    def test_tuple_becomes_list(self):
        result = _make_json_safe((1, 2, 3))
        assert result == [1, 2, 3]

    def test_unsupported_type_becomes_string(self):
        class Custom:
            def __str__(self):
                return "custom_value"
        result = _make_json_safe(Custom())
        assert result == "custom_value"

    def test_nested_structure(self):
        result = _make_json_safe({"a": (1, {2: "x"})})
        assert result == {"a": [1, {"2": "x"}]}


# ---- _render_template_object ----

class TestRenderTemplateObject:
    def test_string_substitution(self):
        result = _render_template_object(
            "Hello {name}", {"name": "World"}
        )
        assert result == "Hello World"

    def test_dict_recursion(self):
        result = _render_template_object(
            {"greeting": "Hello {name}"},
            {"name": "World"},
        )
        assert result == {"greeting": "Hello World"}

    def test_list_recursion(self):
        result = _render_template_object(
            ["Hello {name}", "{name} rocks"],
            {"name": "World"},
        )
        assert result == ["Hello World", "World rocks"]

    def test_exact_match_returns_raw_value(self):
        """When template is exactly {task_X}, return the raw value from task_context."""
        task_context = {"items": ["a", "b", "c"]}
        result = _render_template_object(
            "{task_items}",
            {},
            task_context=task_context,
        )
        assert result == ["a", "b", "c"]

    def test_exact_match_deep_copies(self):
        task_context = {"items": ["a", "b"]}
        result = _render_template_object(
            "{task_items}", {}, task_context=task_context
        )
        result.append("c")
        assert task_context["items"] == ["a", "b"]

    def test_non_string_passthrough(self):
        result = _render_template_object(42, {"name": "World"})
        assert result == 42

    def test_no_match_leaves_string_intact(self):
        result = _render_template_object(
            "Hello {unknown}", {"name": "World"}
        )
        assert result == "Hello {unknown}"


# ---- _task_context_template_vars ----

class TestTaskContextTemplateVars:
    def test_empty_context(self):
        assert _task_context_template_vars(None) == {}
        assert _task_context_template_vars({}) == {}

    def test_scalar_values(self):
        result = _task_context_template_vars({"goal": "test"})
        assert result["task_goal"] == "test"
        assert "task_context_json" in result

    def test_dict_value_becomes_json(self):
        result = _task_context_template_vars({"config": {"a": 1}})
        assert result["task_config"] == '{"a": 1}'

    def test_non_dict_input(self):
        assert _task_context_template_vars("not a dict") == {}
        assert _task_context_template_vars([1, 2]) == {}


# ---- _user_generation_style_instructions ----

class TestUserGenerationStyleInstructions:
    def test_no_task_family(self):
        assert _user_generation_style_instructions({}) == []
        assert _user_generation_style_instructions(None) == []

    def test_vague_human_request(self):
        scenario = {"task_family": {"user_request_style": {"vague_human_request": True}}}
        result = _user_generation_style_instructions(scenario)
        assert any("normal human user" in inst for inst in result)

    def test_reference_mode_title_only(self):
        scenario = {"task_family": {"user_request_style": {"reference_mode": "title_only"}}}
        result = _user_generation_style_instructions(scenario)
        assert any("titles" in inst.lower() for inst in result)

    def test_reference_mode_folder_purpose(self):
        scenario = {"task_family": {"user_request_style": {"reference_mode": "folder_purpose"}}}
        result = _user_generation_style_instructions(scenario)
        assert any("purpose" in inst.lower() for inst in result)

    def test_examples_capped_at_three(self):
        scenario = {
            "task_family": {
                "user_request_style": {
                    "examples": ["a", "b", "c", "d", "e"]
                }
            }
        }
        result = _user_generation_style_instructions(scenario)
        example_lines = [line for line in result if line.startswith("- ")]
        assert len(example_lines) == 3

    def test_allow_exact_paths_false(self):
        scenario = {"task_family": {"user_request_style": {"allow_exact_paths": False}}}
        result = _user_generation_style_instructions(scenario)
        assert any("file or folder paths" in inst for inst in result)


# ---- _clean_path ----

class TestCleanPath:
    def test_backslash_normalization(self):
        assert _clean_path("a\\b\\c") == "a/b/c"

    def test_strip_slashes(self):
        assert _clean_path("/a/b/c/") == "a/b/c"

    def test_none_input(self):
        assert _clean_path(None) == ""

    def test_empty_string(self):
        assert _clean_path("") == ""

    def test_whitespace(self):
        assert _clean_path("  a/b  ") == "a/b"
