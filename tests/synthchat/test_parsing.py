"""Tests for SynthChat.parsing — response parsing and environment normalization."""
from __future__ import annotations

import json
import pytest
from SynthChat.parsing import (
    _normalize_generated_assertion,
    normalize_generated_environment,
    parse_assistant_response,
    parse_json_object,
    stringify_assistant_message,
)


# ---- stringify_assistant_message ----

class TestStringifyAssistantMessage:
    def test_string_passthrough(self):
        assert stringify_assistant_message("hello") == "hello"

    def test_dict_with_content(self):
        result = stringify_assistant_message({"content": "Some text"})
        assert result == "Some text"

    def test_dict_with_tool_calls(self):
        msg = {"content": "", "tool_calls": [{"name": "read"}]}
        result = stringify_assistant_message(msg)
        assert "Tool calls" in result

    def test_dict_with_content_and_tool_calls(self):
        msg = {"content": "Thinking...", "tool_calls": [{"name": "read"}]}
        result = stringify_assistant_message(msg)
        assert "Thinking..." in result
        assert "Tool calls" in result

    def test_dict_empty_content_and_no_tools(self):
        result = stringify_assistant_message({"content": "", "tool_calls": []})
        assert result  # falls back to json.dumps

    def test_non_dict_non_string(self):
        assert stringify_assistant_message(42) == "42"

    def test_none_in_dict(self):
        result = stringify_assistant_message({"content": None, "tool_calls": None})
        assert result  # falls back to json.dumps


# ---- parse_assistant_response ----

class TestParseAssistantResponse:
    def test_plain_text(self):
        result = parse_assistant_response("Hello there", {})
        assert result["role"] == "assistant"
        assert result["content"] == "Hello there"
        assert "tool_calls" not in result

    def test_none_content(self):
        result = parse_assistant_response(None, {})
        assert result["role"] == "assistant"

    def test_direct_json_with_tool_calls(self):
        payload = json.dumps({
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {"name": "useTools", "arguments": {"key": "val"}},
            }]
        })
        result = parse_assistant_response(payload, {})
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "useTools"

    def test_json_with_dict_arguments_serialized(self):
        payload = json.dumps({
            "content": None,
            "tool_calls": [{
                "function": {"name": "tool", "arguments": {"a": 1}},
            }]
        })
        result = parse_assistant_response(payload, {})
        args = result["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"a": 1}

    def test_json_with_null_tool_calls_returns_text(self):
        payload = json.dumps({"content": "Just text", "tool_calls": None})
        result = parse_assistant_response(payload, {})
        assert result["content"] == "Just text"
        assert "tool_calls" not in result

    def test_thinking_block_extraction(self):
        content = "<thinking>Reasoning here</thinking>\n\nSome text output"
        result = parse_assistant_response(content, {})
        assert result["role"] == "assistant"

    def test_code_fence_stripping(self):
        content = '```json\n{"content": "text", "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]}\n```'
        result = parse_assistant_response(content, {})
        assert result["role"] == "assistant"
        assert len(result.get("tool_calls", [])) == 1

    def test_tool_call_ids_auto_generated(self):
        payload = json.dumps({
            "content": None,
            "tool_calls": [
                {"function": {"name": "a", "arguments": "{}"}},
                {"function": {"name": "b", "arguments": "{}"}},
            ]
        })
        result = parse_assistant_response(payload, {})
        ids = [tc["id"] for tc in result["tool_calls"]]
        assert ids == ["call_0001", "call_0002"]


# ---- parse_json_object ----

class TestParseJsonObject:
    def test_valid_json(self):
        result = parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fenced_json(self):
        result = parse_json_object('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        result = parse_json_object('Some prefix {"key": "value"} suffix')
        assert result == {"key": "value"}

    def test_empty_string_returns_none(self):
        assert parse_json_object("") is None

    def test_none_input_returns_none(self):
        assert parse_json_object(None) is None

    def test_non_object_json_returns_none(self):
        assert parse_json_object("[1, 2, 3]") is None

    def test_invalid_json_returns_none(self):
        assert parse_json_object("{broken json") is None

    def test_nested_braces(self):
        result = parse_json_object('{"a": {"b": 1}}')
        assert result == {"a": {"b": 1}}


# ---- _normalize_generated_assertion ----

class TestNormalizeGeneratedAssertion:
    def test_file_contains_content_alias(self):
        result = _normalize_generated_assertion({
            "type": "file_contains",
            "path": "notes/test.md",
            "content": "hello",
        })
        assert result["text"] == "hello"
        assert "content" not in result

    def test_file_not_contains_content_alias(self):
        result = _normalize_generated_assertion({
            "type": "file_not_contains",
            "path": "notes/test.md",
            "content": "bad",
        })
        assert result["text"] == "bad"
        assert "content" not in result

    def test_frontmatter_has_key_alias(self):
        result = _normalize_generated_assertion({
            "type": "frontmatter_has_key",
            "path": "notes/test.md",
            "key": "status",
        })
        assert result["field"] == "status"
        assert "key" not in result

    def test_frontmatter_field_equals_aliases(self):
        result = _normalize_generated_assertion({
            "type": "frontmatter_field_equals",
            "path": "notes/test.md",
            "key": "status",
            "content": "done",
        })
        assert result["field"] == "status"
        assert result["value"] == "done"
        assert "key" not in result
        assert "content" not in result

    def test_frontmatter_field_contains_aliases(self):
        result = _normalize_generated_assertion({
            "type": "frontmatter_field_contains",
            "path": "notes/test.md",
            "key": "tags",
            "content": "important",
        })
        assert result["field"] == "tags"
        assert result["text"] == "important"

    def test_canonical_names_left_unchanged(self):
        original = {
            "type": "file_contains",
            "path": "notes/test.md",
            "text": "hello",
        }
        result = _normalize_generated_assertion(original)
        assert result["text"] == "hello"

    def test_non_dict_passthrough(self):
        assert _normalize_generated_assertion("not a dict") == "not a dict"
        assert _normalize_generated_assertion(42) == 42

    def test_deep_copy(self):
        original = {"type": "file_contains", "path": "a.md", "content": "x"}
        result = _normalize_generated_assertion(original)
        assert "content" in original  # original unchanged
        assert "text" in result


# ---- normalize_generated_environment ----

class TestNormalizeGeneratedEnvironment:
    def test_standard_shape(self):
        payload = {
            "environment": {"fixture": {"files": {"a.md": "content"}}},
            "system_context": {},
            "task_context": {},
        }
        result = normalize_generated_environment(payload)
        assert result["environment"]["fixture"]["files"]["a.md"] == "content"

    def test_flat_fixture_keys_promoted(self):
        payload = {
            "fixture": {"files": {"a.md": "content"}},
            "assertions": [{"type": "path_exists", "path": "a.md"}],
        }
        result = normalize_generated_environment(payload)
        assert "environment" in result
        assert result["environment"]["fixture"]["files"]["a.md"] == "content"

    def test_flat_directory_keys_wrapped(self):
        payload = {
            "directories": ["notes/"],
            "files": {"a.md": "content"},
        }
        result = normalize_generated_environment(payload)
        env = result["environment"]
        assert "fixture" in env
        assert env["fixture"]["directories"] == ["notes/"]

    def test_assertions_normalized(self):
        payload = {
            "environment": {
                "fixture": {},
                "assertions": [
                    {"type": "file_contains", "path": "a.md", "content": "hello"},
                ],
            }
        }
        result = normalize_generated_environment(payload)
        assertion = result["environment"]["assertions"][0]
        assert assertion["text"] == "hello"
        assert "content" not in assertion

    def test_defaults_added(self):
        result = normalize_generated_environment({"environment": {"fixture": {}}})
        assert "system_context" in result
        assert "task_context" in result

    def test_empty_environment_handled(self):
        result = normalize_generated_environment({})
        assert result["environment"] == {}
