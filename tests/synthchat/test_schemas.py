"""Tests for SynthChat.schemas — JSON schema construction for environments and tool responses."""
from __future__ import annotations

import json
import pytest
from SynthChat.schemas.environment_schema import (
    _build_canonical_environment_generation_prompt,
    _build_canonical_environment_schema,
)
from SynthChat.schemas.tool_response_schema import (
    _build_use_tools_generation_prompt,
    _build_use_tools_response_schema,
    _resolve_allowed_tool_names,
    _resolve_context_defaults,
)


# ---- _build_canonical_environment_schema ----

class TestBuildCanonicalEnvironmentSchema:
    def test_schema_is_valid_json_schema(self):
        schema = _build_canonical_environment_schema()
        assert schema["type"] == "object"
        assert "environment" in schema["properties"]
        assert "environment" in schema["required"]

    def test_environment_has_fixture_and_assertions(self):
        schema = _build_canonical_environment_schema()
        env = schema["properties"]["environment"]
        assert "fixture" in env["properties"]
        assert "assertions" in env["properties"]
        assert set(env["required"]) == {"fixture", "assertions"}

    def test_fixture_has_directories_files_notes(self):
        schema = _build_canonical_environment_schema()
        fixture = schema["properties"]["environment"]["properties"]["fixture"]
        assert "directories" in fixture["properties"]
        assert "files" in fixture["properties"]
        assert "notes" in fixture["properties"]

    def test_system_context_and_task_context_present(self):
        schema = _build_canonical_environment_schema()
        assert "system_context" in schema["properties"]
        assert "task_context" in schema["properties"]

    def test_assertions_items_are_anyof(self):
        schema = _build_canonical_environment_schema()
        assertions = schema["properties"]["environment"]["properties"]["assertions"]
        items = assertions["items"]
        assert "anyOf" in items
        types = {
            opt["properties"]["type"]["const"]
            for opt in items["anyOf"]
            if "properties" in opt and "type" in opt["properties"]
        }
        expected_types = {
            "path_exists", "path_not_exists",
            "file_contains", "file_not_contains",
            "dir_contains",
            "frontmatter_has_key", "frontmatter_field_equals", "frontmatter_field_contains",
        }
        assert types == expected_types


# ---- _build_canonical_environment_generation_prompt ----

class TestBuildCanonicalEnvironmentGenerationPrompt:
    def test_contract_prepended(self):
        prompt = _build_canonical_environment_generation_prompt("Generate an environment")
        assert "Return one valid JSON object only" in prompt
        assert "Generate an environment" in prompt

    def test_empty_base_prompt(self):
        prompt = _build_canonical_environment_generation_prompt("")
        assert "Return one valid JSON object only" in prompt
        assert "Task:" not in prompt

    def test_assertion_types_listed(self):
        prompt = _build_canonical_environment_generation_prompt("test")
        for t in ["path_exists", "file_contains", "frontmatter_has_key"]:
            assert t in prompt


# ---- _build_use_tools_response_schema ----

class TestBuildUseToolsResponseSchema:
    def test_default_schema_structure(self):
        schema = _build_use_tools_response_schema()
        assert schema["type"] == "object"
        assert "content" in schema["properties"]
        assert "tool_calls" in schema["properties"]
        assert set(schema["required"]) == {"content", "tool_calls"}

    def test_custom_wrapper_name(self):
        schema = _build_use_tools_response_schema(wrapper_name="myWrapper")
        tool_calls = schema["properties"]["tool_calls"]
        # Navigate to the array option with items
        array_option = [
            opt for opt in tool_calls["anyOf"]
            if opt.get("type") == "array" and opt.get("minItems") == 1
        ][0]
        fn_name = array_option["items"]["properties"]["function"]["properties"]["name"]
        assert fn_name["const"] == "myWrapper"

    def test_allowed_tools_constrain_enum(self):
        schema = _build_use_tools_response_schema(
            allowed_tools=["fileManager_read", "fileManager_write", "searchManager_search"]
        )
        tool_calls = schema["properties"]["tool_calls"]
        array_option = [
            opt for opt in tool_calls["anyOf"]
            if opt.get("type") == "array" and opt.get("minItems") == 1
        ][0]
        arguments = array_option["items"]["properties"]["function"]["properties"]["arguments"]
        calls_items = arguments["properties"]["calls"]["items"]
        agent_enum = calls_items["properties"]["agent"]["enum"]
        tool_enum = calls_items["properties"]["tool"]["enum"]
        assert "fileManager" in agent_enum
        assert "searchManager" in agent_enum
        assert "read" in tool_enum
        assert "write" in tool_enum
        assert "search" in tool_enum

    def test_session_and_workspace_consts(self):
        schema = _build_use_tools_response_schema(
            session_id="sess_123", workspace_id="ws_456"
        )
        tool_calls = schema["properties"]["tool_calls"]
        array_option = [
            opt for opt in tool_calls["anyOf"]
            if opt.get("type") == "array" and opt.get("minItems") == 1
        ][0]
        args = array_option["items"]["properties"]["function"]["properties"]["arguments"]
        context = args["properties"]["context"]
        assert context["properties"]["sessionId"]["const"] == "sess_123"
        assert context["properties"]["workspaceId"]["const"] == "ws_456"

    def test_tool_calls_allows_null(self):
        schema = _build_use_tools_response_schema()
        options = schema["properties"]["tool_calls"]["anyOf"]
        null_option = [opt for opt in options if opt.get("type") == "null"]
        assert len(null_option) == 1

    def test_tool_calls_allows_empty_array(self):
        schema = _build_use_tools_response_schema()
        options = schema["properties"]["tool_calls"]["anyOf"]
        empty_arr = [opt for opt in options if opt.get("type") == "array" and opt.get("maxItems") == 0]
        assert len(empty_arr) == 1


# ---- _build_use_tools_generation_prompt ----

class TestBuildUseToolsGenerationPrompt:
    def test_includes_base_prompt(self):
        prompt = _build_use_tools_generation_prompt(
            base_prompt="Test the tools",
            wrapper_name="useTools",
            allowed_tools=[],
        )
        assert "Test the tools" in prompt

    def test_includes_wrapper_name(self):
        prompt = _build_use_tools_generation_prompt(
            base_prompt="test",
            wrapper_name="myWrapper",
            allowed_tools=[],
        )
        assert "myWrapper" in prompt

    def test_includes_allowed_tools(self):
        prompt = _build_use_tools_generation_prompt(
            base_prompt="test",
            wrapper_name="useTools",
            allowed_tools=["fileManager_read", "searchManager_search"],
        )
        assert "fileManager_read" in prompt
        assert "searchManager_search" in prompt

    def test_no_tools_line_when_empty(self):
        prompt = _build_use_tools_generation_prompt(
            base_prompt="test",
            wrapper_name="useTools",
            allowed_tools=[],
        )
        assert "Allowed concrete tools" not in prompt


# ---- _resolve_allowed_tool_names ----

class TestResolveAllowedToolNames:
    def test_from_scenario_expected_tools(self):
        result = _resolve_allowed_tool_names(
            scenario={"expected_tools": ["fileManager_read"]},
            tool_schema=None,
        )
        assert result == ["fileManager_read"]

    def test_from_scenario_tool(self):
        result = _resolve_allowed_tool_names(
            scenario={"tool": "searchManager_search"},
            tool_schema=None,
        )
        assert result == ["searchManager_search"]

    def test_text_only_filtered(self):
        result = _resolve_allowed_tool_names(
            scenario={"expected_tools": ["TEXT_ONLY", "fileManager_read"]},
            tool_schema=None,
        )
        assert "TEXT_ONLY" not in result
        assert "fileManager_read" in result

    def test_fallback_to_schema(self):
        schema = {
            "tools": {
                "fileManager": [{"name": "read"}, {"name": "write"}],
                "searchManager": [{"name": "search"}],
            }
        }
        result = _resolve_allowed_tool_names(scenario={}, tool_schema=schema)
        assert "fileManager_read" in result
        assert "fileManager_write" in result
        assert "searchManager_search" in result

    def test_deduplicated_and_sorted(self):
        result = _resolve_allowed_tool_names(
            scenario={
                "expected_tools": ["b_tool", "a_tool"],
                "acceptable_tools": ["a_tool", "c_tool"],
            },
            tool_schema=None,
        )
        assert result == sorted(set(result))


# ---- _resolve_context_defaults ----

class TestResolveContextDefaults:
    def test_none_input(self):
        assert _resolve_context_defaults(system_context=None) == (None, None)

    def test_direct_ids(self):
        result = _resolve_context_defaults(
            system_context={"session_id": "s1", "workspace_id": "w1"}
        )
        assert result == ("s1", "w1")

    def test_workspace_from_selected_workspace(self):
        ctx = {"selected_workspace": {"id": "ws_2"}}
        _, workspace_id = _resolve_context_defaults(system_context=ctx)
        assert workspace_id == "ws_2"

    def test_empty_strings_become_none(self):
        ctx = {"session_id": "", "workspace_id": "  "}
        session_id, workspace_id = _resolve_context_defaults(system_context=ctx)
        assert session_id is None
        assert workspace_id is None
