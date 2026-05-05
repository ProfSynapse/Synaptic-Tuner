"""Tests for SynthChat.schemas — JSON schema construction for environments and tool responses."""
from __future__ import annotations

import json
import pytest
from SynthChat.schemas.environment_schema import (
    _build_canonical_environment_generation_prompt,
    _build_canonical_environment_schema,
)
from SynthChat.schemas.tool_response_schema import (
    build_tool_generation_prompt,
    build_tool_response_schema,
    _resolve_allowed_tool_names,
    _resolve_context_defaults,
)
from SynthChat.config.format_resolver import get_default_tool_call_format
from SynthChat.parsing import parse_assistant_response


def _default_fmt(**overrides):
    """Return a copy of the default tool call format config with optional overrides."""
    fmt = get_default_tool_call_format()
    fmt.update(overrides)
    return fmt


# ---- _build_canonical_environment_schema ----

class TestBuildCanonicalEnvironmentSchema:
    def test_schema_is_valid_json_schema(self):
        schema = _build_canonical_environment_schema()
        assert schema["type"] == "object"
        assert "environment" in schema["properties"]
        assert "environment" in schema["required"]
        assert "task_context" in schema["required"]

    def test_environment_has_fixture_and_assertions(self):
        schema = _build_canonical_environment_schema()
        env = schema["properties"]["environment"]
        assert "fixture" in env["properties"]
        assert "assertions" in env["properties"]
        assert "mock_tool_outputs" in env["properties"]
        assert set(env["required"]) == {"fixture", "assertions"}

    def test_environment_mock_tool_outputs_are_declarative(self):
        schema = _build_canonical_environment_schema()
        mock_items = schema["properties"]["environment"]["properties"]["mock_tool_outputs"]["items"]
        assert "tool" in mock_items["properties"]
        assert "match" in mock_items["properties"]
        assert "output" in mock_items["properties"]
        assert "status" in mock_items["properties"]

    def test_fixture_has_directories_files_notes(self):
        schema = _build_canonical_environment_schema()
        fixture = schema["properties"]["environment"]["properties"]["fixture"]
        assert "directories" in fixture["properties"]
        assert "files" in fixture["properties"]
        assert "notes" in fixture["properties"]
        assert fixture["properties"]["files"]["minProperties"] == 1
        assert fixture["properties"]["files"]["additionalProperties"]["minLength"] == 1
        assert fixture["properties"]["notes"]["minItems"] == 1
        note = fixture["properties"]["notes"]["items"]
        assert "body" in note["required"]
        assert note["properties"]["path"]["pattern"].startswith("^[A-Za-z0-9]")
        assert "anyOf" in fixture

    def test_system_context_and_task_context_present(self):
        schema = _build_canonical_environment_schema()
        assert "system_context" in schema["properties"]
        assert "task_context" in schema["properties"]
        assert schema["properties"]["task_context"]["minProperties"] == 1

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

    def test_assertion_paths_reject_placeholder_style_values(self):
        schema = _build_canonical_environment_schema()
        assertions = schema["properties"]["environment"]["properties"]["assertions"]
        path_assertions = [
            opt
            for opt in assertions["items"]["anyOf"]
            if "path" in opt.get("properties", {})
            and opt["properties"]["path"].get("pattern")
        ]
        assert path_assertions
        for assertion in path_assertions:
            path_schema = assertion["properties"]["path"]
            assert path_schema["minLength"] == 1
            assert "`" not in path_schema["pattern"]
            assert "A-Za-z0-9" in path_schema["pattern"]


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

    def test_path_contract_disallows_placeholders(self):
        prompt = _build_canonical_environment_generation_prompt("test")
        assert "plain ASCII relative paths" in prompt
        assert "placeholders" in prompt
        assert "ellipses" in prompt


# ---- build_tool_response_schema ----

class TestBuildToolResponseSchema:
    def test_default_schema_structure(self):
        schema = build_tool_response_schema(format_config=_default_fmt())
        assert schema["type"] == "object"
        assert "content" in schema["properties"]
        assert "tool_calls" in schema["properties"]
        assert set(schema["required"]) == {"content", "tool_calls"}

    def test_custom_wrapper_name(self):
        schema = build_tool_response_schema(format_config=_default_fmt(wrapper_name="myWrapper"))
        tool_calls = schema["properties"]["tool_calls"]
        # Navigate to the array option with items
        array_option = [
            opt for opt in tool_calls["anyOf"]
            if opt.get("type") == "array" and opt.get("minItems") == 1
        ][0]
        fn_name = array_option["items"]["properties"]["function"]["properties"]["name"]
        assert fn_name["const"] == "myWrapper"

    def test_allowed_tools_constrain_enum(self):
        schema = build_tool_response_schema(
            format_config=_default_fmt(),
            allowed_tools=["fileManager_read", "fileManager_write", "searchManager_search"],
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
        schema = build_tool_response_schema(
            format_config=_default_fmt(),
            context_overrides={"sessionId": "sess_123", "workspaceId": "ws_456"},
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
        schema = build_tool_response_schema(format_config=_default_fmt())
        options = schema["properties"]["tool_calls"]["anyOf"]
        null_option = [opt for opt in options if opt.get("type") == "null"]
        assert len(null_option) == 1

    def test_tool_calls_allows_empty_array(self):
        schema = build_tool_response_schema(format_config=_default_fmt())
        options = schema["properties"]["tool_calls"]["anyOf"]
        empty_arr = [opt for opt in options if opt.get("type") == "array" and opt.get("maxItems") == 0]
        assert len(empty_arr) == 1

    def test_wrapper_arguments_can_be_generated_as_object(self):
        fmt = _default_fmt(
            wrapper_name="useTools",
            generation_argument_mode="object",
            argument_fields={
                "required": ["sessionId", "workspaceId", "memory", "goal", "tool"],
                "properties": {
                    "sessionId": {"type": "string"},
                    "workspaceId": {"type": "string"},
                    "memory": {"type": "string"},
                    "goal": {"type": "string"},
                    "tool": {"type": "string"},
                },
            },
            argument_required=["sessionId", "workspaceId", "memory", "goal", "tool"],
        )
        schema = build_tool_response_schema(
            format_config=fmt,
            context_overrides={"sessionId": "sess_123", "workspaceId": "ws_456"},
        )
        tool_calls = schema["properties"]["tool_calls"]
        array_option = [
            opt for opt in tool_calls["anyOf"]
            if opt.get("type") == "array" and opt.get("minItems") == 1
        ][0]
        arguments = array_option["items"]["properties"]["function"]["properties"]["arguments"]

        assert arguments["type"] == "object"
        assert arguments["properties"]["sessionId"]["const"] == "sess_123"
        assert arguments["properties"]["workspaceId"]["const"] == "ws_456"
        assert "tool" in arguments["required"]


# ---- build_tool_generation_prompt ----

class TestBuildToolGenerationPrompt:
    def test_includes_base_prompt(self):
        prompt = build_tool_generation_prompt(
            format_config=_default_fmt(),
            base_prompt="Test the tools",
            allowed_tools=[],
        )
        assert "Test the tools" in prompt

    def test_includes_wrapper_name(self):
        prompt = build_tool_generation_prompt(
            format_config=_default_fmt(wrapper_name="myWrapper"),
            base_prompt="test",
            allowed_tools=[],
        )
        assert "myWrapper" in prompt

    def test_includes_allowed_tools(self):
        prompt = build_tool_generation_prompt(
            format_config=_default_fmt(),
            base_prompt="test",
            allowed_tools=["fileManager_read", "searchManager_search"],
        )
        assert "fileManager_read" in prompt
        assert "searchManager_search" in prompt

    def test_no_tools_line_when_empty(self):
        prompt = build_tool_generation_prompt(
            format_config=_default_fmt(),
            base_prompt="test",
            allowed_tools=[],
        )
        assert "Allowed concrete tools" not in prompt


def test_parse_assistant_response_serializes_object_wrapper_arguments():
    raw = json.dumps({
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": {
                        "workspaceId": "default",
                        "sessionId": "session_eval_001",
                        "memory": "Find the file",
                        "goal": "Read the matching file",
                        "tool": "content read \"docs/guide.txt\" 1",
                        "strategy": "serial",
                    },
                },
            }
        ],
    })

    message = parse_assistant_response(raw, {"type": "tool", "tool": "content read"})

    args = message["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args)["tool"] == "content read \"docs/guide.txt\" 1"


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
