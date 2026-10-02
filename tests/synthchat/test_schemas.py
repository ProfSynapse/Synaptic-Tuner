"""Tests for SynthChat.schemas — JSON schema construction for environments and tool responses."""
from __future__ import annotations

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
from SynthChat.config.format_resolver import get_default_tool_call_format, load_tool_call_formats


def _default_fmt(**overrides):
    """Return a copy of the default tool call format config with optional overrides."""
    fmt = get_default_tool_call_format()
    fmt.update(overrides)
    return fmt


def _configured_fmt(**overrides):
    """Return the configured default tool-call format (tool_call_formats.yaml) with overrides."""
    fmt = dict(load_tool_call_formats()["default"])
    fmt.update(overrides)
    return fmt


def _wrapper_call_schema(schema):
    tool_calls = schema["properties"]["tool_calls"]
    return [
        opt for opt in tool_calls["anyOf"]
        if opt.get("type") == "array" and opt.get("minItems") == 1
    ][0]["items"]


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

    def test_builtin_default_is_native_and_does_not_constrain_tool_names(self):
        # Without a configured wrapper each tool is its own tool_calls entry; the
        # allowed tools are listed in the generation prompt, not the schema.
        schema = build_tool_response_schema(
            format_config=_default_fmt(),
            allowed_tools=["fileManager_read", "fileManager_write", "searchManager_search"],
        )
        call = _wrapper_call_schema(schema)
        function = call["properties"]["function"]["properties"]
        assert function["name"] == {"type": "string", "minLength": 1}
        assert function["arguments"]["type"] == "object"

    def test_configured_wrapper_arguments_describe_configured_fields(self):
        fmt = _configured_fmt()
        schema = build_tool_response_schema(
            format_config=fmt,
            allowed_tools=["fileManager_read", "fileManager_write", "searchManager_search"],
            context_overrides={"sessionId": "sess_123", "workspaceId": "ws_456"},
        )
        call = _wrapper_call_schema(schema)
        function = call["properties"]["function"]["properties"]
        assert function["name"]["const"] == fmt["wrapper_name"]
        arguments = function["arguments"]
        assert arguments["type"] == "string"
        description = arguments["description"]
        assert f"'{fmt['wrapper_name']}' wrapper payload" in description
        configured_fields = list(fmt["argument_fields"]["properties"]) + list(fmt["extra_argument_fields"])
        for field_name in configured_fields:
            assert field_name in description

    def test_context_overrides_name_only_configured_fields(self):
        schema = build_tool_response_schema(
            format_config=_default_fmt(
                wrapper_name="myWrapper",
                argument_fields={"properties": {"sessionId": {"type": "string"}}},
            ),
            context_overrides={"sessionId": "sess_123", "unknownField": "x"},
        )
        description = _wrapper_call_schema(schema)["properties"]["function"]["properties"]["arguments"]["description"]
        assert description.endswith("fields: sessionId")
        assert "unknownField" not in description

    def test_tool_calls_allows_null(self):
        schema = build_tool_response_schema(format_config=_default_fmt())
        options = schema["properties"]["tool_calls"]["anyOf"]
        null_option = [opt for opt in options if opt.get("type") == "null"]
        assert len(null_option) == 1

    def test_tool_calls_every_array_option_has_items(self):
        # Strict-schema providers (OpenAI, Azure) reject an array schema without items,
        # so a text-only response is expressed by the null option alone.
        for fmt in (_default_fmt(), _default_fmt(wrapper_name="myWrapper")):
            schema = build_tool_response_schema(format_config=fmt)
            options = schema["properties"]["tool_calls"]["anyOf"]
            array_options = [opt for opt in options if opt.get("type") == "array"]
            assert array_options
            assert all("items" in opt for opt in array_options)
            assert not any(opt.get("maxItems") == 0 for opt in array_options)


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
        # The configured instructions carry a {wrapper_name} placeholder.
        prompt = build_tool_generation_prompt(
            format_config=_configured_fmt(wrapper_name="myWrapper"),
            base_prompt="test",
            allowed_tools=[],
        )
        assert "function.name is 'myWrapper'" in prompt
        assert "{wrapper_name}" not in prompt

    def test_builtin_default_prompt_names_no_wrapper(self):
        prompt = build_tool_generation_prompt(
            format_config=_default_fmt(),
            base_prompt="test",
            allowed_tools=[],
        )
        assert prompt.startswith("\n".join(_default_fmt()["generation_instructions"]))
        assert "wrapper" not in prompt

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
