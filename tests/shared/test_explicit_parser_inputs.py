"""Explicit parsing contexts have no ambient registry or catalog dependency."""

import json

import pytest

from SynthChat.parsing import parse_assistant_response
from shared.validation.parsing.configured_formats import build_wrapper_specs
from shared.validation.parsing.cli_commands import build_cli_command_catalog, load_cli_command_catalog
from shared.validation.parsing.response_parser import (
    parse_response, extract_arguments_from_response, extract_context_from_response,
)
from shared.environments.tool_executor import expand_cli_wrapper_commands, execute_response_tool_calls
from shared.environments.local_runtime import LocalEnvironmentRuntime
from shared.environments.fixture_parser import EnvironmentFixture


def _format(wrapper, context, command):
    return {"configured": {"wrapper_name": wrapper, "command_field": command,
            "argument_fields": {"required": [context, command], "properties": {
                context: {"type": "string"}, command: {"type": "string"}}}}}


def _response(wrapper, context, command):
    return {"tool_calls": [{"function": {"name": wrapper,
            "arguments": {context: "selected", command: 'memo save "note.txt" "exact text"'}}}]}


def _catalog():
    return {"tools": [{"agent": "memo", "tool": "save", "command": "memo save", "arguments": [
        {"name": "path", "type": "string", "positional": True, "required": True},
        {"name": "content", "type": "string", "positional": True, "required": True},
    ]}]}


@pytest.fixture(autouse=True)
def forbid_ambient_defaults(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit context consulted an ambient default")
    monkeypatch.setattr("shared.validation.parsing.configured_formats.get_configured_wrapper_specs", forbidden)
    monkeypatch.setattr("shared.environments.tool_executor.cli_command_catalog", forbidden)
    monkeypatch.setattr("shared.environments.tool_executor.load_cli_command_catalog", forbidden)
    monkeypatch.setattr("shared.environments.tool_executor.tool_catalog_path", forbidden)


def test_custom_contexts_remain_independent_and_ignore_later_caller_mutation():
    formats = _format("dispatch_alpha", "tenant", "instruction")
    first_specs = build_wrapper_specs(formats)
    response = _response("dispatch_alpha", "tenant", "instruction")
    first = parse_response(response, wrapper_specs=first_specs)
    formats["configured"]["argument_fields"]["properties"]["tenant"]["type"] = "object"
    first_specs[0]["field_names"].clear()
    response["tool_calls"][0]["function"]["arguments"]["tenant"] = "changed"
    second_specs = build_wrapper_specs(_format("dispatch_beta", "account", "action"))
    second = parse_response(_response("dispatch_beta", "account", "action"), wrapper_specs=second_specs)
    assert first.context == {"tenant": "selected"}
    assert second.context == {"account": "selected"}
    assert first.first_tool_call.name == "dispatch_alpha"
    assert second.first_tool_call.name == "dispatch_beta"
    assert first_specs[0]["properties"]["tenant"]["type"] == "string"


@pytest.mark.parametrize("encoded", [False, True])
def test_real_parse_extract_and_synthchat_share_explicit_specs(encoded):
    specs = build_wrapper_specs(_format("dispatch_custom", "tenant", "instruction"))
    response = _response("dispatch_custom", "tenant", "instruction")
    if encoded:
        response["tool_calls"][0]["function"]["arguments"] = json.dumps(
            response["tool_calls"][0]["function"]["arguments"])
    assert extract_context_from_response(response, wrapper_specs=specs) == {"tenant": "selected"}
    assert extract_arguments_from_response(response, wrapper_specs=specs)["instruction"].startswith("memo save")
    message = parse_assistant_response(json.dumps(response), {}, wrapper_specs=specs)
    assert message["tool_calls"][0]["function"]["name"] == "dispatch_custom"
    assert parse_response(message, wrapper_specs=specs).context == {"tenant": "selected"}


def test_lenient_recovery_and_direct_wrapper_json_use_only_explicit_format():
    specs = build_wrapper_specs(_format("dispatch_custom", "tenant", "instruction"))
    arguments = '{"instruction":"memo save note.txt "body"", "tenant":"selected", '
    response = {"tool_calls": [{"function": {"name": "dispatch_custom", "arguments": arguments}}]}
    assert parse_response(response, wrapper_specs=specs).context == {"tenant": "selected"}
    direct = json.dumps({"tenant": "selected", "instruction": "memo save note.txt body"})
    message = parse_assistant_response(direct, {}, wrapper_specs=specs)
    assert message["tool_calls"][0]["function"]["name"] == "dispatch_custom"


def test_explicit_empty_specs_and_catalog_never_adopt_defaults():
    response = _response("dispatch_custom", "tenant", "instruction")
    parsed = parse_response(response, wrapper_specs=[])
    assert parsed.context is None
    assert extract_context_from_response(response, wrapper_specs=[]) is None
    specs = build_wrapper_specs(_format("dispatch_custom", "tenant", "instruction"))
    arguments = response["tool_calls"][0]["function"]["arguments"]
    assert expand_cli_wrapper_commands("dispatch_custom", arguments, wrapper_specs=specs, catalog={}) is None
    assert expand_cli_wrapper_commands("dispatch_custom", arguments, wrapper_specs=[], catalog={}) is None
    message = parse_assistant_response(json.dumps(response), {}, wrapper_specs=[])
    assert parse_response(message, wrapper_specs=[]).context is None


def test_catalog_mapping_path_parity_and_nested_mutation_isolation(tmp_path):
    payload = _catalog()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload))
    catalog = build_cli_command_catalog(payload)
    assert catalog == load_cli_command_catalog(path)
    payload["tools"][0]["arguments"][0]["name"] = "changed"
    assert catalog["memo save"].arguments[0]["name"] == "path"
    assert build_cli_command_catalog({"tools": []}) == {}


def test_explicit_context_access_returns_independent_nested_values():
    formats = _format("dispatch_custom", "tenant", "instruction")
    formats["configured"]["argument_fields"]["properties"]["metadata"] = {"type": "object"}
    response = _response("dispatch_custom", "tenant", "instruction")
    response["tool_calls"][0]["function"]["arguments"]["metadata"] = {"labels": ["original"]}
    parsed = parse_response(response, wrapper_specs=build_wrapper_specs(formats))
    context = parsed.context
    context["metadata"]["labels"].append("changed by caller")
    assert parsed.context["metadata"] == {"labels": ["original"]}


def test_none_preserves_legacy_custom_context_without_deepcopy(monkeypatch):
    class LegacyValue:
        def __deepcopy__(self, memo):
            raise AssertionError("legacy context must not be copied")
    value = LegacyValue()
    formats = _format("dispatch_legacy", "tenant", "instruction")
    formats["configured"]["argument_fields"]["properties"]["metadata"] = {"type": "object"}
    specs = build_wrapper_specs(formats)
    monkeypatch.setattr("shared.validation.parsing.configured_formats.get_configured_wrapper_specs", lambda: specs)
    response = _response("dispatch_legacy", "tenant", "instruction")
    response["tool_calls"][0]["function"]["arguments"]["metadata"] = value
    assert parse_response(response).context["metadata"] is value


@pytest.mark.parametrize("encoding", ["qwen", "qwen-malformed", "mistral", "chatml"])
def test_tagged_response_paths_retain_explicit_context_without_defaults(encoding):
    specs = build_wrapper_specs(_format("dispatch_custom", "tenant", "instruction"))
    arguments = {"tenant": "selected", "instruction": "memo save note.txt body"}
    if encoding == "mistral":
        response = "[TOOL_CALLS] " + json.dumps([{"name": "dispatch_custom", "arguments": arguments}])
    elif encoding == "chatml":
        response = "tool_call: dispatch_custom\narguments: " + json.dumps(arguments)
    else:
        body = json.dumps({"name": "dispatch_custom", "arguments": arguments})
        if encoding == "qwen-malformed":
            body = body[:-1] + ", invalid outer document"
        response = "<tool_call>" + body + "</tool_call>"
    parsed = parse_response(response, wrapper_specs=specs)
    assert parsed.first_tool_call.name == "dispatch_custom"
    assert parsed.context == {"tenant": "selected"}
    assert extract_context_from_response(response, wrapper_specs=specs) == {"tenant": "selected"}


def test_real_parse_and_expansion_execute_custom_declared_action():
    specs = build_wrapper_specs(_format("dispatch_custom", "tenant", "instruction"))
    catalog = build_cli_command_catalog(_catalog())
    response = _response("dispatch_custom", "tenant", "instruction")
    arguments = response["tool_calls"][0]["function"]["arguments"]
    expanded = expand_cli_wrapper_commands("dispatch_custom", arguments, wrapper_specs=specs, catalog=catalog)
    assert len(expanded) == 1 and expanded[0].spec.tool_name == "memo_save"
    assert expanded[0].arguments == {"path": "note.txt", "content": "exact text"}
    runtime = LocalEnvironmentRuntime()
    runtime.setup(EnvironmentFixture())
    try:
        executed, issues = execute_response_tool_calls(
            runtime, response, allowed_tools=["memo_save"], action_hints={"memo_save": "write"},
            wrapper_specs=specs, catalog=catalog,
        )
        assert not issues and len(executed) == 1
        assert executed[0].name == "memo_save"
        assert runtime.read_text("note.txt") == "exact text"
    finally:
        runtime.teardown()
