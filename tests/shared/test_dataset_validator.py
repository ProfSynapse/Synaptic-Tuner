"""Config-first checks in shared.validation.dataset_validator.

Wrapper names, required fields, field constraints, prompt-bound fields and tool
schemas all come from config: the fixtures under
tests/fixtures/tool_call_formats/ or the engine's configured
SynthChat/config/tool_call_formats.yaml and Tools/tool_schemas.json. No format
is hardcoded here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from shared.utilities.paths import get_engine_root
from shared.validation import dataset_validator
from shared.validation.dataset_validator import ValidatorConfig
from shared.validation.parsing.configured_formats import build_wrapper_specs
from SynthChat.config.format_resolver import load_tool_call_formats

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tool_call_formats"
REGISTRY_PATH = FIXTURES / "registry.yaml"
TOOL_SCHEMAS_PATH = FIXTURES / "tool_schemas.json"
FIXTURE_FORMATS = load_tool_call_formats(str(REGISTRY_PATH))
FIXTURE_TOOL_SCHEMAS = dataset_validator.load_tool_schemas(TOOL_SCHEMAS_PATH)
WRAPPER_FORMAT = FIXTURE_FORMATS["ticketed"]
WRAPPERLESS_FORMAT = FIXTURE_FORMATS["direct"]
WRAPPER = WRAPPER_FORMAT["wrapper_name"]
DIRECT_TOOL = next(name for name in FIXTURE_TOOL_SCHEMAS if name != WRAPPER)


def _complete_arguments(fmt):
    """Arguments satisfying every configured property of a wrapper format."""
    properties = dict(fmt["argument_fields"]["properties"])
    properties.update(fmt.get("extra_argument_fields") or {})
    return {
        name: (spec["enum"][0] if "enum" in spec else f"{name} value")
        for name, spec in properties.items()
    }


def _direct_arguments(tool_name):
    return {name: f"{name} value" for name in FIXTURE_TOOL_SCHEMAS[tool_name]["required_params"]}


def _openai_message(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


def _issues(message, config=None, system_prompt=None, level=None):
    report = dataset_validator.ExampleReport(index=1, label=True)
    dataset_validator.validate_assistant_message_openai(message, report, system_prompt, config)
    return [issue.message for issue in report.issues if level is None or issue.level == level]


def _errors(message, config=None, system_prompt=None):
    return _issues(message, config, system_prompt, level="ERROR")


@pytest.fixture
def config():
    return ValidatorConfig(
        wrapper_specs=build_wrapper_specs(FIXTURE_FORMATS),
        tool_schemas=FIXTURE_TOOL_SCHEMAS,
    )


# --- wrapper fields from the tool-call format config ---------------------------------

def test_complete_wrapper_call_passes(config):
    assert _issues(_openai_message(WRAPPER, _complete_arguments(WRAPPER_FORMAT)), config) == []


@pytest.mark.parametrize("missing", WRAPPER_FORMAT["argument_required"])
def test_missing_configured_required_field_is_reported_once(config, missing):
    # Both the wrapper format and the tool schema require it; one finding.
    assert missing in FIXTURE_TOOL_SCHEMAS[WRAPPER]["required_params"]
    arguments = _complete_arguments(WRAPPER_FORMAT)
    del arguments[missing]

    assert _errors(_openai_message(WRAPPER, arguments), config) == [
        f"Tool call #1 ({WRAPPER}): Missing required parameter '{missing}'"
    ]


def test_configured_property_constraints_are_enforced(config):
    enum_field, enum_spec = next(
        (name, spec) for name, spec in WRAPPER_FORMAT["extra_argument_fields"].items() if "enum" in spec
    )
    min_length_field = next(
        name for name, spec in WRAPPER_FORMAT["argument_fields"]["properties"].items() if spec.get("minLength")
    )
    arguments = _complete_arguments(WRAPPER_FORMAT)
    arguments[enum_field] = "".join(enum_spec["enum"]) + "-unlisted"
    arguments[min_length_field] = ""

    errors = _errors(_openai_message(WRAPPER, arguments), config)

    assert len(errors) == 2
    assert any(f"Invalid wrapper field '{enum_field}'" in error for error in errors)
    assert any(f"Invalid wrapper field '{min_length_field}'" in error for error in errors)


def test_chatml_wrapper_call_uses_configured_wrapper(config):
    arguments = _complete_arguments(WRAPPER_FORMAT)
    missing = WRAPPER_FORMAT["argument_required"][0]
    del arguments[missing]
    content = f"tool_call: {WRAPPER}\narguments: {json.dumps(arguments)}"
    report = dataset_validator.ExampleReport(index=1, label=True)

    dataset_validator.validate_assistant_content(content, report, None, config)

    assert [issue.message for issue in report.issues if issue.level == "ERROR"] == [
        f"Tool call #1 ({WRAPPER}): Missing required parameter '{missing}'"
    ]


def test_wrapperless_format_accepts_direct_calls():
    config = ValidatorConfig(
        wrapper_specs=build_wrapper_specs({"direct": WRAPPERLESS_FORMAT}),
        tool_schemas=FIXTURE_TOOL_SCHEMAS,
    )
    assert config.wrapper_specs == []

    assert _issues(_openai_message(DIRECT_TOOL, _direct_arguments(DIRECT_TOOL)), config) == []


def test_direct_call_needs_no_wrapper_fields_beside_wrapper_formats(config):
    assert _issues(_openai_message(DIRECT_TOOL, _direct_arguments(DIRECT_TOOL)), config) == []


def test_non_object_arguments_are_rejected(config):
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"type": "function", "function": {"name": DIRECT_TOOL, "arguments": "[1, 2]"}}],
    }

    assert _errors(message, config) == [f"Tool call #1 ({DIRECT_TOOL}): Arguments must be a JSON object"]


# --- prompt-bound fields --------------------------------------------------------------

def _bound_field_and_sources():
    return next(iter(WRAPPER_FORMAT["prompt_bound_fields"].items()))


def _bound_arguments(field_name):
    return dict(_complete_arguments(WRAPPER_FORMAT), **{field_name: "REF-7"})


def _prompt(*sections):
    return "\n".join(f"<{tag}>\n{body}\n</{tag}>" for tag, body in sections)


def test_prompt_bound_field_matching_the_prompt_passes(config):
    field_name, sources = _bound_field_and_sources()
    first, second = sources
    arguments = _bound_arguments(field_name)
    system_prompt = _prompt(
        (first["in_tag"], first["pattern"].split("\\s*")[0] + f" {arguments[field_name]}"),
    )

    assert _errors(_openai_message(WRAPPER, arguments), config, system_prompt) == []


def test_prompt_bound_field_accepts_any_configured_source(config):
    field_name, sources = _bound_field_and_sources()
    first, second = sources
    arguments = _bound_arguments(field_name)
    system_prompt = _prompt(
        (first["in_tag"], first["pattern"].split("\\s*")[0] + " OTHER-1"),
        (second["in_tag"], second["pattern"].split("\\s*")[0] + f" {arguments[field_name]}"),
    )

    assert _errors(_openai_message(WRAPPER, arguments), config, system_prompt) == []


def test_prompt_bound_field_mismatch_is_reported(config):
    field_name, sources = _bound_field_and_sources()
    first = sources[0]
    arguments = _bound_arguments(field_name)
    system_prompt = _prompt((first["in_tag"], first["pattern"].split("\\s*")[0] + " OTHER-1"))

    assert _errors(_openai_message(WRAPPER, arguments), config, system_prompt) == [
        f"Tool call #1 ({WRAPPER}): {field_name} '{arguments[field_name]}' "
        "does not match system prompt (expected one of: ['OTHER-1'])"
    ]


def test_prompt_bound_field_unchecked_when_prompt_states_nothing(config):
    arguments = _complete_arguments(WRAPPER_FORMAT)

    assert _errors(_openai_message(WRAPPER, arguments), config, "no tagged sections here") == []


# --- tool schema catalog --------------------------------------------------------------

def test_tool_schema_required_parameter_is_enforced(config):
    required = FIXTURE_TOOL_SCHEMAS[DIRECT_TOOL]["required_params"][0]
    arguments = _direct_arguments(DIRECT_TOOL)
    del arguments[required]

    assert _errors(_openai_message(DIRECT_TOOL, arguments), config) == [
        f"Tool call #1 ({DIRECT_TOOL}): Missing required parameter '{required}'"
    ]


def test_undeclared_parameter_warns(config):
    arguments = dict(_direct_arguments(DIRECT_TOOL), undeclared="x")

    assert _issues(_openai_message(DIRECT_TOOL, arguments), config, level="WARN") == [
        f"Tool call #1 ({DIRECT_TOOL}): Unexpected parameter 'undeclared' not in schema"
    ]


def test_tool_without_schema_warns(config):
    assert _issues(_openai_message("uncatalogued_tool", {}), config, level="WARN") == [
        "Tool call #1 (uncatalogued_tool): No schema found for this tool"
    ]


# --- engine defaults ------------------------------------------------------------------

def test_engine_tool_schemas_resolve_from_engine_root(monkeypatch):
    assert dataset_validator.SCHEMAS_FILE == get_engine_root() / "Tools" / "tool_schemas.json"
    assert dataset_validator.SCHEMAS_FILE.is_file()

    monkeypatch.chdir(FIXTURES)
    assert os.getcwd() != str(get_engine_root())
    assert dataset_validator.load_tool_schemas() == json.loads(
        dataset_validator.SCHEMAS_FILE.read_text(encoding="utf-8")
    )


def test_engine_config_is_the_default_and_self_consistent():
    configured = load_tool_call_formats()["default"]
    wrapper_name = configured["wrapper_name"]
    arguments = _complete_arguments(configured)

    # The configured wrapper also has a catalog schema, and the two agree.
    assert wrapper_name in dataset_validator.default_validator_config().tool_schemas
    assert _issues(_openai_message(wrapper_name, arguments)) == []

    missing = configured["argument_required"][-1]
    del arguments[missing]
    assert _errors(_openai_message(wrapper_name, arguments)) == [
        f"Tool call #1 ({wrapper_name}): Missing required parameter '{missing}'"
    ]


# --- CLI ------------------------------------------------------------------------------

def test_cli_validates_against_config_flags(capsys):
    dataset_validator.main(
        [
            str(FIXTURES / "ticketed_dataset.jsonl"),
            "--tool-call-formats", str(REGISTRY_PATH),
            "--tool-schemas", str(TOOL_SCHEMAS_PATH),
        ]
    )

    captured = capsys.readouterr()
    assert "Validated 2 example(s): 0 failed" in captured.out
    assert f"{len(FIXTURE_TOOL_SCHEMAS)} tool schemas loaded" in captured.err


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--tool-call-formats", "Tool-call format registry not found"),
        ("--tool-schemas", "Tool schema catalog not found"),
    ],
)
def test_cli_rejects_missing_config(flag, message):
    with pytest.raises(SystemExit, match=message):
        dataset_validator.main([str(FIXTURES / "ticketed_dataset.jsonl"), flag, str(FIXTURES / "absent")])
