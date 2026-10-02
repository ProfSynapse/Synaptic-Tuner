"""Config-first wrapper checks in shared.validation.dataset_validator.

The wrapper name, required fields and field constraints all come from a
tool-call format registry: the fixture registry under
tests/fixtures/tool_call_formats/ or the engine's configured
SynthChat/config/tool_call_formats.yaml. No format is hardcoded here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.validation import dataset_validator
from shared.validation.parsing.configured_formats import build_wrapper_specs
from SynthChat.config.format_resolver import load_tool_call_formats

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tool_call_formats"
REGISTRY_PATH = FIXTURES / "registry.yaml"
FIXTURE_FORMATS = load_tool_call_formats(str(REGISTRY_PATH))
WRAPPER_FORMAT = FIXTURE_FORMATS["ticketed"]
WRAPPERLESS_FORMAT = FIXTURE_FORMATS["direct"]


def _complete_arguments(fmt):
    """Arguments satisfying every configured property of a wrapper format."""
    properties = dict(fmt["argument_fields"]["properties"])
    properties.update(fmt.get("extra_argument_fields") or {})
    return {
        name: (spec["enum"][0] if "enum" in spec else f"{name} value")
        for name, spec in properties.items()
    }


def _openai_message(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


def _errors(message, wrapper_specs=None):
    report = dataset_validator.ExampleReport(index=1, label=True)
    dataset_validator.validate_assistant_message_openai(message, report, wrapper_specs=wrapper_specs)
    return [issue.message for issue in report.issues if issue.level == "ERROR"]


@pytest.fixture
def fixture_specs():
    return build_wrapper_specs(FIXTURE_FORMATS)


def test_complete_wrapper_call_passes(fixture_specs):
    message = _openai_message(WRAPPER_FORMAT["wrapper_name"], _complete_arguments(WRAPPER_FORMAT))

    assert _errors(message, fixture_specs) == []


@pytest.mark.parametrize("missing", WRAPPER_FORMAT["argument_required"])
def test_missing_configured_required_field_is_reported(fixture_specs, missing):
    arguments = _complete_arguments(WRAPPER_FORMAT)
    del arguments[missing]

    errors = _errors(_openai_message(WRAPPER_FORMAT["wrapper_name"], arguments), fixture_specs)

    assert errors == [
        f"Tool call #1 ({WRAPPER_FORMAT['wrapper_name']}): Missing required '{missing}' field in arguments"
    ]


def test_configured_property_constraints_are_enforced(fixture_specs):
    enum_field, enum_spec = next(
        (name, spec) for name, spec in WRAPPER_FORMAT["extra_argument_fields"].items() if "enum" in spec
    )
    min_length_field = next(
        name for name, spec in WRAPPER_FORMAT["argument_fields"]["properties"].items() if spec.get("minLength")
    )
    arguments = _complete_arguments(WRAPPER_FORMAT)
    arguments[enum_field] = "".join(enum_spec["enum"]) + "-unlisted"
    arguments[min_length_field] = ""

    errors = _errors(_openai_message(WRAPPER_FORMAT["wrapper_name"], arguments), fixture_specs)

    assert len(errors) == 2
    assert any(f"Invalid wrapper field '{enum_field}'" in error for error in errors)
    assert any(f"Invalid wrapper field '{min_length_field}'" in error for error in errors)


def test_chatml_wrapper_call_uses_configured_wrapper(fixture_specs):
    arguments = _complete_arguments(WRAPPER_FORMAT)
    missing = WRAPPER_FORMAT["argument_required"][0]
    del arguments[missing]
    content = f"tool_call: {WRAPPER_FORMAT['wrapper_name']}\narguments: {json.dumps(arguments)}"
    report = dataset_validator.ExampleReport(index=1, label=True)

    dataset_validator.validate_assistant_content(content, report, wrapper_specs=fixture_specs)

    errors = [issue.message for issue in report.issues if issue.level == "ERROR"]
    assert errors == [
        f"Tool call #1 ({WRAPPER_FORMAT['wrapper_name']}): Missing required '{missing}' field in arguments"
    ]


def test_wrapperless_format_accepts_direct_calls():
    specs = build_wrapper_specs({"direct": WRAPPERLESS_FORMAT})
    assert specs == []

    assert _errors(_openai_message("notes_read", {"path": "notes/a.md"}), specs) == []
    assert _errors(_openai_message("notes_list", {}), specs) == []


def test_direct_call_needs_no_wrapper_fields_beside_wrapper_formats(fixture_specs):
    assert _errors(_openai_message("notes_read", {"path": "notes/a.md"}), fixture_specs) == []


def test_non_object_arguments_are_rejected(fixture_specs):
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"type": "function", "function": {"name": "notes_read", "arguments": "[1, 2]"}}],
    }

    assert _errors(message, fixture_specs) == ["Tool call #1 (notes_read): Arguments must be a JSON object"]


def test_engine_registry_is_the_default():
    configured = load_tool_call_formats()["default"]
    wrapper_name = configured["wrapper_name"]
    arguments = _complete_arguments(configured)

    assert _errors(_openai_message(wrapper_name, arguments)) == []

    missing = configured["argument_required"][-1]
    del arguments[missing]
    assert _errors(_openai_message(wrapper_name, arguments)) == [
        f"Tool call #1 ({wrapper_name}): Missing required '{missing}' field in arguments"
    ]


def test_cli_validates_against_registry_flag(capsys):
    dataset_validator.main(
        [str(FIXTURES / "ticketed_dataset.jsonl"), "--tool-call-formats", str(REGISTRY_PATH)]
    )

    assert "Validated 2 example(s): 0 failed" in capsys.readouterr().out


def test_cli_rejects_missing_registry():
    with pytest.raises(SystemExit, match="Tool-call format registry not found"):
        dataset_validator.main(
            [str(FIXTURES / "ticketed_dataset.jsonl"), "--tool-call-formats", str(FIXTURES / "absent.yaml")]
        )
