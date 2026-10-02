"""The configured `command_field` drives wrapper command recovery.

Formats come from the fixture registry under tests/fixtures/tool_call_formats/
or the engine's SynthChat/config/tool_call_formats.yaml; no field name is
hardcoded here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.validation.parsing.configured_formats import (
    build_wrapper_specs,
    extract_lenient_wrapper_arguments,
    get_configured_wrapper_specs,
    sanitize_wrapper_string_fields,
)
from SynthChat.config.format_resolver import load_tool_call_formats

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "tool_call_formats" / "registry.yaml"
FORMATS = load_tool_call_formats(str(REGISTRY_PATH))
SPECS = build_wrapper_specs(FORMATS)
SPEC = SPECS[0]
COMMAND_FIELD = SPEC["command_field"]
OTHER_FIELD = next(name for name in SPEC["required_fields"] if name != COMMAND_FIELD)


def test_spec_carries_the_configured_command_field():
    fmt = next(fmt for fmt in FORMATS.values() if fmt.get("wrapper_name") == SPEC["wrapper_name"])
    assert COMMAND_FIELD == fmt["command_field"]


def test_engine_spec_carries_its_configured_command_field():
    configured = load_tool_call_formats()["default"]
    spec = next(s for s in get_configured_wrapper_specs() if s["wrapper_name"] == configured["wrapper_name"])
    assert spec["command_field"] == configured["command_field"]


def test_command_field_must_be_a_declared_argument_field():
    fmt = dict(FORMATS["ticketed"], command_field="undeclared")

    with pytest.raises(ValueError, match="command_field 'undeclared' is not a declared argument field"):
        build_wrapper_specs({"broken": fmt})


def test_format_without_command_field_has_no_command_recovery():
    fmt = {key: value for key, value in FORMATS["ticketed"].items() if key != "command_field"}
    specs = build_wrapper_specs({"plain": fmt})
    leaked = f'close 42","{OTHER_FIELD}":"x'
    args = {OTHER_FIELD: "x", COMMAND_FIELD: leaked}

    assert specs[0]["command_field"] is None
    assert sanitize_wrapper_string_fields(args, function_name=fmt["wrapper_name"], specs=specs) == args


def test_sanitize_cuts_leaked_fields_from_the_command_field():
    args = {OTHER_FIELD: "T-1", COMMAND_FIELD: f'tickets close 42","{OTHER_FIELD}":"T-1'}

    cleaned = sanitize_wrapper_string_fields(args, function_name=SPEC["wrapper_name"], specs=SPECS)

    assert cleaned == {OTHER_FIELD: "T-1", COMMAND_FIELD: "tickets close 42"}


def test_lenient_recovery_reads_the_command_field_up_to_the_next_field():
    command = 'tickets note 42 "needs "quotes""'
    raw = (
        "{"
        + f'"{COMMAND_FIELD}": "{command}", '
        + f'"{OTHER_FIELD}": {json.dumps("T-42")}'
        + ", "  # truncated: malformed JSON
    )

    extracted = extract_lenient_wrapper_arguments(raw, specs=SPECS)

    assert extracted == {OTHER_FIELD: "T-42", COMMAND_FIELD: command}
