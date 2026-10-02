"""`tools` rules in StructureValidator: the `error` template and `_item_schema`.

Rules come from config: the fixture rules under
tests/fixtures/structure_validator/ and checked-in rubric/fitness configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from shared.validation.validators import StructureValidator
from shared.validation.validators.structure_validator import DEFAULT_TOOL_ERROR_TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
RULES = yaml.safe_load(
    (ROOT / "tests" / "fixtures" / "structure_validator" / "tool_rules.yaml").read_text(encoding="utf-8")
)


def _tool_rule(validations):
    return next(rule for rule in validations if "tools" in rule)


def _call(name, arguments):
    return {"tool_calls": [{"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def _valid_batch_args():
    return {
        "owner": "ops",
        "steps": [{"action": "copy", "options": {}}],
        "labels": ["nightly"],
        "grid": [[1, 2], [3.5]],
    }


def _validate(validations, data):
    return StructureValidator().validate(data, validations)


def test_valid_call_passes():
    assert _validate(RULES["templated"], _call("batchJob", _valid_batch_args())) == (True, [])


def test_error_template_formats_each_failure():
    rule = _tool_rule(RULES["templated"])
    arguments = _valid_batch_args()
    del arguments["owner"]
    arguments["labels"] = ["nightly", 7]

    is_valid, errors = _validate(RULES["templated"], _call("batchJob", arguments))

    assert is_valid is False
    assert errors == [
        rule["error"].format(tool_name="batchJob", details="Missing required field 'owner'"),
        rule["error"].format(tool_name="batchJob", details="Field 'labels[1]' must be string, got int"),
    ]


def test_error_template_covers_unknown_tools_and_bad_json():
    rule = _tool_rule(RULES["templated"])
    data = {
        "tool_calls": [
            {"function": {"name": "unlisted", "arguments": "{}"}},
            {"function": {"name": "batchJob", "arguments": "{not json"}},
        ]
    }

    _, errors = _validate(RULES["templated"], data)

    assert errors == [
        rule["error"].format(tool_name="unlisted", details="Unknown tool 'unlisted' (not in manifest)"),
        rule["error"].format(tool_name="batchJob", details="Invalid JSON in arguments"),
    ]


def test_default_template_when_rule_sets_no_error():
    assert "error" not in _tool_rule(RULES["untemplated"])

    _, errors = _validate(RULES["untemplated"], _call("batchJob", {}))

    assert errors == [
        DEFAULT_TOOL_ERROR_TEMPLATE.format(tool_name="batchJob", details="Missing required field 'owner'")
    ]


def test_item_schema_validates_object_items():
    arguments = _valid_batch_args()
    arguments["steps"] = [{"action": "copy", "options": {}}, {"options": []}, "copy"]

    _, errors = _validate(RULES["templated"], _call("batchJob", arguments))

    details = [error.split(": ", 1)[1] for error in errors]
    assert details == [
        "Missing required field 'steps[1].action'",
        "Field 'steps[1].options' must be object, got list",
        "Field 'steps[2]' must be an object",
    ]


def test_item_schema_requires_an_array():
    arguments = _valid_batch_args()
    arguments["steps"] = {"action": "copy"}

    _, errors = _validate(RULES["templated"], _call("batchJob", arguments))

    assert [error.split(": ", 1)[1] for error in errors] == ["Field 'steps' must be an array"]


def test_nested_item_schema_validates_arrays_of_arrays():
    arguments = _valid_batch_args()
    arguments["grid"] = [[1, "two"], 3]

    _, errors = _validate(RULES["templated"], _call("batchJob", arguments))

    assert [error.split(": ", 1)[1] for error in errors] == [
        "Field 'grid[0][1]' must be number, got str",
        "Field 'grid[1]' must be an array",
    ]


@pytest.mark.parametrize(
    "config_path",
    [
        "SynthChat/rubrics/workspace_use_tools_response.yaml",
        "Trainers/sft/configs/fitness/tool_calling.yaml",
    ],
)
def test_checked_in_configs_apply_their_error_template(config_path):
    validations = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))["validations"]
    rule = _tool_rule(validations)
    tool_name = next(iter(rule["tools"]))

    _, errors = _validate([rule], _call(tool_name, {}))

    assert errors
    prefix = rule["error"].split("{details}", 1)[0].format(tool_name=tool_name)
    assert all(error.startswith(prefix) for error in errors)


def test_item_schema_items_still_check_subtool_params():
    config_path = ROOT / "Trainers" / "sft" / "configs" / "fitness" / "tool_calling.yaml"
    rule = _tool_rule(yaml.safe_load(config_path.read_text(encoding="utf-8"))["validations"])
    tool_name, manifest = next(iter(rule["tools"].items()))
    calls_field, calls_schema = next(
        (field, schema) for field, schema in manifest.items() if isinstance(schema, dict) and "_item_schema" in schema
    )
    agent, subtool, param = next(
        (agent, subtool, schema["_required"][0])
        for agent, tools in calls_schema["_subtools"].items()
        for subtool, schema in tools.items()
        if schema.get("_required")
    )
    keys = calls_schema["_subtool_keys"]
    item = {key: ({} if kind == "object" else "x") for key, kind in calls_schema["_item_schema"].items()}
    item.update({keys["group"]: agent, keys["tool"]: subtool, keys["params"]: {}})

    _, errors = _validate([rule], _call(tool_name, {calls_field: [item]}))

    expected = rule["error"].format(
        tool_name=tool_name,
        details=f"{calls_field}[0] - '{agent}.{subtool}' missing required param '{param}'",
    )
    assert expected in errors


def _steps_schema(rule_name):
    return _tool_rule(RULES[rule_name])["tools"]["batchJob"]["steps"]


def _step(group, tool, params):
    keys = _steps_schema("subtools")["_subtool_keys"]
    return {keys["group"]: group, keys["tool"]: tool, keys["params"]: params}


def test_subtool_keys_select_the_item_fields():
    manifest = _steps_schema("subtools")["_subtools"]
    group, tools = next(iter(manifest.items()))
    tool, schema = next(iter(tools.items()))
    first, second = schema["_required"]

    _, errors = _validate(
        RULES["subtools"],
        _call("batchJob", {"steps": [
            _step(group, tool, {first: "a"}),
            _step(group, tool, {first: 1, second: "b"}),
            _step(group, "unlisted", {}),
        ]}),
    )

    assert [error.split(": ", 1)[1] for error in errors] == [
        f"steps[0] - '{group}.{tool}' missing required param '{second}'",
        f"steps[1] - '{group}.{tool}' param '{first}' must be string, got int",
        f"Unknown subtool '{group}.unlisted'. Valid tools for {group}: {list(tools)}",
    ]


def test_subtools_without_subtool_keys_is_a_config_error():
    assert "_subtool_keys" not in _steps_schema("missing_subtool_keys")

    with pytest.raises(ValueError, match="requires '_subtool_keys'"):
        _validate(RULES["missing_subtool_keys"], _call("batchJob", {"steps": [{"module": "files"}]}))
