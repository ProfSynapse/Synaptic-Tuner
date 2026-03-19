from __future__ import annotations

import json
from pathlib import Path

import yaml

from Evaluator.config_loader import ConfigLoader
from shared.environments import EnvironmentValidator


VAULT_GYM_PATH = (
    Path(__file__).resolve().parents[2] / "Evaluator" / "config" / "scenarios" / "vault_gym.yaml"
)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "Evaluator" / "config"


def _load_vault_case(case_id: str):
    data = yaml.safe_load(VAULT_GYM_PATH.read_text(encoding="utf-8"))
    loader = ConfigLoader(CONFIG_DIR)
    prompt_cases = {
        case.case_id: case for case in loader.load_all_scenarios(["vault_gym.yaml"])
    }
    for case in (data or {}).get("tests") or []:
        if case.get("id") == case_id:
            prompt_case = prompt_cases[case_id]
            return case, prompt_case
    raise AssertionError(f"Missing vault gym case: {case_id}")


def _use_tools_response(calls: list[dict]) -> dict:
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps({"calls": calls}),
                },
            }
        ]
    }


def test_vault_gym_includes_new_environment_behavior_cases():
    data = yaml.safe_load(VAULT_GYM_PATH.read_text(encoding="utf-8"))
    case_ids = {case["id"] for case in data["tests"]}

    assert "vault_archive_empty_test_folder" in case_ids
    assert "vault_update_production_endpoint_note" in case_ids
    assert "vault_archive_only_deprecated_api_notes" in case_ids
    assert "vault_read_before_replace_settings" in case_ids
    assert "vault_continue_inbox_organization" in case_ids


def test_vault_gym_archive_empty_folder_case_passes_with_verified_delete():
    _, prompt_case = _load_vault_case("vault_archive_empty_test_folder")
    validator = EnvironmentValidator(backend="local")

    response = _use_tools_response(
        [
            {
                "agent": "storageManager",
                "tool": "list",
                "params": {"path": "Projects/test/"},
            },
            {
                "agent": "storageManager",
                "tool": "archive",
                "params": {"path": "Projects/test/", "recursive": True},
            },
        ]
    )

    result = validator.validate_response(
        system_prompt=prompt_case.metadata["system"],
        response=response,
        environment_config=prompt_case.metadata["environment"],
        expected_tools=prompt_case.expected_tools,
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == [
        "storageManager_list",
        "storageManager_archive",
    ]


def test_vault_gym_update_production_endpoint_case_passes_with_search_read_update():
    _, prompt_case = _load_vault_case("vault_update_production_endpoint_note")
    validator = EnvironmentValidator(backend="local")

    updated_note = """---
title: Production Config
type: config
environment: production
---
api_base_url: https://api.prod.example.com
retry_policy: exponential
owner: platform
"""

    response = _use_tools_response(
        [
            {
                "agent": "searchManager",
                "tool": "searchContent",
                "params": {"query": "api.old.example.com", "path": "Operations/"},
            },
            {
                "agent": "contentManager",
                "tool": "read",
                "params": {"path": "Operations/production-config.md", "startLine": 1},
            },
            {
                "agent": "contentManager",
                "tool": "update",
                "params": {
                    "path": "Operations/production-config.md",
                    "content": updated_note,
                    "startLine": 1,
                    "overwrite": True,
                },
            },
        ]
    )

    result = validator.validate_response(
        system_prompt=prompt_case.metadata["system"],
        response=response,
        environment_config=prompt_case.metadata["environment"],
        expected_tools=prompt_case.expected_tools,
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == [
        "searchManager_searchContent",
        "contentManager_read",
        "contentManager_update",
    ]


def test_vault_gym_cases_render_mocked_workspace_system_prompt():
    _, prompt_case = _load_vault_case("vault_create_daily_note")
    system_prompt = prompt_case.metadata["system"]

    assert "<available_workspaces>" in system_prompt
    assert '<selected_workspace name="Alpha Lab" id="ws_1732300800000_alphalab">' in system_prompt
    assert "Templates/daily-note.md" in system_prompt
    assert "Projects/Alpha/meeting-notes.md" in system_prompt
