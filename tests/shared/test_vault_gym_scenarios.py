from __future__ import annotations

import json
from pathlib import Path

import yaml

from Evaluator.config_loader import ConfigLoader
from shared.environments import EnvironmentValidator
from SynthChat.config.format_resolver import load_tool_call_formats


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


def _configured_tool_response(prompt_case, *commands: str) -> dict:
    """Build one wrapper call in the configured default tool-call format.

    The wrapper name and required argument fields come from
    SynthChat/config/tool_call_formats.yaml; the session and workspace IDs come
    from the case's expected context.
    """
    tool_call_format = load_tool_call_formats()["default"]
    expected_context = prompt_case.metadata["expected_context"]
    arguments = {
        "sessionId": expected_context["session_id"],
        "workspaceId": expected_context["workspace_id"],
        "memory": "Vault gym regression check.",
        "goal": prompt_case.question,
        "tool": ", ".join(commands),
    }
    missing = set(tool_call_format["argument_required"]) - set(arguments)
    assert not missing, f"configured tool-call format requires {sorted(missing)}"
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": tool_call_format["wrapper_name"],
                    "arguments": json.dumps(arguments),
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

    response = _configured_tool_response(
        prompt_case,
        'storage list "Projects/test/"',
        'storage archive "Projects/test/"',
    )

    result = validator.validate_response(
        system_prompt=prompt_case.metadata["system"],
        response=response,
        environment_config=prompt_case.metadata["environment"],
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == [
        "storageManager_list",
        "storageManager_archive",
    ]


def test_vault_gym_update_production_endpoint_case_passes_with_search_read_update():
    _, prompt_case = _load_vault_case("vault_update_production_endpoint_note")
    validator = EnvironmentValidator(backend="local")

    # The scenario's preferred scoring path: search content -> content read -> content replace.
    response = _configured_tool_response(
        prompt_case,
        'search content "api.old.example.com" --paths \'["Operations/"]\'',
        'content read "Operations/production-config.md" 1',
        (
            'content replace "Operations/production-config.md" '
            '"api_base_url: https://api.old.example.com" '
            '"api_base_url: https://api.prod.example.com" 6 6'
        ),
    )

    result = validator.validate_response(
        system_prompt=prompt_case.metadata["system"],
        response=response,
        environment_config=prompt_case.metadata["environment"],
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == [
        "searchManager_content",
        "contentManager_read",
        "contentManager_replace",
    ]


def test_vault_gym_cases_render_mocked_workspace_system_prompt():
    _, prompt_case = _load_vault_case("vault_create_daily_note")
    system_prompt = prompt_case.metadata["system"]

    assert "<available_workspaces>" in system_prompt
    assert '<selected_workspace name="Alpha Lab" id="ws_1732300800000_alphalab">' in system_prompt
    assert "Templates/daily-note.md" in system_prompt
    assert "Projects/Alpha/meeting-notes.md" in system_prompt
