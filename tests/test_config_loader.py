from __future__ import annotations

from pathlib import Path

from Evaluator.config_loader import ConfigLoader
from shared.environments.fixture_parser import EnvironmentFixture


CONFIG_DIR = Path(__file__).resolve().parents[1] / "Evaluator" / "config"


def test_config_loader_renders_mocked_workspace_prompt_from_system_context():
    loader = ConfigLoader(CONFIG_DIR)
    cases = {case.case_id: case for case in loader.load_all_scenarios(["vault_gym.yaml"])}

    case = cases["vault_triage_rate_limit_note"]
    system_prompt = case.metadata["system"]

    assert "<session_context>" in system_prompt
    assert '<selected_workspace name="Alpha Lab" id="ws_1732300800000_alphalab">' in system_prompt
    assert "<note_contents>" in system_prompt
    assert "Files:" in system_prompt
    assert "Inbox/api-rate-limits.md" in system_prompt
    assert "agent_1732300800000_organizer - Vault Organizer" in system_prompt


def test_config_loader_merges_environment_defaults_into_case_metadata():
    loader = ConfigLoader(CONFIG_DIR)
    cases = {case.case_id: case for case in loader.load_all_scenarios(["vault_gym.yaml"])}

    case = cases["vault_continue_inbox_organization"]
    environment = case.metadata["environment"]

    assert environment["max_steps"] == 8
    assert "storageManager_move" in environment["allowed_tools"]
    assert environment["execution"]["strict_schema"] is True
    assert environment["loop"]["enabled"] is True
    assert environment["loop"]["mode"] == "agentic"
    assert environment["loop"]["stop_on_environment_pass"] is True
    assert environment["loop"]["stop_on_text_response"] is True


def test_config_loader_infers_expected_context_from_system_context():
    loader = ConfigLoader(CONFIG_DIR)
    cases = {case.case_id: case for case in loader.load_all_scenarios(["vault_gym.yaml"])}

    case = cases["vault_promote_inbox_note"]
    expected_context = case.metadata["expected_context"]

    assert expected_context["session_id"] == "session_1732300800000_vault001"
    assert expected_context["workspace_id"] == "ws_1732300800000_alphalab"
    assert "ws_1732300800000_alphalab" in expected_context["workspace_ids"]
    assert "agent_1732300800000_docwriter" in expected_context["agent_ids"]


def test_config_loader_merged_fixture_from_local_path_source(tmp_path: Path):
    source = tmp_path / "real-vault"
    (source / "Notes").mkdir(parents=True)
    (source / "Notes" / "todo.md").write_text("buy milk", encoding="utf-8")

    from Evaluator.config_loader import _merged_fixture_from_config

    fixture = _merged_fixture_from_config(
        {
            "fixture": {
                "source": {
                    "type": "local_path",
                    "path": str(source),
                }
            }
        }
    )

    assert isinstance(fixture, EnvironmentFixture)
    assert "Notes" in fixture.directories
    assert fixture.files["Notes/todo.md"] == "buy milk"
