from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from Evaluator.config_loader import ConfigLoader
from Evaluator.protocols import BackendResponse
from Evaluator.runner import evaluate_cases
from shared.environments import EnvironmentValidator


CONFIG_DIR = Path(__file__).resolve().parents[2] / "Evaluator" / "config"


class _SequenceClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        idx = len(self.calls) - 1
        if idx >= len(self._responses):
            raise AssertionError("No more fake responses configured")
        response = self._responses[idx]
        return BackendResponse(message=response, raw={"message": response}, latency_s=0.1)


def _tool_response(tool_command: str, session_id: str = "session_1732300800000_workspace001"):
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "sessionId": session_id,
                            "workspaceId": "ws_1732300800000_atlasops",
                            "memory": "Continue the workspace task from current state.",
                            "goal": "Inspect relevant files, then make the requested workspace-local change.",
                            "tool": tool_command,
                        }
                    ),
                },
            }
        ]
    }


def test_workspace_multiturn_preset_loads_scenario_pack():
    loader = ConfigLoader(CONFIG_DIR)
    run_config = loader.load_eval_run(preset="workspace_multiturn")
    cases = loader.load_all_scenarios(run_config.scenarios)
    case_ids = {case.case_id for case in cases}

    assert run_config.scenarios == ["workspace_multiturn.yaml"]
    assert {
        "workspace_state_find_read_update_launch_index",
        "workspace_state_continue_prior_search_archive_superseded_rfc",
        "workspace_search_read_create_oauth_brief",
        "workspace_list_disambiguate_move_acme_escalation",
    }.issubset(case_ids)


def test_workspace_multiturn_system_prompt_includes_state_and_fixture_files():
    loader = ConfigLoader(CONFIG_DIR)
    cases = {
        case.case_id: case for case in loader.load_all_scenarios(["workspace_multiturn.yaml"])
    }
    system_prompt = cases["workspace_state_find_read_update_launch_index"].metadata["system"]

    assert '<selected_workspace name="Atlas Ops" id="ws_1732300800000_atlasops">' in system_prompt
    assert "session_launch_readiness" in system_prompt
    assert "Projects/Atlas/Launch/checklists/customer-comms.md" in system_prompt
    assert "Projects/Phoenix/Launch/index.md" in system_prompt


def test_workspace_multiturn_loop_passes_with_search_read_update_sequence(monkeypatch):
    temp_root = CONFIG_DIR.parents[1] / "tmp" / f"workspace_multiturn_test_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)

    class _WorkspaceTemporaryDirectory:
        def __init__(self, prefix=None):
            self.name = str(temp_root / f"{prefix or 'tmp'}{uuid.uuid4().hex}")
            Path(self.name).mkdir(parents=True, exist_ok=False)

        def cleanup(self):
            shutil.rmtree(self.name, ignore_errors=True)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _WorkspaceTemporaryDirectory)

    loader = ConfigLoader(CONFIG_DIR)
    cases = {
        case.case_id: case for case in loader.load_all_scenarios(["workspace_multiturn.yaml"])
    }
    case = cases["workspace_state_find_read_update_launch_index"]
    updated_index = (
        "Atlas Launch status: Final go/no-go: go. "
        "Checklist: [[Projects/Atlas/Launch/checklists/customer-comms]]"
    )
    client = _SequenceClient(
        [
            _tool_response('search content "Customer comms approved for launch"'),
            _tool_response('content read "Projects/Atlas/Launch/checklists/customer-comms.md" 1'),
            _tool_response('content read "Projects/Atlas/Launch/index.md" 1'),
            _tool_response(
                'content write "Projects/Atlas/Launch/index.md" '
                f'"{updated_index}" --overwrite'
            ),
            {"content": "Updated the Atlas launch index with the go/no-go line and checklist link."},
        ]
    )

    try:
        record = evaluate_cases(
            [case],
            client=client,
            environment_validator=EnvironmentValidator(backend="local"),
        )[0]

        assert record.passed is True
        assert record.environment is not None
        assert record.environment.passed is True
        assert record.environment.episode_trace is not None
        assert record.environment.episode_trace.stop_reason == "environment_passed_final_text"
        assert record.scoring is not None
        assert record.scoring.matched_path == "search-read-update-index"
        assert record.conversation_trace is not None
        assert any(entry["kind"] == "tool_feedback" for entry in record.conversation_trace)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
