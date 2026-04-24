from __future__ import annotations

import json
from pathlib import Path

from Evaluator.assertions import evaluate_correctness, select_path
from Evaluator.config_loader import ConfigLoader
from Evaluator.prompt_sets import PromptCase
from Evaluator.protocols import BackendResponse
from Evaluator.reporting import aggregate_stats, record_to_dict
from Evaluator.response_view import build_response_view
from Evaluator.runner import evaluate_cases


class _FakeClient:
    def __init__(self, message):
        self._message = message

    def chat(self, messages):
        return BackendResponse(message=self._message, raw={"message": self._message}, latency_s=0.1)


def test_response_view_preserves_raw_and_parses_tool_arguments_without_semantic_expansion():
    response = {
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "workspaceId": "workspace_main",
                            "sessionId": "session_eval",
                            "memory": "Need to copy the requested file.",
                            "goal": "Copy the runbook into a template.",
                            "constraints": "Use the CLI command exactly.",
                            "tool": 'storage copy "Projects/Runbooks/Incident-Response.md" "Projects/Runbooks/Incident-Response-Template.md"',
                        }
                    ),
                },
            }
        ],
    }

    view = build_response_view(response, raw_response={"choices": [{"message": response}]})

    assert select_path(view, "$.raw.tool_calls[0].function.name") == "useTools"
    assert select_path(view, "$.tool_calls[0].function.name") == "useTools"
    assert select_path(view, "$.tool_calls[0].function.arguments.tool").startswith("storage copy")
    assert "storageManager_copy" not in str(view)


def test_correctness_assertions_match_cli_command_directly():
    view = build_response_view(
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "useTools",
                        "arguments": json.dumps({"tool": 'prompt execute-prompt "weekly planning"'}),
                    }
                }
            ]
        }
    )
    correct = {
        "any": [
            {
                "name": "prompt_cli",
                "assertions": [
                    {"type": "jsonpath_equals", "path": "$.tool_calls[0].function.name", "value": "useTools"},
                    {
                        "type": "jsonpath_regex",
                        "path": "$.tool_calls[0].function.arguments.tool",
                        "pattern": r'^prompt execute-prompt\s+"weekly planning"$',
                    },
                ],
            }
        ]
    }

    result = evaluate_correctness(correct, view)

    assert result.passed
    assert result.matched_path == "prompt_cli"


def test_response_view_parses_plain_tool_call_blocks():
    response = '''tool_call: useTools
arguments: {
  "workspaceId": "ws_1732300800000_atlasroll",
  "sessionId": "session_1732300800000_eval01234",
  "memory": "Need to archive a prompt.",
  "goal": "Archive QA Prototype.",
  "constraints": "Do not modify unrelated prompts.",
  "tool": "prompt archive-prompt \\"agent_1732300800004_qa_prototype\\""
}'''

    view = build_response_view(response)

    assert select_path(view, "$.tool_calls[0].name") == "useTools"
    assert select_path(view, "$.tool_calls[0].arguments.tool").startswith("prompt archive-prompt")


def test_runner_uses_correctness_assertions_as_the_only_pass_fail_contract():
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps({"tool": 'storage copy "a.md" "b.md"'}),
                },
            }
        ]
    }
    case = PromptCase(
        case_id="assertion_case",
        question="Copy a.md to b.md",
        metadata={
            "correct": {
                "any": [
                    {
                        "name": "emitted_cli",
                        "assertions": [
                            {
                                "type": "jsonpath_regex",
                                "path": "$.tool_calls[0].function.arguments.tool",
                                "pattern": r'^storage copy\s+"a\.md"\s+"b\.md"$',
                            }
                        ],
                    }
                ]
            }
        },
    )

    record = evaluate_cases([case], client=_FakeClient(response))[0]

    assert record.status == "pass"
    assert record.correctness is not None
    assert record.correctness.passed


def test_config_loader_accepts_messages_and_correct_assertions():
    config_dir = Path("tests/fixtures/evaluator_assertions")

    cases = ConfigLoader(config_dir).load_all_scenarios(["assertion_smoke.yaml"])

    assert len(cases) == 1
    assert cases[0].question == "Update that file."
    assert "correct" in cases[0].metadata
    assert cases[0].chat_messages()[0]["role"] == "system"


def test_tool_prompt_scenario_is_assertion_only():
    scenario = ConfigLoader(Path("Evaluator/config")).load_all_scenarios(["tool_prompts.yaml"])

    assert scenario
    for case in scenario:
        assert "correct" in case.metadata
        assert "expect" not in case.metadata
        assert "behavior_expectations" not in case.metadata
        assert "expected_response_type" not in case.metadata


def test_reporting_serializes_correctness_results():
    case = PromptCase(
        case_id="text_assertion_case",
        question="Which file?",
        metadata={
            "correct": {
                "any": [
                    {
                        "name": "clarification",
                        "assertions": [
                            {"type": "text_regex", "pattern": r"Which file\?"}
                        ],
                    }
                ]
            }
        },
    )

    record = evaluate_cases([case], client=_FakeClient("Which file?"))[0]
    payload = record_to_dict(record)
    stats = aggregate_stats([record])

    assert payload["correctness"]["passed"] is True
    assert payload["correctness_passed"] is True
    assert stats["correctness_tested"] == 1
    assert stats["correctness_passed"] == 1
