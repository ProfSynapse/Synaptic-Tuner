from __future__ import annotations

import json

from Evaluator.prompt_sets import PromptCase
from Evaluator.protocols import BackendResponse
from Evaluator.reporting import aggregate_stats
from Evaluator.runner import evaluate_cases


class _FakeClient:
    def __init__(self, message):
        self._message = message

    def chat(self, messages):
        return BackendResponse(message=self._message, raw={"message": self._message}, latency_s=0.1)


def test_scoring_prefers_higher_scoring_configured_path():
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "sessionId": "session_1732300800000_eval01234",
                            "workspaceId": "ws_1732300800000_atlasroll",
                            "memory": "Need to locate the template before writing.",
                            "goal": "Create the note using the discovered format.",
                            "constraints": "Use the CLI wrapper.",
                            "tool": (
                                'search search-directory "daily note template" --paths "Templates/", '
                                'content read "Templates/daily-note.md" 1, '
                                'content write "Journal/Daily/2026-03-15.md" "---\\ntype: daily\\n---"'
                            ),
                        }
                    ),
                },
            }
        ]
    }

    case = PromptCase(
        case_id="score_path_case",
        question="Create today's daily note using the vault template.",
        metadata={
            "correct": {
                "any": [
                    {
                        "name": "template_cli",
                        "assertions": [
                            {
                                "type": "jsonpath_regex",
                                "path": "$.tool_calls[0].function.arguments.tool",
                                "pattern": r"search search-directory.*content read.*content write",
                            }
                        ],
                    }
                ]
            },
            "scoring": {
                "paths": [
                    {
                        "name": "wrapper-path",
                        "tier": "preferred",
                        "score": 1.0,
                        "all_tools": ["useTools"],
                    },
                    {
                        "name": "impossible-path",
                        "tier": "acceptable",
                        "score": 0.5,
                        "min_tool_calls": 2,
                    },
                ]
            },
        },
    )

    records = evaluate_cases([case], client=_FakeClient(response))
    record = records[0]

    assert record.status == "pass"
    assert record.scoring is not None
    assert record.scoring.matched_path == "wrapper-path"
    assert record.scoring.matched_tier == "preferred"
    assert record.scoring.awarded_score == 1.0
    assert record.scoring.normalized_score == 1.0

    stats = aggregate_stats(records)
    assert stats["scoring_tested"] == 1
    assert stats["average_score"] == 1.0
    assert stats["normalized_score"] == 1.0


def test_scoring_falls_back_to_lower_configured_path():
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "sessionId": "session_1732300800000_eval01234",
                            "workspaceId": "ws_1732300800000_atlasroll",
                            "memory": "Writing directly.",
                            "goal": "Create the note quickly.",
                            "constraints": "Use the CLI wrapper.",
                            "tool": 'content write "Journal/Daily/2026-03-15.md" "plain body"',
                        }
                    ),
                },
            }
        ]
    }

    case = PromptCase(
        case_id="score_partial_case",
        question="Create today's daily note.",
        metadata={
            "correct": {
                "any": [
                    {
                        "name": "direct_cli",
                        "assertions": [
                            {
                                "type": "jsonpath_regex",
                                "path": "$.tool_calls[0].function.arguments.tool",
                                "pattern": r"^content write",
                            }
                        ],
                    }
                ]
            },
            "scoring": {
                "paths": [
                    {
                        "name": "impossible-path",
                        "tier": "preferred",
                        "score": 1.0,
                        "min_tool_calls": 2,
                    },
                    {
                        "name": "wrapper-path",
                        "tier": "acceptable",
                        "score": 0.4,
                        "all_tools": ["useTools"],
                    },
                ]
            },
        },
    )

    records = evaluate_cases([case], client=_FakeClient(response))
    record = records[0]

    assert record.status == "pass"
    assert record.scoring is not None
    assert record.scoring.matched_path == "wrapper-path"
    assert record.scoring.matched_tier == "acceptable"
    assert record.scoring.awarded_score == 0.4
    assert record.scoring.normalized_score == 0.4
