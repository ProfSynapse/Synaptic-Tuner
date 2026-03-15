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


def test_scoring_prefers_multi_step_path_over_acceptable_direct_path():
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "context": {
                                "sessionId": "session_1732300800000_eval01234",
                                "workspaceId": "ws_1732300800000_atlasroll",
                                "memory": "Need to locate the template before writing.",
                                "goal": "Create the note using the discovered format.",
                            },
                            "calls": [
                                {
                                    "agent": "searchManager",
                                    "tool": "searchDirectory",
                                    "params": {"query": "daily note template", "paths": ["Templates/"]},
                                },
                                {
                                    "agent": "contentManager",
                                    "tool": "read",
                                    "params": {"path": "Templates/daily-note.md", "startLine": 1},
                                },
                                {
                                    "agent": "contentManager",
                                    "tool": "write",
                                    "params": {"path": "Journal/Daily/2026-03-15.md", "content": "---\ntype: daily\n---"},
                                },
                            ],
                        }
                    ),
                },
            }
        ]
    }

    case = PromptCase(
        case_id="score_path_case",
        question="Create today's daily note using the vault template.",
        acceptable_tools=["searchManager_searchDirectory", "contentManager_read", "contentManager_write"],
        metadata={
            "scoring": {
                "paths": [
                    {
                        "name": "template-driven",
                        "tier": "preferred",
                        "score": 1.0,
                        "ordered_tools": [
                            "searchManager_searchDirectory",
                            "contentManager_read",
                            "contentManager_write",
                        ],
                    },
                    {
                        "name": "direct-write",
                        "tier": "acceptable",
                        "score": 0.5,
                        "all_tools": ["contentManager_write"],
                    },
                ]
            }
        },
    )

    records = evaluate_cases([case], client=_FakeClient(response))
    record = records[0]

    assert record.scoring is not None
    assert record.scoring.matched_path == "template-driven"
    assert record.scoring.matched_tier == "preferred"
    assert record.scoring.awarded_score == 1.0
    assert record.scoring.normalized_score == 1.0

    stats = aggregate_stats(records)
    assert stats["scoring_tested"] == 1
    assert stats["average_score"] == 1.0
    assert stats["normalized_score"] == 1.0


def test_scoring_falls_back_to_lower_acceptable_path():
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "context": {
                                "sessionId": "session_1732300800000_eval01234",
                                "workspaceId": "ws_1732300800000_atlasroll",
                                "memory": "Writing directly.",
                                "goal": "Create the note quickly.",
                            },
                            "calls": [
                                {
                                    "agent": "contentManager",
                                    "tool": "write",
                                    "params": {"path": "Journal/Daily/2026-03-15.md", "content": "plain body"},
                                }
                            ],
                        }
                    ),
                },
            }
        ]
    }

    case = PromptCase(
        case_id="score_partial_case",
        question="Create today's daily note.",
        acceptable_tools=["contentManager_write"],
        metadata={
            "scoring": {
                "paths": [
                    {
                        "name": "template-driven",
                        "tier": "preferred",
                        "score": 1.0,
                        "ordered_tools": [
                            "searchManager_searchDirectory",
                            "contentManager_read",
                            "contentManager_write",
                        ],
                    },
                    {
                        "name": "direct-write",
                        "tier": "acceptable",
                        "score": 0.4,
                        "all_tools": ["contentManager_write"],
                    },
                ]
            }
        },
    )

    records = evaluate_cases([case], client=_FakeClient(response))
    record = records[0]

    assert record.scoring is not None
    assert record.scoring.matched_path == "direct-write"
    assert record.scoring.matched_tier == "acceptable"
    assert record.scoring.awarded_score == 0.4
    assert record.scoring.normalized_score == 0.4
