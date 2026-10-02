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


def _wrapper_response(tool_value):
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "sessionId": "session_1732300800000_eval01234",
                            "workspaceId": "ws_1732300800000_atlasroll",
                            "memory": "Find the template, then write the note.",
                            "goal": "Create the daily note.",
                            "tool": tool_value,
                        }
                    ),
                },
            }
        ]
    }


def _path_results(tool_value, paths):
    case = PromptCase(
        case_id="score_levels_case",
        question="Create today's daily note.",
        metadata={"scoring": {"paths": paths}},
    )
    record = evaluate_cases([case], client=_FakeClient(_wrapper_response(tool_value)))[0]
    assert record.scoring is not None
    return {match.name: match.matched for match in record.scoring.matches}


def test_scoring_paths_match_at_the_level_their_tool_names_are_written_in():
    tool_value = (
        'search directory "daily-note" --paths \'["Templates/"]\', '
        'content read "Templates/daily-note.md" 1, '
        'content write "Journal/Daily/2026-03-15.md" "---\\ntype: daily\\n---\\n"'
    )
    paths = [
        {
            "name": "cli-commands",
            "score": 1.0,
            "ordered_tools": ["search directory", "content read", "content write"],
            "first_tool": "search directory",
            "min_tool_calls": 3,
        },
        {"name": "cli-commands-wrong-order", "score": 0.9, "ordered_tools": ["content write", "content read"]},
        {"name": "catalog-tools", "score": 0.8, "all_tools": ["searchManager_directory", "contentManager_write"]},
        {"name": "wrapper-calls", "score": 0.5, "all_tools": ["useTools"], "max_tool_calls": 1},
        {"name": "call-count-only", "score": 0.2, "min_tool_calls": 2},
    ]

    assert _path_results(tool_value, paths) == {
        "cli-commands": True,
        "cli-commands-wrong-order": False,
        "catalog-tools": True,
        "wrapper-calls": True,
        "call-count-only": False,
    }


def test_scoring_command_paths_do_not_match_an_unparseable_wrapper_command():
    paths = [{"name": "cli-commands", "score": 1.0, "all_tools": ["content write"]}]

    assert _path_results('content write "Journal/Daily/2026-03-15.md" "unterminated', paths) == {
        "cli-commands": False,
    }
