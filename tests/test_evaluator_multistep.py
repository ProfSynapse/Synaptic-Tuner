from __future__ import annotations

import json

from Evaluator.prompt_sets import PromptCase
from Evaluator.protocols import BackendResponse
from Evaluator.runner import evaluate_cases
from shared.environments import EnvironmentValidator


class _SequenceClient:
    def __init__(self, responses, expected_substrings=None):
        self._responses = list(responses)
        self._expected_substrings = list(expected_substrings or [])
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        idx = len(self.calls) - 1
        if idx >= len(self._responses):
            raise AssertionError("No more fake responses configured")
        for expected in self._expected_substrings[idx]:
            joined = "\n".join(message.get("content", "") for message in messages)
            assert expected in joined
        return BackendResponse(
            message=self._responses[idx],
            raw={"message": self._responses[idx]},
            latency_s=0.1,
        )


def _tool_response(calls):
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "context": {
                                "sessionId": "session_1732300800000_loopcase",
                                "workspaceId": "ws_1732300800000_loopcase",
                                "memory": "Loop test",
                                "goal": "Complete the task",
                            },
                            "calls": calls,
                        }
                    ),
                },
            }
        ]
    }


def test_multistep_environment_loop_feeds_tool_results_back_to_model():
    responses = [
        _tool_response(
            [
                {
                    "agent": "searchManager",
                    "tool": "searchDirectory",
                    "params": {"query": "daily-note", "paths": ["Templates/"]},
                }
            ]
        ),
        _tool_response(
            [
                {
                    "agent": "contentManager",
                    "tool": "read",
                    "params": {"path": "Templates/daily-note.md", "startLine": 1},
                }
            ]
        ),
        _tool_response(
            [
                {
                    "agent": "contentManager",
                    "tool": "write",
                    "params": {
                        "path": "Journal/Daily/2026-03-15.md",
                        "content": (
                            "---\n"
                            "title: 2026-03-15\n"
                            "type: daily\n"
                            "tags:\n"
                            "  - journal\n"
                            "mood: focused\n"
                            "---\n"
                            "# Daily Note\n\n"
                            "## Linked Notes\n"
                            "- [[Projects/Alpha/meeting-notes]]\n"
                        ),
                        "overwrite": True,
                    },
                }
            ]
        ),
        {"content": "Done. The daily note is created."},
    ]
    client = _SequenceClient(
        responses,
        expected_substrings=[
            [],
            ["Templates/daily-note.md"],
            ["title: Daily Note Template"],
            ["Journal/Daily/2026-03-15.md"],
        ],
    )
    case = PromptCase(
        case_id="loop_daily_note",
        question="Create today's daily note from the vault template.",
        expected_tools=[
            "searchManager_searchDirectory",
            "contentManager_read",
            "contentManager_write",
        ],
        metadata={
            "system": "Loop system prompt",
            "environment": {
                "allowed_tools": [
                    "searchManager_searchDirectory",
                    "contentManager_read",
                    "contentManager_write",
                ],
                "max_steps": 4,
                "loop": {
                    "enabled": True,
                    "max_turns": 4,
                    "max_tool_steps": 4,
                    "stop_on_text_response": True,
                },
                "fixture": {
                    "directories": ["Journal/Daily"],
                    "notes": [
                        {
                            "path": "Templates/daily-note.md",
                            "frontmatter": {
                                "title": "Daily Note Template",
                                "type": "daily",
                                "tags": ["journal"],
                                "mood": "neutral",
                            },
                            "body": "# Daily Note\n\n## Linked Notes\n",
                        },
                        {
                            "path": "Projects/Alpha/meeting-notes.md",
                            "frontmatter": {"title": "Meeting Notes", "type": "meeting"},
                            "body": "Meeting summary.",
                        },
                    ],
                },
                "assertions": [
                    {"type": "path_exists", "path": "Journal/Daily/2026-03-15.md"},
                    {"type": "frontmatter_field_equals", "path": "Journal/Daily/2026-03-15.md", "field": "mood", "value": "focused"},
                    {"type": "file_contains", "path": "Journal/Daily/2026-03-15.md", "text": "[[Projects/Alpha/meeting-notes]]"},
                ],
            }
        },
    )

    record = evaluate_cases(
        [case],
        client=client,
        environment_validator=EnvironmentValidator(backend="local"),
    )[0]

    assert record.passed is True
    assert record.validator is not None
    assert [tool.name for tool in record.validator.tool_calls] == [
        "searchManager_searchDirectory",
        "contentManager_read",
        "contentManager_write",
    ]
    assert record.environment is not None
    assert record.environment.passed is True
    assert record.environment.episode_trace is not None
    assert record.environment.episode_trace.total_turns == 4
    assert record.environment.episode_trace.stop_reason == "text_response"
    assert record.conversation_trace is not None
    assert [entry["kind"] for entry in record.conversation_trace[:2]] == ["prompt_message", "prompt_message"]
    assert any(entry["kind"] == "tool_feedback" for entry in record.conversation_trace)
    assert record.conversation_trace[-1]["kind"] == "assistant_response"


def test_multistep_environment_loop_can_stop_when_environment_passes():
    client = _SequenceClient(
        [
            _tool_response(
                [
                    {
                        "agent": "contentManager",
                        "tool": "write",
                        "params": {
                            "path": "Inbox/final.md",
                            "content": "complete",
                            "overwrite": True,
                        },
                    }
                ]
            )
        ],
        expected_substrings=[[]],
    )
    case = PromptCase(
        case_id="loop_stop_on_environment",
        question="Write the final file.",
        metadata={
            "system": "Loop system prompt",
            "environment": {
                "allowed_tools": ["contentManager_write"],
                "max_steps": 2,
                "loop": {
                    "enabled": True,
                    "max_turns": 3,
                    "stop_on_environment_pass": True,
                },
                "fixture": {"directories": ["Inbox"]},
                "assertions": [{"type": "path_exists", "path": "Inbox/final.md"}],
            }
        },
    )

    record = evaluate_cases(
        [case],
        client=client,
        environment_validator=EnvironmentValidator(backend="local"),
    )[0]

    assert record.passed is True
    assert record.environment is not None
    assert record.environment.episode_trace is not None
    assert record.environment.episode_trace.total_turns == 1
    assert record.environment.episode_trace.stop_reason == "environment_passed"


def test_multistep_environment_loop_can_require_final_text_after_pass():
    client = _SequenceClient(
        [
            _tool_response(
                [
                    {
                        "agent": "contentManager",
                        "tool": "write",
                        "params": {
                            "path": "Inbox/final.md",
                            "content": "complete",
                            "overwrite": True,
                        },
                    }
                ]
            ),
            {"content": "Done. I wrote the file.", "tool_calls": None},
        ],
        expected_substrings=[
            [],
            ["Reply to the user with a brief final text-only response."],
        ],
    )
    case = PromptCase(
        case_id="loop_final_text_after_pass",
        question="Write the final file.",
        metadata={
            "system": "Loop system prompt",
            "environment": {
                "allowed_tools": ["contentManager_write"],
                "max_steps": 2,
                "loop": {
                    "enabled": True,
                    "max_turns": 3,
                    "stop_on_environment_pass": True,
                    "require_final_text_after_pass": True,
                    "final_text_prompt": "Reply to the user with a brief final text-only response.",
                },
                "fixture": {"directories": ["Inbox"]},
                "assertions": [{"type": "path_exists", "path": "Inbox/final.md"}],
            }
        },
    )

    record = evaluate_cases(
        [case],
        client=client,
        environment_validator=EnvironmentValidator(backend="local"),
    )[0]

    assert record.passed is True
    assert record.environment is not None
    assert record.environment.episode_trace is not None
    assert record.environment.episode_trace.total_turns == 2
    assert record.environment.episode_trace.stop_reason == "environment_passed_final_text"
    assert record.conversation_trace is not None
    assert any(entry["kind"] == "final_text_request" for entry in record.conversation_trace)


def test_multistep_environment_loop_prefers_success_over_tool_budget_on_same_turn():
    client = _SequenceClient(
        [
            _tool_response(
                [
                    {
                        "agent": "contentManager",
                        "tool": "write",
                        "params": {
                            "path": "Inbox/final.md",
                            "content": "complete",
                            "overwrite": True,
                        },
                    },
                    {
                        "agent": "contentManager",
                        "tool": "read",
                        "params": {"path": "Inbox/final.md", "startLine": 1},
                    },
                ]
            ),
            {"content": "Done. I wrote the file.", "tool_calls": None},
        ],
        expected_substrings=[
            [],
            ["Reply to the user with a brief final text-only response."],
        ],
    )
    case = PromptCase(
        case_id="loop_success_beats_budget",
        question="Write the final file.",
        metadata={
            "system": "Loop system prompt",
            "environment": {
                "allowed_tools": ["contentManager_write", "contentManager_read"],
                "max_steps": 3,
                "loop": {
                    "enabled": True,
                    "max_turns": 3,
                    "max_tool_steps": 1,
                    "stop_on_environment_pass": True,
                    "require_final_text_after_pass": True,
                    "final_text_prompt": "Reply to the user with a brief final text-only response.",
                },
                "fixture": {"directories": ["Inbox"]},
                "assertions": [{"type": "path_exists", "path": "Inbox/final.md"}],
            }
        },
    )

    record = evaluate_cases(
        [case],
        client=client,
        environment_validator=EnvironmentValidator(backend="local"),
    )[0]

    assert record.passed is True
    assert record.environment is not None
    assert record.environment.episode_trace is not None
    assert record.environment.episode_trace.stop_reason == "environment_passed_final_text"


def test_multistep_environment_loop_rejects_text_before_completion_when_final_text_required():
    client = _SequenceClient(
        [
            {"content": "Done. I handled it.", "tool_calls": None},
        ],
        expected_substrings=[[]],
    )
    case = PromptCase(
        case_id="loop_text_before_completion",
        question="Write the final file.",
        metadata={
            "system": "Loop system prompt",
            "environment": {
                "allowed_tools": ["contentManager_write"],
                "max_steps": 2,
                "loop": {
                    "enabled": True,
                    "max_turns": 2,
                    "stop_on_text_response": True,
                    "require_final_text_after_pass": True,
                },
                "fixture": {"directories": ["Inbox"]},
                "assertions": [{"type": "path_exists", "path": "Inbox/final.md"}],
            }
        },
    )

    record = evaluate_cases(
        [case],
        client=client,
        environment_validator=EnvironmentValidator(backend="local"),
    )[0]

    assert record.passed is False
    assert record.environment is not None
    assert record.environment.episode_trace is not None
    assert record.environment.episode_trace.stop_reason == "text_response_before_completion"


def test_agentic_loop_can_recover_after_a_bad_first_step():
    client = _SequenceClient(
        [
            _tool_response(
                [
                    {
                        "agent": "contentManager",
                        "tool": "read",
                        "params": {"path": "Journal/Daily/2026-03-15.md", "startLine": 1},
                    }
                ]
            ),
            _tool_response(
                [
                    {
                        "agent": "searchManager",
                        "tool": "searchDirectory",
                        "params": {"query": "daily-note", "paths": ["Templates/"]},
                    }
                ]
            ),
            _tool_response(
                [
                    {
                        "agent": "contentManager",
                        "tool": "write",
                        "params": {
                            "path": "Journal/Daily/2026-03-15.md",
                            "content": (
                                "---\n"
                                "title: 2026-03-15\n"
                                "type: daily\n"
                                "tags:\n"
                                "  - journal\n"
                                "mood: focused\n"
                                "---\n"
                                "# Daily Note\n"
                            ),
                            "overwrite": True,
                        },
                    }
                ]
            ),
        ],
        expected_substrings=[
            [],
            ["Tool execution results", "Tool 'contentManager_read' failed"],
            ["Templates/daily-note.md"],
        ],
    )
    case = PromptCase(
        case_id="loop_recovery_case",
        question="Create today's daily note.",
        metadata={
            "system": "Loop system prompt",
            "environment": {
                "allowed_tools": [
                    "contentManager_read",
                    "searchManager_searchDirectory",
                    "contentManager_write",
                ],
                "max_steps": 4,
                "loop": {
                    "enabled": True,
                    "mode": "agentic",
                    "max_turns": 4,
                    "continue_on_execution_error": True,
                    "stop_on_environment_pass": True,
                },
                "fixture": {
                    "directories": ["Journal/Daily"],
                    "notes": [
                        {
                            "path": "Templates/daily-note.md",
                            "frontmatter": {"title": "Daily Note Template", "type": "daily"},
                            "body": "# Daily Note\n",
                        }
                    ],
                },
                "assertions": [
                    {"type": "path_exists", "path": "Journal/Daily/2026-03-15.md"},
                    {"type": "frontmatter_field_equals", "path": "Journal/Daily/2026-03-15.md", "field": "mood", "value": "focused"},
                ],
            }
        },
    )

    record = evaluate_cases(
        [case],
        client=client,
        environment_validator=EnvironmentValidator(backend="local"),
    )[0]

    assert record.passed is True
    assert record.environment is not None
    assert record.environment.episode_trace is not None
    assert record.environment.episode_trace.recovered_after_error is True
    assert record.environment.episode_trace.stop_reason == "environment_passed"


def test_agentic_loop_stops_repeated_failed_steps_as_stuck():
    repeated_failure = _tool_response(
        [
            {
                "agent": "contentManager",
                "tool": "read",
                "params": {"path": "Inbox/missing.md", "startLine": 1},
            }
        ]
    )
    client = _SequenceClient(
        [repeated_failure, repeated_failure, repeated_failure],
        expected_substrings=[
            [],
            ["Tool execution results", "missing.md"],
            ["Tool execution results", "missing.md"],
        ],
    )
    case = PromptCase(
        case_id="loop_stuck_case",
        question="Read the missing file.",
        metadata={
            "system": "Loop system prompt",
            "environment": {
                "allowed_tools": ["contentManager_read"],
                "max_steps": 8,
                "loop": {
                    "enabled": True,
                    "mode": "agentic",
                    "max_turns": 6,
                    "max_tool_steps": 8,
                    "continue_on_execution_error": True,
                    "stuck_repeat_limit": 2,
                    "no_progress_window": 3,
                },
                "fixture": {"directories": ["Inbox"]},
                "assertions": [{"type": "path_exists", "path": "Inbox/missing.md"}],
            }
        },
    )

    record = evaluate_cases(
        [case],
        client=client,
        environment_validator=EnvironmentValidator(backend="local"),
    )[0]

    assert record.failed is True
    assert record.environment is not None
    assert record.environment.episode_trace is not None
    assert record.environment.episode_trace.stop_reason == "stuck_repeated_failure"
    assert record.environment.episode_trace.total_turns == 2
    assert all(step.state_changed is False for step in record.environment.episode_trace.steps)
