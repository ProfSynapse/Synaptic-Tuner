from __future__ import annotations

import json
from pathlib import Path

from shared.environments import EnvironmentValidator
from shared.environments.fixture_parser import EnvironmentFixture, merge_environment_fixture


def test_merge_environment_fixture_supports_obsidian_note_shorthand():
    merged = merge_environment_fixture(
        EnvironmentFixture(directories=["Inbox"], files={"README.md": "# Root"}),
        {
            "directories": ["Projects/Alpha"],
            "notes": [
                {
                    "path": "Inbox/alpha-prototype.md",
                    "frontmatter": {
                        "title": "Alpha Prototype",
                        "status": "inbox",
                        "tags": ["fleeting", "alpha"],
                    },
                    "body": "Need to compare RAG vs fine-tune for support.",
                }
            ],
        },
    )

    assert "Inbox" in merged.directories
    assert "Projects/Alpha" in merged.directories
    assert merged.files["README.md"] == "# Root"
    note = merged.files["Inbox/alpha-prototype.md"]
    assert note.startswith("---\n")
    assert "status: inbox" in note
    assert "- fleeting" in note
    assert "Need to compare RAG vs fine-tune for support." in note


def test_merge_environment_fixture_can_load_from_local_path(tmp_path: Path):
    source = tmp_path / "vault"
    (source / "Inbox").mkdir(parents=True)
    (source / "Inbox" / "capture.md").write_text("real note body", encoding="utf-8")
    (source / "README.md").write_text("# Real Vault", encoding="utf-8")

    merged = merge_environment_fixture(
        EnvironmentFixture(),
        {
            "source": {
                "type": "local_path",
                "path": str(source),
            }
        },
    )

    assert "Inbox" in merged.directories
    assert merged.files["Inbox/capture.md"] == "real note body"
    assert merged.files["README.md"] == "# Real Vault"


def test_environment_validator_applies_explicit_fixture_and_frontmatter_assertions():
    validator = EnvironmentValidator(backend="local")

    result = validator.validate_response(
        system_prompt="",
        response={"content": "No tool call needed."},
        environment_config={
            "fixture": {
                "notes": [
                    {
                        "path": "Inbox/capture.md",
                        "frontmatter": {"title": "Capture", "status": "inbox"},
                        "body": "Raw note body.",
                    }
                ]
            },
            "assertions": [
                {"type": "path_exists", "path": "Inbox/capture.md"},
                {"type": "frontmatter_has_key", "path": "Inbox/capture.md", "field": "title"},
                {"type": "frontmatter_has_keys", "path": "Inbox/capture.md", "fields": ["title", "status"]},
                {"type": "frontmatter_field_equals", "path": "Inbox/capture.md", "field": "status", "value": "inbox"},
            ],
        },
    )

    assert result.passed is True
    assert result.assertions_run == 4


def test_environment_validator_frontmatter_has_keys_reports_missing_fields():
    validator = EnvironmentValidator(backend="local")

    result = validator.validate_response(
        system_prompt="",
        response={"content": "No tool call needed."},
        environment_config={
            "fixture": {
                "notes": [
                    {
                        "path": "Journal/Daily/2023-10-05.md",
                        "frontmatter": {"date": "2023-10-05"},
                        "body": "## Summary",
                    }
                ]
            },
            "assertions": [
                {"type": "frontmatter_has_keys", "path": "Journal/Daily/2023-10-05.md", "fields": ["title", "date"]},
            ],
        },
    )

    assert result.passed is False
    assert any("missing required keys: title" in issue.message for issue in result.issues)


def test_environment_validator_can_copy_real_local_path_into_runtime(tmp_path: Path):
    source = tmp_path / "workspace"
    (source / "Docs").mkdir(parents=True)
    (source / "Docs" / "spec.md").write_text("spec v1", encoding="utf-8")

    validator = EnvironmentValidator(backend="local")
    result = validator.validate_response(
        system_prompt="",
        response={"content": "No tool call needed."},
        environment_config={
            "fixture": {
                "local_path": str(source),
            },
            "assertions": [
                {"type": "path_exists", "path": "Docs/spec.md"},
                {"type": "file_contains", "path": "Docs/spec.md", "text": "spec v1"},
            ],
        },
    )

    assert result.passed is True


def test_environment_validator_updates_note_and_checks_frontmatter_contains():
    validator = EnvironmentValidator(backend="local")

    updated_note = """---
title: Alpha Prototype
status: active
tags:
  - alpha
  - project
---
Need to compare RAG vs fine-tune for support.
"""

    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "contentManager_write",
                    "arguments": json.dumps(
                        {
                            "path": "Projects/Alpha/alpha-prototype.md",
                            "content": updated_note,
                            "overwrite": True,
                        }
                    ),
                },
            }
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        environment_config={
            "fixture": {
                "directories": ["Projects/Alpha"],
            },
            "assertions": [
                {"type": "path_exists", "path": "Projects/Alpha/alpha-prototype.md"},
                {
                    "type": "frontmatter_field_equals",
                    "path": "Projects/Alpha/alpha-prototype.md",
                    "field": "status",
                    "value": "active",
                },
                {
                    "type": "frontmatter_field_contains",
                    "path": "Projects/Alpha/alpha-prototype.md",
                    "field": "tags",
                    "value": "project",
                },
                {
                    "type": "file_contains",
                    "path": "Projects/Alpha/alpha-prototype.md",
                    "text": "Need to compare RAG vs fine-tune",
                },
            ],
        },
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == ["contentManager_write"]


def test_environment_validator_session_persists_runtime_across_multiple_steps():
    validator = EnvironmentValidator(backend="local")
    session = validator.start_session(
        system_prompt="",
        environment_config={
            "fixture": {"directories": ["Inbox"]},
            "assertions": [
                {"type": "path_exists", "path": "Inbox/step-two.md"},
            ],
        },
    )

    try:
        session.execute_response(
            {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "contentManager_write",
                            "arguments": json.dumps(
                                {
                                    "path": "Inbox/step-one.md",
                                    "content": "first",
                                    "overwrite": True,
                                }
                            ),
                        },
                    }
                ]
            }
        )
        session.execute_response(
            {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "storageManager_move",
                            "arguments": json.dumps(
                                {
                                    "path": "Inbox/step-one.md",
                                    "newPath": "Inbox/step-two.md",
                                }
                            ),
                        },
                    }
                ]
            }
        )
        result = session.finalize(total_turns=2, stop_reason="test_complete")
    finally:
        session.close()

    assert result.passed is True
    assert result.episode_trace is not None
    assert result.episode_trace.total_turns == 2
    assert result.episode_trace.total_tool_calls == 2
    assert result.episode_trace.stop_reason == "test_complete"
    assert [tool.name for tool in result.executed_tools] == [
        "contentManager_write",
        "storageManager_move",
    ]


def test_environment_validator_supports_precise_regex_and_line_assertions():
    validator = EnvironmentValidator(backend="local")

    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "contentManager_write",
                    "arguments": json.dumps(
                        {
                            "path": "Settings/settings.yaml",
                            "content": (
                                "service: api\n"
                                "database_url: postgresql://prod-user:prod-pass@db.prod.internal:5432/app\n"
                                "retries: 3\n"
                            ),
                            "overwrite": True,
                        }
                    ),
                },
            }
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        environment_config={
            "fixture": {"directories": ["Settings"]},
            "assertions": [
                {
                    "type": "file_line_contains",
                    "path": "Settings/settings.yaml",
                    "line": 2,
                    "text": "database_url: postgresql://prod-user:prod-pass@db.prod.internal:5432/app",
                },
                {
                    "type": "file_line_not_contains",
                    "path": "Settings/settings.yaml",
                    "line": 2,
                    "text": "localhost",
                },
                {
                    "type": "file_matches_regex",
                    "path": "Settings/settings.yaml",
                    "pattern": r"database_url:\s+postgresql://prod-user:prod-pass@db\.prod\.internal:5432/app",
                },
            ],
        },
    )

    assert result.passed is True


def test_environment_validator_can_return_line_numbered_read_output():
    validator = EnvironmentValidator(backend="local")

    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "contentManager_read",
                    "arguments": json.dumps(
                        {
                            "path": "Docs/example.md",
                            "startLine": 2,
                            "endLine": 3,
                        }
                    ),
                },
            }
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        environment_config={
            "fixture": {
                "files": {
                    "Docs/example.md": "first\nsecond\nthird\nfourth\n",
                },
            },
            "execution": {
                "read_output": {
                    "include_line_numbers": True,
                    "honor_line_range": True,
                },
            },
        },
    )

    assert result.passed is True
    assert result.executed_tools[0].output == "2: second\n3: third"


def test_environment_validator_line_numbers_cli_wrapper_reads():
    validator = EnvironmentValidator(backend="local")

    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "workspaceId": "default",
                            "sessionId": "session_test",
                            "memory": "Need to inspect a file.",
                            "goal": "Read the example file.",
                            "tool": "content read Docs/example.md 1",
                            "strategy": "serial",
                        }
                    ),
                },
            }
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        environment_config={
            "fixture": {
                "files": {
                    "Docs/example.md": "first\nsecond\nthird\n",
                },
            },
        },
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == ["contentManager_read"]
    assert result.executed_tools[0].output == "1: first\n2: second\n3: third"


def test_environment_validator_replaces_line_range_from_cli_wrapper():
    validator = EnvironmentValidator(backend="local")

    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "workspaceId": "default",
                            "sessionId": "session_test",
                            "memory": "Need to update one line.",
                            "goal": "Replace the target setting.",
                            "tool": (
                                "content replace Docs/example.md "
                                "\"status=draft\" \"status=ready\" 3 3"
                            ),
                            "strategy": "serial",
                        }
                    ),
                },
            }
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        environment_config={
            "fixture": {
                "files": {
                    "Docs/example.md": "# Example\nanchor\nstatus=draft\nkeep\n",
                },
            },
            "assertions": [
                {"type": "file_contains", "path": "Docs/example.md", "text": "status=ready"},
                {"type": "file_not_contains", "path": "Docs/example.md", "text": "status=draft"},
                {"type": "file_contains", "path": "Docs/example.md", "text": "anchor"},
            ],
        },
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == ["contentManager_replace"]
    assert result.executed_tools[0].output == "replaced"


def test_environment_validator_rejects_replace_wrong_line_with_found_line_hint():
    validator = EnvironmentValidator(backend="local")

    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "workspaceId": "default",
                            "sessionId": "session_test",
                            "memory": "Need to update one line.",
                            "goal": "Replace the target setting.",
                            "tool": (
                                "content replace Docs/example.md "
                                "\"status=draft\" \"status=ready\" 1 1"
                            ),
                            "strategy": "serial",
                        }
                    ),
                },
            }
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        environment_config={
            "fixture": {
                "files": {
                    "Docs/example.md": "# Example\nanchor\nstatus=draft\nkeep\n",
                },
            },
            "loop": {"continue_on_execution_error": True},
            "assertions": [
                {"type": "file_contains", "path": "Docs/example.md", "text": "status=ready"},
            ],
        },
    )

    assert result.passed is False
    assert result.executed_tools[0].status == "error"
    assert "matching text starts at line 3" in (result.executed_tools[0].error or "")


def test_environment_validator_supports_invariant_style_contains_any_assertions():
    validator = EnvironmentValidator(backend="local")

    result = validator.validate_response(
        system_prompt="",
        response={"content": "No tool call needed."},
        environment_config={
            "fixture": {
                "files": {
                    "Policies/admin.md": (
                        "Access policy updated for contractors.\n"
                        "Effective immediately.\n"
                    )
                }
            },
            "assertions": [
                {
                    "type": "file_contains_any",
                    "path": "Policies/admin.md",
                    "texts": ["contractors", "vendors"],
                },
                {
                    "type": "file_contains_all",
                    "path": "Policies/admin.md",
                    "texts": ["Access policy", "Effective immediately"],
                },
                {
                    "type": "file_not_contains_any",
                    "path": "Policies/admin.md",
                    "texts": ["deprecated language", "legacy override"],
                },
            ],
        },
    )

    assert result.passed is True
