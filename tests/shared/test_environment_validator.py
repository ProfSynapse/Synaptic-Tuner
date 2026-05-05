from __future__ import annotations

import json
from pathlib import Path

from shared.environments import EnvironmentValidator
from shared.environments.types import ExecutedToolCall
from shared.environments.tool_executor import _expand_allowed_tool_identifiers
from shared.environments.validator import _called_tool_identifiers
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


def test_environment_validator_replace_edits_old_content_without_clobbering_file():
    validator = EnvironmentValidator(backend="local")
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "contentManager_replace",
                    "arguments": json.dumps(
                        {
                            "path": "Projects/Alpha/status.md",
                            "oldContent": "pending",
                            "newContent": "completed",
                            "startLine": 5,
                            "endLine": 5,
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
                "notes": [
                    {
                        "path": "Projects/Alpha/status.md",
                        "frontmatter": {"title": "Alpha Status"},
                        "body": "Release 1.2 status: pending",
                    }
                ],
            },
            "assertions": [
                {
                    "type": "file_contains",
                    "path": "Projects/Alpha/status.md",
                    "text": "Release 1.2 status",
                },
                {"type": "file_contains", "path": "Projects/Alpha/status.md", "text": "completed"},
                {
                    "type": "file_not_contains",
                    "path": "Projects/Alpha/status.md",
                    "text": "pending",
                },
            ],
        },
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == ["contentManager_replace"]


def test_environment_validator_uses_configured_mock_tool_output_for_non_filesystem_tool():
    validator = EnvironmentValidator(backend="local")
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "memoryManager_loadWorkspace",
                    "arguments": json.dumps({"id": "Team Workspace", "limit": 5, "recursive": True}),
                },
            }
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        environment_config={
            "allowed_tools": ["memory load-workspace"],
            "mock_tool_outputs": [
                {
                    "tool": "memory load-workspace",
                    "match": {"id": "Team Workspace"},
                    "output": {
                        "context": {"name": "Team Workspace", "purpose": "Project delivery"},
                        "keyFiles": ["reports/status.md"],
                        "workflows": [{"name": "Weekly review", "steps": ["Read status", "Summarize blockers"]}],
                    },
                }
            ],
        },
        expected_tools=["memory load-workspace"],
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == ["memoryManager_loadWorkspace"]
    assert result.executed_tools[0].status == "ok"
    assert '"keyFiles"' in result.executed_tools[0].output
    assert "simulated" not in result.executed_tools[0].output


def test_environment_validator_surfaces_configured_mock_tool_error():
    validator = EnvironmentValidator(backend="local")
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "memoryManager_loadState",
                    "arguments": json.dumps({"name": "missing-state"}),
                },
            }
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        environment_config={
            "allowed_tools": ["memory load-state"],
            "mock_tool_outputs": [
                {
                    "tool": "memory load-state",
                    "match": {"name": "missing-state"},
                    "status": "error",
                    "error": "State not found: missing-state",
                }
            ],
        },
        expected_tools=["memory load-state"],
    )

    assert result.passed is False
    assert result.executed_tools[0].status == "error"
    assert any("State not found" in issue.message for issue in result.issues)


def test_environment_validator_can_require_all_tool_statuses_ok():
    validator = EnvironmentValidator(backend="local")
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "contentManager_replace",
                    "arguments": json.dumps(
                        {
                            "path": "status.md",
                            "oldContent": "missing",
                            "newContent": "completed",
                            "startLine": 1,
                            "endLine": 1,
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
            "require_all_tools_ok": True,
            "loop": {"continue_on_execution_error": True},
            "fixture": {"files": {"status.md": "already completed"}},
            "assertions": [{"type": "file_contains", "path": "status.md", "text": "already completed"}],
        },
    )

    assert result.passed is False
    assert any("did not complete successfully" in issue.message for issue in result.issues)


def test_environment_validator_can_require_expected_tool_order():
    validator = EnvironmentValidator(backend="local")
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "contentManager_read",
                    "arguments": json.dumps({"path": "target.md", "startLine": 1}),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "searchManager_searchContent",
                    "arguments": json.dumps({"query": "answer"}),
                },
            },
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        expected_tools=["search search-content", "content read"],
        environment_config={
            "require_expected_tool_order": True,
            "fixture": {"files": {"target.md": "answer: 42"}},
            "assertions": [{"type": "file_contains", "path": "target.md", "text": "answer: 42"}],
        },
    )

    assert result.passed is False
    assert any("Expected tool order not satisfied" in issue.message for issue in result.issues)


def test_environment_validator_can_forbid_unexpected_tools():
    validator = EnvironmentValidator(backend="local")
    response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "searchManager_searchContent",
                    "arguments": json.dumps({"query": "answer"}),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "contentManager_replace",
                    "arguments": json.dumps(
                        {
                            "path": "target.md",
                            "oldContent": "answer: 42",
                            "newContent": "answer: changed",
                            "startLine": 1,
                            "endLine": 1,
                        }
                    ),
                },
            },
        ]
    }

    result = validator.validate_response(
        system_prompt="",
        response=response,
        expected_tools=["search search-content"],
        environment_config={
            "forbid_unexpected_tools": True,
            "fixture": {"files": {"target.md": "answer: 42"}},
        },
    )

    assert result.passed is False
    assert any("Unexpected tool(s) executed" in issue.message for issue in result.issues)


def test_called_tool_identifiers_include_cli_commands_from_schema():
    identifiers = _called_tool_identifiers(
        [ExecutedToolCall(name="storageManager_list")],
        {
            "tools": {
                "storageManager": [
                    {"name": "list", "command": "storage list"},
                ]
            }
        },
    )

    assert "storageManager_list" in identifiers
    assert "storage list" in identifiers


def test_allowed_tool_identifiers_include_schema_expanded_executor_names():
    identifiers = _expand_allowed_tool_identifiers(
        {"storage list"},
        {
            "tools": {
                "storageManager": [
                    {"name": "list", "command": "storage list"},
                ]
            }
        },
    )

    assert "storage list" in identifiers
    assert "storageManager_list" in identifiers


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
