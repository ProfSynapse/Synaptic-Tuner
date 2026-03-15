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
                {"type": "frontmatter_field_equals", "path": "Inbox/capture.md", "field": "status", "value": "inbox"},
            ],
        },
    )

    assert result.passed is True
    assert result.assertions_run == 3


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
