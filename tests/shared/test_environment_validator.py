from __future__ import annotations

import json

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
