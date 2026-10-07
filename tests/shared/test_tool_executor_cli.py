"""CLI command strings executed by the local environment runtime.

The wrapper name, its required fields and the CLI escape table come from the
configured default tool-call format (SynthChat/config/tool_call_formats.yaml);
commands come from the CLI catalog (cli-first-tool-schemas.json).
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from shared.environments import EnvironmentValidator
from shared.validation.parsing.cli_commands import tokenize_cli_commands
from SynthChat.config.format_resolver import load_tool_call_formats

CATALOG_PATH = Path(__file__).resolve().parents[2] / "cli-first-tool-schemas.json"
_TOOL_CALL_FORMAT = load_tool_call_formats()["default"]

FRONTMATTER_NOTE = (
    "---\n"
    "title: 2026-03-15\n"
    "type: daily\n"
    "tags:\n"
    "  - journal\n"
    "mood: focused\n"
    "---\n"
    "# Daily Note\n"
    "\n"
    "## Linked Notes\n"
    "- [[Projects/Alpha/meeting-notes]]\n"
)


def _cli_quote(value: str) -> str:
    """Double-quote ``value`` the way the CLI expects (escape ``\\`` and ``"``)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _tool_response(*commands: str) -> dict:
    arguments = {
        "sessionId": "session_1732300800000_clitest",
        "workspaceId": "ws_1732300800000_clitest",
        "memory": "CLI quoting check.",
        "goal": "Write the note exactly.",
        "tool": ", ".join(commands),
    }
    missing = set(_TOOL_CALL_FORMAT["argument_required"]) - set(arguments)
    assert not missing, f"configured tool-call format requires {sorted(missing)}"
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_CALL_FORMAT["wrapper_name"],
                    "arguments": json.dumps(arguments),
                },
            }
        ]
    }


def _run(*commands: str, fixture: dict | None = None, read_paths=()):
    """Execute commands in a local session; return (result, {path: content})."""
    session = EnvironmentValidator(backend="local").start_session(
        system_prompt="",
        environment_config={"fixture": fixture or {"directories": ["Journal/Daily"]}},
    )
    try:
        session.execute_response(_tool_response(*commands))
        contents = {path: session.runtime.read_text(path) for path in read_paths}
        return session.finalize(total_turns=1, stop_reason="single_response"), contents
    finally:
        session.close()


def _catalog_examples():
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return [example for tool in payload["tools"] for example in tool.get("examples") or []]


@pytest.mark.parametrize("example", _catalog_examples())
def test_tokenizer_matches_posix_shell_quoting_for_catalog_examples(example):
    assert tokenize_cli_commands(example, {}) == [shlex.split(example)]


@pytest.mark.parametrize(
    "command",
    [
        'content write "a.md" "line one\nline two\n\tindented"',
        'content write "a.md" "say \\"hi\\" and C:\\\\path and \\d"',
        "search content \"x\" --paths '[\"a\\nb\"]'",
        "content write a\\ b.md text\\ here",
        'content write "a.md" ""',
    ],
)
def test_tokenizer_without_escapes_is_shlex_split(command):
    assert tokenize_cli_commands(command, {}) == [shlex.split(command)]


def test_tokenizer_splits_commands_on_top_level_commas_only():
    commands = tokenize_cli_commands(
        'content write "a.md" "x, y", search directory "d" --paths ["A/", "B/"],content read a.md 1',
        {},
    )

    assert commands == [
        ["content", "write", "a.md", "x, y"],
        ["search", "directory", "d", "--paths", "[A/,", "B/]"],
        ["content", "read", "a.md", "1"],
    ]


def test_tokenizer_decodes_configured_escapes_inside_double_quotes_only():
    escapes = {"n": "\n", "t": "\t"}

    assert tokenize_cli_commands('content write "a.md" "a\\nb\\tc\\\\nd"', escapes) == [
        ["content", "write", "a.md", "a\nb\tc\\nd"]
    ]
    assert tokenize_cli_commands("content write 'a.md' 'a\\nb'", escapes) == [
        ["content", "write", "a.md", "a\\nb"]
    ]


def test_tokenizer_rejects_unterminated_quotes():
    with pytest.raises(ValueError):
        tokenize_cli_commands('content write "a.md" "unterminated', {})


def test_cli_write_round_trips_multiline_frontmatter_content():
    path = "Journal/Daily/2026-03-15.md"
    result, contents = _run(
        f"content write {_cli_quote(path)} {_cli_quote(FRONTMATTER_NOTE)}",
        read_paths=[path],
    )

    assert result.passed is True
    assert [tool.name for tool in result.executed_tools] == ["contentManager_write"]
    assert contents[path] == FRONTMATTER_NOTE


def test_cli_write_keeps_value_starting_with_dashes_as_argument():
    path = "Notes/rule.md"
    content = "---\n--- not a flag\n---"
    result, contents = _run(
        f"content write {_cli_quote(path)} {_cli_quote(content)} --overwrite",
        fixture={"directories": ["Notes"]},
        read_paths=[path],
    )

    assert result.passed is True
    assert result.executed_tools[0].arguments == {"path": path, "content": content, "overwrite": True}
    assert contents[path] == content


def test_cli_write_flag_value_may_start_with_frontmatter_dashes():
    path = "Notes/flagged.md"
    result, contents = _run(
        f"content write --path {_cli_quote(path)} --content {_cli_quote(FRONTMATTER_NOTE)}",
        fixture={"directories": ["Notes"]},
        read_paths=[path],
    )

    assert result.passed is True
    assert contents[path] == FRONTMATTER_NOTE


def test_cli_write_preserves_embedded_quotes_and_whitespace():
    path = "Notes/quotes.md"
    content = 'He said "ship it" \u2014 it\'s \u201cdone\u201d.\n\n    code  block\twith tab\n'
    result, contents = _run(
        f"content write {_cli_quote(path)} {_cli_quote(content)}",
        fixture={"directories": ["Notes"]},
        read_paths=[path],
    )

    assert result.passed is True
    assert contents[path] == content


def test_cli_write_decodes_escaped_newlines_from_the_configured_format():
    assert _TOOL_CALL_FORMAT["command_escapes"]["n"] == "\n"
    path = "Notes/escaped.md"
    # The decoded command string holds a literal backslash-n, as the CLI schema
    # tells models to write multi-line content.
    result, contents = _run(
        'content write "Notes/escaped.md" "---\\ntitle: Escaped\\n---\\n# Body\\n"',
        fixture={"directories": ["Notes"]},
        read_paths=[path],
    )

    assert result.passed is True
    assert contents[path] == "---\ntitle: Escaped\n---\n# Body\n"


def test_cli_replace_matches_multiline_old_content():
    path = "Notes/plan.md"
    old_content, new_content = _cli_quote("a\nb"), _cli_quote("A\nB")
    result, contents = _run(
        f"content replace {_cli_quote(path)} {old_content} {new_content} 2 3",
        fixture={"files": {path: "top\na\nb\nend\n"}},
        read_paths=[path],
    )

    assert result.passed is True
    assert contents[path] == "top\nA\nB\nend\n"
