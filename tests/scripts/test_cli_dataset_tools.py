"""Dataset tools parse CLI wrapper strings with the shared CLI parser."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from SynthChat.config.format_resolver import load_tool_call_formats

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "cli-first-tool-schemas.json"
_TOOL_CALL_FORMAT = load_tool_call_formats()["default"]
FRONTMATTER_NOTE = "---\ntitle: Alley Meeting\ntags: [noir]\n---\n\nThe rain was a \"greasy\" curtain.\n"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli_schema_utils(monkeypatch):
    # The migration scripts import their sibling ``utils`` module by its bare name.
    monkeypatch.syspath_prepend(str(REPO_ROOT / "Tools" / "migrations"))
    previous_utils = sys.modules.get("utils")
    yield _load_module("cli_schema_utils_under_test", REPO_ROOT / "Tools" / "migrations" / "cli_schema_utils.py")
    if previous_utils is None:
        sys.modules.pop("utils", None)
    else:
        sys.modules["utils"] = previous_utils


@pytest.fixture
def analyze_tool_coverage():
    return _load_module("analyze_tool_coverage_under_test", REPO_ROOT / "tools" / "analyze_tool_coverage.py")


def _assistant_message(tool_value: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_CALL_FORMAT["wrapper_name"],
                    "arguments": json.dumps(
                        {
                            "sessionId": "session_1732300800000_dataset",
                            "workspaceId": "default",
                            "memory": "Write the chapter.",
                            "goal": "Save the chapter note.",
                            "tool": tool_value,
                        }
                    ),
                },
            }
        ],
    }


def _write_command(content: str) -> str:
    quoted = content.replace("\\", "\\\\").replace('"', '\\"')
    return f'content write "Chapters/Chapter_04.md" "{quoted}" --overwrite'


def test_migration_tools_keep_frontmatter_content_and_render_round_trips(cli_schema_utils):
    catalog = cli_schema_utils.load_target_catalog(SCHEMA_PATH)
    example = {"conversations": [_assistant_message(_write_command(FRONTMATTER_NOTE) + ', content read "Chapters/Chapter_04.md" 1')]}

    calls = cli_schema_utils.extract_normalized_calls(example, catalog)

    assert [(call["agent"], call["tool"]) for call in calls] == [("contentManager", "write"), ("contentManager", "read")]
    assert calls[0]["params"] == {"path": "Chapters/Chapter_04.md", "content": FRONTMATTER_NOTE, "overwrite": True}
    rendered = cli_schema_utils.render_cli_command("contentManager", "write", calls[0]["params"], catalog)
    assert cli_schema_utils.parse_cli_tool_string(rendered, catalog, {})[0]["params"] == calls[0]["params"]


def test_migration_tools_decode_configured_escapes(cli_schema_utils):
    catalog = cli_schema_utils.load_target_catalog(SCHEMA_PATH)
    example = {"conversations": [_assistant_message('content write "a.md" "---\\ntitle: A\\n---"')]}

    calls = cli_schema_utils.extract_normalized_calls(example, catalog)

    assert calls[0]["params"]["content"] == "---\ntitle: A\n---"


def test_tool_coverage_counts_commands_with_frontmatter_values(analyze_tool_coverage):
    _, _, command_lookup = analyze_tool_coverage.load_tool_schema(SCHEMA_PATH)
    message = _assistant_message(_write_command(FRONTMATTER_NOTE) + ', storage copy "a.md" "b.md"')

    assert analyze_tool_coverage.extract_tools_from_assistant_message(message, command_lookup) == [
        "contentManager_write",
        "storageManager_copy",
    ]
