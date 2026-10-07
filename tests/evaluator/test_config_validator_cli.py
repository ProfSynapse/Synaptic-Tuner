"""ConfigDrivenValidator expands CLI wrapper calls with the shared CLI parser."""

from __future__ import annotations

import json
from pathlib import Path

from Evaluator.config_validator import ConfigDrivenValidator
from SynthChat.config.format_resolver import load_tool_call_formats

CONFIG_DIR = Path(__file__).resolve().parents[2] / "Evaluator" / "config"
_TOOL_CALL_FORMAT = load_tool_call_formats()["default"]


def _wrapper_response(tool_value: str) -> dict:
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_CALL_FORMAT["wrapper_name"],
                    "arguments": json.dumps(
                        {
                            "sessionId": "session_1732300800000_cfgval",
                            "workspaceId": "ws_1732300800000_cfgval",
                            "memory": "Write the note.",
                            "goal": "Create the daily note.",
                            "tool": tool_value,
                        }
                    ),
                },
            }
        ]
    }


def test_config_validator_expands_multiline_frontmatter_write_and_following_command():
    validator = ConfigDrivenValidator(CONFIG_DIR)
    note = "---\ntitle: 2026-03-15\ntype: daily\n---\n# Daily Note, with a comma\n"
    tool_value = (
        'content write "Journal/Daily/2026-03-15.md" "' + note + '", '
        'content read "Journal/Daily/2026-03-15.md" 1'
    )

    parsed = validator.parse_response(_wrapper_response(tool_value))

    assert [call.name for call in parsed.tool_calls] == ["contentManager_write", "contentManager_read"]
    assert parsed.tool_calls[0].agent == "contentManager"
    assert parsed.tool_calls[0].tool == "write"
    assert parsed.tool_calls[0].context["workspaceId"] == "ws_1732300800000_cfgval"


def test_config_validator_keeps_wrapper_call_when_command_is_unknown():
    validator = ConfigDrivenValidator(CONFIG_DIR)

    parsed = validator.parse_response(_wrapper_response('content teleport "a.md"'))

    assert [call.name for call in parsed.tool_calls] == [_TOOL_CALL_FORMAT["wrapper_name"]]
