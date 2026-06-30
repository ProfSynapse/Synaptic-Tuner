import json
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2] / ".skills" / "transcript-distillation" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from adapters.claude_code import ClaudeCodeAdapter  # noqa: E402
from adapters.codex import CodexAdapter  # noqa: E402
from distill import emit_rows, redact_nested  # noqa: E402
from sanitize import Redactor  # noqa: E402


LAB = {
    "correction_markers": ["no"],
    "interrupt_markers": ["[request interrupted"],
    "max_correction_chars": 180,
}


def test_native_render_preserves_legacy_fields_and_native_tool_messages():
    events = [
        {"role": "human", "text": "Run the command"},
        {
            "role": "assistant",
            "text": "I'll run it.",
            "tool_calls": [{"id": "call_1", "name": "shell", "input": {"command": "echo ok"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool_error": False,
            "command": "echo ok",
            "output": "ok",
        },
        {"role": "assistant", "text": "Done.", "tool_calls": []},
    ]

    rows = emit_rows(
        events,
        source_kind="unit",
        project="proj",
        rel_id="session.jsonl",
        lab=LAB,
        ctx_budget={"max_context_tokens": 8192, "chars_per_token": 4.0, "max_context_messages": 10},
        render_mode="native",
    )

    tool_call_row = rows[0]
    assert tool_call_row["prompt"] == "Run the command"
    assert tool_call_row["completion"] == "I'll run it."
    assert tool_call_row["label"] is True
    assistant = tool_call_row["conversations"][-1]
    assert assistant == {
        "role": "assistant",
        "content": "I'll run it.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"command": "echo ok"}'},
            }
        ],
    }

    final_row = rows[1]
    assert final_row["completion"] == "Done."
    tool_msg = final_row["conversations"][-2]
    assert tool_msg == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


def test_flat_render_remains_default_tool_calls_text_shape():
    rows = emit_rows(
        [
            {"role": "human", "text": "Run it"},
            {
                "role": "assistant",
                "text": "",
                "tool_calls": [{"id": "call_1", "name": "shell", "input": {"command": "echo ok"}}],
            },
        ],
        source_kind="unit",
        project="proj",
        rel_id="session.jsonl",
        lab=LAB,
        ctx_budget={},
    )

    assert rows[0]["completion"].startswith("Tool calls: ")
    assert rows[0]["conversations"][-1] == {
        "role": "assistant",
        "content": rows[0]["completion"],
    }


def test_native_redaction_covers_serialized_tool_arguments():
    redactor = Redactor({
        "enabled": True,
        "replacement": "[REDACTED:{name}]",
        "patterns": {"openai_key": "sk-[A-Za-z0-9_-]{20,}"},
    })
    counts = Counter()

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": '{"command": "echo sk-abcdefghijklmnopqrst"}',
            },
        }
    ]

    redacted = redact_nested(tool_calls, redactor, counts)
    assert redacted[0]["function"]["arguments"] == '{"command": "echo [REDACTED:openai_key]"}'
    assert counts["openai_key"] == 1


def test_codex_adapter_links_call_id_to_tool_result(tmp_path):
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(
        "\n".join([
            json.dumps({"type": "session_meta", "payload": {"cwd": str(tmp_path)}}),
            json.dumps({
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "shell",
                    "arguments": json.dumps({"cmd": "pytest"}),
                },
            }),
            json.dumps({
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "1 passed",
                },
            }),
        ])
    )

    events, _ = CodexAdapter().parse(str(transcript))
    assert events[0]["tool_calls"][0]["id"] == "call_abc"
    assert events[1]["tool_call_id"] == "call_abc"
    assert events[1]["command"] == "pytest"


def test_claude_adapter_links_tool_use_id_to_tool_result(tmp_path):
    transcript = tmp_path / "claude.jsonl"
    transcript.write_text(
        "\n".join([
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                         "input": {"command": "pytest"}}
                    ]
                },
            }),
            json.dumps({
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "1 passed"}
                    ]
                },
            }),
        ])
    )

    events, _ = ClaudeCodeAdapter().parse(str(transcript))
    assert events[0]["tool_calls"][0]["id"] == "toolu_1"
    assert events[1]["tool_call_id"] == "toolu_1"
    assert events[1]["command"] == "pytest"
