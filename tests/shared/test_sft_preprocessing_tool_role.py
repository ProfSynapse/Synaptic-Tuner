"""Tests for tool-role support in shared.sft_preprocessing.

Regression-locks the byte output of existing single-turn (system/user/assistant
+ tool_calls) sanitization, then verifies the additive `tool` role rendering for
multi-turn trajectories.
"""
from __future__ import annotations

from shared.sft_preprocessing import (
    render_tool_call_content,
    render_tool_result_content,
    sanitize_messages_for_chat_template,
)


def _single_turn_with_tool_calls():
    """The canonical single-turn shape: assistant text + OpenAI tool_calls, content None."""
    return [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "useTools", "arguments": '{"a": 1}'},
                }
            ],
        },
    ]


# The byte-exact expected output BEFORE tool-role support was added. This is the
# regression lock: the new `tool` branch must not perturb these rows.
def _expected_legacy_sanitize(messages):
    import json

    sanitized = []
    for message in messages:
        normalized = dict(message)
        content = normalized.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        tool_calls = normalized.get("tool_calls") or []
        if tool_calls:
            tool_content = render_tool_call_content(tool_calls)
            content = f"{content}\n\n{tool_content}".strip() if content else tool_content
        normalized["content"] = content
        normalized.pop("tool_calls", None)
        sanitized.append(normalized)
    return sanitized


class TestSingleTurnRegressionLock:
    def test_single_turn_tool_calls_byte_identical(self):
        messages = _single_turn_with_tool_calls()
        got = sanitize_messages_for_chat_template(messages)
        expected = _expected_legacy_sanitize(_single_turn_with_tool_calls())
        assert got == expected

    def test_plain_conversation_unchanged(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        got = sanitize_messages_for_chat_template(messages)
        assert got == messages
        # roles untouched, no tool_calls key introduced
        assert [m["role"] for m in got] == ["system", "user", "assistant"]

    def test_none_content_becomes_empty_string(self):
        got = sanitize_messages_for_chat_template([{"role": "assistant", "content": None}])
        assert got[0]["content"] == ""


class TestToolRoleRendering:
    def test_tool_role_rendered_and_retagged_to_user(self):
        messages = [
            {"role": "user", "content": "search the vault"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "useTools", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "Tool execution results:\nfound 3 files"},
            {"role": "assistant", "content": "Done."},
        ]
        got = sanitize_messages_for_chat_template(messages)
        # tool message re-tagged to user for template compatibility
        assert got[2]["role"] == "user"
        # tool output preserved inside the templated text
        assert "found 3 files" in got[2]["content"]
        assert got[2]["content"].startswith("tool_result:")
        # no tool_calls leak on the result message
        assert "tool_calls" not in got[2]

    def test_tool_role_with_dict_content_json_rendered(self):
        messages = [{"role": "tool", "content": {"status": "ok", "n": 2}}]
        got = sanitize_messages_for_chat_template(messages)
        assert got[0]["role"] == "user"
        assert '"status": "ok"' in got[0]["content"]
        assert '"n": 2' in got[0]["content"]

    def test_multi_turn_templates_without_error(self):
        """A full multi-turn trajectory sanitizes into a coherent user/assistant transcript."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "result body"},
            {"role": "assistant", "content": "final answer"},
        ]
        got = sanitize_messages_for_chat_template(messages)
        roles = [m["role"] for m in got]
        # only roles a generic chat template understands remain
        assert set(roles).issubset({"system", "user", "assistant"})
        assert all(isinstance(m["content"], str) for m in got)
        assert "result body" in got[3]["content"]


class TestRenderToolResultContent:
    def test_string_body(self):
        assert render_tool_result_content("hi") == "tool_result:\nhi"

    def test_empty_body(self):
        assert render_tool_result_content("") == "tool_result:"
        assert render_tool_result_content(None) == "tool_result:"

    def test_dict_body_pretty_printed(self):
        out = render_tool_result_content({"k": "v"})
        assert out.startswith("tool_result:\n")
        assert '"k": "v"' in out
