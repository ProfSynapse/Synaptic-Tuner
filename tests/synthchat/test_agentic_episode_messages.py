from types import SimpleNamespace

from SynthChat.agentic.episode import _messages_with_terminal_assistant


def test_messages_include_terminal_assistant_after_schema_failure():
    episode = SimpleNamespace(
        messages=[{"role": "user", "content": "Read the status."}],
        conversation_trace=[
            {"role": "user", "kind": "prompt_message", "content": "Read the status."},
            {
                "role": "assistant",
                "kind": "assistant_response",
                "content": "Tool calls: bad",
            },
        ],
        final_response={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "useTools", "arguments": "{...}"},
                }
            ],
        },
    )

    messages = _messages_with_terminal_assistant(episode)

    assert len(messages) == 2
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["tool_calls"][0]["function"]["arguments"] == "{...}"


def test_messages_do_not_duplicate_appended_assistant():
    episode = SimpleNamespace(
        messages=[
            {"role": "user", "content": "Read the status."},
            {"role": "assistant", "content": "already appended"},
        ],
        conversation_trace=[
            {"role": "user", "kind": "prompt_message", "content": "Read the status."},
            {
                "role": "assistant",
                "kind": "assistant_response",
                "content": "already appended",
            },
        ],
        final_response={"role": "assistant", "content": "already appended"},
    )

    messages = _messages_with_terminal_assistant(episode)

    assert messages == episode.messages
