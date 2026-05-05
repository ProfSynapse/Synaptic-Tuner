import json

from SynthChat.config import ConfigLoader
from SynthChat.services.parsing import ScopeExtractor
from SynthChat.services.scope_handlers.response_handler import ResponseHandler


def test_response_scope_extracts_latest_assistant_message():
    scope_config = ConfigLoader().load()
    extractor = ScopeExtractor(scope_config)

    example = {
        "conversations": [
            {"role": "user", "content": "Find the status."},
            {"role": "assistant", "content": "first assistant response"},
            {"role": "user", "content": "Tool result"},
            {"role": "assistant", "content": "latest assistant response"},
        ]
    }

    assert extractor.extract(example, "response") == "latest assistant response"


def test_response_improvement_applies_to_latest_assistant_message():
    scope_config = ConfigLoader().load()
    extractor = ScopeExtractor(scope_config)
    handler = ResponseHandler(scope_config, extractor)

    example = {
        "conversations": [
            {"role": "user", "content": "Find the status."},
            {"role": "assistant", "content": "first assistant response"},
            {"role": "user", "content": "Tool result"},
            {"role": "assistant", "content": "Tool calls: bad placeholder"},
        ]
    }
    improved_message = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_0002",
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": json.dumps(
                        {
                            "workspaceId": "default",
                            "sessionId": "session_1",
                            "memory": "Read status",
                            "goal": "Read status",
                            "tool": "content read \"project/docs/status.md\" 1",
                            "strategy": "serial",
                        }
                    ),
                },
            }
        ],
    }

    updated = handler.apply_improvement(
        example,
        json.dumps(improved_message),
        output_format={"type": "assistant_message"},
    )

    assert updated["conversations"][1]["content"] == "first assistant response"
    assert updated["conversations"][3]["content"] is None
    assert updated["conversations"][3]["tool_calls"][0]["function"]["name"] == "useTools"
