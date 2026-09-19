"""Validation-feedback repair turns in the shared environment episode runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from shared.agentic_loop import (
    AgenticModelResponse,
    format_validation_feedback_message,
    run_environment_episode,
)


@dataclass
class _Issue:
    level: str
    message: str


@dataclass
class _Validation:
    passed: bool
    issues: List[_Issue] = field(default_factory=list)
    tool_calls: List[Any] = field(default_factory=list)


@dataclass
class _Step:
    hard_error: bool = False
    recoverable_error: bool = False
    executed_tools: List[Any] = field(default_factory=list)
    issues: List[Any] = field(default_factory=list)


@dataclass
class _Preview:
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed}


class _Session:
    def __init__(self) -> None:
        self.steps: List[Any] = []
        self.executed_tools: List[Any] = []
        self.executed: List[Any] = []

    def execute_response(self, message: Any) -> _Step:
        self.executed.append(message)
        return _Step()

    def finalize(self, *, expected_tools=None, total_turns=0, stop_reason="") -> _Preview:
        return _Preview(passed=False)

    def close(self) -> None:
        pass


def _validate(message: Any) -> _Validation:
    if isinstance(message, dict) and message.get("tool_calls"):
        return _Validation(passed=False, issues=[_Issue("ERROR", "Failed to parse arguments JSON for tool useTools")])
    return _Validation(passed=True)


def _responder(scripted: List[Dict[str, Any]]):
    seen: List[List[Dict[str, Any]]] = []

    def respond(messages, turn_index):
        seen.append([dict(m) for m in messages])
        return AgenticModelResponse(message=dict(scripted[turn_index - 1]))

    respond.seen = seen  # type: ignore[attr-defined]
    return respond


_BAD = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "useTools", "arguments": "{'tool':'storage list'}"}}]}
_GOOD_TEXT = {"role": "assistant", "content": "Here is the answer.", "tool_calls": None}
_INITIAL = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do it"}]


def test_validation_failure_ends_episode_by_default():
    session = _Session()
    result = run_environment_episode(
        initial_messages=_INITIAL,
        session=session,
        respond=_responder([_BAD, _GOOD_TEXT]),
        validate=_validate,
        max_turns=4,
    )
    assert result.stop_reason == "schema_validation_failed"
    assert result.validation_retries == 0
    assert session.executed == []
    assert [m["role"] for m in result.messages] == ["system", "user"]


def test_validation_failure_gets_feedback_turn_when_enabled():
    session = _Session()
    respond = _responder([_BAD, _GOOD_TEXT])
    result = run_environment_episode(
        initial_messages=_INITIAL,
        session=session,
        respond=respond,
        validate=_validate,
        max_turns=4,
        continue_on_validation_error=True,
        max_validation_retries=2,
        validation_feedback_prompt="Fix the wrapper payload.",
    )
    assert result.stop_reason == "text_response"
    assert result.validation_retries == 1
    # Only the corrected turn reached the environment; the rejected one never did.
    assert [m.get("content") for m in session.executed] == ["Here is the answer."]
    # The model saw its own rejected message followed by the feedback before turn 2.
    second_turn_input = respond.seen[1]
    assert second_turn_input[-2]["role"] == "assistant"
    assert second_turn_input[-1]["role"] == "user"
    assert second_turn_input[-1]["content"].startswith("Fix the wrapper payload.")
    assert "Failed to parse arguments JSON" in second_turn_input[-1]["content"]
    kinds = [entry.get("kind") for entry in result.conversation_trace]
    assert kinds.count("validation_feedback") == 1
    assert [m["role"] for m in result.messages] == ["system", "user", "assistant", "user", "assistant"]


def test_validation_retry_budget_is_bounded():
    session = _Session()
    result = run_environment_episode(
        initial_messages=_INITIAL,
        session=session,
        respond=_responder([_BAD, _BAD, _BAD, _GOOD_TEXT]),
        validate=_validate,
        max_turns=6,
        continue_on_validation_error=True,
        max_validation_retries=2,
    )
    assert result.stop_reason == "schema_validation_failed"
    assert result.validation_retries == 2
    assert len(result.turns) == 3
    assert session.executed == []


def test_format_validation_feedback_message_renders_issues_and_default_prompt():
    text = format_validation_feedback_message(_Validation(passed=False, issues=[_Issue("error", "bad json")]))
    assert "failed response validation" in text
    assert "- ERROR: bad json" in text
    empty = format_validation_feedback_message({"passed": False, "issues": []})
    assert "- ERROR: response did not pass validation" in empty
