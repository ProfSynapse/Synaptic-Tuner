from types import SimpleNamespace

from shared.agentic_judge import AgenticJudgeResult
from shared.agentic_loop import AgenticModelResponse, run_environment_episode, _detect_stuck_episode


class _Session:
    def __init__(self):
        self.validator = SimpleNamespace(tool_schema={})
        self.executed_tools = []
        self.steps = []

    def execute_response(self, response):
        step = SimpleNamespace(
            hard_error=False,
            recoverable_error=False,
            executed_tools=[],
            issues=[],
            state_changed=False,
            action_signature=None,
            issue_signature=None,
            to_dict=lambda: {
                "executed_tools": [],
                "issues": [],
                "hard_error": False,
                "recoverable_error": False,
            },
        )
        self.steps.append(step)
        return step

    def finalize(self, **kwargs):
        return SimpleNamespace(
            passed=False,
            to_dict=lambda: {
                "passed": False,
                "issues": [],
                "executed_tools": [],
            },
        )


class _ExpectedToolsSession:
    def __init__(self):
        self.validator = SimpleNamespace(tool_schema={})
        self.executed = []
        self.steps = []

    def execute_response(self, response):
        tool_name = response["tool_calls"][0]["name"]
        self.executed.append(tool_name)
        step = SimpleNamespace(
            hard_error=False,
            recoverable_error=False,
            executed_tools=[SimpleNamespace(name=tool_name, status="ok", output="ok", error=None)],
            issues=[],
            state_changed=True,
            action_signature=tool_name,
            issue_signature=None,
            to_dict=lambda: {
                "executed_tools": [{"name": tool_name, "status": "ok"}],
                "issues": [],
                "hard_error": False,
                "recoverable_error": False,
            },
        )
        self.steps.append(step)
        return step

    def finalize(self, **kwargs):
        expected_tools = kwargs.get("expected_tools") or []
        missing = [tool for tool in expected_tools if tool not in self.executed]
        passed = not missing
        return SimpleNamespace(
            passed=passed,
            issues=[{"message": f"missing {tool}"} for tool in missing],
            executed_tools=list(self.executed),
            to_dict=lambda: {
                "passed": passed,
                "issues": [{"message": f"missing {tool}"} for tool in missing],
                "executed_tools": [{"name": tool, "status": "ok"} for tool in self.executed],
            },
        )


def test_validation_error_can_feed_back_and_continue_episode():
    calls = []

    def respond(messages, turn_index):
        calls.append((turn_index, list(messages)))
        if turn_index == 1:
            return AgenticModelResponse(message={"role": "assistant", "content": {"bad": True}})
        return AgenticModelResponse(message={"role": "assistant", "content": "Corrected text."})

    def validate(message):
        if isinstance(message.get("content"), dict):
            return SimpleNamespace(
                passed=False,
                issues=[SimpleNamespace(level="ERROR", message="content must be a string")],
                tool_calls=[],
            )
        return SimpleNamespace(passed=True, issues=[], tool_calls=[])

    result = run_environment_episode(
        initial_messages=[{"role": "user", "content": "Help"}],
        session=_Session(),
        respond=respond,
        validate=validate,
        continue_on_validation_error=True,
        stop_on_text_response=True,
        max_turns=3,
    )

    assert len(calls) == 2
    assert result.stop_reason == "text_response"
    assert any(message["role"] == "user" and "schema validation" in message["content"] for message in calls[1][1])
    assert any(
        message["role"] == "user" and "single-quoted pseudo-JSON" in message["content"] for message in calls[1][1]
    )
    assert any(item["kind"] == "validation_feedback" for item in result.conversation_trace)


def test_environment_pass_preview_respects_required_expected_tools():
    responses = [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "storage list"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "content read"}]},
    ]

    def respond(messages, turn_index):
        return AgenticModelResponse(message=responses[turn_index - 1])

    def validate(message):
        return SimpleNamespace(passed=True, issues=[], tool_calls=message.get("tool_calls", []))

    result = run_environment_episode(
        initial_messages=[{"role": "user", "content": "Explore then answer"}],
        session=_ExpectedToolsSession(),
        respond=respond,
        validate=validate,
        stop_on_environment_pass=True,
        expected_tools=["storage list", "content read"],
        require_expected_tools=True,
        max_turns=3,
    )

    assert result.stop_reason == "environment_passed"
    assert len(result.turns) == 2
    assert result.environment_result.passed is True


def test_final_text_turn_retries_when_judge_feedback_is_visible():
    responses = [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "storage list"}]},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [{"name": "storage list"}],
        },
        {"role": "assistant", "content": "Done.", "tool_calls": []},
    ]

    def respond(messages, turn_index):
        return AgenticModelResponse(message=responses[turn_index - 1])

    def validate(message):
        return SimpleNamespace(passed=True, issues=[], tool_calls=message.get("tool_calls", []))

    def judge_turn(payload):
        if payload.get("environment_step") is not None:
            return AgenticJudgeResult(passed=True, should_stop=True)
        if payload.get("environment_step") is None and payload["response_message"].get("tool_calls"):
            return AgenticJudgeResult(
                passed=False,
                feedback_to_model="Reply with text only and no tool_calls.",
                feedback_for_trace="Final answer included an extra tool call.",
            )
        return AgenticJudgeResult(passed=True, should_stop=True)

    result = run_environment_episode(
        initial_messages=[{"role": "user", "content": "List files"}],
        session=_ExpectedToolsSession(),
        respond=respond,
        validate=validate,
        stop_on_environment_pass=True,
        expected_tools=["storage list"],
        require_expected_tools=True,
        require_final_text_after_pass=True,
        judge_turn=judge_turn,
        judge_feedback_visible_to_model=True,
        max_turns=4,
    )

    assert result.stop_reason == "environment_passed_final_text"
    assert result.final_text_satisfied is True
    assert len(result.turns) == 3
    assert any(
        item["kind"] == "judge_feedback" and "text only" in item["content"]
        for item in result.conversation_trace
    )


def test_successful_tool_turns_remain_structured_in_episode_messages():
    response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": "{\"tool\":\"storage list --path \\\"\\\"\"}",
                },
            }
        ],
    }

    def respond(messages, turn_index):
        return AgenticModelResponse(message=response)

    def validate(message):
        return SimpleNamespace(passed=True, issues=[], tool_calls=message.get("tool_calls", []))

    result = run_environment_episode(
        initial_messages=[{"role": "user", "content": "List files"}],
        session=_Session(),
        respond=respond,
        validate=validate,
        stop_on_text_response=False,
        max_turns=1,
    )

    assistant_messages = [message for message in result.messages if message.get("role") == "assistant"]
    assert assistant_messages
    assert assistant_messages[-1]["content"] is None
    assert assistant_messages[-1]["tool_calls"] == response["tool_calls"]


def test_no_progress_stuck_detection_treats_successful_read_steps_as_progress():
    ok_step = SimpleNamespace(
        state_changed=False,
        executed_tools=[SimpleNamespace(name="search", status="ok")],
        issues=[],
        action_signature="search",
        issue_signature=None,
    )
    error_step = SimpleNamespace(
        state_changed=False,
        executed_tools=[SimpleNamespace(name="read", status="error")],
        issues=[SimpleNamespace(level="error")],
        action_signature="read",
        issue_signature="missing path",
    )

    assert _detect_stuck_episode(
        [ok_step, error_step, error_step],
        repeat_limit=4,
        no_progress_window=3,
    ) is None
