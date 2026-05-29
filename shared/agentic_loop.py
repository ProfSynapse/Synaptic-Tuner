"""Shared environment-backed agentic episode runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from shared.agentic_judge import AgenticJudgeResult
from shared.environments import EnvironmentSession
from shared.environments.tool_executor import format_tool_results_message


@dataclass
class AgenticModelResponse:
    """Normalized model response for one agentic turn."""

    message: Any
    raw: Optional[Dict[str, Any]] = None
    latency_s: float = 0.0


@dataclass
class AgenticEpisodeTurn:
    """Trace for one agentic turn."""

    turn_index: int
    response: AgenticModelResponse
    validation: Any
    environment_step: Any = None
    judge_result: Optional[AgenticJudgeResult] = None
    final_text_turn: bool = False


@dataclass
class AgenticEpisodeResult:
    """Aggregate result for a shared environment-backed episode."""

    final_response: Any
    final_raw: Optional[Dict[str, Any]]
    total_latency_s: float
    conversation_trace: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    turns: List[AgenticEpisodeTurn] = field(default_factory=list)
    judge_trace: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "max_turns_reached"
    environment_result: Any = None
    final_text_required: bool = False
    final_text_satisfied: bool = False


def run_environment_episode(
    *,
    initial_messages: Sequence[Mapping[str, Any]],
    session: EnvironmentSession,
    respond: Callable[[Sequence[Mapping[str, Any]], int], AgenticModelResponse],
    validate: Callable[[Any], Any],
    max_turns: int = 6,
    max_tool_steps: int = 0,
    stop_on_text_response: bool = True,
    stop_on_environment_pass: bool = False,
    continue_on_execution_error: bool = False,
    stuck_repeat_limit: int = 2,
    no_progress_window: int = 3,
    tool_result_format: str = "json",
    expected_tools: Optional[Sequence[str]] = None,
    require_expected_tools: bool = False,
    stringify_response: Optional[Callable[[Any], str]] = None,
    judge_turn: Optional[Callable[[Dict[str, Any]], AgenticJudgeResult]] = None,
    judge_feedback_visible_to_model: bool = False,
    judge_stop_on_hard_failure: bool = False,
    require_final_text_after_pass: bool = False,
    final_text_prompt: Optional[str] = None,
) -> AgenticEpisodeResult:
    """Run a multi-turn environment episode with shared loop semantics."""
    messages = [dict(message) for message in initial_messages]
    conversation_trace = _messages_to_trace(messages)
    turns: List[AgenticEpisodeTurn] = []
    final_response: Any = None
    final_raw: Optional[Dict[str, Any]] = None
    total_latency_s = 0.0
    stop_reason = "max_turns_reached"
    stringify = stringify_response or _default_stringify_response
    judge_trace: List[Dict[str, Any]] = []
    awaiting_final_text = False
    final_text_satisfied = False

    for turn_index in range(1, max_turns + 1):
        response = respond(messages, turn_index)
        total_latency_s += float(response.latency_s or 0.0)
        final_response = response.message
        final_raw = response.raw

        conversation_trace.append(
            {
                "role": "assistant",
                "kind": "assistant_response",
                "content": stringify(response.message),
                "raw": response.message,
                "turn_index": turn_index,
            }
        )

        validation = validate(response.message)
        turns.append(AgenticEpisodeTurn(turn_index=turn_index, response=response, validation=validation))
        if not _validation_passed(validation):
            stop_reason = "schema_validation_failed"
            break

        messages.append({"role": "assistant", "content": stringify(response.message)})

        if awaiting_final_text:
            turns[-1].final_text_turn = True
            judge_result = _run_turn_judge(
                judge_turn=judge_turn,
                judge_trace=judge_trace,
                messages=messages,
                response=response,
                validation=validation,
                environment_step=None,
                turn_index=turn_index,
                environment_preview=None,
                tool_feedback=None,
            )
            turns[-1].judge_result = judge_result
            if judge_result is not None and judge_result.hard_failure and judge_stop_on_hard_failure:
                stop_reason = "judge_hard_failure"
                break
            if _response_has_tool_calls(validation, response.message):
                stop_reason = "final_text_tool_calls_emitted"
                break
            if not _extract_text_content(response.message).strip():
                stop_reason = "final_text_missing"
                break
            final_text_satisfied = True
            stop_reason = "environment_passed_final_text"
            break

        step = session.execute_response(response.message)
        turns[-1].environment_step = step

        if step.hard_error:
            stop_reason = "environment_execution_failed"
            break

        expected_tools_for_check = expected_tools if require_expected_tools else None
        environment_preview = session.finalize(
            expected_tools=expected_tools_for_check,
            total_turns=turn_index,
            stop_reason="preview",
        )

        has_tool_calls = _response_has_tool_calls(validation, response.message)
        feedback = None
        if has_tool_calls or (step.recoverable_error and continue_on_execution_error):
            feedback = format_tool_results_message(
                executions=step.executed_tools,
                issues=step.issues,
                format_name=tool_result_format,
            )
            messages.append({"role": "user", "content": feedback})
            conversation_trace.append(
                {
                    "role": "user",
                    "kind": "tool_feedback",
                    "content": feedback,
                    "turn_index": turn_index,
                }
            )

        judge_result = _run_turn_judge(
            judge_turn=judge_turn,
            judge_trace=judge_trace,
            messages=messages,
            response=response,
            validation=validation,
            environment_step=step,
            turn_index=turn_index,
            environment_preview=environment_preview,
            tool_feedback=feedback,
        )
        turns[-1].judge_result = judge_result
        if judge_result is not None and judge_result.hard_failure and judge_stop_on_hard_failure:
            stop_reason = "judge_hard_failure"
            break
        if judge_result is not None and judge_feedback_visible_to_model and judge_result.feedback_to_model:
            messages.append({"role": "user", "content": judge_result.feedback_to_model})
            conversation_trace.append(
                {
                    "role": "user",
                    "kind": "judge_feedback",
                    "content": judge_result.feedback_to_model,
                    "turn_index": turn_index,
                }
            )
        if stop_on_environment_pass and environment_preview.passed:
            if require_final_text_after_pass:
                awaiting_final_text = True
                completion_prompt = final_text_prompt or (
                    "The task is complete. Reply to the user with a brief final text-only response. "
                    "Do not call any more tools."
                )
                messages.append({"role": "user", "content": completion_prompt})
                conversation_trace.append(
                    {
                        "role": "user",
                        "kind": "final_text_request",
                        "content": completion_prompt,
                        "turn_index": turn_index,
                    }
                )
                continue
            stop_reason = "environment_passed"
            break

        if max_tool_steps and len(session.executed_tools) > max_tool_steps:
            stop_reason = "max_tool_steps_exceeded"
            break

        stuck_reason = _detect_stuck_episode(
            session.steps,
            repeat_limit=stuck_repeat_limit,
            no_progress_window=no_progress_window,
        )
        if stuck_reason:
            stop_reason = stuck_reason
            break

        if has_tool_calls or (step.recoverable_error and continue_on_execution_error) or (
            judge_result is not None and judge_feedback_visible_to_model and judge_result.feedback_to_model
        ):
            continue

        if require_final_text_after_pass and not environment_preview.passed:
            stop_reason = "text_response_before_completion"
            break

        if stop_on_text_response:
            stop_reason = "text_response"
            break

    environment_result = session.finalize(
        expected_tools=expected_tools if require_expected_tools else None,
        total_turns=len(turns),
        stop_reason=stop_reason,
    )
    return AgenticEpisodeResult(
        final_response=final_response,
        final_raw=final_raw,
        total_latency_s=total_latency_s,
        conversation_trace=conversation_trace,
        messages=messages,
        turns=turns,
        judge_trace=judge_trace,
        stop_reason=stop_reason,
        environment_result=environment_result,
        final_text_required=require_final_text_after_pass,
        final_text_satisfied=final_text_satisfied,
    )


def _validation_passed(validation: Any) -> bool:
    if validation is None:
        return False
    passed = getattr(validation, "passed", None)
    if passed is not None:
        return bool(passed)
    if isinstance(validation, dict):
        return bool(validation.get("passed"))
    return False


def _response_has_tool_calls(validation: Any, response_message: Any) -> bool:
    tool_calls = getattr(validation, "tool_calls", None)
    if tool_calls is None and isinstance(validation, dict):
        tool_calls = validation.get("tool_calls")
    if tool_calls is not None:
        return bool(tool_calls)
    if isinstance(response_message, dict):
        return bool(response_message.get("tool_calls"))
    return False


def _messages_to_trace(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    for index, message in enumerate(messages, start=1):
        trace.append(
            {
                "index": index,
                "role": str(message.get("role", "")),
                "kind": "prompt_message",
                "content": message.get("content"),
            }
        )
    return trace


def _default_stringify_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content")
        tool_calls = response.get("tool_calls") or []
        parts: List[str] = []
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        if tool_calls:
            parts.append(f"Tool calls: {tool_calls}")
        return "\n\n".join(parts).strip() or str(response)
    return str(response)


def _extract_text_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content")
        return content if isinstance(content, str) else ""
    return str(response or "")


def _validation_to_dict(validation: Any) -> Dict[str, Any]:
    if validation is None:
        return {}
    if hasattr(validation, "to_dict"):
        return validation.to_dict()
    if isinstance(validation, dict):
        return dict(validation)
    return {"value": str(validation)}


def _run_turn_judge(
    *,
    judge_turn: Optional[Callable[[Dict[str, Any]], AgenticJudgeResult]],
    judge_trace: List[Dict[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    response: AgenticModelResponse,
    validation: Any,
    environment_step: Any,
    turn_index: int,
    environment_preview: Any,
    tool_feedback: Optional[str],
) -> Optional[AgenticJudgeResult]:
    if judge_turn is None:
        return None
    payload = {
        "messages": [dict(message) for message in messages],
        "response_message": response.message,
        "response_raw": response.raw,
        "response_latency_s": response.latency_s,
        "validation": _validation_to_dict(validation),
        "environment_step": environment_step.to_dict() if hasattr(environment_step, "to_dict") else environment_step,
        "environment_preview": environment_preview.to_dict() if hasattr(environment_preview, "to_dict") else environment_preview,
        "tool_feedback": tool_feedback,
        "turn_index": turn_index,
    }
    result = judge_turn(payload)
    if result is not None:
        judge_trace.append({"turn_index": turn_index, **result.to_dict()})
    return result


def _detect_stuck_episode(
    steps,
    *,
    repeat_limit: int,
    no_progress_window: int,
) -> Optional[str]:
    if not steps:
        return None

    repeat_limit = max(int(repeat_limit or 0), 2)
    no_progress_window = max(int(no_progress_window or 0), 2)

    tail = steps[-repeat_limit:]
    if len(tail) == repeat_limit:
        first = tail[0]
        if (
            first.issue_signature
            and all(step.issue_signature == first.issue_signature for step in tail)
            and all(step.action_signature == first.action_signature for step in tail)
            and all(not step.state_changed for step in tail)
            and all(any(issue.level.lower() == "error" for issue in step.issues) for step in tail)
        ):
            return "stuck_repeated_failure"

    window = steps[-no_progress_window:]
    if (
        len(window) == no_progress_window
        and all(not step.state_changed for step in window)
        and all(step.executed_tools for step in window)
        and any(any(issue.level.lower() == "error" for issue in step.issues) for step in window)
    ):
        return "stuck_no_progress"

    return None
