"""Shared environment-backed agentic episode runner."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from shared.agentic_judge import AgenticJudgeResult
from shared.environments import EnvironmentSession
from shared.environments.tool_executor import (
    format_environment_payload_for_model,
    format_tool_results_message,
)


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
    continue_on_validation_error: bool = False,
    stuck_repeat_limit: int = 2,
    no_progress_window: int = 3,
    tool_result_format: str = "json",
    tool_result_name_format: str = "executor",
    expected_tools: Optional[Sequence[str]] = None,
    require_expected_tools: bool = False,
    stringify_response: Optional[Callable[[Any], str]] = None,
    judge_turn: Optional[Callable[[Dict[str, Any]], AgenticJudgeResult]] = None,
    judge_feedback_visible_to_model: bool = False,
    judge_stop_on_hard_failure: bool = False,
    require_final_text_after_pass: bool = False,
    final_text_prompt: Optional[str] = None,
    debug_event_writer: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
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
        _emit_debug_event(
            debug_event_writer,
            "agentic_turn_start",
            {
                "turn_index": turn_index,
                "message_count": len(messages),
                "messages": messages,
            },
        )
        response = respond(messages, turn_index)
        total_latency_s += float(response.latency_s or 0.0)
        final_response = response.message
        final_raw = response.raw
        _emit_debug_event(
            debug_event_writer,
            "agentic_turn_response",
            {
                "turn_index": turn_index,
                "response": response.message,
                "raw": response.raw,
                "latency_s": response.latency_s,
            },
        )

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
        _emit_debug_event(
            debug_event_writer,
            "agentic_turn_validation",
            {
                "turn_index": turn_index,
                "validation": _validation_to_dict(validation),
            },
        )
        turns.append(AgenticEpisodeTurn(turn_index=turn_index, response=response, validation=validation))
        if not _validation_passed(validation):
            if continue_on_validation_error:
                feedback = _format_validation_feedback_message(validation, stringify(response.message))
                messages.append({"role": "user", "content": feedback})
                conversation_trace.append(
                    {
                        "role": "user",
                        "kind": "validation_feedback",
                        "content": feedback,
                        "turn_index": turn_index,
                    }
                )
                _emit_debug_event(
                    debug_event_writer,
                    "agentic_validation_feedback",
                    {
                        "turn_index": turn_index,
                        "feedback": feedback,
                    },
                )
                continue
            stop_reason = "schema_validation_failed"
            break

        messages.append(_assistant_message_for_history(response.message, stringify))

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
                tool_schema=session.validator.tool_schema,
                tool_name_format=tool_result_name_format,
            )
            turns[-1].judge_result = judge_result
            if judge_result is not None:
                _emit_debug_event(
                    debug_event_writer,
                    "agentic_turn_judge",
                    {
                        "turn_index": turn_index,
                        "final_text_turn": True,
                        "judge_result": judge_result.to_dict(),
                    },
                )
            if judge_result is not None and judge_result.hard_failure and judge_stop_on_hard_failure:
                stop_reason = "judge_hard_failure"
                break
            judge_requested_correction = bool(
                judge_result is not None
                and judge_feedback_visible_to_model
                and judge_result.feedback_to_model
                and not judge_result.passed
            )
            if judge_requested_correction:
                messages.append({"role": "user", "content": judge_result.feedback_to_model})
                conversation_trace.append(
                    {
                        "role": "user",
                        "kind": "judge_feedback",
                        "content": judge_result.feedback_to_model,
                        "turn_index": turn_index,
                    }
                )
                _emit_debug_event(
                    debug_event_writer,
                    "agentic_judge_feedback",
                    {
                        "turn_index": turn_index,
                        "final_text_turn": True,
                        "feedback": judge_result.feedback_to_model,
                    },
                )
                continue
            if _response_has_tool_calls(validation, response.message):
                stop_reason = "final_text_tool_calls_emitted"
                break
            if not _extract_text_content(response.message).strip():
                stop_reason = "final_text_missing"
                break
            if judge_result is not None and not judge_result.passed:
                stop_reason = "final_text_judge_failed"
                break
            final_text_satisfied = True
            stop_reason = "environment_passed_final_text"
            break

        step = session.execute_response(response.message)
        turns[-1].environment_step = step
        _emit_debug_event(
            debug_event_writer,
            "agentic_environment_step",
            {
                "turn_index": turn_index,
                "step": step.to_dict() if hasattr(step, "to_dict") else step,
            },
        )

        if step.hard_error:
            stop_reason = "environment_execution_failed"
            break

        preview_expected_tools = expected_tools if require_expected_tools else None
        environment_preview = session.finalize(
            expected_tools=preview_expected_tools,
            total_turns=turn_index,
            stop_reason="preview",
        )
        _emit_debug_event(
            debug_event_writer,
            "agentic_environment_preview",
            {
                "turn_index": turn_index,
                "preview": environment_preview.to_dict() if hasattr(environment_preview, "to_dict") else environment_preview,
            },
        )

        has_tool_calls = _response_has_tool_calls(validation, response.message)
        feedback = None
        if has_tool_calls or (step.recoverable_error and continue_on_execution_error):
            feedback = format_tool_results_message(
                executions=step.executed_tools,
                issues=step.issues,
                format_name=tool_result_format,
                tool_schema=session.validator.tool_schema,
                tool_name_format=tool_result_name_format,
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
            _emit_debug_event(
                debug_event_writer,
                "agentic_tool_feedback",
                {
                    "turn_index": turn_index,
                    "feedback": feedback,
                },
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
            tool_schema=session.validator.tool_schema,
            tool_name_format=tool_result_name_format,
        )
        turns[-1].judge_result = judge_result
        if judge_result is not None:
            _emit_debug_event(
                debug_event_writer,
                "agentic_turn_judge",
                {
                    "turn_index": turn_index,
                    "judge_result": judge_result.to_dict(),
                },
            )
        if judge_result is not None and judge_result.hard_failure and judge_stop_on_hard_failure:
            stop_reason = "judge_hard_failure"
            break
        judge_requested_correction = bool(
            judge_result is not None
            and judge_feedback_visible_to_model
            and judge_result.feedback_to_model
            and not judge_result.passed
        )
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
            _emit_debug_event(
                debug_event_writer,
                "agentic_judge_feedback",
                {
                    "turn_index": turn_index,
                    "feedback": judge_result.feedback_to_model,
                },
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
                _emit_debug_event(
                    debug_event_writer,
                    "agentic_final_text_request",
                    {
                        "turn_index": turn_index,
                        "prompt": completion_prompt,
                    },
                )
                continue
            stop_reason = "environment_passed"
            break

        if judge_requested_correction:
            continue

        if judge_result is not None and judge_result.should_stop:
            stop_reason = "judge_requested_stop"
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

        if has_tool_calls or (step.recoverable_error and continue_on_execution_error) or judge_requested_correction:
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
    _emit_debug_event(
        debug_event_writer,
        "agentic_episode_done",
        {
            "stop_reason": stop_reason,
            "turn_count": len(turns),
            "environment_result": environment_result.to_dict() if hasattr(environment_result, "to_dict") else environment_result,
            "judge_trace": judge_trace,
            "conversation_trace": conversation_trace,
        },
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


def _format_validation_feedback_message(validation: Any, response_text: str) -> str:
    issues = []
    raw_issues = getattr(validation, "issues", None)
    if raw_issues is None and isinstance(validation, dict):
        raw_issues = validation.get("issues")
    for issue in raw_issues or []:
        level = getattr(issue, "level", None)
        message = getattr(issue, "message", None)
        if isinstance(issue, dict):
            level = issue.get("level", level)
            message = issue.get("message", message)
        text = str(message or issue).strip()
        if text:
            prefix = str(level or "ERROR").upper()
            issues.append(f"- {prefix}: {text}")
    issue_text = "\n".join(issues) if issues else "- ERROR: response did not pass validation"
    return (
        "Your previous assistant response failed schema validation and was not executed.\n"
        "Return a corrected assistant response using the configured format.\n\n"
        "Correction rules:\n"
        "- function.arguments must be a valid JSON object serialized as a string.\n"
        "- Use double quotes for every JSON key and string value.\n"
        "- Escape quotes inside nested CLI commands with backslashes.\n"
        "- Do not use single-quoted pseudo-JSON, HTML entities, non-breaking spaces, ellipses, or placeholder macros.\n\n"
        "Validation issues:\n"
        f"{issue_text}\n\n"
        "Previous response:\n"
        f"{response_text}"
    )


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
            try:
                rendered_tool_calls = json.dumps(tool_calls, ensure_ascii=False)
            except TypeError:
                rendered_tool_calls = str(tool_calls)
            parts.append(f"Tool calls: {rendered_tool_calls}")
        return "\n\n".join(parts).strip() or str(response)
    return str(response)


def _assistant_message_for_history(response: Any, stringify: Callable[[Any], str]) -> Dict[str, Any]:
    """Preserve structured assistant messages while retaining text fallback."""
    if isinstance(response, dict):
        message = {"role": "assistant", "content": response.get("content")}
        if response.get("tool_calls") is not None:
            message["tool_calls"] = response.get("tool_calls")
        if message.get("content") is None and not message.get("tool_calls"):
            message["content"] = stringify(response)
        return message
    return {"role": "assistant", "content": stringify(response)}


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


def _emit_debug_event(
    debug_event_writer: Optional[Callable[[str, Dict[str, Any]], bool]],
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    if debug_event_writer is None:
        return
    try:
        debug_event_writer(event_type, _json_safe(payload))
    except Exception:
        return


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return str(value)


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
    tool_schema: Optional[Dict[str, Any]] = None,
    tool_name_format: str = "executor",
) -> Optional[AgenticJudgeResult]:
    if judge_turn is None:
        return None
    payload = {
        "messages": [dict(message) for message in messages],
        "response_message": response.message,
        "response_raw": response.raw,
        "response_latency_s": response.latency_s,
        "validation": _validation_to_dict(validation),
        "environment_step": format_environment_payload_for_model(
            environment_step.to_dict() if hasattr(environment_step, "to_dict") else environment_step,
            tool_schema,
            tool_name_format,
        ),
        "environment_preview": format_environment_payload_for_model(
            environment_preview.to_dict() if hasattr(environment_preview, "to_dict") else environment_preview,
            tool_schema,
            tool_name_format,
        ),
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
        and all(any(issue.level.lower() == "error" for issue in step.issues) for step in window)
    ):
        return "stuck_no_progress"

    return None
