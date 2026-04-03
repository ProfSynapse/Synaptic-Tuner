"""SynthChat Agentic Episode - Multi-turn agentic episode generation.

Location: SynthChat/agentic/episode.py
Purpose: Generate multi-turn agentic rollouts using the shared environment
         episode runner. Builds turn judges, handles loop responses, and
         validates agentic assistant responses.
Usage: Called by SynthChatGenerator._generate_agentic_episode in generator.py
       when a scenario is configured for environment-backed agentic execution.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Mapping, Tuple

from ..template_utils import _make_json_safe

try:
    from shared.agentic_judge import AgenticTurnJudge
    from shared.agentic_loop import AgenticModelResponse, run_environment_episode
    from Evaluator.schema_validator import ValidationResult, ValidatorIssue, validate_assistant_response
except ImportError:
    AgenticTurnJudge = None
    AgenticModelResponse = None
    run_environment_episode = None
    ValidationResult = None
    ValidatorIssue = None
    validate_assistant_response = None


def validate_agentic_synthchat_response(message: Any):
    """Relax eval-only generator-format checks for synthetic loop rollouts."""
    result = validate_assistant_response(message, None)
    filtered_issues = []
    for issue in result.issues:
        message_text = str(issue.message)
        if "does not match generator format" in message_text:
            continue
        filtered_issues.append(issue)

    passed = all(str(issue.level).lower() != "error" for issue in filtered_issues)
    return ValidationResult(
        passed=passed,
        issues=[
            issue
            if isinstance(issue, ValidatorIssue)
            else ValidatorIssue(level=getattr(issue, "level", "ERROR"), message=getattr(issue, "message", str(issue)))
            for issue in filtered_issues
        ],
        tool_calls=result.tool_calls,
        context_validation=None,
    )


def build_turn_judge_template_vars(
    *,
    scenario_key: str,
    scenario: Dict[str, Any],
    assistant_prompt: str,
    system_context: Dict[str, Any],
    task_context: Dict[str, Any],
    hard_requirements: List[Dict[str, Any]],
    quality_rubric: List[str],
    turn_payload: Dict[str, Any],
) -> Dict[str, str]:
    """Build the template variable dict for a turn-level judge prompt."""
    messages = turn_payload.get("messages") or []
    safe_scenario = _make_json_safe(scenario or {})
    safe_system_context = _make_json_safe(system_context or {})
    safe_task_context = _make_json_safe(task_context or {})
    safe_hard_requirements = _make_json_safe(hard_requirements or [])
    safe_quality_rubric = _make_json_safe(quality_rubric or [])
    safe_turn_payload = _make_json_safe(turn_payload or {})
    latest_user = ""
    for message in reversed(messages):
        if str(message.get("role", "")).strip() == "user":
            latest_user = str(message.get("content") or "")
            break
    return {
        "scenario_key": scenario_key,
        "assistant_prompt": assistant_prompt,
        "scenario_json": json.dumps(safe_scenario, ensure_ascii=False, indent=2),
        "system_context_json": json.dumps(safe_system_context, ensure_ascii=False, indent=2),
        "task_context_json": json.dumps(safe_task_context, ensure_ascii=False, indent=2),
        "hard_requirements_json": json.dumps(safe_hard_requirements, ensure_ascii=False, indent=2),
        "quality_rubric_json": json.dumps(safe_quality_rubric, ensure_ascii=False, indent=2),
        "messages_json": json.dumps(_make_json_safe(messages), ensure_ascii=False, indent=2),
        "latest_user_message": latest_user,
        "assistant_response_json": json.dumps(safe_turn_payload.get("response_message"), ensure_ascii=False, indent=2),
        "validation_json": json.dumps(safe_turn_payload.get("validation") or {}, ensure_ascii=False, indent=2),
        "environment_step_json": json.dumps(safe_turn_payload.get("environment_step") or {}, ensure_ascii=False, indent=2),
        "environment_preview_json": json.dumps(safe_turn_payload.get("environment_preview") or {}, ensure_ascii=False, indent=2),
        "tool_feedback": str(safe_turn_payload.get("tool_feedback") or ""),
        "turn_index": str(turn_payload.get("turn_index") or ""),
    }


def build_turn_judge(
    *,
    scenario_key: str,
    scenario: Dict[str, Any],
    assistant_prompt: str,
    system_context: Dict[str, Any],
    task_context: Dict[str, Any],
    hard_requirements: List[Dict[str, Any]],
    quality_rubric: List[str],
    judge_config: Dict[str, Any],
    llm_client: Any,
    get_stage_llm_clients: Callable,
    logger: Any = None,
):
    """Build a turn-level judge callback for use in agentic episode runs."""
    if AgenticTurnJudge is None or not judge_config:
        return None
    if not bool(judge_config.get("enabled")):
        return None
    prompt_template = str(judge_config.get("prompt") or "").strip()
    if not prompt_template:
        return None
    judge = AgenticTurnJudge(
        llm_client=llm_client,
        llm_clients=get_stage_llm_clients(judge_config),
        prompt_template=prompt_template,
        system_prompt=judge_config.get("system"),
        output_schema=judge_config.get("output_schema"),
        temperature=float(judge_config.get("temperature", 0.2) or 0.2),
        max_tokens=judge_config.get("max_tokens"),
        max_retries=int(judge_config.get("max_retries", 3) or 3),
    )

    def run_judge(turn_payload: Dict[str, Any]):
        template_vars = build_turn_judge_template_vars(
            scenario_key=scenario_key,
            scenario=scenario,
            assistant_prompt=assistant_prompt,
            system_context=system_context,
            task_context=task_context,
            hard_requirements=hard_requirements,
            quality_rubric=quality_rubric,
            turn_payload=turn_payload,
        )
        result = judge.judge(template_vars)
        if logger:
            logger.info(
                f"[{scenario_key}] turn_judge done "
                f"(turn={turn_payload.get('turn_index')} passed={result.passed} "
                f"hard_failure={result.hard_failure} stop={result.should_stop})"
            )
        return result

    return run_judge


def synthchat_loop_response(
    *,
    scenario: Dict[str, Any],
    system_context: Dict[str, Any],
    messages: Sequence[Mapping[str, Any]],
    assistant_prompt: str,
    randomize_params: bool,
    scenario_key: str,
    turn_index: int,
    thinking_content: Optional[str],
    build_loop_assistant_context: Callable,
    generate_assistant_response: Callable,
    parse_response: Callable,
) -> Any:
    """Generate one assistant turn for a shared agentic episode."""
    assistant_context = build_loop_assistant_context(list(messages))
    if thinking_content:
        assistant_context = f"{assistant_context}\n\nYour prior thinking:\n{thinking_content}"

    trace_label = f"{scenario_key}:assistant_turn_{turn_index}"
    assistant_content = generate_assistant_response(
        scenario=scenario,
        system_context=system_context,
        assistant_context=assistant_context,
        assistant_prompt=assistant_prompt,
        randomize_params=randomize_params,
        trace_label=trace_label,
    )
    assistant_msg = parse_response(assistant_content, scenario)
    if thinking_content and assistant_msg.get("tool_calls"):
        assistant_msg["content"] = f"<thinking>{thinking_content}</thinking>"
    return AgenticModelResponse(
        message=assistant_msg,
        raw={"message": assistant_msg},
        latency_s=0.0,
    )


def generate_agentic_episode(
    *,
    scenario_key: str,
    scenario: Dict[str, Any],
    example: Dict[str, Any],
    assistant_prompt: str,
    randomize_params: bool,
    resolved_system_context: Dict[str, Any],
    resolved_environment_config: Dict[str, Any],
    resolved_task_context: Dict[str, Any],
    hard_requirements: List[Dict[str, Any]],
    quality_rubric: List[str],
    thinking_content: Optional[str],
    stage_failures: List[str],
    environment_validator: Any,
    llm_client: Any,
    get_stage_llm_clients: Callable,
    log_stage: Callable,
    build_loop_assistant_context: Callable,
    generate_assistant_response: Callable,
    parse_response: Callable,
    stringify_response: Callable,
    logger: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Generate a multi-turn agentic rollout using the shared episode runner."""
    loop_cfg = resolved_environment_config.get("loop") if isinstance(resolved_environment_config, dict) else {}
    max_turns = int(loop_cfg.get("max_turns", 6) or 6)
    max_tool_steps = int(loop_cfg.get("max_tool_steps", resolved_environment_config.get("max_steps", 0)) or 0)
    stop_on_text_response = bool(loop_cfg.get("stop_on_text_response", True))
    stop_on_environment_pass = bool(loop_cfg.get("stop_on_environment_pass", False))
    require_final_text_after_pass = bool(
        loop_cfg.get("require_final_text_after_pass", loop_cfg.get("require_final_text", False))
    )
    final_text_prompt = loop_cfg.get("final_text_prompt")
    continue_on_execution_error = bool(
        loop_cfg.get("continue_on_execution_error", str(loop_cfg.get("mode", "strict")).strip().lower() == "agentic")
    )
    stuck_repeat_limit = int(loop_cfg.get("stuck_repeat_limit", 2) or 2)
    no_progress_window = int(loop_cfg.get("no_progress_window", 3) or 3)
    tool_result_format = str(loop_cfg.get("tool_result_format", "json") or "json")
    judge_cfg = scenario.get("judge") if isinstance(scenario.get("judge"), dict) else {}
    in_loop_judge_cfg = judge_cfg.get("in_loop") if isinstance(judge_cfg.get("in_loop"), dict) else {}
    judge_feedback_visible_to_model = bool(in_loop_judge_cfg.get("feedback_visible_to_model", False))
    judge_stop_on_hard_failure = bool(in_loop_judge_cfg.get("stop_on_hard_failure", False))
    turn_judge = build_turn_judge(
        scenario_key=scenario_key,
        scenario=scenario,
        assistant_prompt=assistant_prompt,
        system_context=resolved_system_context,
        task_context=resolved_task_context,
        hard_requirements=hard_requirements,
        quality_rubric=quality_rubric,
        judge_config=in_loop_judge_cfg,
        llm_client=llm_client,
        get_stage_llm_clients=get_stage_llm_clients,
        logger=logger,
    )

    system_prompt_text = ""
    for msg in example["conversations"]:
        if msg.get("role") == "system":
            system_prompt_text = msg.get("content") or ""
            break

    session = environment_validator.start_session(
        system_prompt=system_prompt_text,
        environment_config=resolved_environment_config,
    )
    try:
        log_stage(scenario_key, "assistant_loop", "start")
        episode = run_environment_episode(
            initial_messages=example["conversations"],
            session=session,
            respond=lambda messages, turn_index: synthchat_loop_response(
                scenario=scenario,
                system_context=resolved_system_context,
                messages=messages,
                assistant_prompt=assistant_prompt,
                randomize_params=randomize_params,
                scenario_key=scenario_key,
                turn_index=turn_index,
                thinking_content=thinking_content,
                build_loop_assistant_context=build_loop_assistant_context,
                generate_assistant_response=generate_assistant_response,
                parse_response=parse_response,
            ),
            validate=validate_agentic_synthchat_response,
            max_turns=max_turns,
            max_tool_steps=max_tool_steps,
            stop_on_text_response=stop_on_text_response,
            stop_on_environment_pass=stop_on_environment_pass,
            continue_on_execution_error=continue_on_execution_error,
            stuck_repeat_limit=stuck_repeat_limit,
            no_progress_window=no_progress_window,
            tool_result_format=tool_result_format,
            expected_tools=scenario.get("expected_tools") or ([scenario.get("tool")] if scenario.get("tool") else None),
            require_expected_tools=bool(resolved_environment_config.get("require_expected_tools")),
            stringify_response=stringify_response,
            judge_turn=turn_judge,
            judge_feedback_visible_to_model=judge_feedback_visible_to_model,
            judge_stop_on_hard_failure=judge_stop_on_hard_failure,
            require_final_text_after_pass=require_final_text_after_pass,
            final_text_prompt=final_text_prompt,
        )
    finally:
        session.close()

    if not episode.environment_result.passed:
        stage_failures.append("environment")

    if episode.stop_reason in {
        "schema_validation_failed",
        "final_text_tool_calls_emitted",
        "final_text_missing",
        "judge_hard_failure",
        "judge_requested_stop",
    }:
        stage_failures.append("response")

    example["conversations"] = [dict(message) for message in episode.messages]
    example["conversation_trace"] = episode.conversation_trace
    final_response = episode.final_response if isinstance(episode.final_response, dict) else {"role": "assistant", "content": str(episode.final_response or "")}
    log_stage(
        scenario_key,
        "assistant_loop",
        "done",
        extra=f"turns={len(episode.turns)} stop={episode.stop_reason}",
    )
    environment_trace = episode.environment_result.to_dict()
    environment_trace["final_text_required"] = episode.final_text_required
    environment_trace["final_text_satisfied"] = episode.final_text_satisfied
    return final_response, example, {
        **environment_trace,
        "judge_trace": list(episode.judge_trace),
    }
