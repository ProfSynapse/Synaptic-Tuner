"""Environment-backed reward helpers for env-GRPO.

The reward function is intentionally config-driven. It can score generic
environment progress signals emitted by the rollout bridge without assuming a
specific scenario, tool catalog, or wrapper shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Dict, List

from shared.validation.parsing.response_parser import parse_response


def build_env_reward_function(reward_cfg: Dict[str, Any]) -> Callable[..., List[float]]:
    success_reward = float(reward_cfg.get("success_reward", 1.0))
    failure_penalty = float(reward_cfg.get("failure_penalty", -1.0))
    step_penalty = float(reward_cfg.get("step_penalty", 0.0))
    max_tool_steps_penalty = float(reward_cfg.get("max_tool_steps_penalty", 0.0))
    text_before_completion_penalty = float(reward_cfg.get("text_before_completion_penalty", 0.0))
    require_final_text_satisfied = bool(reward_cfg.get("require_final_text_satisfied", False))
    final_text_failure_penalty = float(reward_cfg.get("final_text_failure_penalty", 0.0))
    no_tool_call_penalty = float(reward_cfg.get("no_tool_call_penalty", 0.0))
    tool_call_parse_reward = float(reward_cfg.get("tool_call_parse_reward", 0.0))
    expected_tool_reward = float(reward_cfg.get("expected_tool_reward", 0.0))
    expected_tool_order_reward = float(reward_cfg.get("expected_tool_order_reward", 0.0))
    final_text_satisfied_reward = float(reward_cfg.get("final_text_satisfied_reward", 0.0))
    tool_text_mix_penalty = float(reward_cfg.get("tool_text_mix_penalty", 0.0))
    placeholder_path_penalty = float(reward_cfg.get("placeholder_path_penalty", 0.0))
    invalid_wrapper_arg_penalty = float(reward_cfg.get("invalid_wrapper_arg_penalty", 0.0))
    tool_status_rewards = _float_map(reward_cfg.get("tool_status_rewards") or {})
    environment_issue_penalties = _float_map(reward_cfg.get("environment_issue_penalties") or {})
    max_progress_reward = _optional_float(reward_cfg.get("max_progress_reward"))
    wrapper_name = str(reward_cfg.get("wrapper_name") or "").strip()
    wrapper_cli_field = str(reward_cfg.get("wrapper_cli_field") or "tool").strip()
    placeholder_patterns = [
        str(item)
        for item in reward_cfg.get("placeholder_patterns", [])
        if str(item).strip()
    ]

    def reward_from_env(completions, **kwargs) -> List[float]:
        env_passed = kwargs.get("env_passed") or []
        final_text_satisfied = kwargs.get("final_text_satisfied") or []
        stop_reasons = kwargs.get("stop_reason") or []
        total_turns = kwargs.get("total_turns") or []
        total_tool_calls = kwargs.get("total_tool_calls") or []
        executed_tool_names = kwargs.get("executed_tool_names") or []
        executed_tool_statuses = kwargs.get("executed_tool_statuses") or []
        environment_issue_levels = kwargs.get("environment_issue_levels") or []
        expected_tool_names = kwargs.get("expected_tool_names") or []

        rewards: List[float] = []
        for index, completion in enumerate(completions):
            passed = bool(env_passed[index]) if index < len(env_passed) else False
            final_text_ok = bool(final_text_satisfied[index]) if index < len(final_text_satisfied) else False
            if require_final_text_satisfied and not final_text_ok:
                passed = False
            stop_reason = stop_reasons[index] if index < len(stop_reasons) else ""
            turns = int(total_turns[index]) if index < len(total_turns) else 0
            tool_calls = int(total_tool_calls[index]) if index < len(total_tool_calls) else 0
            executed = _as_string_list(executed_tool_names[index] if index < len(executed_tool_names) else [])
            statuses = _as_string_list(executed_tool_statuses[index] if index < len(executed_tool_statuses) else [])
            issue_levels = _as_string_list(environment_issue_levels[index] if index < len(environment_issue_levels) else [])
            expected = _as_string_list(expected_tool_names[index] if index < len(expected_tool_names) else [])

            reward = success_reward if passed else failure_penalty
            reward -= step_penalty * max(turns - 1, 0)
            progress_reward = 0.0

            completion_text = _completion_to_text(completion)
            parsed = parse_response(completion_text)
            if parsed.has_tool_calls:
                progress_reward += tool_call_parse_reward
                if parsed.text_content.strip():
                    reward -= tool_text_mix_penalty
                if _has_invalid_wrapper_arg(
                    parsed.tool_calls,
                    wrapper_name=wrapper_name,
                    wrapper_cli_field=wrapper_cli_field,
                ):
                    reward -= invalid_wrapper_arg_penalty
            elif expected:
                reward -= no_tool_call_penalty

            if placeholder_patterns and _contains_any(completion_text, placeholder_patterns):
                reward -= placeholder_path_penalty

            if expected:
                matched_count = _count_unique_matches(executed, expected)
                progress_reward += expected_tool_reward * matched_count
                progress_reward += expected_tool_order_reward * _ordered_prefix_matches(executed, expected)

            for status in statuses:
                progress_reward += tool_status_rewards.get(status.lower(), 0.0)

            for level in issue_levels:
                reward -= environment_issue_penalties.get(level.lower(), 0.0)

            if final_text_ok:
                progress_reward += final_text_satisfied_reward

            if max_progress_reward is not None:
                progress_reward = min(progress_reward, max_progress_reward)
            reward += progress_reward

            if stop_reason == "max_tool_steps_exceeded":
                reward -= max_tool_steps_penalty
            elif stop_reason == "text_response_before_completion":
                reward -= text_before_completion_penalty
            elif stop_reason in {"final_text_missing", "final_text_tool_calls_emitted"}:
                reward -= final_text_failure_penalty

            rewards.append(float(reward))

        return rewards

    return reward_from_env


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, float] = {}
    for key, raw in value.items():
        try:
            result[str(key).strip().lower()] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
    return str(completion or "")


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _count_unique_matches(executed: List[str], expected: List[str]) -> int:
    executed_set = set(executed)
    return sum(1 for item in dict.fromkeys(expected) if item in executed_set)


def _ordered_prefix_matches(executed: List[str], expected: List[str]) -> int:
    if not executed or not expected:
        return 0
    matches = 0
    search_from = 0
    for expected_name in expected:
        try:
            found_at = executed.index(expected_name, search_from)
        except ValueError:
            break
        matches += 1
        search_from = found_at + 1
    return matches


def _contains_any(text: str, patterns: List[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _has_invalid_wrapper_arg(tool_calls: Any, *, wrapper_name: str, wrapper_cli_field: str) -> bool:
    if not wrapper_name or not wrapper_cli_field:
        return False
    for call in tool_calls or []:
        name = str(getattr(call, "name", "")).strip()
        if name != wrapper_name:
            continue
        args = getattr(call, "arguments", {})
        if not isinstance(args, dict):
            return True
        if not isinstance(args.get(wrapper_cli_field), str):
            return True
    return False
