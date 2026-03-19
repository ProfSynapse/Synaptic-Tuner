"""Environment-backed reward helpers for env-GRPO."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


def build_env_reward_function(reward_cfg: Dict[str, Any]) -> Callable[..., List[float]]:
    success_reward = float(reward_cfg.get("success_reward", 1.0))
    failure_penalty = float(reward_cfg.get("failure_penalty", -1.0))
    step_penalty = float(reward_cfg.get("step_penalty", 0.0))
    max_tool_steps_penalty = float(reward_cfg.get("max_tool_steps_penalty", 0.0))
    text_before_completion_penalty = float(reward_cfg.get("text_before_completion_penalty", 0.0))

    def reward_from_env(completions, **kwargs) -> List[float]:
        env_passed = kwargs.get("env_passed") or []
        stop_reasons = kwargs.get("stop_reason") or []
        total_turns = kwargs.get("total_turns") or []

        rewards: List[float] = []
        for index, _completion in enumerate(completions):
            passed = bool(env_passed[index]) if index < len(env_passed) else False
            stop_reason = stop_reasons[index] if index < len(stop_reasons) else ""
            turns = int(total_turns[index]) if index < len(total_turns) else 0

            reward = success_reward if passed else failure_penalty
            reward -= step_penalty * max(turns - 1, 0)

            if stop_reason == "max_tool_steps_exceeded":
                reward -= max_tool_steps_penalty
            elif stop_reason == "text_response_before_completion":
                reward -= text_before_completion_penalty

            rewards.append(float(reward))

        return rewards

    return reward_from_env

