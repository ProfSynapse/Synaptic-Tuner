import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from env_rewards import build_env_reward_function


def test_build_env_reward_function_applies_stop_reason_penalties():
    reward = build_env_reward_function(
        {
            "success_reward": 1.0,
            "failure_penalty": -1.0,
            "step_penalty": 0.1,
            "max_tool_steps_penalty": 0.5,
            "text_before_completion_penalty": 0.25,
        }
    )

    scores = reward(
        ["a", "b"],
        env_passed=[True, False],
        stop_reason=["environment_passed_final_text", "max_tool_steps_exceeded"],
        total_turns=[2, 4],
    )

    assert scores[0] == 0.9
    assert scores[1] == -1.8

