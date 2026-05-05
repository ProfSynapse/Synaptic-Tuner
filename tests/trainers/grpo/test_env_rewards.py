import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from env_rewards import build_env_reward_function


def test_build_env_reward_function_applies_stop_reason_penalties():
    reward = build_env_reward_function(
        {
            "default": 0.0,
            "rules": [
                {
                    "type": "add_if",
                    "when": {"type": "field_equals", "field": "env_passed", "value": True},
                    "score": 1.0,
                },
                {
                    "type": "add_if",
                    "when": {"type": "field_equals", "field": "env_passed", "value": False},
                    "score": -1.0,
                },
                {
                    "type": "linear",
                    "field": "total_turns",
                    "baseline": 1,
                    "min_delta": 0,
                    "weight": -0.1,
                },
                {
                    "type": "add_if",
                    "when": {
                        "type": "field_equals",
                        "field": "stop_reason",
                        "value": "max_tool_steps_exceeded",
                    },
                    "score": -0.5,
                },
            ],
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
