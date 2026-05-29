import sys

from datasets import Dataset

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from env_dataset import (
    filter_env_rollout_dataset,
    format_dataset_for_env_grpo,
    format_dataset_for_env_grpo_with_options,
)


def test_filter_and_format_env_rollout_dataset():
    dataset = Dataset.from_list(
        [
            {
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "assistant"},
                ],
                "metadata": {
                    "scenario": "demo",
                    "environment": {"passed": True},
                    "stage_reviews": {
                        "environment_generation": {"passed": True},
                        "user_generation": {"passed": True},
                        "final": {"passed": True},
                    },
                    "generated_environment": {
                        "environment": {
                            "fixture": {"directories": [], "files": [], "notes": []},
                            "assertions": [],
                        }
                    },
                    "task_context": {"target": "file.md"},
                    "hard_requirements": ["env_passed"],
                    "quality_rubric": ["be correct"],
                },
            },
            {
                "conversations": [{"role": "system", "content": "bad"}],
                "metadata": {
                    "scenario": "bad",
                    "environment": {"passed": False},
                    "stage_reviews": {},
                },
            },
        ]
    )

    filtered = filter_env_rollout_dataset(
        dataset,
        required_stage_reviews=["environment_generation", "user_generation", "final"],
    )
    assert len(filtered) == 1

    formatted = format_dataset_for_env_grpo(filtered)
    row = formatted[0]
    assert row["example_id"].startswith("demo:")
    assert row["prompt_messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert row["resolved_environment_config"]["fixture"] == {
        "directories": [],
        "files": [],
        "notes": [],
    }


def test_format_env_rollout_dataset_can_append_configured_system_prompt():
    dataset = Dataset.from_list(
        [
            {
                "conversations": [
                    {"role": "system", "content": "Base instructions."},
                    {"role": "user", "content": "Do the task."},
                    {"role": "assistant", "content": "assistant"},
                ],
                "metadata": {
                    "scenario": "demo",
                    "environment": {"passed": True},
                    "stage_reviews": {"final": {"passed": True}},
                    "generated_environment": {
                        "environment": {
                            "fixture": {"directories": [], "files": []},
                            "assertions": [],
                        }
                    },
                    "task_context": {"target": "file.md"},
                },
            }
        ]
    )

    formatted = format_dataset_for_env_grpo_with_options(
        dataset,
        prompt_augmentation={
            "system_append": "Continue with the next useful step when the request is already authorized.",
        },
    )

    messages = formatted[0]["prompt_messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == (
        "Base instructions.\n\n"
        "Continue with the next useful step when the request is already authorized."
    )


def test_format_env_rollout_dataset_can_insert_configured_system_prompt():
    dataset = Dataset.from_list(
        [
            {
                "conversations": [
                    {"role": "user", "content": "Do the task."},
                    {"role": "assistant", "content": "assistant"},
                ],
                "metadata": {
                    "scenario": "demo",
                    "environment": {"passed": True},
                    "generated_environment": {
                        "environment": {
                            "fixture": {"directories": [], "files": []},
                            "assertions": [],
                        }
                    },
                    "task_context": {"target": "file.md"},
                },
            }
        ]
    )

    formatted = format_dataset_for_env_grpo_with_options(
        dataset,
        prompt_augmentation={
            "system_append": "Use the available environment tools to complete the task.",
            "insert_if_missing": True,
        },
    )

    assert formatted[0]["prompt_messages"] == [
        {"role": "system", "content": "Use the available environment tools to complete the task."},
        {"role": "user", "content": "Do the task."},
    ]
