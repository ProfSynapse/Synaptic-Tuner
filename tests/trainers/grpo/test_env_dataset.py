import sys

from datasets import Dataset

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from env_dataset import filter_env_rollout_dataset, format_dataset_for_env_grpo


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
