import sys
from pathlib import Path

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from env_rollout import build_prompt_registry


def test_build_prompt_registry_rejects_duplicate_prompts():
    dataset = Dataset.from_list(
        [
            {
                "prompt": "same",
                "prompt_messages": [],
                "resolved_environment_config": {},
                "task_context": {},
                "metadata": {"scenario": "one"},
            },
            {
                "prompt": "same",
                "prompt_messages": [],
                "resolved_environment_config": {},
                "task_context": {},
                "metadata": {"scenario": "two"},
            },
        ]
    )

    try:
        build_prompt_registry(dataset)
    except ValueError as exc:
        assert "Duplicate prompt" in str(exc)
    else:
        raise AssertionError("Expected duplicate prompt registry failure")
