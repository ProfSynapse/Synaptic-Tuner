"""Helpers for cloud-first environment-backed GRPO datasets.

This module prepares canonical SynthChat rollout artifacts for a future
multi-step environment-backed GRPO trainer. It does not hardcode scenario
families; it just extracts the replayable environment state and initial
messages from rollout records.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset


def load_env_rollout_dataset(
    *,
    dataset_name: Optional[str] = None,
    data_files: Optional[str] = None,
    local_file: Optional[str] = None,
    num_proc: int = 1,
) -> Dataset:
    """Load canonical rollout rows for env-GRPO."""
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if local_file:
        dataset = load_dataset("json", data_files=local_file, split="train", cache_dir=cache_dir)
    elif dataset_name:
        if data_files:
            dataset = load_dataset(
                dataset_name,
                data_files=data_files,
                num_proc=num_proc,
                cache_dir=cache_dir,
            )["train"]
        else:
            dataset = load_dataset(dataset_name, num_proc=num_proc, cache_dir=cache_dir)["train"]
    else:
        raise ValueError("Must provide either dataset_name or local_file")
    return dataset


def filter_env_rollout_dataset(
    dataset: Dataset,
    *,
    require_environment_passed: bool = True,
    required_stage_reviews: Optional[Iterable[str]] = None,
    require_environment_config: bool = True,
) -> Dataset:
    """Filter canonical rollout rows down to replayable, clean examples."""
    required_reviews = list(required_stage_reviews or [])

    def _keep(example: Dict[str, Any]) -> bool:
        metadata = example.get("metadata") or {}
        if not isinstance(metadata, dict):
            return False

        if require_environment_passed:
            environment = metadata.get("environment") or {}
            if not isinstance(environment, dict) or not bool(environment.get("passed")):
                return False

        stage_reviews = metadata.get("stage_reviews") or {}
        if not isinstance(stage_reviews, dict):
            stage_reviews = {}
        for stage_name in required_reviews:
            review = stage_reviews.get(stage_name) or {}
            if not isinstance(review, dict) or not bool(review.get("passed")):
                return False

        if require_environment_config and not _resolve_environment_config(metadata):
            return False

        return True

    return dataset.filter(_keep, desc="Filtering env rollout rows")


def format_dataset_for_env_grpo(dataset: Dataset) -> Dataset:
    """Project canonical rollout rows into replay-ready env examples."""

    def _format(example: Dict[str, Any]) -> Dict[str, Any]:
        metadata = example.get("metadata") or {}
        conversations = example.get("conversations") or []
        initial_messages = _extract_initial_messages(conversations)
        task_context = metadata.get("task_context") or {}
        environment_config = _resolve_environment_config(metadata) or {}
        scenario = metadata.get("scenario") or "unknown"
        seed_meta = metadata.get("environment_seed") or {}
        example_id = _build_example_id(
            scenario=scenario,
            initial_messages=initial_messages,
            seed_meta=seed_meta,
        )
        result = dict(example)
        result["example_id"] = example_id
        result["prompt_messages"] = initial_messages
        result["resolved_environment_config"] = environment_config
        result["task_context"] = task_context
        result["hard_requirements"] = metadata.get("hard_requirements") or []
        result["quality_rubric"] = metadata.get("quality_rubric") or []
        result["environment_seed"] = seed_meta
        return result

    return dataset.map(_format, desc="Formatting env rollout dataset")


def _extract_initial_messages(conversations: Any) -> List[Dict[str, Any]]:
    if not isinstance(conversations, list):
        return []

    prompt_messages: List[Dict[str, Any]] = []
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        if role == "assistant":
            break
        prompt_messages.append(
            {
                "role": role,
                "content": item.get("content", ""),
            }
        )
    return prompt_messages


def _resolve_environment_config(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    resolved = metadata.get("resolved_environment_config")
    if isinstance(resolved, dict):
        return resolved

    generated = metadata.get("generated_environment")
    if isinstance(generated, dict):
        environment = generated.get("environment")
        if isinstance(environment, dict):
            return environment
    return None


def _build_example_id(
    *,
    scenario: str,
    initial_messages: List[Dict[str, Any]],
    seed_meta: Dict[str, Any],
) -> str:
    payload = {
        "scenario": scenario,
        "seed_meta": seed_meta,
        "initial_messages": initial_messages,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"{scenario}:{digest}"
