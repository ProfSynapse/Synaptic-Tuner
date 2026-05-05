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
        dataset = _load_local_jsonl(local_file)
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


def _load_local_jsonl(local_file: str) -> Dataset:
    """Load JSONL without Arrow inferring heterogeneous nested metadata.

    Canonical rollout rows often contain rich, model-generated nested metadata
    whose shape can vary row to row. Store those nested values as JSON strings
    and decode them in the env formatting helpers.
    """
    rows: List[Dict[str, Any]] = []
    with open(local_file, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row {line_number} in {local_file}: {exc}") from exc
            rows.append(
                {
                    "conversations": _json_text(raw.get("conversations") or []),
                    "metadata": _json_text(raw.get("metadata") or {}),
                    "scenario": raw.get("scenario") or (raw.get("metadata") or {}).get("scenario") or "",
                }
            )
    return Dataset.from_list(rows)


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
        metadata = _as_mapping(example.get("metadata"))
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


def format_dataset_for_env_grpo(
    dataset: Dataset,
    *,
    prompt_message_roles: Optional[Iterable[str]] = None,
    user_prompt_prefix: Optional[str] = None,
    user_prompt_suffix: Optional[str] = None,
) -> Dataset:
    """Project canonical rollout rows into replay-ready env examples."""
    allowed_roles = None
    if prompt_message_roles is not None:
        allowed_roles = {
            str(role).strip()
            for role in prompt_message_roles
            if str(role).strip()
        }

    def _format(example: Dict[str, Any]) -> Dict[str, Any]:
        metadata = _as_mapping(example.get("metadata"))
        conversations = _as_list(example.get("conversations"))
        initial_messages = _extract_initial_messages(
            conversations,
            allowed_roles=allowed_roles,
            user_prompt_prefix=user_prompt_prefix,
            user_prompt_suffix=user_prompt_suffix,
        )
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


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        decoded = json.loads(value)
        return decoded if isinstance(decoded, list) else []
    return []


def _extract_initial_messages(
    conversations: Any,
    *,
    allowed_roles: Optional[set[str]] = None,
    user_prompt_prefix: Optional[str] = None,
    user_prompt_suffix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(conversations, list):
        return []

    prompt_messages: List[Dict[str, Any]] = []
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        if role == "assistant":
            break
        if allowed_roles is not None and role not in allowed_roles:
            continue
        content = item.get("content", "")
        if role == "user" and isinstance(content, str):
            content = f"{user_prompt_prefix or ''}{content}{user_prompt_suffix or ''}"
        prompt_messages.append({"role": role, "content": content})
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
