"""SynthChat Target Spec Handling - Normalization and extraction of generation target configurations.

Location: SynthChat/targets.py
Purpose: Single source of truth for target spec logic, eliminating the DRY violation
         between generator.py and run.py that previously had identical implementations.
Usage: from SynthChat.targets import _normalize_target_spec, _extract_shared_seed_spec
"""

from typing import Any, Dict, List, Optional, Tuple


def _normalize_target_spec(raw_target: Any) -> Dict[str, int]:
    """Normalize target count config into explicit seed/rollout counts."""
    if isinstance(raw_target, bool):
        raise ValueError("Boolean target specs are not supported")
    if isinstance(raw_target, int):
        if raw_target < 0:
            raise ValueError("Target counts must be non-negative")
        return {"seed_count": raw_target, "rollouts_per_seed": 1}
    if not isinstance(raw_target, dict):
        raise ValueError(f"Unsupported target spec: {raw_target!r}")

    count = raw_target.get("count")
    seed_count = raw_target.get("seed_count", count if count is not None else 1)
    rollouts_per_seed = raw_target.get("rollouts_per_seed", 1)

    seed_count = int(seed_count)
    rollouts_per_seed = int(rollouts_per_seed)
    if seed_count < 0 or rollouts_per_seed < 0:
        raise ValueError("Target specs must use non-negative integers")
    return {
        "seed_count": seed_count,
        "rollouts_per_seed": rollouts_per_seed,
    }


def _extract_shared_seed_spec(targets: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Split special shared-seed config from normal scenario targets."""
    if not isinstance(targets, dict):
        raise ValueError("Targets must be a dictionary")

    cleaned_targets = dict(targets)
    raw_spec = cleaned_targets.pop("_shared_seed", None)
    if raw_spec is None:
        return None, cleaned_targets
    if not isinstance(raw_spec, dict):
        raise ValueError("_shared_seed must be an object")

    scenario_key = str(raw_spec.get("scenario") or raw_spec.get("scenario_key") or "").strip()
    if not scenario_key:
        raise ValueError("_shared_seed requires a 'scenario' key")

    seed_count = int(raw_spec.get("seed_count", 1) or 1)
    if seed_count < 0:
        raise ValueError("_shared_seed.seed_count must be non-negative")

    raw_targets = raw_spec.get("targets") or raw_spec.get("scenarios") or []
    target_keys = [str(item).strip() for item in raw_targets if str(item).strip()]

    return (
        {
            "scenario": scenario_key,
            "seed_count": seed_count,
            "targets": target_keys,
        },
        cleaned_targets,
    )


def _apply_stage_review_result(
    stage_failures: List[str],
    stage_reviews: Dict[str, Any],
    stage_name: str,
    review: Optional[Dict[str, Any]],
) -> None:
    if review is None:
        return
    stage_reviews[stage_name] = review
    enforce = review.get("enforce", True)
    passed = review.get("passed")
    if passed is False and enforce:
        if stage_name not in stage_failures:
            stage_failures.append(stage_name)
        return
    if passed is True and stage_name in stage_failures:
        stage_failures.remove(stage_name)
