"""Config-driven environment-backed reward helpers for env-GRPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List


def build_env_reward_function(reward_cfg: Dict[str, Any]) -> Callable[..., List[float]]:
    """Build a TRL reward function from declarative reward rules.

    The code intentionally knows only generic scoring primitives. Field names,
    pass/fail meanings, stop reasons, and weights must come from YAML.
    """

    cfg = reward_cfg or {}
    rules = cfg.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError("rewards.rules must be a list")

    default_score = float(cfg.get("default", cfg.get("base", 0.0)) or 0.0)
    clamp_cfg = cfg.get("clamp") or {}

    def reward_from_env(completions, **kwargs) -> List[float]:
        rewards: List[float] = []
        environments = kwargs.get("environments") or kwargs.get("environment") or []

        for index, completion in enumerate(completions):
            payload = _build_payload(
                index=index,
                completion=completion,
                kwargs=kwargs,
                environments=environments,
            )
            score = default_score
            for rule in rules:
                score += _evaluate_rule(rule, payload)
            rewards.append(float(_clamp(score, clamp_cfg)))

        return rewards

    return reward_from_env


def _build_payload(
    *,
    index: int,
    completion: Any,
    kwargs: Mapping[str, Any],
    environments: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "index": index,
        "completion": completion,
        "kwargs": {},
    }

    for key, value in kwargs.items():
        if key in {"environment", "environments"}:
            continue
        item_value = _item_at(value, index)
        payload[key] = item_value
        payload["kwargs"][key] = item_value

    env = _item_at(environments, index)
    if env is not None:
        payload["environment"] = _environment_to_mapping(env)

    return payload


def _evaluate_rule(rule: Mapping[str, Any], payload: Mapping[str, Any]) -> float:
    rule_type = str(rule.get("type", "add_if"))

    if rule_type == "constant":
        return float(rule.get("score", 0.0) or 0.0)

    if rule_type == "add_if":
        if _condition_passes(rule.get("when"), payload):
            return float(rule.get("score", 0.0) or 0.0)
        return 0.0

    if rule_type == "linear":
        if not _condition_passes(rule.get("when"), payload):
            return 0.0
        value = _coerce_float(_resolve_path(payload, str(rule.get("field", ""))))
        baseline = float(rule.get("baseline", 0.0) or 0.0)
        delta = value - baseline
        if "min_delta" in rule:
            delta = max(delta, float(rule["min_delta"]))
        if "max_delta" in rule:
            delta = min(delta, float(rule["max_delta"]))
        return float(rule.get("weight", 1.0) or 0.0) * delta

    raise ValueError(f"Unsupported env reward rule type: {rule_type}")


def _condition_passes(condition: Any, payload: Mapping[str, Any]) -> bool:
    if not condition:
        return True
    if isinstance(condition, list):
        return all(_condition_passes(item, payload) for item in condition)
    if not isinstance(condition, Mapping):
        raise ValueError("Reward rule condition must be a mapping or list")

    condition_type = str(condition.get("type", "field_truthy"))

    if condition_type == "all":
        return all(_condition_passes(item, payload) for item in condition.get("conditions", []) or [])
    if condition_type == "any":
        return any(_condition_passes(item, payload) for item in condition.get("conditions", []) or [])
    if condition_type == "not":
        return not _condition_passes(condition.get("condition"), payload)

    field = str(condition.get("field", ""))
    value = _resolve_path(payload, field)

    if condition_type == "field_exists":
        return value is not None
    if condition_type == "field_truthy":
        return bool(value)
    if condition_type == "field_falsy":
        return not bool(value)
    if condition_type == "field_equals":
        return value == condition.get("value")
    if condition_type == "field_not_equals":
        return value != condition.get("value")
    if condition_type == "field_in":
        values = condition.get("values", []) or []
        return value in values
    if condition_type == "numeric_compare":
        return _numeric_compare(
            _coerce_float(value),
            str(condition.get("op", "==")),
            float(condition.get("value", 0.0) or 0.0),
        )

    raise ValueError(f"Unsupported env reward condition type: {condition_type}")


def _numeric_compare(left: float, op: str, right: float) -> bool:
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    raise ValueError(f"Unsupported numeric comparison operator: {op}")


def _resolve_path(payload: Mapping[str, Any], path: str) -> Any:
    if not path:
        return None

    current: Any = payload
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _item_at(value: Any, index: int) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if index < len(value):
            return value[index]
        if len(value) == 1:
            return value[0]
        return None
    return value


def _environment_to_mapping(environment: Any) -> Any:
    if isinstance(environment, Mapping):
        return dict(environment)
    to_dict = getattr(environment, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if hasattr(environment, "__dict__"):
        return {
            key: value
            for key, value in vars(environment).items()
            if not key.startswith("_") and _is_jsonish(value)
        }
    return environment


def _is_jsonish(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_jsonish(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_jsonish(item) for key, item in value.items())
    return False


def _coerce_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(score: float, clamp_cfg: Mapping[str, Any]) -> float:
    if not clamp_cfg:
        return score
    if "min" in clamp_cfg:
        score = max(score, float(clamp_cfg["min"]))
    if "max" in clamp_cfg:
        score = min(score, float(clamp_cfg["max"]))
    return score
