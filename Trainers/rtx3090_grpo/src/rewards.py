"""
GRPO/GSPO Reward Functions - YAML-Driven

All reward logic is defined in configs/rewards/*.yaml rubrics.
This module provides a thin execution layer that:
1. Loads rubric YAMLs
2. Uses shared/validation for extraction and validation
3. Converts validation results to reward scores

No hardcoded field names or structures - everything from YAML.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

# Import shared validation infrastructure
try:
    from shared.validation.validators import StructureValidator
except ImportError:
    # Fallback for standalone testing
    StructureValidator = None


class RewardRubric:
    """
    A single reward rubric loaded from YAML.

    Provides:
    - Validation rules (loaded from YAML)
    - Scoring strategy (loaded from YAML)
    - Evaluation method that returns a reward score
    """

    def __init__(self, rubric_path: Path, schema_config: Dict = None):
        """
        Load rubric from YAML file.

        Args:
            rubric_path: Path to rubric YAML
            schema_config: Shared schema config (from _schema.yaml)
        """
        self.rubric_path = rubric_path
        self.schema_config = schema_config or {}

        with open(rubric_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        self.name = self.config.get("name", rubric_path.stem)
        self.description = self.config.get("description", "")
        self.scope = self.config.get("scope", "response")
        self.validation = self.config.get("validation", [])
        self.scoring = self.config.get("scoring", {})
        self.ground_truth = self.config.get("ground_truth", {})

        # Initialize structure validator if available
        self.validator = StructureValidator() if StructureValidator else None

    def evaluate(self, completion_text: str, **kwargs) -> float:
        """
        Evaluate completion against this rubric.

        Args:
            completion_text: Raw model output text
            **kwargs: Additional context (ground_truth_tool, ground_truth_args_json, etc.)

        Returns:
            Reward score (0.0 - 1.0)
        """
        # Extract data from completion
        data = self._extract_data(completion_text)

        # Get scoring strategy
        strategy = self.scoring.get("strategy", "binary")

        if strategy == "binary":
            return self._score_binary(data, completion_text)
        elif strategy == "proportional":
            return self._score_proportional(data)
        elif strategy == "tiered":
            return self._score_tiered(data, kwargs)
        elif strategy == "weighted":
            return self._score_weighted(data, kwargs)
        else:
            return self._score_binary(data, completion_text)

    def _extract_data(self, text: str) -> Dict:
        """Extract structured data from completion text."""
        # Find arguments JSON in tool call
        args = self._parse_arguments(text)
        if not args:
            return {}

        return {
            "context": args.get("context", {}),
            "calls": args.get("calls", []),
            "raw_text": text,
        }

    def _parse_arguments(self, text: str) -> Optional[Dict]:
        """Parse arguments from tool call output."""
        # Try escaped JSON string
        match = re.search(r'"arguments"\s*:\s*"(\{.*?\})"', text, re.DOTALL)
        if match:
            try:
                args_str = match.group(1)
                args_str = args_str.replace('\\"', '"').replace('\\n', '\n')
                return json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try unescaped JSON
        match = re.search(r'"arguments"\s*:\s*(\{)', text)
        if match:
            start = match.start(1)
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break

        return None

    def _score_binary(self, data: Dict, raw_text: str) -> float:
        """Binary scoring: 1.0 if all validations pass, 0.0 otherwise."""
        if not self.validation:
            # No validation rules - check if we have expected structure
            if self.scope == "context":
                return 1.0 if data.get("context") else 0.0
            elif self.scope == "calls":
                return 1.0 if data.get("calls") else 0.0
            elif self.scope == "arguments":
                return 1.0 if (data.get("context") and data.get("calls")) else 0.0
            else:
                # Format check - look for useTools pattern
                if '"name"' in raw_text and '"arguments"' in raw_text:
                    return self.scoring.get("on_pass", 1.0)
                return self.scoring.get("on_fail", 0.0)

        # Run validations
        if self.validator:
            is_valid, errors = self.validator.validate(
                data, self.validation, raw_text
            )
            if is_valid:
                return self.scoring.get("on_pass", 1.0)
            return self.scoring.get("on_fail", 0.0)

        return 0.0

    def _score_proportional(self, data: Dict) -> float:
        """Proportional scoring: score = fields_valid / fields_total."""
        context = data.get("context", {})
        if not context:
            return 0.0

        # Get required fields from schema config
        required_fields = self._get_required_fields()
        if not required_fields:
            return 1.0 if context else 0.0

        # Count present fields
        present = sum(1 for f in required_fields if context.get(f))
        score = present / len(required_fields)

        # Optional bonus
        optional_bonus = self.scoring.get("optional_bonus", 0.0)
        optional_fields = self._get_optional_fields()
        if optional_fields:
            optional_present = sum(1 for f in optional_fields if context.get(f))
            if optional_present > 0:
                score = min(1.0, score + optional_bonus)

        return score * self.scoring.get("max_score", 1.0)

    def _score_tiered(self, data: Dict, kwargs: Dict) -> float:
        """Tiered scoring based on match conditions."""
        # Get ground truth
        gt_field = self.ground_truth.get("field", "ground_truth_tool")
        ground_truth = kwargs.get(gt_field)

        if not ground_truth:
            return 0.0

        # Extract predicted tool
        calls = data.get("calls", [])
        if not calls:
            return 0.0

        first_call = calls[0]
        agent = first_call.get("agent", "")
        tool = first_call.get("tool", "")
        predicted = f"{agent}_{tool}" if agent and tool else ""

        if not predicted:
            return 0.0

        # Check tiers
        tiers = self.scoring.get("tiers", [])
        for tier in tiers:
            condition = tier.get("condition", "")
            score = tier.get("score", 0.0)

            if condition == "exact_match" and predicted == ground_truth:
                return score
            elif condition == "agent_match":
                pred_agent = predicted.split("_")[0] if "_" in predicted else ""
                gt_agent = ground_truth.split("_")[0] if "_" in ground_truth else ""
                if pred_agent and pred_agent == gt_agent:
                    return score
            elif condition == "no_match":
                return score

        return 0.0

    def _score_weighted(self, data: Dict, kwargs: Dict) -> float:
        """
        Weighted scoring: compare predicted args against ground truth.

        Compares context fields, tool selection, and params based on
        weights defined in the rubric's scoring.weights config.
        """
        # Get ground truth config
        gt_field = self.ground_truth.get("field", "ground_truth_args_json")
        gt_parse = self.ground_truth.get("parse", "json")

        gt_raw = kwargs.get(gt_field)
        if not gt_raw:
            return 0.0

        # Parse ground truth
        try:
            if gt_parse == "json" and isinstance(gt_raw, str):
                gt_args = json.loads(gt_raw)
            else:
                gt_args = gt_raw
        except (json.JSONDecodeError, ValueError):
            return 0.0

        # Get comparison config
        comparison = self.config.get("comparison", {})
        context_fields = comparison.get("context_fields", ["workspaceId", "sessionId", "memory", "goal"])
        call_fields = comparison.get("call_fields", ["agent", "tool", "params"])

        # Get weights
        weights = self.scoring.get("weights", {})
        context_weight = weights.get("context_match", 0.4)
        tool_weight = weights.get("tool_match", 0.3)
        params_weight = weights.get("params_match", 0.3)

        total_score = 0.0

        # --- 1. Context matching ---
        pred_context = data.get("context", {})
        gt_context = gt_args.get("context", {})

        if context_fields and gt_context:
            context_matches = 0
            for field in context_fields:
                pred_val = pred_context.get(field)
                gt_val = gt_context.get(field)
                if pred_val is not None and gt_val is not None:
                    # For strings, check equality; for dicts, check key overlap
                    if isinstance(gt_val, dict) and isinstance(pred_val, dict):
                        # Key overlap scoring for nested objects
                        if gt_val:
                            overlap = len(set(pred_val.keys()) & set(gt_val.keys()))
                            context_matches += overlap / len(gt_val)
                    elif str(pred_val) == str(gt_val):
                        context_matches += 1

            context_score = context_matches / len(context_fields) if context_fields else 0.0
            total_score += context_weight * context_score

        # --- 2. Tool matching (agent + tool) ---
        pred_calls = data.get("calls", [])
        gt_calls = gt_args.get("calls", [])

        if pred_calls and gt_calls:
            pred_call = pred_calls[0]
            gt_call = gt_calls[0]

            pred_agent = pred_call.get("agent", "")
            pred_tool = pred_call.get("tool", "")
            gt_agent = gt_call.get("agent", "")
            gt_tool = gt_call.get("tool", "")

            # Exact match on agent + tool
            if pred_agent == gt_agent and pred_tool == gt_tool:
                total_score += tool_weight * 1.0
            elif pred_agent == gt_agent:
                # Partial credit for correct agent
                total_score += tool_weight * 0.3

        # --- 3. Params matching ---
        if pred_calls and gt_calls:
            pred_params = pred_calls[0].get("params", {})
            gt_params = gt_calls[0].get("params", {})

            params_strategy = self.scoring.get("params_strategy", "key_overlap")

            if params_strategy == "key_overlap" and gt_params:
                # Score based on key overlap
                gt_keys = set(gt_params.keys())
                pred_keys = set(pred_params.keys())

                if gt_keys:
                    overlap = len(gt_keys & pred_keys)
                    params_score = overlap / len(gt_keys)

                    # Bonus for value matches
                    value_matches = 0
                    for key in gt_keys & pred_keys:
                        if str(pred_params.get(key)) == str(gt_params.get(key)):
                            value_matches += 1

                    if overlap > 0:
                        value_bonus = (value_matches / overlap) * 0.5
                        params_score = min(1.0, params_score + value_bonus)

                    total_score += params_weight * params_score

            elif params_strategy == "exact" and gt_params:
                # Exact match required
                if pred_params == gt_params:
                    total_score += params_weight * 1.0

        return total_score

    def _get_required_fields(self) -> List[str]:
        """Get required context fields from schema config."""
        # Try to get from loaded tool schema
        tool_schema_path = self.schema_config.get("tool_schema_path")
        if tool_schema_path:
            schema_path = self.rubric_path.parent / tool_schema_path
            if schema_path.exists():
                try:
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema = yaml.safe_load(f) or {}
                    context_cfg = (
                        schema.get("tool_format", {})
                        .get("wrapper_structure", {})
                        .get("context", {})
                    )
                    return context_cfg.get("required", [])
                except Exception:
                    pass

        # Fallback to schema config
        return self.schema_config.get("context_required", [])

    def _get_optional_fields(self) -> List[str]:
        """Get optional context fields from schema config."""
        tool_schema_path = self.schema_config.get("tool_schema_path")
        if tool_schema_path:
            schema_path = self.rubric_path.parent / tool_schema_path
            if schema_path.exists():
                try:
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema = yaml.safe_load(f) or {}
                    context_cfg = (
                        schema.get("tool_format", {})
                        .get("wrapper_structure", {})
                        .get("context", {})
                    )
                    return context_cfg.get("optional", [])
                except Exception:
                    pass

        return self.schema_config.get("context_optional", [])


class RewardEngine:
    """
    Loads all reward rubrics and builds combined reward function.
    """

    def __init__(self, rewards_dir: Path):
        """
        Initialize reward engine.

        Args:
            rewards_dir: Directory containing reward YAML files
        """
        self.rewards_dir = Path(rewards_dir)
        self.rubrics: Dict[str, RewardRubric] = {}
        self.schema_config: Dict = {}

        self._load_schema_config()
        self._load_rubrics()

    def _load_schema_config(self) -> None:
        """Load shared schema config from _schema.yaml."""
        schema_file = self.rewards_dir / "_schema.yaml"
        if schema_file.exists():
            with open(schema_file, "r", encoding="utf-8") as f:
                self.schema_config = yaml.safe_load(f) or {}

    def _load_rubrics(self) -> None:
        """Load all rubric YAML files."""
        for yaml_file in self.rewards_dir.glob("*.yaml"):
            if yaml_file.name.startswith("_"):
                continue  # Skip schema/config files

            try:
                rubric = RewardRubric(yaml_file, self.schema_config)
                self.rubrics[rubric.name] = rubric
            except Exception as e:
                print(f"Warning: Failed to load rubric {yaml_file}: {e}")

    def get_rubric(self, name: str) -> Optional[RewardRubric]:
        """Get a specific rubric by name."""
        return self.rubrics.get(name)

    def list_rubrics(self) -> List[str]:
        """List all available rubric names."""
        return list(self.rubrics.keys())


def _coerce_to_text(completion: Any) -> str:
    """Convert completion to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first.get("content") or "")
    if isinstance(completion, dict) and "content" in completion:
        return str(completion.get("content") or "")
    return str(completion)


def _expand_to_match_completions(values: Any, completions_len: int) -> List[Any]:
    """Expand ground truth values to match completions length."""
    if completions_len <= 0:
        return []

    if isinstance(values, list):
        if len(values) == completions_len:
            return values
        if len(values) == 1:
            return values * completions_len
        if len(values) > 0 and completions_len % len(values) == 0:
            factor = completions_len // len(values)
            expanded: List[Any] = []
            for v in values:
                expanded.extend([v] * factor)
            return expanded
        padded = (values + [values[-1]] * completions_len)[:completions_len]
        return padded

    return [values] * completions_len


def _coerce_rewards(result: Any, expected_len: int) -> List[float]:
    """Coerce reward result to list of floats."""
    if expected_len <= 0:
        return []
    if isinstance(result, list):
        if len(result) == expected_len:
            return [float(x) for x in result]
        if len(result) == 1:
            return [float(result[0])] * expected_len
        raise ValueError(f"Reward returned {len(result)} values, expected {expected_len}")
    try:
        return [float(x) for x in list(result)]
    except Exception:
        return [float(result)] * expected_len


def build_rubric_reward(rubric: RewardRubric) -> Callable:
    """Build a reward function from a rubric."""

    def _reward(completions, prompts=None, **kwargs):
        rewards = []
        n_completions = len(completions)

        # TRL batches kwargs - expand them to match completions
        expanded_kwargs = {}
        for key, value in kwargs.items():
            expanded_kwargs[key] = _expand_to_match_completions(value, n_completions)

        for i, completion in enumerate(completions):
            text = _coerce_to_text(completion)
            # Get kwargs for this specific completion
            item_kwargs = {k: v[i] for k, v in expanded_kwargs.items()}
            score = rubric.evaluate(text, **item_kwargs)
            rewards.append(score)
        return rewards

    return _reward


def build_combined_reward_function(
    rewards_config: Dict[str, Any],
    base_dir: Path,
) -> Tuple[Callable, List[Dict]]:
    """
    Build combined reward function from config.

    Args:
        rewards_config: Rewards section from config.yaml
        base_dir: Base directory for relative paths

    Returns:
        (combined_reward_fn, plan_items)
    """
    rewards_dir = base_dir / "configs" / "rewards"

    # Load reward engine
    engine = RewardEngine(rewards_dir) if rewards_dir.exists() else None

    items = rewards_config.get("items", []) or []
    plan: List[Dict] = []
    components: List[Tuple[str, float, Callable]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        if not name:
            continue

        weight = float(item.get("weight", 1.0))
        params = item.get("params", {}) or {}

        # Try to load from rubric YAML first
        if engine:
            rubric = engine.get_rubric(name)
            if rubric:
                func = build_rubric_reward(rubric)
                components.append((name, weight, func))
                plan.append({
                    "type": "rubric",
                    "name": name,
                    "weight": weight,
                    "source": str(rubric.rubric_path),
                })
                continue

        # Rubric not found
        print(f"Warning: Reward rubric '{name}' not found in {rewards_dir}")

    # Handle custom rewards (from module/file)
    custom = rewards_config.get("custom", {}) or {}
    if bool(custom.get("enabled")):
        module_name = custom.get("module")
        file_path = custom.get("file")

        if file_path:
            path = Path(file_path)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            if path.exists():
                spec = importlib.util.spec_from_file_location("custom_rewards", str(path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for fn_cfg in custom.get("functions", []) or []:
                        fn_name = fn_cfg.get("name")
                        if not fn_name:
                            continue
                        weight = float(fn_cfg.get("weight", 1.0))
                        func = getattr(module, fn_name, None)
                        if func and callable(func):
                            components.append((f"custom:{fn_name}", weight, func))
                            plan.append({
                                "type": "custom",
                                "name": fn_name,
                                "weight": weight,
                            })

    if not components:
        raise ValueError(
            f"No reward components loaded. "
            f"Check that rubric YAMLs exist in {rewards_dir}"
        )

    def combined_reward(completions, prompts=None, **kwargs):
        total = [0.0] * len(completions)

        for name, weight, func in components:
            if weight == 0:
                continue
            rewards = func(completions, prompts=prompts, **kwargs)
            rewards_list = _coerce_rewards(rewards, len(completions))
            for i, r in enumerate(rewards_list):
                total[i] += float(weight) * float(r)

        return total

    return combined_reward, plan
