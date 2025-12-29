"""
GRPO/GSPO Reward Functions - YAML-Driven

All reward logic is defined in configs/rewards/*.yaml rubrics.
This module provides a thin execution layer that:
1. Loads rubric YAMLs
2. Extracts tool call data directly from model output (Qwen/Mistral format)
3. Compares against ground truth using configurable field mappings

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
        """
        Extract structured data from completion text.

        Directly extracts from model's native format (e.g., Qwen's <tool_call> tags).
        No unnecessary format conversion - just get the arguments and compare.
        """
        result = {"raw_text": text}

        # Try Qwen format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        tool_call_match = re.search(
            r'<tool_call>\s*([\s\S]*?)\s*</tool_call>',
            text,
            re.IGNORECASE
        )

        if tool_call_match:
            json_content = tool_call_match.group(1).strip()
            try:
                tool_obj = json.loads(json_content)
                if isinstance(tool_obj, dict):
                    result["tool_name"] = tool_obj.get("name", "")
                    args = tool_obj.get("arguments", {})
                    # Arguments might be JSON string or dict
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    result["parsed_args"] = args if isinstance(args, dict) else {}
                    return result
            except json.JSONDecodeError:
                pass

        # Fallback: Try Mistral format [TOOL_CALLS] [...]
        if "[TOOL_CALLS]" in text:
            match = re.search(r'\[TOOL_CALLS\]\s*(\[[\s\S]*?\])', text)
            if match:
                try:
                    tool_calls = json.loads(match.group(1))
                    if isinstance(tool_calls, list) and tool_calls:
                        tc = tool_calls[0]
                        result["tool_name"] = tc.get("name", "")
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        result["parsed_args"] = args if isinstance(args, dict) else {}
                        return result
                except json.JSONDecodeError:
                    pass

        return result

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

        Compares fields based on mappings defined in rubric's comparison config.
        All paths and fields come from YAML - no hardcoding.
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

        # Get comparison config from rubric
        comparison = self.config.get("comparison", {})
        mappings = comparison.get("mappings", [])

        # If no explicit mappings, use legacy field-based comparison
        if not mappings:
            return self._score_weighted_legacy(data, gt_args, comparison, kwargs)

        # Score based on explicit field mappings
        total_score = 0.0
        total_weight = 0.0
        pred_args = data.get("parsed_args", {})

        for mapping in mappings:
            pred_path = mapping.get("pred_path", "")
            gt_path = mapping.get("gt_path", "")
            weight = float(mapping.get("weight", 1.0))
            strategy = mapping.get("strategy", "equals")

            # Special case: tool_name comparison uses different fields
            if mapping.get("use_tool_name"):
                pred_val = data.get("tool_name", "")
                gt_val = kwargs.get("ground_truth_tool", "")
            elif pred_path == "" and strategy == "key_overlap":
                # Empty pred_path with key_overlap means compare all parsed args
                pred_val = pred_args
                gt_val = self._get_nested_value(gt_args, gt_path)
            else:
                # Extract values using paths
                pred_val = self._get_nested_value(pred_args, pred_path)
                gt_val = self._get_nested_value(gt_args, gt_path)

            # Compare based on strategy
            field_score = self._compare_values(pred_val, gt_val, strategy)
            total_score += weight * field_score
            total_weight += weight

        # Normalize score
        return total_score / total_weight if total_weight > 0 else 0.0

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get value from nested dict using dot notation path."""
        if not path or not obj:
            return None

        parts = path.split(".")
        current = obj

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if idx < len(current) else None
            else:
                return None

            if current is None:
                return None

        return current

    def _compare_values(self, pred: Any, gt: Any, strategy: str) -> float:
        """Compare two values using the specified strategy."""
        if pred is None or gt is None:
            return 0.0

        if strategy == "equals":
            return 1.0 if str(pred) == str(gt) else 0.0

        elif strategy == "contains":
            return 1.0 if str(gt) in str(pred) else 0.0

        elif strategy == "key_overlap":
            if isinstance(pred, dict) and isinstance(gt, dict) and gt:
                overlap = len(set(pred.keys()) & set(gt.keys()))
                return overlap / len(gt)
            return 0.0

        elif strategy == "tool_name_match":
            # Compare tool names (handle agent_tool format)
            pred_str = str(pred)
            gt_str = str(gt)
            if pred_str == gt_str:
                return 1.0
            # Partial: same agent
            if "_" in pred_str and "_" in gt_str:
                if pred_str.split("_")[0] == gt_str.split("_")[0]:
                    return 0.3
            return 0.0

        return 0.0

    def _score_weighted_legacy(self, data: Dict, gt_args: Dict, comparison: Dict, kwargs: Dict) -> float:
        """Legacy weighted scoring using field lists (backwards compatible)."""
        context_fields = comparison.get("context_fields", [])
        call_fields = comparison.get("call_fields", [])

        weights = self.scoring.get("weights", {})
        context_weight = weights.get("context_match", 0.4)
        tool_weight = weights.get("tool_match", 0.3)
        params_weight = weights.get("params_match", 0.3)

        total_score = 0.0
        pred_args = data.get("parsed_args", {})

        # Context field matching
        if context_fields:
            gt_context = gt_args.get("context", {}) if isinstance(gt_args, dict) else {}
            if isinstance(gt_context, dict) and gt_context:
                matches = sum(
                    1 for f in context_fields
                    if str(pred_args.get(f, "")) == str(gt_context.get(f, ""))
                )
                total_score += context_weight * (matches / len(context_fields))

        # Tool name matching
        pred_tool = data.get("tool_name", "")
        gt_tool = kwargs.get("ground_truth_tool", "")
        if pred_tool and gt_tool:
            if pred_tool == gt_tool:
                total_score += tool_weight * 1.0
            elif "_" in pred_tool and "_" in gt_tool:
                if pred_tool.split("_")[0] == gt_tool.split("_")[0]:
                    total_score += tool_weight * 0.3

        # Params matching (key overlap)
        gt_calls = gt_args.get("calls", []) if isinstance(gt_args, dict) else []
        if gt_calls and isinstance(gt_calls, list) and gt_calls:
            gt_params = gt_calls[0].get("params", {}) if isinstance(gt_calls[0], dict) else {}
            if isinstance(gt_params, dict) and gt_params:
                gt_keys = set(gt_params.keys())
                pred_keys = set(k for k in pred_args.keys() if k not in ["sessionId", "workspaceId"])
                if gt_keys:
                    overlap = len(gt_keys & pred_keys)
                    total_score += params_weight * (overlap / len(gt_keys))

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
