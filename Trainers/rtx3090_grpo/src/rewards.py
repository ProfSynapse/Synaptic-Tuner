"""
Reward functions and config-driven reward composition for GRPO/GSPO.

Built-in rewards are selected in YAML (`rewards.items`) and combined into a
single reward function passed to TRL's GRPOTrainer.

Custom reward functions can be loaded via:
  - import path: rewards.custom.module
  - file path:  rewards.custom.file

TODO: Add an argument-matching reward (compare predicted tool args vs ground_truth_args).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


CONTEXT_FIELDS_DEFAULT = [
    "sessionId",
    "workspaceId",
    "sessionDescription",
    "sessionMemory",
    "toolContext",
    "primaryGoal",
    "subgoal",
]


def _coerce_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first.get("content") or "")
    if isinstance(completion, dict) and "content" in completion:
        return str(completion.get("content") or "")
    return str(completion)


def _extract_balanced_json_object(text: str, start: int) -> Optional[str]:
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_tool_call(text: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    Extract tool name and arguments from model output.

    Handles multiple formats:
      - tool_call: toolName\\narguments: {...}
      - <tool_call>{...}</tool_call> (JSON inside tags)
      - JSON snippets containing {"name": "...", "arguments": {...}}
      - Fallback: any tool name pattern like vaultManager_xxx / vaultLibrarian_xxx
    """
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None

    # Pattern A: <tool_call> JSON </tool_call>
    tag_match = re.search(r"<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", text, re.DOTALL)
    if tag_match:
        try:
            obj = json.loads(tag_match.group(1))
            tool_name = obj.get("name") or obj.get("tool") or obj.get("function")
            args = obj.get("arguments") or obj.get("args") or {}
            if isinstance(args, str):
                args = json.loads(args) if args.strip().startswith("{") else {}
            tool_args = args if isinstance(args, dict) else {}
            return tool_name, tool_args
        except Exception:
            pass

    # Pattern B: tool_call: name + arguments: { ... }
    tc_match = re.search(r"tool_call:\\s*(\\w+)", text)
    if tc_match:
        tool_name = tc_match.group(1)
        args_idx = text.find("arguments:", tc_match.end())
        if args_idx != -1:
            brace_idx = text.find("{", args_idx)
            if brace_idx != -1:
                obj_str = _extract_balanced_json_object(text, brace_idx)
                if obj_str:
                    try:
                        tool_args = json.loads(obj_str)
                    except Exception:
                        tool_args = {}
        return tool_name, tool_args or {}

    # Pattern C: JSON fragment with "name" and "arguments"
    fn_match = re.search(r"\"name\"\\s*:\\s*\"([^\"]+)\"", text)
    if fn_match:
        tool_name = fn_match.group(1)
        args_match = re.search(r"\"arguments\"\\s*:\\s*(\"\\{.*?\\}\"|\\{.*?\\})", text, re.DOTALL)
        if args_match:
            raw = args_match.group(1)
            try:
                if raw.startswith('"'):
                    raw = json.loads(raw)
                if isinstance(raw, str) and raw.strip().startswith("{"):
                    tool_args = json.loads(raw)
                elif isinstance(raw, dict):
                    tool_args = raw
                else:
                    tool_args = {}
            except Exception:
                tool_args = {}
        return tool_name, tool_args or {}

    # Fallback: find any tool-like name, attempt to parse a JSON object containing "context"
    name_match = re.search(r"(\\w+Manager_\\w+|\\w+Librarian_\\w+)", text)
    if name_match:
        tool_name = name_match.group(1)
        brace_idx = text.find("{")
        if brace_idx != -1:
            obj_str = _extract_balanced_json_object(text, brace_idx)
            if obj_str:
                try:
                    tool_args = json.loads(obj_str)
                except Exception:
                    tool_args = {}
        return tool_name, tool_args or {}

    return None, None


def _expand_to_match_completions(values: Any, completions_len: int) -> List[Any]:
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
        # Fall back: pad/truncate
        padded = (values + [values[-1]] * completions_len)[:completions_len]
        return padded

    return [values] * completions_len


def _coerce_rewards(result: Any, expected_len: int) -> List[float]:
    if expected_len <= 0:
        return []
    if isinstance(result, list):
        if len(result) == expected_len:
            return [float(x) for x in result]
        if len(result) == 1:
            return [float(result[0])] * expected_len
        raise ValueError(f"Reward function returned {len(result)} rewards, expected {expected_len}.")
    try:
        # numpy / torch
        return [float(x) for x in list(result)]
    except Exception:
        return [float(result)] * expected_len


def build_tool_selection_reward(
    ground_truth_field: str = "ground_truth_tool",
    partial_credit: float = 0.3,
) -> Callable:
    def _reward(completions, prompts=None, **kwargs):
        ground_truth = kwargs.get(ground_truth_field)
        gt_list = _expand_to_match_completions(ground_truth, len(completions))

        rewards: List[float] = []
        for completion, gt in zip(completions, gt_list):
            text = _coerce_to_text(completion)
            pred_tool, _ = extract_tool_call(text)

            if pred_tool and gt and pred_tool == gt:
                rewards.append(1.0)
                continue

            if pred_tool and gt:
                pred_agent = pred_tool.split("_")[0] if "_" in pred_tool else ""
                true_agent = gt.split("_")[0] if "_" in gt else ""
                rewards.append(float(partial_credit) if pred_agent and pred_agent == true_agent else 0.0)
                continue

            rewards.append(0.0)

        return rewards

    return _reward


def build_json_structure_reward(reward_value: float = 0.3) -> Callable:
    def _reward(completions, prompts=None, **kwargs):
        rewards: List[float] = []
        for completion in completions:
            text = _coerce_to_text(completion)
            _, args = extract_tool_call(text)
            ok = isinstance(args, dict) and len(args) > 0
            rewards.append(float(reward_value) if ok else 0.0)
        return rewards

    return _reward


def build_context_completeness_reward(
    reward_value: float = 0.5,
    required_fields: Optional[List[str]] = None,
) -> Callable:
    required = required_fields or CONTEXT_FIELDS_DEFAULT
    required = [str(f) for f in required]

    def _reward(completions, prompts=None, **kwargs):
        rewards: List[float] = []
        for completion in completions:
            text = _coerce_to_text(completion)
            _, args = extract_tool_call(text)
            if not isinstance(args, dict):
                rewards.append(0.0)
                continue

            ctx = args.get("context")
            if not isinstance(ctx, dict):
                rewards.append(0.0)
                continue

            present = sum(1 for field in required if ctx.get(field))
            frac = present / len(required) if required else 0.0
            rewards.append(float(reward_value) * frac)
        return rewards

    return _reward


def build_format_reward(reward_value: float = 0.2) -> Callable:
    def _reward(completions, prompts=None, **kwargs):
        rewards: List[float] = []
        for completion in completions:
            text = _coerce_to_text(completion).lower()
            has_tool_call = (
                "tool_call" in text
                or "arguments" in text
                or "<tool_call>" in text
                or re.search(r"\\w+manager_\\w+|\\w+librarian_\\w+", text) is not None
            )
            rewards.append(float(reward_value) if has_tool_call else 0.0)
        return rewards

    return _reward


BUILTIN_REWARD_BUILDERS: Dict[str, Callable[..., Callable]] = {
    "tool_selection": build_tool_selection_reward,
    "json_structure": build_json_structure_reward,
    "context_completeness": build_context_completeness_reward,
    "format": build_format_reward,
}


def _load_custom_module(
    module_name: Optional[str],
    file_path: Optional[str],
    base_dir: Path,
) -> ModuleType:
    if file_path:
        path = Path(file_path)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Custom reward file not found: {path}")

        spec = importlib.util.spec_from_file_location("custom_rewards", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to import custom rewards from: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    if module_name:
        return importlib.import_module(module_name)

    raise ValueError("Custom rewards enabled but neither module nor file is set.")


def build_combined_reward_function(
    rewards_config: Dict[str, Any],
    base_dir: Path,
) -> tuple[Callable, list[dict]]:
    """
    Build a single combined reward function based on YAML config.

    Returns:
        (combined_reward_fn, plan_items)
        plan_items: list of dicts describing enabled reward components.
    """
    rewards_config = rewards_config or {}
    items = rewards_config.get("items", []) or []
    custom = rewards_config.get("custom", {}) or {}

    plan: list[dict] = []
    components: list[tuple[str, float, Callable]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        weight = float(item.get("weight", 1.0))
        params = item.get("params", {}) or {}

        builder = BUILTIN_REWARD_BUILDERS.get(name)
        if builder is None:
            raise ValueError(f"Unknown built-in reward: '{name}'. Available: {sorted(BUILTIN_REWARD_BUILDERS.keys())}")

        func = builder(**params)
        components.append((name, weight, func))
        plan.append({"type": "builtin", "name": name, "weight": weight, "params": params})

    if bool(custom.get("enabled")):
        module_name = custom.get("module")
        file_path = custom.get("file")
        module = _load_custom_module(module_name, file_path, base_dir=base_dir)

        for fn_cfg in custom.get("functions", []) or []:
            if not isinstance(fn_cfg, dict):
                continue
            fn_name = fn_cfg.get("name")
            if not fn_name:
                continue
            weight = float(fn_cfg.get("weight", 1.0))
            params = fn_cfg.get("params", {}) or {}

            func = getattr(module, fn_name, None)
            if func is None or not callable(func):
                raise AttributeError(f"Custom reward function '{fn_name}' not found/callable in custom rewards module.")

            def _wrap_custom(f, static_params: Dict[str, Any]):
                def _wrapped(completions, prompts=None, **kwargs):
                    merged = {**static_params, **kwargs}
                    return f(completions=completions, prompts=prompts, **merged)

                return _wrapped

            wrapped = _wrap_custom(func, params)
            components.append((f"custom:{fn_name}", weight, wrapped))
            plan.append({"type": "custom", "name": fn_name, "weight": weight, "params": params})

    if not components:
        raise ValueError("No reward components configured. Set rewards.items and/or rewards.custom.")

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
