"""Functional equivalence reward for PivotRL.
Loaded via the ``custom`` reward mechanism in rewards.py.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _normalize_value(value: Any) -> Any:
    """Normalise a scalar/collection for comparison."""
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped.replace("\\", "/")


def _normalize_tool_call(tool_name: str, args: Dict) -> Tuple[str, Dict]:
    """Return normalised (tool_name, sorted_args) tuple."""
    return tool_name.strip().lower(), {k: _normalize_value(v) for k, v in sorted(args.items())}


def _parse_args(raw: Any) -> Dict:
    """Coerce raw arguments to dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _extract_tool_call(text: str) -> Optional[Tuple[str, Dict]]:
    """Extract (tool_name, arguments_dict) from Qwen/Mistral/plain output."""
    # Qwen format
    m = re.search(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text, re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj.get("name", ""), _parse_args(obj.get("arguments", {}))
        except (json.JSONDecodeError, ValueError):
            pass
    # Mistral format
    if "[TOOL_CALLS]" in text:
        m = re.search(r"\[TOOL_CALLS\]\s*(\[[\s\S]*?\])", text)
        if m:
            try:
                calls = json.loads(m.group(1))
                if isinstance(calls, list) and calls:
                    tc = calls[0]
                    return tc.get("name", ""), _parse_args(tc.get("arguments", {}))
            except (json.JSONDecodeError, ValueError):
                pass
    # Plain format
    m = re.search(r"tool_call:\s*(\S+)\s*\narguments:\s*(\{[\s\S]*?\})", text)
    if m:
        try:
            args = json.loads(m.group(2).strip())
            if isinstance(args, dict):
                return m.group(1).strip(), args
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _compare_args(predicted: Dict, expected: Dict) -> float:
    """Compare argument dicts: ``key_overlap * 0.4 + value_match * 0.6``."""
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    pred_keys, exp_keys = set(predicted.keys()), set(expected.keys())
    all_keys = pred_keys | exp_keys
    common = pred_keys & exp_keys
    key_overlap = len(common) / len(all_keys) if all_keys else 0.0
    if not common:
        return key_overlap * 0.4
    value_scores: List[float] = []
    for key in common:
        pv, ev = _normalize_value(predicted[key]), _normalize_value(expected[key])
        if pv == ev:
            value_scores.append(1.0)
        elif isinstance(pv, str) and isinstance(ev, str) and (pv in ev or ev in pv):
            value_scores.append(0.5)
        else:
            value_scores.append(0.0)
    return key_overlap * 0.4 + (sum(value_scores) / len(value_scores)) * 0.6


def functional_equivalence_reward(
    completions: List[str],
    prompts: List[str] | None = None,
    **kwargs: Any,
) -> List[float]:
    """TRL-compatible reward entry point.  Kwargs: ``ground_truth_tool``,
    ``ground_truth_args_json`` (per-item lists expanded by TRL).
    """
    gt_tools = kwargs.get("ground_truth_tool") or []
    gt_args_raw = kwargs.get("ground_truth_args_json") or []
    scores: List[float] = []
    for idx, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else str(completion)
        gt_tool = gt_tools[idx] if idx < len(gt_tools) else None
        gt_json = gt_args_raw[idx] if idx < len(gt_args_raw) else None
        if not gt_tool:
            scores.append(0.0)
            continue
        extracted = _extract_tool_call(text)
        if extracted is None:
            scores.append(0.0)
            continue
        pred_name, pred_args = extracted
        # Parse ground-truth args
        gt_args: Dict = {}
        if gt_json:
            try:
                parsed = json.loads(gt_json) if isinstance(gt_json, str) else gt_json
                gt_args = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                gt_args = {}
        norm_pred_name, norm_pred_args = _normalize_tool_call(pred_name, pred_args)
        norm_gt_name, norm_gt_args = _normalize_tool_call(gt_tool, gt_args)
        if norm_pred_name != norm_gt_name:
            scores.append(0.0)
            continue
        scores.append(_compare_args(norm_pred_args, norm_gt_args))
    return scores
