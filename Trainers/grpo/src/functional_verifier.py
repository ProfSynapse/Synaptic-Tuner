"""Functional equivalence reward for PivotRL.
Loaded via the ``custom`` reward mechanism in rewards.py.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from shared.verifiers.extraction import extract
from shared.verifiers.builtins.args_match import (
    compare_args_overlap,
    normalize_tool_call,
    normalize_value,
)

logger = logging.getLogger(__name__)


def _extract_tool_call(text: str) -> Optional[Tuple[str, Dict]]:
    """Extract (tool_name, arguments_dict) via the canonical tool-call parser."""
    ea = extract(text, mode="tool_call")
    if ea.found:
        return ea.tool_name, ea.arguments
    return None


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
        norm_pred_name, norm_pred_args = normalize_tool_call(pred_name, pred_args)
        norm_gt_name, norm_gt_args = normalize_tool_call(gt_tool, gt_args)
        if norm_pred_name != norm_gt_name:
            scores.append(0.0)
            continue
        scores.append(compare_args_overlap(norm_pred_args, norm_gt_args))
    return scores
