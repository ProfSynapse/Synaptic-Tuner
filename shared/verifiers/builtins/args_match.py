"""Argument/value comparison verifier and the canonical comparison toolkit.

This module is the SINGLE HOME for the tool-call argument comparison logic that
previously lived in three copies across the GRPO trainer
(``rewards.RewardRubric._compare_values`` / ``_get_nested_value`` /
``_score_weighted`` mapping loop / ``_score_weighted_legacy`` and
``functional_verifier._normalize_value`` / ``_normalize_tool_call`` /
``_compare_args``).

The module exposes plain functions (the building blocks the GRPO reward paths
delegate to) plus an ``args_match`` :class:`Verifier` that wires those building
blocks into the shared verifier contract. Everything is config-driven; no tool
formats or field names are hardcoded.

``shared.verifiers`` MUST NOT import ``Trainers/`` or ``Evaluator/`` — this
module keeps the comparison logic dependency-free so the GRPO code can depend on
it (and not the reverse).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Tuple

from ..contract import VerifierInput, VerifierOutput
from ..extraction import extract
from ..registry import register


# ---------------------------------------------------------------------------
# Normalization (moved verbatim from functional_verifier)
# ---------------------------------------------------------------------------

def normalize_value(value: Any) -> Any:
    """Normalise a scalar/collection for comparison."""
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
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


def normalize_tool_call(tool_name: str, args: Dict) -> Tuple[str, Dict]:
    """Return normalised (tool_name, sorted_args) tuple."""
    return tool_name.strip().lower(), {k: normalize_value(v) for k, v in sorted(args.items())}


# ---------------------------------------------------------------------------
# Nested access + per-value comparison (moved verbatim from RewardRubric)
# ---------------------------------------------------------------------------

def get_nested_value(obj: Any, path: str) -> Any:
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


def compare_values(pred: Any, gt: Any, strategy: str) -> float:
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


# ---------------------------------------------------------------------------
# Overlap scheme (moved verbatim from functional_verifier._compare_args)
# ---------------------------------------------------------------------------

def compare_args_overlap(predicted: Dict, expected: Dict) -> float:
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
        pv, ev = normalize_value(predicted[key]), normalize_value(expected[key])
        if pv == ev:
            value_scores.append(1.0)
        elif isinstance(pv, str) and isinstance(ev, str) and (pv in ev or ev in pv):
            value_scores.append(0.5)
        else:
            value_scores.append(0.0)
    return key_overlap * 0.4 + (sum(value_scores) / len(value_scores)) * 0.6


# ---------------------------------------------------------------------------
# Mapping-based weighted scheme (the loop body of RewardRubric._score_weighted,
# after ground-truth parsing)
# ---------------------------------------------------------------------------

def score_mappings(
    pred_args: Dict,
    pred_tool: str,
    gt_args: Any,
    gt_tool: str,
    mappings: List[Mapping[str, Any]],
) -> float:
    """Score predicted args against ground truth using explicit field mappings.

    This is the mapping loop from ``RewardRubric._score_weighted`` (the part
    AFTER ground-truth parsing), verbatim. Returns the normalized
    ``total_score / total_weight`` (``0.0`` if no positive weight).

    Args:
        pred_args: Predicted parsed arguments.
        pred_tool: Predicted tool name (used by ``use_tool_name`` mappings).
        gt_args: Parsed ground-truth args object (dict-like for nested lookups).
        gt_tool: Ground-truth tool name (used by ``use_tool_name`` mappings).
        mappings: List of mapping configs (``pred_path``/``gt_path``/``weight``/
            ``strategy``/``use_tool_name``).
    """
    total_score = 0.0
    total_weight = 0.0

    for mapping in mappings:
        pred_path = mapping.get("pred_path", "")
        gt_path = mapping.get("gt_path", "")
        weight = float(mapping.get("weight", 1.0))
        strategy = mapping.get("strategy", "equals")

        # Special case: tool_name comparison uses different fields
        if mapping.get("use_tool_name"):
            pred_val = pred_tool
            gt_val = gt_tool
        elif pred_path == "" and strategy == "key_overlap":
            # Empty pred_path with key_overlap means compare all parsed args
            pred_val = pred_args
            gt_val = get_nested_value(gt_args, gt_path)
        else:
            # Extract values using paths
            pred_val = get_nested_value(pred_args, pred_path)
            gt_val = get_nested_value(gt_args, gt_path)

        # Compare based on strategy
        field_score = compare_values(pred_val, gt_val, strategy)
        total_score += weight * field_score
        total_weight += weight

    # Normalize score
    return total_score / total_weight if total_weight > 0 else 0.0


# ---------------------------------------------------------------------------
# Legacy field-list weighted scheme (moved verbatim from
# RewardRubric._score_weighted_legacy)
# ---------------------------------------------------------------------------

def score_legacy_fields(
    pred_args: Dict,
    pred_tool: str,
    gt_args: Any,
    gt_tool: str,
    comparison: Mapping[str, Any],
    weights: Mapping[str, Any],
) -> float:
    """Legacy weighted scoring using field lists (backwards compatible).

    Verbatim port of ``RewardRubric._score_weighted_legacy``.

    Args:
        pred_args: Predicted parsed arguments.
        pred_tool: Predicted tool name.
        gt_args: Parsed ground-truth args (expects ``context`` / ``calls``).
        gt_tool: Ground-truth tool name.
        comparison: Rubric ``comparison`` config (``context_fields`` /
            ``call_fields``).
        weights: Scoring ``weights`` config (``context_match`` / ``tool_match`` /
            ``params_match``).
    """
    context_fields = comparison.get("context_fields", [])
    call_fields = comparison.get("call_fields", [])

    context_weight = weights.get("context_match", 0.4)
    tool_weight = weights.get("tool_match", 0.3)
    params_weight = weights.get("params_match", 0.3)

    total_score = 0.0

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


# ---------------------------------------------------------------------------
# Verifier wiring
# ---------------------------------------------------------------------------

class ArgsMatchVerifier:
    """Verifier that scores predicted tool-call args against ground truth.

    The predicted tool/args are extracted from ``sample.completion_text`` via
    :func:`shared.verifiers.extraction.extract` (``mode="tool_call"``). The
    expected tool/args come from ``sample.ground_truth``.

    The ``scheme`` selects the comparison building block:
    - ``"overlap"``: normalized tool-call equality + :func:`compare_args_overlap`
      (the functional-equivalence scheme).
    - ``"mappings"``: :func:`score_mappings` driven by ``params.mappings``.
    - ``"legacy"``: :func:`score_legacy_fields` driven by ``params.comparison``
      and ``params.weights``.

    Ground-truth field names are config-driven via ``gt_tool_field`` /
    ``gt_args_field``; nothing is hardcoded.
    """

    def __init__(
        self,
        name: str = "args_match",
        scheme: str = "overlap",
        gt_tool_field: str = "tool_name",
        gt_args_field: str = "arguments",
        mappings: List[Mapping[str, Any]] | None = None,
        comparison: Mapping[str, Any] | None = None,
        weights: Mapping[str, Any] | None = None,
        pass_threshold: float = 0.5,
    ):
        if scheme not in ("overlap", "mappings", "legacy"):
            raise ValueError(
                f"args_match 'scheme' must be one of "
                f"('overlap', 'mappings', 'legacy'), got {scheme!r}"
            )
        self.name = name
        self.scheme = scheme
        self.gt_tool_field = gt_tool_field
        self.gt_args_field = gt_args_field
        self.mappings = list(mappings) if mappings else []
        self.comparison = dict(comparison) if comparison else {}
        self.weights = dict(weights) if weights else {}
        self.pass_threshold = pass_threshold

    def _expected(self, sample: VerifierInput) -> Tuple[str, Any]:
        gt = sample.ground_truth or {}
        gt_tool = gt.get(self.gt_tool_field, "") or ""
        gt_args_raw = gt.get(self.gt_args_field)
        gt_args: Any = {}
        if gt_args_raw is not None:
            if isinstance(gt_args_raw, str):
                try:
                    gt_args = json.loads(gt_args_raw)
                except (json.JSONDecodeError, ValueError):
                    gt_args = {}
            else:
                gt_args = gt_args_raw
        return str(gt_tool), gt_args

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        ea = extract(sample.completion_text, mode="tool_call")
        pred_tool = ea.tool_name if ea.found else ""
        pred_args = ea.arguments if ea.found else {}

        gt_tool, gt_args = self._expected(sample)

        if self.scheme == "overlap":
            norm_pred_name, norm_pred_args = normalize_tool_call(pred_tool, pred_args)
            gt_args_dict = gt_args if isinstance(gt_args, dict) else {}
            norm_gt_name, norm_gt_args = normalize_tool_call(gt_tool, gt_args_dict)
            if not gt_tool or not ea.found or norm_pred_name != norm_gt_name:
                score = 0.0
            else:
                score = compare_args_overlap(norm_pred_args, norm_gt_args)
        elif self.scheme == "mappings":
            score = score_mappings(
                pred_args, pred_tool, gt_args, gt_tool, self.mappings
            )
        else:  # legacy
            score = score_legacy_fields(
                pred_args, pred_tool, gt_args, gt_tool, self.comparison, self.weights
            )

        return VerifierOutput(
            score=score,
            passed=score >= self.pass_threshold,
            detail={
                "scheme": self.scheme,
                "pred_tool": pred_tool,
                "gt_tool": gt_tool,
            },
        )


@register("args_match")
def _build_args_match(spec: Mapping) -> ArgsMatchVerifier:
    params = spec.get("params", spec)
    return ArgsMatchVerifier(
        name=spec.get("name", "args_match"),
        scheme=params.get("scheme", "overlap"),
        gt_tool_field=params.get("gt_tool_field", "tool_name"),
        gt_args_field=params.get("gt_args_field", "arguments"),
        mappings=params.get("mappings"),
        comparison=params.get("comparison"),
        weights=params.get("weights"),
        pass_threshold=params.get("pass_threshold", 0.5),
    )
