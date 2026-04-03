"""
Metrics-weighted merge surgery operation.

Location: shared/evolutionary/surgery/operations/metrics_weighted_merge.py
Purpose: Merge N checkpoints weighted by their eval scores using softmax
         normalization.
Used by: LoRASurgeon orchestrator via the operation registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict

import torch

from ..config import MetricsWeightedMergeConfig, OperationResult, SurgeryConfig
from ..registry import register_operation
from ..utils import copy_adapter, load_all_weights, save_all_weights, softmax

logger = logging.getLogger(__name__)


@register_operation
class MetricsWeightedMergeOperation:
    """Merge N checkpoints weighted by their eval scores."""

    name = "metrics_weighted_merge"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        op_config: MetricsWeightedMergeConfig = config.metrics_weighted_merge_config
        paths = op_config.checkpoint_paths
        scores = op_config.checkpoint_scores

        if len(paths) < 2 or len(paths) != len(scores):
            return OperationResult(
                operation="metrics_weighted_merge",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "insufficient_checkpoints_or_scores"},
            )

        # Load all checkpoint weights
        all_weights = [load_all_weights(p) for p in paths]
        merge_weights = softmax(scores, temperature=1.0)

        # Weighted sum
        merged: Dict[str, torch.Tensor] = {}
        reference_keys = list(all_weights[0].keys())

        for key in reference_keys:
            tensors = []
            for w_dict in all_weights:
                if key in w_dict:
                    tensors.append(w_dict[key])
            if len(tensors) != len(paths):
                # Key not present in all checkpoints, skip interpolation
                merged[key] = all_weights[0][key].clone()
                continue
            weighted = sum(w * t for w, t in zip(merge_weights, tensors))
            merged[key] = weighted

        variant_dir = os.path.join(work_dir, "metrics_merge")
        copy_adapter(adapter_path, variant_dir)
        save_all_weights(variant_dir, merged)

        score = await evaluate_fn(variant_dir)

        return OperationResult(
            operation="metrics_weighted_merge",
            variants_tried=1,
            best_variant="metrics_merge",
            best_score=score,
            improvement=score - baseline_score,
            adapter_path=variant_dir if score > baseline_score else adapter_path,
            details={"merge_weights": merge_weights, "score": score},
        )
