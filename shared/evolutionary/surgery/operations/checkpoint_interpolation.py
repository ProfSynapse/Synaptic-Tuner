"""
Checkpoint interpolation surgery operation.

Location: shared/evolutionary/surgery/operations/checkpoint_interpolation.py
Purpose: Blend two checkpoints at various ratios by linearly interpolating
         their weight tensors.
Used by: LoRASurgeon orchestrator via the operation registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict

import torch

from ..config import CheckpointInterpolationConfig, OperationResult, SurgeryConfig
from ..registry import register_operation
from ..utils import copy_adapter, load_all_weights, save_all_weights

logger = logging.getLogger(__name__)


@register_operation
class CheckpointInterpolationOperation:
    """Blend two checkpoints at various ratios."""

    name = "checkpoint_interpolation"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        op_config: CheckpointInterpolationConfig = config.checkpoint_interpolation_config
        other_path = op_config.other_checkpoint_path

        if not other_path or not os.path.isdir(other_path):
            return OperationResult(
                operation="checkpoint_interpolation",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_other_checkpoint"},
            )

        weights_a = load_all_weights(adapter_path)
        weights_b = load_all_weights(other_path)

        # Only interpolate keys present in both
        common_keys = set(weights_a.keys()) & set(weights_b.keys())

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"blend_scores": {}}

        for ratio in op_config.blend_ratios:
            variant_dir = os.path.join(work_dir, f"blend_{ratio:.2f}")
            copy_adapter(adapter_path, variant_dir)

            blended: Dict[str, torch.Tensor] = {}
            for key in weights_a:
                if key in common_keys:
                    blended[key] = ratio * weights_a[key] + (1.0 - ratio) * weights_b[key]
                else:
                    blended[key] = weights_a[key].clone()

            save_all_weights(variant_dir, blended)

            score = await evaluate_fn(variant_dir)
            variants_tried += 1
            details["blend_scores"][str(ratio)] = score
            logger.info("  blend ratio=%.2f  score=%.4f", ratio, score)

            if score > best_score:
                best_score = score
                best_variant = f"blend_{ratio:.2f}"
                best_path = variant_dir

        return OperationResult(
            operation="checkpoint_interpolation",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )
