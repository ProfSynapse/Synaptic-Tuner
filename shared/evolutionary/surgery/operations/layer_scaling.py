"""
Layer scaling surgery operation.

Location: shared/evolutionary/surgery/operations/layer_scaling.py
Purpose: Scale individual layer weights by various factors to find which
         layers matter most for performance.
Used by: LoRASurgeon orchestrator via the operation registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict

from ..config import LayerScalingConfig, OperationResult, SurgeryConfig
from ..registry import register_operation
from ..utils import (
    copy_adapter,
    get_layer_indices,
    load_all_weights,
    save_all_weights,
)

logger = logging.getLogger(__name__)


@register_operation
class LayerScalingOperation:
    """Scale individual layers to find importance."""

    name = "layer_scaling"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        op_config: LayerScalingConfig = config.layer_scaling_config
        weights = load_all_weights(adapter_path)
        layer_indices = get_layer_indices(list(weights.keys()))

        if not layer_indices:
            return OperationResult(
                operation="layer_scaling",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_lora_layers_found"},
            )

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"layer_scores": {}}

        for layer_idx in layer_indices:
            layer_pattern = f"layers.{layer_idx}."
            for scale in op_config.scales:
                variant_dir = os.path.join(
                    work_dir, f"layer{layer_idx}_scale{scale}"
                )
                copy_adapter(adapter_path, variant_dir)

                modified_weights = dict(weights)
                for key, tensor in weights.items():
                    if layer_pattern in key:
                        modified_weights[key] = tensor * scale
                    else:
                        modified_weights[key] = tensor.clone()

                save_all_weights(variant_dir, modified_weights)

                score = await evaluate_fn(variant_dir)
                variants_tried += 1

                variant_label = f"layer{layer_idx}_scale{scale}"
                details["layer_scores"][variant_label] = score
                logger.info(
                    "  layer=%d scale=%.2f  score=%.4f", layer_idx, scale, score
                )

                if score > best_score:
                    best_score = score
                    best_variant = variant_label
                    best_path = variant_dir

        return OperationResult(
            operation="layer_scaling",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )
