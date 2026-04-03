"""
DARE drop-and-rescale surgery operation.

Location: shared/evolutionary/surgery/operations/dare_drop_rescale.py
Purpose: Randomly drop weights and rescale survivors so expected value is
         preserved. Evaluates several drop rates.
Used by: LoRASurgeon orchestrator via the operation registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict

import torch

from ..config import DAREConfig, OperationResult, SurgeryConfig
from ..registry import register_operation
from ..utils import copy_adapter, load_all_weights, save_all_weights

logger = logging.getLogger(__name__)


@register_operation
class DAREDropRescaleOperation:
    """DARE: randomly drop weights and rescale survivors."""

    name = "dare_drop_rescale"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        op_config: DAREConfig = config.dare_config
        weights = load_all_weights(adapter_path)

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"dare_scores": {}}

        for drop_rate in op_config.drop_rates:
            # Clamp drop_rate to [0.0, 0.99] to prevent division by zero
            if drop_rate >= 1.0:
                logger.warning(
                    "DARE drop_rate %.2f >= 1.0 would cause division by zero, "
                    "clamping to 0.99",
                    drop_rate,
                )
                drop_rate = 0.99
            drop_rate = max(0.0, drop_rate)

            variant_dir = os.path.join(work_dir, f"dare_{drop_rate:.2f}")
            copy_adapter(adapter_path, variant_dir)

            modified_weights: Dict[str, torch.Tensor] = {}
            for key, tensor in weights.items():
                mask = (torch.rand_like(tensor.float()) > drop_rate).to(tensor.dtype)
                modified_weights[key] = tensor * mask / (1.0 - drop_rate)

            save_all_weights(variant_dir, modified_weights)

            score = await evaluate_fn(variant_dir)
            variants_tried += 1
            details["dare_scores"][str(drop_rate)] = score
            logger.info("  dare drop=%.2f  score=%.4f", drop_rate, score)

            if score > best_score:
                best_score = score
                best_variant = f"dare_{drop_rate:.2f}"
                best_path = variant_dir

        return OperationResult(
            operation="dare_drop_rescale",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )
