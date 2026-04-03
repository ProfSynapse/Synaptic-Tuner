"""
Alpha sweep surgery operation.

Location: shared/evolutionary/surgery/operations/alpha_sweep.py
Purpose: Modify adapter_config.json lora_alpha values and evaluate each variant.
         No weight changes — config-only operation.
Used by: LoRASurgeon orchestrator via the operation registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict

from ..config import AlphaSweepConfig, OperationResult, SurgeryConfig
from ..registry import register_operation
from ..utils import copy_adapter, load_adapter_config, save_adapter_config

logger = logging.getLogger(__name__)


@register_operation
class AlphaSweepOperation:
    """Modify adapter_config.json lora_alpha, no weight changes."""

    name = "alpha_sweep"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        op_config: AlphaSweepConfig = config.alpha_sweep_config
        original_config = load_adapter_config(adapter_path)
        current_alpha = original_config.get("lora_alpha", 16)
        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"alpha_scores": {}}

        for mult in op_config.multipliers:
            new_alpha = int(round(current_alpha * mult))
            if new_alpha == current_alpha:
                continue

            variant_dir = os.path.join(work_dir, f"alpha_{new_alpha}")
            copy_adapter(adapter_path, variant_dir)

            variant_config = load_adapter_config(variant_dir)
            variant_config["lora_alpha"] = new_alpha
            save_adapter_config(variant_dir, variant_config)

            score = await evaluate_fn(variant_dir)
            variants_tried += 1
            details["alpha_scores"][str(new_alpha)] = score

            logger.info("  alpha=%d  score=%.4f", new_alpha, score)

            if score > best_score:
                best_score = score
                best_variant = f"alpha={new_alpha}"
                best_path = variant_dir

        return OperationResult(
            operation="alpha_sweep",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )
