"""
Module ablation surgery operation.

Location: shared/evolutionary/surgery/operations/module_ablation.py
Purpose: Zero all weights of a module type (q_proj, k_proj, etc.) one at a
         time and measure the impact on eval score.
Used by: LoRASurgeon orchestrator via the operation registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict

import torch

from ..config import OperationResult, SurgeryConfig
from ..registry import register_operation
from ..utils import copy_adapter, get_module_types, load_all_weights, save_all_weights

logger = logging.getLogger(__name__)


@register_operation
class ModuleAblationOperation:
    """Zero all weights of a module type one at a time, measure impact."""

    name = "module_ablation"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        weights = load_all_weights(adapter_path)
        module_types = get_module_types(list(weights.keys()))

        if not module_types:
            return OperationResult(
                operation="module_ablation",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_module_types_found"},
            )

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"module_scores": {}}

        for mod_type in module_types:
            variant_dir = os.path.join(work_dir, f"ablate_{mod_type}")
            copy_adapter(adapter_path, variant_dir)

            modified_weights = {}
            for key, tensor in weights.items():
                if f".{mod_type}." in key:
                    modified_weights[key] = torch.zeros_like(tensor)
                else:
                    modified_weights[key] = tensor.clone()

            save_all_weights(variant_dir, modified_weights)

            score = await evaluate_fn(variant_dir)
            variants_tried += 1
            details["module_scores"][mod_type] = score
            logger.info("  ablate %s  score=%.4f", mod_type, score)

            if score > best_score:
                best_score = score
                best_variant = f"ablate_{mod_type}"
                best_path = variant_dir

        return OperationResult(
            operation="module_ablation",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )
