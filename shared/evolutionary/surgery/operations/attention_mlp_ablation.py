"""
Attention/MLP ablation surgery operation.

Location: shared/evolutionary/surgery/operations/attention_mlp_ablation.py
Purpose: Zero all attention LoRA weights vs all MLP LoRA weights to measure
         which module type matters more for the adapter's performance.
Used by: LoRASurgeon orchestrator via the operation registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict

import torch

from ..config import OperationResult, SurgeryConfig
from ..registry import register_operation
from ..utils import (
    copy_adapter,
    is_attention_key,
    is_mlp_key,
    load_all_weights,
    save_all_weights,
)

logger = logging.getLogger(__name__)


@register_operation
class AttentionMLPAblationOperation:
    """Zero all attention LoRA vs all MLP LoRA, measure which matters more."""

    name = "attention_mlp_ablation"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        weights = load_all_weights(adapter_path)

        has_attn = any(is_attention_key(k) for k in weights)
        has_mlp = any(is_mlp_key(k) for k in weights)

        if not has_attn and not has_mlp:
            return OperationResult(
                operation="attention_mlp_ablation",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_attention_or_mlp_keys"},
            )

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"ablation_scores": {}}

        # Ablate attention
        if has_attn:
            variant_dir = os.path.join(work_dir, "ablate_attention")
            copy_adapter(adapter_path, variant_dir)

            modified = {}
            for key, tensor in weights.items():
                if is_attention_key(key):
                    modified[key] = torch.zeros_like(tensor)
                else:
                    modified[key] = tensor.clone()
            save_all_weights(variant_dir, modified)

            score = await evaluate_fn(variant_dir)
            variants_tried += 1
            details["ablation_scores"]["zero_attention"] = score
            logger.info("  ablate attention  score=%.4f", score)

            if score > best_score:
                best_score = score
                best_variant = "ablate_attention"
                best_path = variant_dir

        # Ablate MLP
        if has_mlp:
            variant_dir = os.path.join(work_dir, "ablate_mlp")
            copy_adapter(adapter_path, variant_dir)

            modified = {}
            for key, tensor in weights.items():
                if is_mlp_key(key):
                    modified[key] = torch.zeros_like(tensor)
                else:
                    modified[key] = tensor.clone()
            save_all_weights(variant_dir, modified)

            score = await evaluate_fn(variant_dir)
            variants_tried += 1
            details["ablation_scores"]["zero_mlp"] = score
            logger.info("  ablate mlp  score=%.4f", score)

            if score > best_score:
                best_score = score
                best_variant = "ablate_mlp"
                best_path = variant_dir

        return OperationResult(
            operation="attention_mlp_ablation",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )
