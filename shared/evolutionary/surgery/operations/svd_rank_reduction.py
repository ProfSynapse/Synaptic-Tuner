"""
SVD rank reduction surgery operation.

Location: shared/evolutionary/surgery/operations/svd_rank_reduction.py
Purpose: Compress LoRA via truncated SVD. For each LoRA pair (A, B),
         compose W = B @ A, perform SVD, truncate to a lower rank, and
         reconstruct new A/B matrices.
Used by: LoRASurgeon orchestrator via the operation registry.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict

import torch

from ..config import OperationResult, SVDRankReductionConfig, SurgeryConfig
from ..registry import register_operation
from ..utils import (
    copy_adapter,
    find_lora_pairs,
    load_adapter_config,
    load_all_weights,
    save_adapter_config,
    save_all_weights,
)

logger = logging.getLogger(__name__)


@register_operation
class SVDRankReductionOperation:
    """Compress LoRA via truncated SVD."""

    name = "svd_rank_reduction"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        op_config: SVDRankReductionConfig = config.svd_rank_reduction_config
        weights = load_all_weights(adapter_path)
        adapter_config = load_adapter_config(adapter_path)
        original_rank = adapter_config.get("r", adapter_config.get("lora_rank", 16))

        if original_rank <= 1:
            return OperationResult(
                operation="svd_rank_reduction",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "rank_too_small"},
            )

        # Group LoRA A/B pairs
        lora_pairs = find_lora_pairs(weights)

        if not lora_pairs:
            return OperationResult(
                operation="svd_rank_reduction",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_lora_pairs_found"},
            )

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"svd_scores": {}}

        for fraction in op_config.rank_fractions:
            new_rank = max(1, int(round(original_rank * fraction)))
            if new_rank >= original_rank:
                continue

            variant_dir = os.path.join(work_dir, f"svd_rank{new_rank}")
            copy_adapter(adapter_path, variant_dir)

            modified_weights = dict(weights)
            for prefix, (a_key, b_key) in lora_pairs.items():
                a_tensor = weights[a_key]  # shape: (r, in_features)
                b_tensor = weights[b_key]  # shape: (out_features, r)

                # Compose: W = B @ A -> (out_features, in_features)
                w = b_tensor.float() @ a_tensor.float()
                u, s, vh = torch.linalg.svd(w, full_matrices=False)

                # Truncate to new_rank
                u_trunc = u[:, :new_rank]
                s_trunc = s[:new_rank]
                vh_trunc = vh[:new_rank, :]

                # Reconstruct as new A and B
                # New A = sqrt(S) @ Vh  -> (new_rank, in_features)
                # New B = U @ sqrt(S)   -> (out_features, new_rank)
                sqrt_s = torch.diag(torch.sqrt(s_trunc))
                new_a = (sqrt_s @ vh_trunc).to(a_tensor.dtype)
                new_b = (u_trunc @ sqrt_s).to(b_tensor.dtype)

                modified_weights[a_key] = new_a
                modified_weights[b_key] = new_b

            save_all_weights(variant_dir, modified_weights)

            # Update adapter_config with new rank
            variant_config = load_adapter_config(variant_dir)
            variant_config["r"] = new_rank
            save_adapter_config(variant_dir, variant_config)

            score = await evaluate_fn(variant_dir)
            variants_tried += 1
            details["svd_scores"][str(new_rank)] = score
            logger.info("  svd rank=%d  score=%.4f", new_rank, score)

            if score > best_score:
                best_score = score
                best_variant = f"svd_rank{new_rank}"
                best_path = variant_dir

        return OperationResult(
            operation="svd_rank_reduction",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )
