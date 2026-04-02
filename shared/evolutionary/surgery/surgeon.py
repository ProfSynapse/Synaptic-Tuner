"""
LoRASurgeon thin orchestrator.

Location: shared/evolutionary/surgery/surgeon.py
Purpose: Coordinate surgery operations by iterating through the configured
         operation list, delegating to registered strategy objects, and
         tracking the best adapter across all operations.
Used by: tuner/handlers/surgery_handler.py, tests.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from typing import List

from shared.eval_backend import EvalBackend, EvalResult

from .config import OperationResult, SurgeryConfig, SurgeryResult
from .registry import get_operation
from .utils import check_dependencies, copy_adapter

logger = logging.getLogger(__name__)


class LoRASurgeon:
    """Performs eval-guided post-training weight surgery on LoRA adapters.

    The surgeon iterates through a configurable list of operations. Each
    operation produces one or more modified adapter variants. The variant
    with the highest eval score becomes the new baseline for the next
    operation, provided it exceeds the minimum improvement threshold.

    Args:
        adapter_path: Path to the LoRA adapter directory.
        eval_backend: Object implementing the ``EvalBackend`` protocol.
        eval_scenario: Scenario identifier passed to the eval backend.
        config: Surgery configuration.
    """

    def __init__(
        self,
        adapter_path: str,
        eval_backend: EvalBackend,
        eval_scenario: str,
        config: SurgeryConfig,
    ) -> None:
        check_dependencies()
        self.adapter_path = adapter_path
        self.eval_backend = eval_backend
        self.eval_scenario = eval_scenario
        self.config = config
        self._work_dir = os.path.join(config.output_dir, "_surgery_work")

    # ------------------------------------------------------------------
    # Context manager & cleanup
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "LoRASurgeon":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the temporary work directory created during surgery."""
        if os.path.isdir(self._work_dir):
            shutil.rmtree(self._work_dir)
            logger.info("Cleaned up work directory: %s", self._work_dir)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run_surgery(self) -> SurgeryResult:
        """Run all enabled operations sequentially, keeping improvements.

        Returns:
            SurgeryResult summarising the full pipeline.
        """
        start_time = time.time()

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self._work_dir, exist_ok=True)

        try:
            baseline_score = await self._evaluate(self.adapter_path)
            best_score = baseline_score
            best_adapter = self.adapter_path
            operations_applied: List[OperationResult] = []

            logger.info(
                "Starting surgery on %s  baseline_score=%.4f",
                self.adapter_path,
                baseline_score,
            )

            for operation_name in self.config.operations:
                try:
                    operation = get_operation(operation_name)
                except ValueError:
                    logger.warning("Unknown operation: %s, skipping", operation_name)
                    continue

                logger.info("Running operation: %s", operation_name)

                try:
                    result = await operation.execute(
                        adapter_path=best_adapter,
                        baseline_score=best_score,
                        work_dir=self._work_dir,
                        config=self.config,
                        evaluate_fn=self._evaluate,
                    )
                except Exception:
                    logger.exception("Operation %s failed", operation_name)
                    continue

                if result.improvement > self.config.min_improvement:
                    best_score = result.best_score
                    best_adapter = result.adapter_path
                    operations_applied.append(result)
                    logger.info(
                        "Operation %s improved score: %.4f -> %.4f (+%.4f)",
                        operation_name,
                        best_score - result.improvement,
                        best_score,
                        result.improvement,
                    )
                else:
                    logger.info(
                        "Operation %s did not improve score (best=%.4f, improvement=%.4f < min=%.4f)",
                        operation_name,
                        result.best_score,
                        result.improvement,
                        self.config.min_improvement,
                    )

            # Copy best adapter to final output location
            final_path = os.path.join(self.config.output_dir, "best_adapter")
            if best_adapter != self.adapter_path:
                copy_adapter(best_adapter, final_path)
            else:
                copy_adapter(self.adapter_path, final_path)

            duration = time.time() - start_time

            surgery_result = SurgeryResult(
                baseline_score=baseline_score,
                final_score=best_score,
                total_improvement=best_score - baseline_score,
                operations_applied=operations_applied,
                best_adapter_path=final_path,
                duration_seconds=duration,
            )

            # Save report
            self._save_report(surgery_result)

            return surgery_result
        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evaluate(self, adapter_path: str) -> float:
        """Evaluate an adapter using the configured eval backend."""
        result = await self.eval_backend.run_eval(adapter_path, self.eval_scenario)
        return result.eval_score

    # ------------------------------------------------------------------
    # Backward-compat proxy methods (delegate to registered operations)
    # ------------------------------------------------------------------

    async def _run_operation_by_name(
        self, name: str, adapter_path: str, baseline_score: float
    ) -> "OperationResult":
        """Run a named operation via the registry, for backward compat."""
        os.makedirs(self._work_dir, exist_ok=True)
        operation = get_operation(name)
        return await operation.execute(
            adapter_path=adapter_path,
            baseline_score=baseline_score,
            work_dir=self._work_dir,
            config=self.config,
            evaluate_fn=self._evaluate,
        )

    async def alpha_sweep(self, adapter_path: str, baseline_score: float) -> "OperationResult":
        return await self._run_operation_by_name("alpha_sweep", adapter_path, baseline_score)

    async def layer_scaling(self, adapter_path: str, baseline_score: float) -> "OperationResult":
        return await self._run_operation_by_name("layer_scaling", adapter_path, baseline_score)

    async def module_ablation(self, adapter_path: str, baseline_score: float) -> "OperationResult":
        return await self._run_operation_by_name("module_ablation", adapter_path, baseline_score)

    async def checkpoint_interpolation(self, adapter_path: str, baseline_score: float) -> "OperationResult":
        return await self._run_operation_by_name("checkpoint_interpolation", adapter_path, baseline_score)

    async def dare_drop_rescale(self, adapter_path: str, baseline_score: float) -> "OperationResult":
        return await self._run_operation_by_name("dare_drop_rescale", adapter_path, baseline_score)

    async def metrics_weighted_merge(self, adapter_path: str, baseline_score: float) -> "OperationResult":
        return await self._run_operation_by_name("metrics_weighted_merge", adapter_path, baseline_score)

    async def svd_rank_reduction(self, adapter_path: str, baseline_score: float) -> "OperationResult":
        return await self._run_operation_by_name("svd_rank_reduction", adapter_path, baseline_score)

    async def attention_mlp_ablation(self, adapter_path: str, baseline_score: float) -> "OperationResult":
        return await self._run_operation_by_name("attention_mlp_ablation", adapter_path, baseline_score)

    def _save_report(self, result: SurgeryResult) -> None:
        """Save surgery results as a JSON report."""
        report_path = os.path.join(self.config.output_dir, "surgery_report.json")
        report_data = {
            "baseline_score": result.baseline_score,
            "final_score": result.final_score,
            "total_improvement": result.total_improvement,
            "best_adapter_path": result.best_adapter_path,
            "duration_seconds": result.duration_seconds,
            "operations_applied": [asdict(op) for op in result.operations_applied],
        }
        with open(report_path, "w") as fh:
            json.dump(report_data, fh, indent=2)
        logger.info("Surgery report saved to %s", report_path)
