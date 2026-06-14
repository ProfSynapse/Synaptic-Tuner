"""
Embedding-trainer callbacks — adapt Trainers/shared/callbacks to the ST trainer.

Location: Trainers/embedding/src/callbacks.py
Purpose:  Provide a metrics/logging callback for the embedding trainer by
          subclassing the shared BaseMetricsCallback (the same machinery SFT/KTO
          use). SentenceTransformerTrainer subclasses the HuggingFace Trainer, so
          a transformers TrainerCallback attaches to it unchanged.
Used by:  Trainers/embedding/train_embedding.py.

Contract: docs/architecture/embedding-reranker-phase1/01_CONTRACTS.md §6.

The embedding training loop reports a contrastive `loss` (and optionally
grad_norm / learning_rate) — there is no reward/margin axis (that is KTO). So the
health checker mirrors the SFT loss/grad-norm shape, and the row format surfaces
loss + grad-norm + throughput.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Repo root on sys.path so `Trainers.shared...` imports resolve when this module
# is loaded from the embedding trainer's bootstrap (mirrors the SFT trainer).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Trainers.shared.callbacks.base import BaseMetricsCallback, format_time
from Trainers.shared.callbacks.health_checks import HealthChecker, _grad_norm_warning, _print_warnings


class EmbeddingHealthChecker(HealthChecker):
    """Embedding: loss-range + grad-norm-clip (no reward axis).

    Mirrors SFTHealthChecker's loss/grad-norm checks; contrastive losses (MNRL,
    triplet) sit in a normal positive range, so the same 0<loss<100 sanity bound
    and >100 grad-norm warning apply.
    """

    def check(self, logs: Dict[str, Any], step: int, max_grad_norm: Optional[float]) -> None:
        warnings: list[str] = []
        loss = logs.get("loss", 0.0)
        if not (0 < loss < 100):
            warnings.append(f"⚠ Unusual loss value: {loss:.4f}")
        grad_warning = _grad_norm_warning(logs, max_grad_norm)
        if grad_warning:
            warnings.append(grad_warning)
        _print_warnings(warnings, max_grad_norm, logs.get("grad_norm", 0.0))


class EmbeddingMetricsCallback(BaseMetricsCallback):
    """Table-output + JSONL-logging callback for embedding training.

    Defaults match SFT/KTO behavior (fields_win_on_collision=False ->
    `{**our_fields, **capacity, **logs}`; health/interval-time update every
    on_log call). The row surfaces loss, grad-norm, and throughput.
    """

    default_output_dir = "./embedding_output"
    start_banner = "EMBEDDING TRAINING STARTED"
    completion_banner = "EMBEDDING TRAINING COMPLETED"
    training_type_label = "embedding"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.health_checker = EmbeddingHealthChecker()

    def _print_header(self) -> None:
        print("\n" + "=" * 100)
        print(
            f"{'Step':>8} | {'Loss':>10} | {'GradNorm':>9} | {'LR':>10} | "
            f"{'Samp/s':>8} | {'Time':>8} | {'ETA':>8} | {'Progress':>12}"
        )
        print("-" * 100)

    def _print_row(
        self,
        *,
        step: int,
        state: Any,
        args: Any,
        logs: Dict[str, Any],
        capacity_snapshot: Dict[str, Any],
        interval_time: float,
        samples_per_sec: float,
        eta: str,
        progress: str,
    ) -> None:
        loss = logs.get("loss", 0.0)
        grad_norm = logs.get("grad_norm", 0.0)
        lr = logs.get("learning_rate", 0.0)
        print(
            f"{step:>8,} | {loss:>10.4f} | {grad_norm:>9.3f} | {lr:>10.2e} | "
            f"{samples_per_sec:>8.1f} | {format_time(interval_time):>8} | "
            f"{eta:>8} | {progress:>12}"
        )
