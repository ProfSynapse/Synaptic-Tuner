#!/usr/bin/env python3
"""Per-trainer KTO concrete callback subclasses and re-exports.

Public symbols `MetricsTableCallback`, `CheckpointMonitorCallback`,
`LiveDashboardCallback`, `TwoStageLRCallback`, `DASHBOARD_AVAILABLE`,
`RICH_AVAILABLE` are re-exported at their original paths. Shared lifecycle
lives in `Trainers.shared.callbacks`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# sys.path bootstrap is handled by Trainers/shared/callbacks/__init__.py.
from Trainers.shared.callbacks import (
    BaseLiveDashboardCallback,
    BaseMetricsCallback,
    CheckpointMonitorCallback,
    DASHBOARD_AVAILABLE,
    KTOHealthChecker,
    RICH_AVAILABLE,
    TwoStageLRCallback,
)


class MetricsTableCallback(BaseMetricsCallback):
    """KTO table output: loss/LR/chosen/reject/margin/gpu/samples/eta; writes JSONL every on_log."""

    default_output_dir = "./kto_output"
    start_banner = "TRAINING STARTED"
    completion_banner = "TRAINING COMPLETED"
    log_every_write = True  # KTO writes JSONL on every on_log, not only at interval.

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./kto_output",
        previous_log_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            log_every_n_steps=log_every_n_steps,
            output_dir=output_dir,
            previous_log_entries=previous_log_entries,
        )
        self.health_checker = KTOHealthChecker()

    def _print_header(self):
        print("\n" + "=" * 110)
        print(" " * 47 + "TRAINING METRICS")
        print("=" * 110)
        print("   Step      |   Loss   |    LR     | Chosen | Reject | Margin | GPU Mem  | Time/5s | Samp/sec |    ETA    ")
        print("-" * 110)

    def _print_row(self, *, step, state, args, logs, capacity_snapshot, interval_time, samples_per_sec, eta, progress):
        gpu_mem_value = capacity_snapshot.get("gpu_memory_gb")
        gpu_mem = f"{gpu_mem_value:.1f}GB" if isinstance(gpu_mem_value, (int, float)) else "N/A"
        loss = logs.get("loss", 0.0)
        learning_rate = logs.get("learning_rate", 0.0)
        kto_chosen = logs.get("rewards/chosen", 0.0)
        kto_rejected = logs.get("rewards/rejected", 0.0)
        kto_margin = logs.get("rewards/margins", 0.0)
        print(
            f" {progress:>12} | {loss:>8.4f} | {learning_rate:>9.2e} | "
            f"{kto_chosen:>6.3f} | {kto_rejected:>6.3f} | {kto_margin:>6.3f} | "
            f"{gpu_mem:>8} | {interval_time:>7.1f}s | {samples_per_sec:>8.1f} | {eta:>9} "
        )


class LiveDashboardCallback(BaseLiveDashboardCallback):
    """KTO live dashboard: margin + KL-with-logps-fallback."""

    default_output_dir = "./kto_output"
    default_title = "KTO Training"
    training_type_attr = "kto"
    completion_banner = "KTO TRAINING COMPLETED"

    def _dashboard_metrics(self, logs, capacity_snapshot):
        return {
            "kl": logs.get("kl", logs.get("logps/rejected", 0)),
            "margin": logs.get("rewards/margins"),
        }

    def _fallback_row(self, *, state, logs, capacity_snapshot):
        loss = logs.get("loss", 0)
        margin = logs.get("rewards/margins", 0)
        gpu_mem_value = capacity_snapshot.get("gpu_memory_gb")
        gpu_mem = f"{gpu_mem_value:.1f}GB" if isinstance(gpu_mem_value, (int, float)) else "N/A"
        print(f"  Step {state.global_step}/{self.total_steps} | Loss: {loss:.4f} | Margin: {margin:.4f} | GPU: {gpu_mem}")


__all__ = [
    "MetricsTableCallback",
    "LiveDashboardCallback",
    "TwoStageLRCallback",
    "CheckpointMonitorCallback",
    "DASHBOARD_AVAILABLE",
    "RICH_AVAILABLE",
]
