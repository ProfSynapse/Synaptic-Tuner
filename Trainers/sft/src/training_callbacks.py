#!/usr/bin/env python3
"""Per-trainer SFT concrete callback subclasses and re-exports.

Public symbols `MetricsTableCallback`, `CheckpointMonitorCallback`,
`LiveDashboardCallback`, `TwoStageLRCallback`, `suppress_training_logs`,
`DASHBOARD_AVAILABLE`, `RICH_AVAILABLE` are re-exported at their original
paths. Shared lifecycle lives in `Trainers.shared.callbacks`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# sys.path bootstrap is handled by Trainers/shared/callbacks/__init__.py.
from Trainers.shared.callbacks import (
    BaseLiveDashboardCallback,
    BaseMetricsCallback,
    CheckpointMonitorCallback,
    DASHBOARD_AVAILABLE,
    RICH_AVAILABLE,
    SFTHealthChecker,
    TwoStageLRCallback,
    format_time,
    suppress_training_logs,
)


class MetricsTableCallback(BaseMetricsCallback):
    """SFT table output: loss/LR/gradnorm/epoch/gpu/samples/eta."""

    default_output_dir = "./sft_output"
    start_banner = "TRAINING STARTED"
    completion_banner = "TRAINING COMPLETED"
    log_every_write = False  # Write JSONL only at log_every_n_steps boundary.
    # Pre-refactor SFT gated the entire on_log body on the interval multiple, so
    # health_checker.check() and last_log_time update fired only at printed-row cadence.
    health_check_every_on_log = False
    interval_time_updates_every_on_log = False

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./sft_output",
        previous_log_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            log_every_n_steps=log_every_n_steps,
            output_dir=output_dir,
            previous_log_entries=previous_log_entries,
        )
        self.health_checker = SFTHealthChecker()

    def _print_header(self):
        print("\n" + "=" * 110)
        print(" " * 47 + "TRAINING METRICS")
        print("=" * 110)
        print("   Step      |   Loss   |    LR     | GradNorm | Epoch  | GPU Mem  | Time/5s | Samp/sec |    ETA    ")
        print("-" * 110)

    def _print_row(self, *, step, state, args, logs, capacity_snapshot, interval_time, samples_per_sec, eta, progress):
        gpu_mem_value = capacity_snapshot.get("gpu_memory_gb")
        gpu_mem = f"{gpu_mem_value:.1f}GB" if isinstance(gpu_mem_value, (int, float)) else "N/A"
        loss = logs.get("loss", 0.0)
        learning_rate = logs.get("learning_rate", 0.0)
        grad_norm = logs.get("grad_norm", 0.0)
        epoch = logs.get("epoch", 0.0)
        print(
            f" {progress:>12} | {loss:>8.4f} | {learning_rate:>9.2e} | "
            f"{grad_norm:>8.3f} | {epoch:>6.2f} | {gpu_mem:>8} | "
            f"{interval_time:>7.1f}s | {samples_per_sec:>8.1f} | {eta:>9} "
        )


class LiveDashboardCallback(BaseLiveDashboardCallback):
    """SFT live dashboard: loss/LR/kl/margin/gpu; accepts `training_type` param."""

    default_output_dir = "./sft_output"
    completion_banner = "TRAINING COMPLETED"

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./sft_output",
        training_type: str = "sft",
        previous_log_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        self.training_type = training_type
        self.default_title = f"{training_type.upper()} Training"
        self.training_type_attr = training_type
        super().__init__(
            log_every_n_steps=log_every_n_steps,
            output_dir=output_dir,
            previous_log_entries=previous_log_entries,
        )

    def _dashboard_metrics(self, logs, capacity_snapshot):
        return {
            "kl": logs.get("kl"),
            "margin": logs.get("rewards/margins"),
        }


__all__ = [
    "MetricsTableCallback",
    "LiveDashboardCallback",
    "TwoStageLRCallback",
    "CheckpointMonitorCallback",
    "suppress_training_logs",
    "DASHBOARD_AVAILABLE",
    "RICH_AVAILABLE",
    "format_time",
]
