#!/usr/bin/env python3
"""Per-trainer DPO concrete callback subclasses and re-exports.

Public symbols `MetricsTableCallback`, `CheckpointMonitorCallback`,
`LiveDashboardCallback`, `TwoStageLRCallback`, `DASHBOARD_AVAILABLE`,
`RICH_AVAILABLE` are re-exported at their original paths. Shared lifecycle
lives in `Trainers.shared.callbacks`.

DEVIATION FROM KTO (flagged): DPO reuses KTOHealthChecker rather than adding a
new DPOHealthChecker to the shared callbacks package. DPO and KTO both surface
TRL preference-pair metrics under the same log keys (rewards/chosen,
rewards/rejected, rewards/margins), so the KTO health heuristics apply
unchanged. Keeping the reuse here (inside Trainers/dpo) avoids editing the
shared Trainers/shared/callbacks package, which is outside this trainer's
mirror scope.
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
    """DPO table output: loss/LR/chosen/reject/margin/gpu/samples/eta; writes JSONL every on_log."""

    default_output_dir = "./dpo_output"
    start_banner = "TRAINING STARTED"
    completion_banner = "TRAINING COMPLETED"
    log_every_write = True  # DPO writes JSONL on every on_log, matching KTO.

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./dpo_output",
        previous_log_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            log_every_n_steps=log_every_n_steps,
            output_dir=output_dir,
            previous_log_entries=previous_log_entries,
        )
        # DPO shares KTO's preference-pair metric keys; reuse its health checker.
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
        dpo_chosen = logs.get("rewards/chosen", 0.0)
        dpo_rejected = logs.get("rewards/rejected", 0.0)
        dpo_margin = logs.get("rewards/margins", 0.0)
        print(
            f" {progress:>12} | {loss:>8.4f} | {learning_rate:>9.2e} | "
            f"{dpo_chosen:>6.3f} | {dpo_rejected:>6.3f} | {dpo_margin:>6.3f} | "
            f"{gpu_mem:>8} | {interval_time:>7.1f}s | {samples_per_sec:>8.1f} | {eta:>9} "
        )


class LiveDashboardCallback(BaseLiveDashboardCallback):
    """DPO live dashboard: margin + KL-with-logps-fallback."""

    default_output_dir = "./dpo_output"
    default_title = "DPO Training"
    training_type_attr = "dpo"
    completion_banner = "DPO TRAINING COMPLETED"

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
