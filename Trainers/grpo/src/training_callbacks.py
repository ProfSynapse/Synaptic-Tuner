#!/usr/bin/env python3
"""Per-trainer GRPO/GSPO concrete callback subclasses and re-exports.

Public symbols `MetricsTableCallback`, `LiveDashboardCallback`,
`DASHBOARD_AVAILABLE`, `RICH_AVAILABLE` are re-exported at their original
paths. Shared lifecycle lives in `Trainers.shared.callbacks`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# sys.path bootstrap is handled by Trainers/shared/callbacks/__init__.py.
from Trainers.shared.callbacks import (
    BaseLiveDashboardCallback,
    BaseMetricsCallback,
    DASHBOARD_AVAILABLE,
    RICH_AVAILABLE,
)


class MetricsTableCallback(BaseMetricsCallback):
    """GRPO compact table output: step/loss/reward/LR/gpu; silent log-write failures."""

    default_output_dir = "./grpo_output"
    start_banner = "TRAINING STARTED"
    log_every_write = True  # Write JSONL every on_log.
    log_write_swallow_errors = True  # GRPO swallows file-write exceptions.
    print_checkpoint_on_save = False  # GRPO does not print checkpoint lines.
    print_completion_banner = False  # GRPO does not print completion banner in MetricsTable.
    interval_key_name = "interval_seconds"  # GRPO's original schema uses this key.
    # Pre-refactor GRPO built `entry = dict(logs); entry[k]=v; entry.update(cap)`,
    # so our fields + capacity override logs on key collisions.
    fields_win_on_collision = True

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./grpo_output",
        previous_log_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            log_every_n_steps=log_every_n_steps,
            output_dir=output_dir,
            previous_log_entries=previous_log_entries,
        )
        # health_checker defaults to NoOpHealthChecker in BaseMetricsCallback.__init__.

    def _print_header(self):
        print("\n" + "=" * 60)
        print("   Step     |   Loss  | Reward  |    LR     | GPU Mem")
        print("-" * 60)

    def _print_row(self, *, step, state, args, logs, capacity_snapshot, interval_time, samples_per_sec, eta, progress):
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        reward = (
            logs.get("reward")
            or logs.get("rewards")
            or logs.get("rewards/mean")
            or logs.get("mean_reward")
        )
        gpu_mem_value = capacity_snapshot.get("gpu_memory_gb")
        gpu_mem = f"{gpu_mem_value:.1f}GB" if isinstance(gpu_mem_value, (int, float)) else "N/A"

        def fmt(x, default="-"):
            if x is None:
                return default
            try:
                return f"{float(x):.4f}"
            except Exception:
                return str(x)

        lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else (lr if lr is not None else "-")
        print(
            f"{state.global_step:>10} | {fmt(loss):>7} | {fmt(reward):>7} | "
            f"{lr_str:>9} | {gpu_mem:>7}"
        )


class LiveDashboardCallback(BaseLiveDashboardCallback):
    """GRPO live dashboard: reward/reward_std/kl_penalty/advantage with multi-key fallbacks."""

    default_output_dir = "./grpo_output"
    default_title = "GRPO Training"
    training_type_attr = "grpo"
    completion_banner = "GRPO TRAINING COMPLETED"
    log_write_swallow_errors = True  # GRPO swallows file-write exceptions on dashboard path too.

    def _dashboard_metrics(self, logs, capacity_snapshot):
        reward = (
            logs.get("reward")
            or logs.get("rewards")
            or logs.get("rewards/mean")
            or logs.get("mean_reward")
            or 0
        )
        reward_std = logs.get("reward_std") or logs.get("rewards/std") or 0
        kl_penalty = logs.get("kl") or logs.get("kl_penalty") or logs.get("kl_div") or 0
        advantage = logs.get("advantage") or logs.get("advantages/mean") or 0
        return {
            "reward": reward,
            "reward_std": reward_std,
            "kl_penalty": kl_penalty,
            "advantage": advantage,
        }

    def _fallback_row(self, *, state, logs, capacity_snapshot):
        loss = logs.get("loss", 0)
        reward = (
            logs.get("reward")
            or logs.get("rewards")
            or logs.get("rewards/mean")
            or logs.get("mean_reward")
            or 0
        )
        print(f"  Step {state.global_step}/{self.total_steps} | Loss: {loss:.4f} | Reward: {reward:.4f}")


__all__ = [
    "MetricsTableCallback",
    "LiveDashboardCallback",
    "DASHBOARD_AVAILABLE",
    "RICH_AVAILABLE",
]
