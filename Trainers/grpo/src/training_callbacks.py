#!/usr/bin/env python3
"""
Custom training callbacks for GRPO/GSPO.
Saves training metrics as JSONL and prints a compact table periodically.
"""

from __future__ import annotations

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState

# Add shared to path for UI components
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.training_capacity import capture_runtime_capacity_snapshot, reset_capacity_peaks

# Try to import LiveDashboard
try:
    from shared.ui import LiveDashboard, RICH_AVAILABLE
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    RICH_AVAILABLE = False


class MetricsTableCallback(TrainerCallback):
    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./grpo_output",
    ):
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"training_{timestamp}.jsonl"
        self.latest_log = self.logs_dir / "training_latest.jsonl"

        self.start_time: Optional[datetime] = None
        self.last_log_time: Optional[datetime] = None
        self.header_printed = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
        self.header_printed = False
        reset_capacity_peaks(torch)

        if self.latest_log.exists():
            try:
                self.latest_log.unlink()
            except Exception:
                pass
        try:
            self.latest_log.symlink_to(self.log_file.name)
        except Exception:
            pass

        print("\n" + "=" * 100)
        print("TRAINING STARTED")
        print("=" * 100)
        print(f"Detailed metrics logging to: {self.log_file}")
        print(f"View in real-time: tail -f {self.log_file}")
        print(f"Or use latest: tail -f {self.latest_log}")
        print("=" * 100)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        if not logs:
            return

        current_time = datetime.now()
        interval_time = (current_time - self.last_log_time).total_seconds() if self.last_log_time else 0.0
        self.last_log_time = current_time
        elapsed = (current_time - self.start_time).total_seconds() if self.start_time else 0.0
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0.0
        samples_per_sec = (
            state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
        ) / elapsed if elapsed > 0 else 0.0
        capacity_snapshot = capture_runtime_capacity_snapshot(torch)
        cloud_provider = os.environ.get("CLOUD_PROVIDER", "").strip()
        if cloud_provider:
            capacity_snapshot.setdefault("cloud_provider", cloud_provider)
        cloud_gpu_type = os.environ.get("CLOUD_GPU_TYPE", "").strip()
        if cloud_gpu_type:
            capacity_snapshot.setdefault("cloud_gpu_type", cloud_gpu_type)

        entry = dict(logs)
        entry["step"] = int(state.global_step)
        entry["timestamp"] = current_time.isoformat()
        entry["interval_seconds"] = interval_time
        entry["elapsed_seconds"] = round(elapsed, 3)
        entry["steps_per_second"] = round(steps_per_sec, 3)
        entry["samples_per_sec"] = round(samples_per_sec, 3)
        entry.update(capacity_snapshot)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        if state.global_step % self.log_every_n_steps != 0:
            return

        if not self.header_printed or state.global_step % (self.log_every_n_steps * 20) == 0:
            self._print_header()
            self.header_printed = True

        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        reward = logs.get("reward") or logs.get("rewards") or logs.get("rewards/mean") or logs.get("mean_reward")

        gpu_mem_value = capacity_snapshot.get("gpu_memory_gb")
        gpu_mem = f"{gpu_mem_value:.1f}GB" if isinstance(gpu_mem_value, (int, float)) else "N/A"

        def fmt(x, default="-"):
            if x is None:
                return default
            try:
                return f"{float(x):.4f}"
            except Exception:
                return str(x)

        print(
            f"{state.global_step:>10} | {fmt(loss):>7} | {fmt(reward):>7} | "
            f"{(f'{lr:.2e}' if isinstance(lr, (int, float)) else (lr if lr is not None else '-')):>9} | "
            f"{gpu_mem:>7}"
        )

    def _print_header(self):
        print("\n" + "=" * 60)
        print("   Step     |   Loss  | Reward  |    LR     | GPU Mem")
        print("-" * 60)


class LiveDashboardCallback(TrainerCallback):
    """
    Training callback that displays a live dashboard with real-time metrics.
    Specialized for GRPO training - shows rewards, KL penalty, advantages.
    """

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./grpo_output",
        previous_log_entries: list = None
    ):
        self.log_every_n_steps = log_every_n_steps
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"training_{timestamp}.jsonl"
        self.latest_log = self.logs_dir / "training_latest.jsonl"

        self.start_time: Optional[datetime] = None
        self.dashboard: Optional[LiveDashboard] = None
        self.total_steps = 0
        self.total_epochs = 1

        if previous_log_entries:
            self._prepopulate_log(previous_log_entries)

    def _prepopulate_log(self, previous_entries: list):
        with open(self.log_file, "w") as f:
            for entry in previous_entries:
                f.write(json.dumps(entry) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.total_steps = state.max_steps if state.max_steps > 0 else 1000
        self.total_epochs = args.num_train_epochs
        reset_capacity_peaks(torch)

        if self.latest_log.exists():
            try:
                self.latest_log.unlink()
            except Exception:
                pass
        try:
            self.latest_log.symlink_to(self.log_file.name)
        except Exception:
            pass

        if DASHBOARD_AVAILABLE and RICH_AVAILABLE:
            self.dashboard = LiveDashboard(
                title="GRPO Training",
                total_epochs=int(self.total_epochs),
                total_steps=self.total_steps,
                training_type="grpo",  # This enables GRPO-specific display
                show_sparklines=True,
                log_lines=3,
            )
            self.dashboard.__enter__()
        else:
            print(f"\n{'=' * 60}")
            print("GRPO TRAINING STARTED")
            print(f"{'=' * 60}")
            print(f"Log file: {self.log_file}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        if not logs:
            return

        if state.global_step % self.log_every_n_steps != 0:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0.0
        samples_per_sec = (
            state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
        ) / elapsed if elapsed > 0 else 0.0
        capacity_snapshot = capture_runtime_capacity_snapshot(torch)
        cloud_provider = os.environ.get("CLOUD_PROVIDER", "").strip()
        if cloud_provider:
            capacity_snapshot.setdefault("cloud_provider", cloud_provider)
        cloud_gpu_type = os.environ.get("CLOUD_GPU_TYPE", "").strip()
        if cloud_gpu_type:
            capacity_snapshot.setdefault("cloud_gpu_type", cloud_gpu_type)

        # Save to log file
        log_entry = {
            "step": state.global_step,
            "timestamp": datetime.now().isoformat(),
            "total_steps": self.total_steps,
            "total_epochs": int(self.total_epochs),
            "elapsed_seconds": round(elapsed, 3),
            "steps_per_second": round(steps_per_sec, 3),
            "samples_per_sec": round(samples_per_sec, 3),
            **capacity_snapshot,
            **logs
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Extract GRPO-specific metrics (handle various key names)
        reward = logs.get("reward") or logs.get("rewards") or logs.get("rewards/mean") or logs.get("mean_reward") or 0
        reward_std = logs.get("reward_std") or logs.get("rewards/std") or 0
        kl_penalty = logs.get("kl") or logs.get("kl_penalty") or logs.get("kl_div") or 0
        advantage = logs.get("advantage") or logs.get("advantages/mean") or 0

        # Update dashboard with GRPO-specific metrics
        if self.dashboard:
            self.dashboard.update(
                step=state.global_step,
                epoch=int(logs.get('epoch', 0)),
                loss=logs.get('loss'),
                learning_rate=logs.get('learning_rate'),
                # GRPO-specific metrics
                reward=reward,
                reward_std=reward_std,
                kl_penalty=kl_penalty,
                advantage=advantage,
                gpu_memory_gb=capacity_snapshot.get("gpu_memory_gb"),
            )
        else:
            loss = logs.get('loss', 0)
            print(f"  Step {state.global_step}/{self.total_steps} | Loss: {loss:.4f} | Reward: {reward:.4f}")

    def on_save(self, args, state, control, **kwargs):
        if self.dashboard:
            self.dashboard.update(log_message=f"Checkpoint saved at step {state.global_step}")
        else:
            print(f"  >> Checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.dashboard:
            self.dashboard.__exit__(None, None, None)
            self.dashboard = None

        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{'=' * 60}")
        print("GRPO TRAINING COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total time: {elapsed // 3600:.0f}h {(elapsed % 3600) // 60:.0f}m {elapsed % 60:.0f}s")
        print(f"Total steps: {state.global_step:,}")
        print(f"Log file: {self.log_file}")
