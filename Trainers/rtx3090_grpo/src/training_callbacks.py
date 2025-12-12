#!/usr/bin/env python3
"""
Custom training callbacks for GRPO/GSPO.
Saves training metrics as JSONL and prints a compact table periodically.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState


class MetricsTableCallback(TrainerCallback):
    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./grpo_output_rtx3090",
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

        entry = dict(logs)
        entry["step"] = int(state.global_step)
        entry["timestamp"] = current_time.isoformat()
        entry["interval_seconds"] = interval_time

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

        gpu_mem = "N/A"
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_reserved() / 1e9:.1f}GB"

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
