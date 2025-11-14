#!/usr/bin/env python3
"""
Custom training callbacks for KTO fine-tuning.
Provides real-time metrics tracking and pretty table output.
"""

from transformers import TrainerCallback, TrainerState, TrainerControl
import torch
from datetime import datetime
from typing import Dict, Any
import json
from pathlib import Path


class MetricsTableCallback(TrainerCallback):
    """
    Custom callback that prints training metrics in a nice table format.
    Shows metrics every N steps to track training progress.
    """

    def __init__(self, log_every_n_steps: int = 5, output_dir: str = "./kto_output_rtx3090"):
        """
        Args:
            log_every_n_steps: Print table every N training steps
            output_dir: Directory to save detailed logs
        """
        self.log_every_n_steps = log_every_n_steps
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"training_{timestamp}.jsonl"

        # Also create a symlink to "latest" for easy access
        self.latest_log = self.logs_dir / "training_latest.jsonl"

        self.start_time = None
        self.step_times = []
        self.header_printed = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        self.header_printed = False

        # Create symlink to latest log for easy access
        if self.latest_log.exists():
            self.latest_log.unlink()
        try:
            self.latest_log.symlink_to(self.log_file.name)
        except (OSError, NotImplementedError):
            # Symlinks might not work on all filesystems (like WSL sometimes)
            # Just skip the symlink in that case
            pass

        print("\n" + "=" * 100)
        print("TRAINING STARTED")
        print("=" * 100)
        print(f"Detailed metrics logging to: {self.log_file}")
        print(f"View in real-time: tail -f {self.log_file}")
        print(f"Or use latest: tail -f {self.latest_log}")
        print("=" * 100)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        """Called when logging occurs."""
        if logs is None:
            return

        # Save full metrics to file (every step)
        self._save_metrics_to_file(logs, state.global_step)

        # Check training health and warn if needed
        self._check_training_health(logs, state.global_step)

        # Only print table at specified intervals
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Print header every 20 rows for readability
        if not self.header_printed or state.global_step % (self.log_every_n_steps * 20) == 0:
            self._print_header()
            self.header_printed = True

        # Calculate metrics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
        samples_per_sec = (state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps) / elapsed if elapsed > 0 else 0

        # Get GPU memory if available (use reserved memory for accurate total)
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_reserved() / 1e9:.1f}GB"

        # Extract metrics from logs
        loss = logs.get('loss', 0.0)
        learning_rate = logs.get('learning_rate', 0.0)
        kto_chosen = logs.get('rewards/chosen', 0.0)
        kto_rejected = logs.get('rewards/rejected', 0.0)
        kto_margin = logs.get('rewards/margins', 0.0)
        kl_div = logs.get('logps/rejected', 0.0)  # KL divergence approximation

        # Calculate ETA
        if state.max_steps > 0:
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta = self._format_time(eta_seconds)
            progress = f"{state.global_step}/{state.max_steps}"
        else:
            eta = "N/A"
            progress = f"{state.global_step}"

        # Print table row
        print(f" {progress:>12} | {loss:>8.4f} | {learning_rate:>9.2e} | "
              f"{kto_chosen:>6.3f} | {kto_rejected:>6.3f} | {kto_margin:>6.3f} | "
              f"{gpu_mem:>8} | {samples_per_sec:>8.1f} | {eta:>9} ")

    def _save_metrics_to_file(self, logs: Dict[str, Any], step: int):
        """Save detailed metrics to JSONL file."""
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **logs
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _check_training_health(self, logs: Dict[str, Any], step: int):
        """Check if training metrics are healthy and warn if not."""
        warnings = []

        # Check for NaN or Inf
        loss = logs.get('loss', 0.0)
        if not (0 < loss < 100):  # Loss should be positive and reasonable
            warnings.append(f"⚠ Unusual loss value: {loss:.4f}")

        # Check KTO margins (should be positive and increasing over time)
        margin = logs.get('rewards/margins', 0.0)
        if margin < -1.0:  # Very negative margin is bad
            warnings.append(f"⚠ Very negative margin: {margin:.4f} (chosen model may be worse than reference)")

        # Check for reward collapse (both chosen and rejected near zero)
        chosen = logs.get('rewards/chosen', 0.0)
        rejected = logs.get('rewards/rejected', 0.0)
        if abs(chosen) < 0.001 and abs(rejected) < 0.001 and step > 10:
            warnings.append("⚠ Reward collapse detected (both rewards near zero)")

        # Check gradient norm (very high = instability)
        grad_norm = logs.get('grad_norm', 0.0)
        if grad_norm > 100.0:
            warnings.append(f"⚠ High gradient norm: {grad_norm:.2f} (may cause instability)")

        # Print warnings if any
        if warnings:
            print("\n" + "!" * 100)
            for warning in warnings:
                print(f"  {warning}")
            print("  Consider: reducing learning rate or reverting to last checkpoint")
            print("!" * 100 + "\n")

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        print("-" * 100)
        print(f">> CHECKPOINT SAVED at step {state.global_step:,} -> {args.output_dir}/checkpoint-{state.global_step}")
        print("-" * 100)

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        print("=" * 100)
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print("\n" + "=" * 100)
        print("TRAINING COMPLETED")
        print("=" * 100)
        print(f"Total time: {self._format_time(elapsed)}")
        print(f"Total steps: {state.global_step:,}")
        print(f"Average speed: {state.global_step / elapsed:.2f} steps/sec")
        print("=" * 100 + "\n")

    def _print_header(self):
        """Print the table header."""
        print("\n" + "=" * 100)
        print(" " * 42 + "TRAINING METRICS")
        print("=" * 100)
        print("   Step      |   Loss   |    LR     | Chosen | Reject | Margin | GPU Mem  | Samp/sec |    ETA    ")
        print("-" * 100)

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class CheckpointMonitorCallback(TrainerCallback):
    """
    Callback to monitor and display checkpoint information.
    Helps track which checkpoints are being kept/deleted.
    """

    def on_save(self, args, state, control, **kwargs):
        """Called when saving a checkpoint."""
        # This is already handled by MetricsTableCallback
        # but we keep this for extensibility
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        """Display checkpoint configuration at start."""
        print(f"\nCheckpoint Configuration:")
        print(f"  Save every: {args.save_steps} steps")
        print(f"  Keep last: {args.save_total_limit} checkpoints")
        print(f"  Output dir: {args.output_dir}")
        print()
