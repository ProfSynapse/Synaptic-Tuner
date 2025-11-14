#!/usr/bin/env python3
"""
Custom training callbacks for KTO fine-tuning.
Provides real-time metrics tracking and pretty table output.
"""

from transformers import TrainerCallback, TrainerState, TrainerControl
import torch
from datetime import datetime
from typing import Dict, Any


class MetricsTableCallback(TrainerCallback):
    """
    Custom callback that prints training metrics in a nice table format.
    Shows metrics every N steps to track training progress.
    """

    def __init__(self, log_every_n_steps: int = 5):
        """
        Args:
            log_every_n_steps: Print table every N training steps
        """
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None
        self.step_times = []
        self.header_printed = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        self.header_printed = False
        print("\n" + "=" * 100)
        print("TRAINING STARTED")
        print("=" * 100)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        """Called when logging occurs."""
        if logs is None:
            return

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

        # Get GPU memory if available
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"

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
        print(f"│ {progress:>12} │ {loss:>8.4f} │ {learning_rate:>9.2e} │ "
              f"{kto_chosen:>7.3f} │ {kto_rejected:>7.3f} │ {kto_margin:>7.3f} │ "
              f"{gpu_mem:>8} │ {samples_per_sec:>8.1f} │ {eta:>10} │")

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        print("├" + "─" * 98 + "┤")
        print(f"│ 💾 CHECKPOINT SAVED at step {state.global_step:,} → {args.output_dir}/checkpoint-{state.global_step}" + " " * (98 - len(f"CHECKPOINT SAVED at step {state.global_step:,} → {args.output_dir}/checkpoint-{state.global_step}") - 4) + "│")
        print("├" + "─" * 98 + "┤")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        print("└" + "─" * 98 + "┘")
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
        print("\n┌" + "─" * 98 + "┐")
        print("│ " + " " * 40 + "TRAINING METRICS" + " " * 41 + "│")
        print("├" + "─" * 98 + "┤")
        print("│    Step      │   Loss   │    LR     │ Chosen │ Reject │ Margin │ GPU Mem  │ Samp/sec │    ETA     │")
        print("├" + "─" * 98 + "┤")

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
