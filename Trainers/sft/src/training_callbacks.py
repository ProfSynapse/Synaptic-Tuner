#!/usr/bin/env python3
"""
Custom training callbacks for SFT fine-tuning.
Provides real-time metrics tracking and pretty table output.
"""

from transformers import TrainerCallback, TrainerState, TrainerControl
import torch
from datetime import datetime
from typing import Dict, Any, Optional
import json
import sys
import logging
import os
from pathlib import Path

# Add shared to path for UI components
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.training_capacity import capture_runtime_capacity_snapshot, reset_capacity_peaks

# Try to import LiveDashboard
try:
    from shared.ui import LiveDashboard, suppress_logs, RICH_AVAILABLE
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    RICH_AVAILABLE = False


def _append_final_training_summary(log_file: Path, *, step: int, total_steps: int, total_epochs: int, elapsed: float) -> None:
    """Persist a final summary row so bucketed logs retain runtime + peak capacity."""
    capacity_snapshot = capture_runtime_capacity_snapshot(torch)
    entry = {
        "event": "train_end",
        "step": int(step),
        "timestamp": datetime.now().isoformat(),
        "total_steps": int(total_steps),
        "total_epochs": int(total_epochs),
        "train_runtime": round(elapsed, 3),
        "train_steps_per_second": round((step / elapsed), 3) if elapsed > 0 else 0.0,
        "train_samples_per_second": None,
        **capacity_snapshot,
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _resolve_cloud_provider(args: Any) -> Optional[str]:
    """Resolve cloud provider metadata without assuming trainer args carry custom CLI fields."""
    cloud_provider = os.environ.get("CLOUD_PROVIDER", "").strip()
    if cloud_provider:
        return cloud_provider
    return getattr(args, "cloud_provider", None)


class MetricsTableCallback(TrainerCallback):
    """
    Custom callback that prints training metrics in a nice table format.
    Shows metrics every N steps to track training progress.
    """

    def __init__(self, log_every_n_steps: int = 5, output_dir: str = "./sft_output",
                 previous_log_entries: list = None):
        """
        Args:
            log_every_n_steps: Print table every N training steps
            output_dir: Directory to save detailed logs
            previous_log_entries: Optional list of log entries from a previous run to prepopulate the log
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
        self.last_log_time = None
        self.step_times = []
        self.header_printed = False

        # Prepopulate log file with previous entries if resuming
        if previous_log_entries:
            self._prepopulate_log(previous_log_entries)

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        self.last_log_time = datetime.now()
        self.header_printed = False
        reset_capacity_peaks(torch)

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

        # Only log at specified intervals
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Calculate time since last log
        current_time = datetime.now()
        interval_time = (current_time - self.last_log_time).total_seconds() if self.last_log_time else 0
        self.last_log_time = current_time

        elapsed = (current_time - self.start_time).total_seconds()
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
        samples_per_sec = (state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps) / elapsed if elapsed > 0 else 0
        capacity_snapshot = capture_runtime_capacity_snapshot(torch)
        cloud_provider = _resolve_cloud_provider(args)
        if cloud_provider:
            capacity_snapshot.setdefault("cloud_provider", cloud_provider)
        cloud_gpu_type = os.environ.get("CLOUD_GPU_TYPE", "").strip()
        if cloud_gpu_type:
            capacity_snapshot.setdefault("cloud_gpu_type", cloud_gpu_type)

        # Save full metrics to file (every log_every_n_steps)
        self._save_metrics_to_file(
            logs,
            state.global_step,
            interval_time,
            elapsed,
            steps_per_sec,
            samples_per_sec,
            capacity_snapshot,
        )

        # Check training health and warn if needed
        self._check_training_health(logs, state.global_step, args.max_grad_norm)

        # Print header every 20 rows for readability
        if not self.header_printed or state.global_step % (self.log_every_n_steps * 20) == 0:
            self._print_header()
            self.header_printed = True

        gpu_mem_value = capacity_snapshot.get("gpu_memory_gb")
        gpu_mem = f"{gpu_mem_value:.1f}GB" if isinstance(gpu_mem_value, (int, float)) else "N/A"

        # Extract metrics from logs
        loss = logs.get('loss', 0.0)
        learning_rate = logs.get('learning_rate', 0.0)
        grad_norm = logs.get('grad_norm', 0.0)
        epoch = logs.get('epoch', 0.0)

        # Calculate ETA
        if state.max_steps > 0:
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta = self._format_time(eta_seconds)
            progress = f"{state.global_step}/{state.max_steps}"
        else:
            eta = "N/A"
            progress = f"{state.global_step}"

        # Print table row (SFT-specific metrics)
        print(f" {progress:>12} | {loss:>8.4f} | {learning_rate:>9.2e} | "
              f"{grad_norm:>8.3f} | {epoch:>6.2f} | {gpu_mem:>8} | "
              f"{interval_time:>7.1f}s | {samples_per_sec:>8.1f} | {eta:>9} ")

    def _save_metrics_to_file(
        self,
        logs: Dict[str, Any],
        step: int,
        interval_time: float,
        elapsed: float,
        steps_per_sec: float,
        samples_per_sec: float,
        capacity_snapshot: Dict[str, Any],
    ):
        """Save detailed metrics to JSONL file."""
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "interval_time": interval_time,
            "elapsed_seconds": round(elapsed, 3),
            "steps_per_second": round(steps_per_sec, 3),
            "samples_per_sec": round(samples_per_sec, 3),
            **capacity_snapshot,
            **logs
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _prepopulate_log(self, previous_entries: list):
        """Prepopulate log file with entries from a previous run.

        Args:
            previous_entries: List of dict entries to write to the log file
        """
        print(f"\n✓ Prepopulating log with {len(previous_entries)} entries from previous run")

        with open(self.log_file, "w") as f:
            for entry in previous_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"  Log file: {self.log_file}")
        print(f"  Steps included: 0-{previous_entries[-1]['step'] if previous_entries else 0}\n")

    def _check_training_health(self, logs: Dict[str, Any], step: int, max_grad_norm: float = None):
        """Check if training metrics are healthy and warn if not (SFT-specific checks)."""
        warnings = []

        # Check for NaN or Inf
        loss = logs.get('loss', 0.0)
        if not (0 < loss < 100):  # Loss should be positive and reasonable
            warnings.append(f"⚠ Unusual loss value: {loss:.4f}")

        # Check gradient norm (show both raw and clipped values)
        grad_norm = logs.get('grad_norm', 0.0)
        if grad_norm > 100.0:
            if max_grad_norm is not None:
                clipped_norm = min(grad_norm, max_grad_norm)
                warnings.append(
                    f"⚠ High gradient norm: {grad_norm:.2f} → {clipped_norm:.2f} (clipped)"
                )
            else:
                warnings.append(f"⚠ High gradient norm: {grad_norm:.2f} (may cause instability)")

        # Check if loss is not decreasing (compare to first few steps)
        if step > 50:  # After warm-up
            # This is a basic check - in production you'd track loss history
            if loss > 2.0:  # Still high after 50 steps
                warnings.append(f"⚠ Loss remains high after {step} steps: {loss:.4f} (may need longer training or LR adjustment)")

        # Print warnings if any
        if warnings:
            print("\n" + "!" * 100)
            for warning in warnings:
                print(f"  {warning}")
            if max_grad_norm is None or grad_norm > max_grad_norm * 10:
                print("  Consider: reducing learning rate or using tighter gradient clipping")
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
        _append_final_training_summary(
            self.log_file,
            step=state.global_step,
            total_steps=state.max_steps if state.max_steps > 0 else state.global_step,
            total_epochs=int(state.epoch or 0),
            elapsed=elapsed,
        )
        print("\n" + "=" * 100)
        print("TRAINING COMPLETED")
        print("=" * 100)
        print(f"Total time: {self._format_time(elapsed)}")
        print(f"Total steps: {state.global_step:,}")
        print(f"Average speed: {state.global_step / elapsed:.2f} steps/sec")
        print("=" * 100 + "\n")

    def _print_header(self):
        """Print the table header."""
        print("\n" + "=" * 110)
        print(" " * 47 + "TRAINING METRICS")
        print("=" * 110)
        print("   Step      |   Loss   |    LR     | GradNorm | Epoch  | GPU Mem  | Time/5s | Samp/sec |    ETA    ")
        print("-" * 110)

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


class TwoStageLRCallback(TrainerCallback):
    """
    Custom callback that implements a two-stage learning rate schedule.

    This reduces the learning rate at a specified step to prevent optimization
    instability while maintaining fast early learning.

    Example:
        Steps 1-50:  LR = 5e-7 (fast early learning)
        Steps 51+:   LR = 2.5e-7 (reduced to prevent overshoot/instability)
    """

    def __init__(self, initial_lr: float, reduced_lr: float, reduction_step: int):
        """
        Args:
            initial_lr: Learning rate for early training (e.g., 5e-7)
            reduced_lr: Reduced learning rate after reduction_step (e.g., 2.5e-7)
            reduction_step: Step at which to reduce LR (e.g., 50)
        """
        self.initial_lr = initial_lr
        self.reduced_lr = reduced_lr
        self.reduction_step = reduction_step
        self.lr_reduced = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Display two-stage LR schedule configuration."""
        print("\n" + "=" * 100)
        print("TWO-STAGE LEARNING RATE SCHEDULE")
        print("=" * 100)
        print(f"  Steps 1-{self.reduction_step}:  LR = {self.initial_lr:.2e} (fast early learning)")
        print(f"  Steps {self.reduction_step+1}+:     LR = {self.reduced_lr:.2e} ({(self.reduced_lr/self.initial_lr)*100:.0f}% of initial, prevents instability)")
        print(f"  Reduction ratio: {self.reduced_lr/self.initial_lr:.1%}")
        print("=" * 100 + "\n")

    def on_step_begin(self, args, state, control, **kwargs):
        """Check if we need to reduce learning rate at this step."""
        if state.global_step == self.reduction_step and not self.lr_reduced:
            # Reduce learning rate for all parameter groups
            optimizer = kwargs.get('optimizer')
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.reduced_lr

                self.lr_reduced = True

                print("\n" + "!" * 100)
                print(f"🔧 LEARNING RATE REDUCED at step {state.global_step}")
                print(f"   {self.initial_lr:.2e} → {self.reduced_lr:.2e} (50% reduction)")
                print(f"   Reason: Preemptive intervention before step 60 instability zone")
                print("!" * 100 + "\n")


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


class LiveDashboardCallback(TrainerCallback):
    """
    Training callback that displays a live dashboard with real-time metrics.

    Uses the shared.ui LiveDashboard for a rich, animated display.
    Falls back to MetricsTableCallback if dashboard is not available.
    """

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: str = "./sft_output",
        training_type: str = "sft",
        previous_log_entries: list = None
    ):
        """
        Args:
            log_every_n_steps: Update dashboard every N training steps
            output_dir: Directory to save detailed logs
            training_type: 'sft' or 'kto' (affects displayed metrics)
            previous_log_entries: Optional list of log entries from a previous run
        """
        self.log_every_n_steps = log_every_n_steps
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.training_type = training_type

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"training_{timestamp}.jsonl"
        self.latest_log = self.logs_dir / "training_latest.jsonl"

        self.start_time = None
        self.dashboard: Optional[LiveDashboard] = None
        self.total_steps = 0
        self.total_epochs = 1

        # Prepopulate if resuming
        if previous_log_entries:
            self._prepopulate_log(previous_log_entries)

    def _prepopulate_log(self, previous_entries: list):
        """Prepopulate log file with entries from a previous run."""
        with open(self.log_file, "w") as f:
            for entry in previous_entries:
                f.write(json.dumps(entry) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        """Start the live dashboard."""
        self.start_time = datetime.now()
        self.total_steps = state.max_steps if state.max_steps > 0 else 1000
        self.total_epochs = args.num_train_epochs
        reset_capacity_peaks(torch)

        # Create symlink to latest log
        if self.latest_log.exists():
            self.latest_log.unlink()
        try:
            self.latest_log.symlink_to(self.log_file.name)
        except (OSError, NotImplementedError):
            pass

        # Start dashboard if available
        if DASHBOARD_AVAILABLE and RICH_AVAILABLE:
            self.dashboard = LiveDashboard(
                title=f"{self.training_type.upper()} Training",
                total_epochs=int(self.total_epochs),
                total_steps=self.total_steps,
                training_type=self.training_type,
                show_sparklines=True,
                log_lines=3,
            )
            self.dashboard.__enter__()
        else:
            # Fallback: print basic info
            print(f"\n{'=' * 60}")
            print(f"TRAINING STARTED - {self.training_type.upper()}")
            print(f"{'=' * 60}")
            print(f"Log file: {self.log_file}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        """Update the dashboard with new metrics."""
        if logs is None:
            return

        # Only update at specified intervals
        if state.global_step % self.log_every_n_steps != 0:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0.0
        samples_per_sec = (
            state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
        ) / elapsed if elapsed > 0 else 0.0
        capacity_snapshot = capture_runtime_capacity_snapshot(torch)
        cloud_provider = _resolve_cloud_provider(args)
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
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update dashboard
        if self.dashboard:
            self.dashboard.update(
                step=state.global_step,
                epoch=int(logs.get('epoch', 0)),
                loss=logs.get('loss'),
                learning_rate=logs.get('learning_rate'),
                kl=logs.get('kl'),
                margin=logs.get('rewards/margins'),
                gpu_memory_gb=capacity_snapshot.get("gpu_memory_gb"),
            )
        else:
            # Fallback: print step info
            loss = logs.get('loss', 0)
            lr = logs.get('learning_rate', 0)
            gpu_mem_value = capacity_snapshot.get("gpu_memory_gb")
            gpu_mem = f"{gpu_mem_value:.1f}GB" if isinstance(gpu_mem_value, (int, float)) else "N/A"
            print(f"  Step {state.global_step}/{self.total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | GPU: {gpu_mem}")

    def on_save(self, args, state, control, **kwargs):
        """Log checkpoint save."""
        if self.dashboard:
            self.dashboard.update(log_message=f"Checkpoint saved at step {state.global_step}")
        else:
            print(f"  >> Checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        """Stop the dashboard."""
        if self.dashboard:
            self.dashboard.__exit__(None, None, None)
            self.dashboard = None

        elapsed = (datetime.now() - self.start_time).total_seconds()
        _append_final_training_summary(
            self.log_file,
            step=state.global_step,
            total_steps=self.total_steps,
            total_epochs=int(self.total_epochs),
            elapsed=elapsed,
        )
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total time: {elapsed // 3600:.0f}h {(elapsed % 3600) // 60:.0f}m {elapsed % 60:.0f}s")
        print(f"Total steps: {state.global_step:,}")
        print(f"Log file: {self.log_file}")


def suppress_training_logs():
    """
    Suppress verbose logs from training libraries.

    Call this before importing unsloth/transformers to quiet the output.
    Returns a context manager that can be used to restore logging.
    """
    # Suppress common noisy loggers
    noisy_loggers = [
        'unsloth',
        'transformers',
        'datasets',
        'accelerate',
        'trl',
        'peft',
        'bitsandbytes',
        'torch',
        'huggingface_hub',
    ]

    if DASHBOARD_AVAILABLE:
        return suppress_logs(noisy_loggers, level=logging.WARNING)
    else:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()
