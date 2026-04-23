"""Shared base callbacks for sft/kto/grpo trainers.

`BaseMetricsCallback` owns the table-output + JSONL-logging lifecycle.
`BaseLiveDashboardCallback` wraps `shared.ui.LiveDashboard` lifecycle.
Per-trainer subclasses inject strategies (HealthChecker, metric extraction,
row formatting) without overriding `on_train_begin` / `on_train_end`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl

# sys.path bootstrap is handled by Trainers/shared/callbacks/__init__.py
# so `shared.*` imports below resolve whenever this module is loaded via
# `from Trainers.shared.callbacks import ...`.
from shared.training_capacity import (
    capture_runtime_capacity_snapshot,
    reset_capacity_peaks,
)

try:
    from shared.ui import LiveDashboard, RICH_AVAILABLE
    DASHBOARD_AVAILABLE = True
except ImportError:
    LiveDashboard = None  # type: ignore[assignment]
    DASHBOARD_AVAILABLE = False
    RICH_AVAILABLE = False

from .health_checks import HealthChecker, NoOpHealthChecker


def resolve_cloud_provider(args: Any) -> Optional[str]:
    """Resolve cloud provider metadata: env first, then args attribute."""
    cloud_provider = os.environ.get("CLOUD_PROVIDER", "").strip()
    if cloud_provider:
        return cloud_provider
    return getattr(args, "cloud_provider", None)


def _annotate_cloud(capacity_snapshot: Dict[str, Any], args: Any) -> None:
    """Add cloud_provider + cloud_gpu_type to the snapshot in-place."""
    # Invoked from both BaseMetricsCallback.on_log and BaseLiveDashboardCallback.on_log.
    cloud_provider = resolve_cloud_provider(args)
    if cloud_provider:
        capacity_snapshot.setdefault("cloud_provider", cloud_provider)
    cloud_gpu_type = os.environ.get("CLOUD_GPU_TYPE", "").strip()
    if cloud_gpu_type:
        capacity_snapshot.setdefault("cloud_gpu_type", cloud_gpu_type)


def append_final_training_summary(
    log_file: Path,
    *,
    step: int,
    total_steps: int,
    total_epochs: int,
    elapsed: float,
) -> None:
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


def format_time(seconds: float) -> str:
    """Human-readable duration used by SFT/KTO table footers."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _setup_latest_symlink(latest_log: Path, log_file_name: str) -> None:
    """Recreate the training_latest.jsonl symlink; swallow WSL/fs errors."""
    if latest_log.exists():
        try:
            latest_log.unlink()
        except Exception:
            pass
    try:
        latest_log.symlink_to(log_file_name)
    except (OSError, NotImplementedError):
        pass


def _prepopulate_log_file(log_file: Path, previous_entries: List[Dict[str, Any]]) -> None:
    with open(log_file, "w") as f:
        for entry in previous_entries:
            f.write(json.dumps(entry) + "\n")


class BaseMetricsCallback(TrainerCallback):
    """Table-output + JSONL-logging lifecycle. Subclasses inject row format & health strategy.

    Subclass contract:
      - set `default_output_dir`, `training_type_label`, optionally
        `log_every_write`, `log_write_swallow_errors`.
      - set `self.health_checker` in `__init__` (default: NoOpHealthChecker).
      - override `_print_header()` and `_print_row(...)`.
    """

    default_output_dir: str = "./output"
    start_banner: str = "TRAINING STARTED"
    completion_banner: str = "TRAINING COMPLETED"
    log_every_write: bool = False
    log_write_swallow_errors: bool = False
    print_checkpoint_on_save: bool = True
    print_completion_banner: bool = True
    interval_key_name: str = "interval_time"  # GRPO overrides to "interval_seconds"
    # Pre-refactor SFT gated the ENTIRE on_log body on the interval multiple, so the
    # health-check and last_log_time update fired only at printed-row cadence. KTO/GRPO
    # fired them every on_log call. Default True matches KTO/GRPO; SFT overrides to False.
    health_check_every_on_log: bool = True
    interval_time_updates_every_on_log: bool = True
    # Pre-refactor GRPO built the JSONL entry as `dict(logs); entry[k]=v; entry.update(cap)`,
    # where our fields + capacity won over `logs` on key collisions. Pre-refactor SFT and
    # KTO built `{**our_fields, **capacity, **logs}`, where `logs` won. Default False
    # preserves SFT/KTO; GRPO overrides to True.
    fields_win_on_collision: bool = False

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: Optional[str] = None,
        previous_log_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.output_dir = Path(output_dir if output_dir is not None else self.default_output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"training_{timestamp}.jsonl"
        self.latest_log = self.logs_dir / "training_latest.jsonl"

        self.start_time: Optional[datetime] = None
        self.last_log_time: Optional[datetime] = None
        self.header_printed = False
        self.health_checker: HealthChecker = NoOpHealthChecker()

        if previous_log_entries:
            self._prepopulate_log(previous_log_entries)

    def _prepopulate_log(self, previous_entries: List[Dict[str, Any]]):
        print(f"\n✓ Prepopulating log with {len(previous_entries)} entries from previous run")
        _prepopulate_log_file(self.log_file, previous_entries)
        print(f"  Log file: {self.log_file}")
        print(f"  Steps included: 0-{previous_entries[-1]['step'] if previous_entries else 0}\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
        self.header_printed = False
        reset_capacity_peaks(torch)
        _setup_latest_symlink(self.latest_log, self.log_file.name)

        print("\n" + "=" * 100)
        print(self.start_banner)
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
        on_interval = state.global_step % self.log_every_n_steps == 0
        # SFT (override False) only updates last_log_time at interval multiples, so
        # the printed `Time/5s` column measures row-to-row gaps, not on_log-to-on_log.
        # KTO/GRPO (default True) update every call — matches their pre-refactor behavior.
        if self.interval_time_updates_every_on_log or on_interval:
            self.last_log_time = current_time
        elapsed = (current_time - self.start_time).total_seconds() if self.start_time else 0.0
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0.0
        samples_per_sec = (
            state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
        ) / elapsed if elapsed > 0 else 0.0
        capacity_snapshot = capture_runtime_capacity_snapshot(torch)
        _annotate_cloud(capacity_snapshot, args)

        should_write_jsonl = self.log_every_write or on_interval
        if should_write_jsonl:
            self._write_log_row(
                logs=logs,
                step=state.global_step,
                interval_time=interval_time,
                elapsed=elapsed,
                steps_per_sec=steps_per_sec,
                samples_per_sec=samples_per_sec,
                capacity_snapshot=capacity_snapshot,
                current_time=current_time,
            )

        # SFT (override False) only runs health checks at interval multiples — matches
        # pre-refactor top-of-on_log early return. KTO/GRPO (default True) every call.
        if self.health_check_every_on_log or on_interval:
            self.health_checker.check(logs, state.global_step, args.max_grad_norm)

        if not on_interval:
            return

        if not self.header_printed or state.global_step % (self.log_every_n_steps * 20) == 0:
            self._print_header()
            self.header_printed = True

        eta, progress = self._compute_eta_progress(state, steps_per_sec)
        self._print_row(
            step=state.global_step,
            state=state,
            args=args,
            logs=logs,
            capacity_snapshot=capacity_snapshot,
            interval_time=interval_time,
            samples_per_sec=samples_per_sec,
            eta=eta,
            progress=progress,
        )

    def _write_log_row(
        self,
        *,
        logs: Dict[str, Any],
        step: int,
        interval_time: float,
        elapsed: float,
        steps_per_sec: float,
        samples_per_sec: float,
        capacity_snapshot: Dict[str, Any],
        current_time: datetime,
    ):
        our_fields = {
            "step": int(step),
            "timestamp": current_time.isoformat(),
            self.interval_key_name: interval_time,
            "elapsed_seconds": round(elapsed, 3),
            "steps_per_second": round(steps_per_sec, 3),
            "samples_per_sec": round(samples_per_sec, 3),
        }
        # GRPO (override True): logs is the base, our fields + capacity override — matches
        # pre-refactor `entry = dict(logs); entry[k]=v; entry.update(cap)`.
        # SFT/KTO (default False): our fields first, logs wins on collision — matches
        # pre-refactor `{..our_fields, **capacity, **logs}`.
        if self.fields_win_on_collision:
            entry = {**logs, **our_fields, **capacity_snapshot}
        else:
            entry = {**our_fields, **capacity_snapshot, **logs}
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            if not self.log_write_swallow_errors:
                raise

    def _compute_eta_progress(self, state, steps_per_sec):
        if state.max_steps > 0:
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            return format_time(eta_seconds), f"{state.global_step}/{state.max_steps}"
        return "N/A", f"{state.global_step}"

    def on_save(self, args, state, control, **kwargs):
        if not self.print_checkpoint_on_save:
            return
        print("-" * 100)
        print(f">> CHECKPOINT SAVED at step {state.global_step:,} -> {args.output_dir}/checkpoint-{state.global_step}")
        print("-" * 100)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
        append_final_training_summary(
            self.log_file,
            step=state.global_step,
            total_steps=state.max_steps if state.max_steps > 0 else state.global_step,
            total_epochs=int(state.epoch or 0),
            elapsed=elapsed,
        )
        if not self.print_completion_banner:
            return
        print("\n" + "=" * 100)
        print(self.completion_banner)
        print("=" * 100)
        print(f"Total time: {format_time(elapsed)}")
        print(f"Total steps: {state.global_step:,}")
        if elapsed > 0:
            print(f"Average speed: {state.global_step / elapsed:.2f} steps/sec")
        print("=" * 100 + "\n")

    # Subclass hooks ---------------------------------------------------

    def _print_header(self):  # pragma: no cover - abstract-ish
        raise NotImplementedError

    def _print_row(
        self,
        *,
        step: int,
        state: TrainerState,
        args: Any,
        logs: Dict[str, Any],
        capacity_snapshot: Dict[str, Any],
        interval_time: float,
        samples_per_sec: float,
        eta: str,
        progress: str,
    ):  # pragma: no cover - abstract-ish
        raise NotImplementedError


class BaseLiveDashboardCallback(TrainerCallback):
    """Wraps `shared.ui.LiveDashboard`; subclasses inject per-trainer metric extraction.

    Subclass contract:
      - set `default_output_dir`, `default_title`, `training_type_attr`,
        `completion_banner`.
      - override `_dashboard_metrics(logs, capacity_snapshot)` to return kwargs
        for `dashboard.update(...)`.
      - override `_fallback_row(...)` to print when the dashboard is unavailable.
    """

    default_output_dir: str = "./output"
    default_title: str = "Training"
    training_type_attr: str = "sft"
    completion_banner: str = "TRAINING COMPLETED"
    log_write_swallow_errors: bool = False  # GRPO overrides True; mirrors BaseMetricsCallback.

    def __init__(
        self,
        log_every_n_steps: int = 5,
        output_dir: Optional[str] = None,
        previous_log_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.output_dir = Path(output_dir if output_dir is not None else self.default_output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"training_{timestamp}.jsonl"
        self.latest_log = self.logs_dir / "training_latest.jsonl"

        self.start_time: Optional[datetime] = None
        self.dashboard: Optional[Any] = None
        self.total_steps = 0
        self.total_epochs = 1  # Sentinel; overwritten by on_train_begin from args.num_train_epochs.

        if previous_log_entries:
            _prepopulate_log_file(self.log_file, previous_log_entries)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.total_steps = state.max_steps if state.max_steps > 0 else 1000
        self.total_epochs = args.num_train_epochs
        reset_capacity_peaks(torch)
        _setup_latest_symlink(self.latest_log, self.log_file.name)

        if DASHBOARD_AVAILABLE and RICH_AVAILABLE:
            self.dashboard = LiveDashboard(
                title=self.default_title,
                total_epochs=int(self.total_epochs),
                total_steps=self.total_steps,
                training_type=self.training_type_attr,
                show_sparklines=True,
                log_lines=3,
            )
            self.dashboard.__enter__()
        else:
            print(f"\n{'=' * 60}")
            print(f"{self.training_type_attr.upper()} TRAINING STARTED")
            print(f"{'=' * 60}")
            print(f"Log file: {self.log_file}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        if logs is None:
            return
        if state.global_step % self.log_every_n_steps != 0:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0.0
        samples_per_sec = (
            state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
        ) / elapsed if elapsed > 0 else 0.0
        capacity_snapshot = capture_runtime_capacity_snapshot(torch)
        _annotate_cloud(capacity_snapshot, args)

        log_entry = {
            "step": state.global_step,
            "timestamp": datetime.now().isoformat(),
            "total_steps": self.total_steps,
            "total_epochs": int(self.total_epochs),
            "elapsed_seconds": round(elapsed, 3),
            "steps_per_second": round(steps_per_sec, 3),
            "samples_per_sec": round(samples_per_sec, 3),
            **capacity_snapshot,
            **logs,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            if not self.log_write_swallow_errors:
                raise

        if self.dashboard:
            epoch = float(logs.get("epoch", 0.0) or 0.0)
            metrics = self._dashboard_metrics(logs, capacity_snapshot)
            self.dashboard.update(
                step=state.global_step,
                epoch=epoch,
                loss=logs.get("loss"),
                learning_rate=logs.get("learning_rate"),
                gpu_memory_gb=capacity_snapshot.get("gpu_memory_gb"),
                **metrics,
            )
        else:
            self._fallback_row(state=state, logs=logs, capacity_snapshot=capacity_snapshot)

    def on_save(self, args, state, control, **kwargs):
        if self.dashboard:
            self.dashboard.update(log_message=f"Checkpoint saved at step {state.global_step}")
        else:
            print(f"  >> Checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.dashboard:
            self.dashboard.__exit__(None, None, None)
            self.dashboard = None

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
        append_final_training_summary(
            self.log_file,
            step=state.global_step,
            total_steps=self.total_steps,
            total_epochs=int(self.total_epochs),
            elapsed=elapsed,
        )
        print(f"\n{'=' * 60}")
        print(self.completion_banner)
        print(f"{'=' * 60}")
        print(f"Total time: {elapsed // 3600:.0f}h {(elapsed % 3600) // 60:.0f}m {elapsed % 60:.0f}s")
        print(f"Total steps: {state.global_step:,}")
        print(f"Log file: {self.log_file}")

    # Subclass hooks ---------------------------------------------------

    def _dashboard_metrics(self, logs: Dict[str, Any], capacity_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Return extra kwargs for `dashboard.update(...)` (step/epoch/loss/lr/gpu are always passed)."""
        return {}

    def _fallback_row(self, *, state: TrainerState, logs: Dict[str, Any], capacity_snapshot: Dict[str, Any]):
        """Print a single-line fallback when the rich dashboard is unavailable."""
        loss = logs.get("loss", 0)
        lr = logs.get("learning_rate", 0)
        gpu_mem_value = capacity_snapshot.get("gpu_memory_gb")
        gpu_mem = f"{gpu_mem_value:.1f}GB" if isinstance(gpu_mem_value, (int, float)) else "N/A"
        print(f"  Step {state.global_step}/{self.total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | GPU: {gpu_mem}")
