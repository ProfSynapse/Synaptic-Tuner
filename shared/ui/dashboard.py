"""
Live Training Dashboard for Synaptic Tuner

Real-time display of training metrics, progress, and logs.
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from threading import Thread, Event

from .theme import COLORS, BOX
from .console import console, RICH_AVAILABLE, clear_screen
from .widgets import sparkline


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: float = 0.0
    total_epochs: int = 1
    step: int = 0
    total_steps: int = 1
    loss: float = 0.0
    learning_rate: float = 0.0

    # KTO-specific
    kl: float = 0.0
    margin: float = 0.0  # rewards/margins

    # GRPO-specific
    reward: float = 0.0
    reward_std: float = 0.0
    kl_penalty: float = 0.0
    advantage: float = 0.0

    # GPU memory (current usage only - can overflow to RAM with offloading)
    gpu_memory_gb: float = 0.0

    # History for sparklines (training type determines which to show)
    loss_history: List[float] = field(default_factory=list)
    kl_history: List[float] = field(default_factory=list)
    margin_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)

    # Timing
    start_time: float = field(default_factory=time.time)
    steps_per_second: float = 0.0
    best_loss_seen: Optional[float] = None

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0
        return (self.step / self.total_steps) * 100

    @property
    def elapsed_str(self) -> str:
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def eta_str(self) -> str:
        if self.steps_per_second <= 0 or self.step == 0:
            return "calculating..."
        remaining_steps = self.total_steps - self.step
        remaining_secs = remaining_steps / self.steps_per_second
        hours, remainder = divmod(int(remaining_secs), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def best_loss(self) -> float:
        if self.best_loss_seen is not None:
            return self.best_loss_seen
        if not self.loss_history:
            return self.loss
        return min(self.loss_history)

    @property
    def epoch_progress_str(self) -> str:
        total_epochs = max(1, int(self.total_epochs))
        return f"{self.epoch:.2f}/{total_epochs}"


class LiveDashboard:
    """
    Real-time training dashboard with metrics, progress, and sparklines.

    Usage:
        dashboard = LiveDashboard(
            title="SFT Training",
            total_epochs=3,
            total_steps=1500,
        )

        with dashboard:
            for step in range(1500):
                # Training step...
                dashboard.update(
                    step=step,
                    loss=current_loss,
                    learning_rate=lr,
                )

    Or with log file watching:
        dashboard = LiveDashboard.from_log_file(
            log_path="logs/training_latest.jsonl",
            title="SFT Training",
        )
        dashboard.run()  # Blocks until training completes
    """

    def __init__(
        self,
        title: str = "Training",
        total_epochs: int = 1,
        total_steps: int = 100,
        training_type: str = "sft",  # "sft" or "kto"
        show_sparklines: bool = True,
        log_lines: int = 5,
    ):
        self.title = title
        self.training_type = training_type
        self.show_sparklines = show_sparklines
        self.log_lines = log_lines

        self.metrics = TrainingMetrics(
            total_epochs=total_epochs,
            total_steps=total_steps,
        )

        self.log_buffer: List[str] = []
        self._stop_event = Event()
        self._live = None
        self._last_update_time = 0.0
        self._min_update_interval = 0.25  # Max 4 updates per second
        self._trend_bucket_count = 40
        self._history_counts: Dict[str, List[int]] = {
            "loss_history": [],
            "kl_history": [],
            "margin_history": [],
            "reward_history": [],
        }

    def update(
        self,
        step: int = None,
        epoch: float = None,
        loss: float = None,
        learning_rate: float = None,
        # KTO metrics
        kl: float = None,
        margin: float = None,
        # GRPO metrics
        reward: float = None,
        reward_std: float = None,
        kl_penalty: float = None,
        advantage: float = None,
        # Other
        gpu_memory_gb: float = None,
        log_message: str = None,
    ):
        """Update dashboard metrics."""
        if step is not None:
            elapsed = time.time() - self.metrics.start_time
            if elapsed > 0:
                self.metrics.steps_per_second = step / elapsed
            self.metrics.step = step

        if epoch is not None:
            self.metrics.epoch = epoch

        if loss is not None:
            self.metrics.loss = loss
            self.metrics.best_loss_seen = (
                loss
                if self.metrics.best_loss_seen is None
                else min(self.metrics.best_loss_seen, loss)
            )
            self._append_history("loss_history", loss)

        if learning_rate is not None:
            self.metrics.learning_rate = learning_rate

        # KTO metrics
        if kl is not None:
            self.metrics.kl = kl
            self._append_history("kl_history", kl)

        if margin is not None:
            self.metrics.margin = margin
            self._append_history("margin_history", margin)

        # GRPO metrics
        if reward is not None:
            self.metrics.reward = reward
            self._append_history("reward_history", reward)

        if reward_std is not None:
            self.metrics.reward_std = reward_std

        if kl_penalty is not None:
            self.metrics.kl_penalty = kl_penalty

        if advantage is not None:
            self.metrics.advantage = advantage

        if gpu_memory_gb is not None:
            self.metrics.gpu_memory_gb = gpu_memory_gb

        # Auto-fetch GPU memory if not provided
        if gpu_memory_gb is None and step is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.metrics.gpu_memory_gb = torch.cuda.memory_reserved() / 1e9
            except:
                pass

        if log_message:
            timestamp = time.strftime("%H:%M:%S")
            self.log_buffer.append(f"[{timestamp}] {log_message}")
            if len(self.log_buffer) > self.log_lines:
                self.log_buffer = self.log_buffer[-self.log_lines:]

        # Refresh display if live (with throttle to reduce flicker)
        if self._live:
            now = time.time()
            if now - self._last_update_time >= self._min_update_interval:
                self._live.update(self._build_display())
                self._last_update_time = now

    def _bucket_index_for_step(self, step: int) -> int:
        """Map a training step into a fixed-width full-run trend bucket."""
        total_steps = max(1, int(self.metrics.total_steps))
        bucket_count = max(1, min(self._trend_bucket_count, total_steps))
        clamped_step = max(0, min(int(step), total_steps))
        return min(bucket_count - 1, int(clamped_step * bucket_count / total_steps))

    def _append_history(self, history_name: str, value: float):
        """
        Update a compacted full-run history series.

        The chart should show the whole run, not just the trailing window.
        We therefore map each point into a fixed number of step buckets and
        average repeated updates within the same bucket.
        """
        history = getattr(self.metrics, history_name)
        counts = self._history_counts[history_name]
        bucket_index = self._bucket_index_for_step(self.metrics.step)

        while len(history) < bucket_index:
            fill_value = history[-1] if history else value
            history.append(fill_value)
            counts.append(0)

        if len(history) == bucket_index:
            history.append(value)
            counts.append(1)
            return

        count = counts[bucket_index]
        if count <= 0:
            history[bucket_index] = value
            counts[bucket_index] = 1
            return

        history[bucket_index] = ((history[bucket_index] * count) + value) / (count + 1)
        counts[bucket_index] = count + 1

    def _build_chart(self, title: str, history: List[float], width: int = 40, higher_is_better: bool = False) -> List:
        """Build a chart panel for the right column."""
        from rich.text import Text

        lines = []
        lines.append(Text(title, style=f"bold {COLORS['sky']}"))
        lines.append(Text(""))

        # Sparkline
        spark = sparkline(history, width=width)
        spark_text = Text()
        spark_text.append(spark, style=COLORS['aqua'])
        lines.append(spark_text)

        # Range with delta
        if len(history) >= 2:
            start_val = history[0]
            end_val = history[-1]
            delta = end_val - start_val

            range_text = Text()
            range_text.append(f"\n{start_val:.4f}", style=COLORS['orange'])
            range_text.append(" → ", style="dim")
            range_text.append(f"{end_val:.4f}", style=COLORS['aqua'])

            # Color delta based on whether higher or lower is better
            if higher_is_better:
                delta_style = COLORS['aqua'] if delta > 0 else COLORS['orange']
            else:
                delta_style = COLORS['aqua'] if delta < 0 else COLORS['orange']
            range_text.append(f" ({delta:+.4f})", style=delta_style)
            lines.append(range_text)

        # Min/Max stats
        stats_text = Text()
        stats_text.append(f"\nMin: {min(history):.4f}", style="dim")
        stats_text.append(f"  Max: {max(history):.4f}", style="dim")
        stats_text.append(f"  Avg: {sum(history) / len(history):.4f}", style="dim")
        if len(history) > 10:
            tail_size = min(10, len(history))
            tail_avg = sum(history[-tail_size:]) / tail_size
            stats_text.append(f"  Tail avg({tail_size}): {tail_avg:.4f}", style="dim")
        lines.append(stats_text)

        return lines

    def _build_display(self):
        """Build the dashboard display."""
        if not RICH_AVAILABLE:
            return self._build_text_display()

        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich.console import Group
        from rich.columns import Columns
        from rich import box

        m = self.metrics

        # Title
        title_text = Text()
        title_text.append(f" {BOX['star']} ", style=COLORS['orange'])
        title_text.append(self.title.upper(), style=f"bold {COLORS['aqua']}")
        title_text.append(f" {BOX['star']} ", style=COLORS['orange'])

        # Progress bar (wider)
        prog_width = 30
        filled = int(m.progress_pct / 100 * prog_width)
        progress_bar = f"[{COLORS['aqua']}]{'█' * filled}[/][{COLORS['cello']}]{'░' * (prog_width - filled)}[/] {m.progress_pct:.1f}%"

        # LEFT COLUMN: Metrics table (varies by training type)
        metrics_table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        metrics_table.add_column("Label", style=COLORS['sky'], width=10)
        metrics_table.add_column("Value", style="white")

        metrics_table.add_row("Progress", progress_bar)
        metrics_table.add_row("Epoch", m.epoch_progress_str)
        metrics_table.add_row("Step", f"{m.step:,}/{m.total_steps:,}")
        metrics_table.add_row("Loss", f"[bold]{m.loss:.4f}[/] (best: {m.best_loss:.4f})")

        if self.training_type == "kto":
            # KTO-specific metrics
            metrics_table.add_row("KL", f"{m.kl:.4f}")
            metrics_table.add_row("Margin", f"{m.margin:.4f}")
            if m.kl > 0:
                score = m.margin / m.kl
                score_style = COLORS['aqua'] if score > 1 else COLORS['orange']
                metrics_table.add_row("Score", f"[bold {score_style}]{score:.2f}[/]")

        elif self.training_type == "grpo":
            # GRPO-specific metrics
            reward_style = COLORS['aqua'] if m.reward > 0 else COLORS['orange']
            metrics_table.add_row("Reward", f"[bold {reward_style}]{m.reward:.4f}[/] ±{m.reward_std:.3f}")
            metrics_table.add_row("KL Pen", f"{m.kl_penalty:.4f}")
            if m.advantage != 0:
                metrics_table.add_row("Advntg", f"{m.advantage:.4f}")

        metrics_table.add_row("LR", f"{m.learning_rate:.2e}")

        if m.gpu_memory_gb > 0:
            metrics_table.add_row("GPU Mem", f"{m.gpu_memory_gb:.1f} GB")

        metrics_table.add_row("Elapsed", m.elapsed_str)
        metrics_table.add_row("ETA", m.eta_str)
        metrics_table.add_row("Speed", f"{m.steps_per_second:.1f} steps/s")

        # RIGHT COLUMN: Chart (varies by training type)
        chart_lines = []
        spark_width = 40

        if self.training_type == "kto" and m.margin_history:
            # KTO: Show margin trend (what we're optimizing)
            chart_lines = self._build_chart(
                title="Margin Trend",
                history=m.margin_history,
                width=spark_width,
                higher_is_better=True,
            )
        elif self.training_type == "grpo" and m.reward_history:
            # GRPO: Show reward trend
            chart_lines = self._build_chart(
                title="Reward Trend",
                history=m.reward_history,
                width=spark_width,
                higher_is_better=True,
            )
        elif m.loss_history:
            # SFT or fallback: Show loss trend
            chart_lines = self._build_chart(
                title="Loss Trend",
                history=m.loss_history,
                width=spark_width,
                higher_is_better=False,
            )
        else:
            chart_lines.append(Text("Waiting for data...", style="dim"))

        chart_content = Group(*chart_lines)
        chart_panel = Panel(
            chart_content,
            border_style=COLORS['cello'],
            box=box.ROUNDED,
            padding=(1, 2),
        )

        # Two-column layout
        left_panel = Panel(
            metrics_table,
            border_style=COLORS['cello'],
            box=box.ROUNDED,
            padding=(0, 1),
        )

        columns = Columns([left_panel, chart_panel], equal=True, expand=True)

        # Log panel - pad to fixed height to prevent flicker
        log_lines_list = self.log_buffer.copy() if self.log_buffer else ["[dim]Waiting for events...[/dim]"]
        while len(log_lines_list) < self.log_lines:
            log_lines_list.append("")
        log_content = "\n".join(log_lines_list[-self.log_lines:])

        log_panel = Panel(
            log_content,
            title="Log",
            title_align="left",
            border_style=COLORS['cello'],
            box=box.ROUNDED,
            height=self.log_lines + 2,
        )

        # Main container with title
        main_panel = Panel(
            Group(columns, Text(""), log_panel),
            title=title_text,
            title_align="center",
            border_style=COLORS['aqua'],
            box=box.DOUBLE,
            padding=(1, 1),
        )

        return main_panel

    def _build_text_display(self) -> str:
        """Plain text fallback display."""
        m = self.metrics
        lines = [
            f"\n{'=' * 60}",
            f"  {self.title.upper()}",
            f"{'=' * 60}",
            f"  Progress: {'█' * int(m.progress_pct / 5)}{'░' * (20 - int(m.progress_pct / 5))} {m.progress_pct:.1f}%",
            f"  Epoch: {m.epoch_progress_str}",
            f"  Step: {m.step:,}/{m.total_steps:,}",
            f"  Loss: {m.loss:.4f} (best: {m.best_loss:.4f})",
        ]

        if self.training_type == "kto":
            lines.extend([
                f"  KL: {m.kl:.4f}",
                f"  Margin: {m.margin:.4f}",
            ])

        if m.gpu_memory_gb > 0:
            lines.append(f"  GPU Mem: {m.gpu_memory_gb:.1f} GB")

        lines.extend([
            f"  LR: {m.learning_rate:.2e}",
            f"  Elapsed: {m.elapsed_str}",
            f"  ETA: {m.eta_str}",
            f"{'=' * 60}",
        ])

        return "\n".join(lines)

    def __enter__(self):
        """Start live display."""
        if RICH_AVAILABLE:
            from rich.live import Live
            clear_screen()
            self._live = Live(
                self._build_display(),
                console=console,
                refresh_per_second=2,  # Reduced to minimize flicker
                transient=False,
                vertical_overflow="crop",
                screen=True,  # Alternate screen buffer - much less flicker
            )
            self._live.__enter__()
        return self

    def __exit__(self, *args):
        """Stop live display."""
        if self._live:
            self._live.__exit__(*args)
            self._live = None
        self._stop_event.set()

    @classmethod
    def from_log_file(
        cls,
        log_path: str,
        title: str = "Training",
        training_type: str = "sft",
        poll_interval: float = 0.5,
    ) -> "LiveDashboard":
        """
        Create dashboard that watches a training log file.

        Args:
            log_path: Path to JSONL log file
            title: Dashboard title
            training_type: "sft" or "kto"
            poll_interval: Seconds between file checks
        """
        dashboard = cls(
            title=title,
            training_type=training_type,
        )
        dashboard._log_path = Path(log_path)
        dashboard._poll_interval = poll_interval
        return dashboard

    def watch_log_file(self):
        """Watch log file and update dashboard (blocking)."""
        if not hasattr(self, '_log_path'):
            raise ValueError("Dashboard not created with from_log_file()")

        log_path = self._log_path
        last_position = 0

        with self:
            while not self._stop_event.is_set():
                try:
                    if log_path.exists():
                        with open(log_path, 'r') as f:
                            f.seek(last_position)
                            for line in f:
                                line = line.strip()
                                if line:
                                    self._process_log_line(line)
                            last_position = f.tell()
                except Exception as e:
                    self.update(log_message=f"Error reading log: {e}")

                time.sleep(self._poll_interval)

    def _process_log_line(self, line: str):
        """Process a single log line (JSONL format)."""
        try:
            data = json.loads(line)

            # Update totals first so bucket placement uses the real run shape.
            if 'total_steps' in data:
                self.metrics.total_steps = data['total_steps']
            if 'total_epochs' in data:
                self.metrics.total_epochs = data['total_epochs']

            # Update metrics from log data
            self.update(
                step=data.get('step'),
                epoch=data.get('epoch'),
                loss=data.get('loss'),
                learning_rate=data.get('learning_rate'),
                kl=data.get('kl'),
                margin=data.get('rewards/margins'),
                gpu_memory_gb=data.get('gpu_memory_gb') or data.get('gpu_memory_reserved_gb'),
            )

        except json.JSONDecodeError:
            # Not JSON, treat as log message
            self.update(log_message=line)

    def stop(self):
        """Signal dashboard to stop."""
        self._stop_event.set()


# =============================================================================
# QUICK HELPERS
# =============================================================================

def show_training_progress(
    log_path: str,
    title: str = "Training Progress",
    training_type: str = "sft",
):
    """
    Show live training progress from log file.

    This is a convenience function for monitoring training.

    Args:
        log_path: Path to training log file (JSONL)
        title: Dashboard title
        training_type: "sft" or "kto"

    Example:
        # In another terminal while training runs:
        from shared.ui.dashboard import show_training_progress
        show_training_progress("logs/training_latest.jsonl", "SFT Training")
    """
    dashboard = LiveDashboard.from_log_file(
        log_path=log_path,
        title=title,
        training_type=training_type,
    )

    try:
        dashboard.watch_log_file()
    except KeyboardInterrupt:
        dashboard.stop()
        print("\n  Dashboard stopped.")
