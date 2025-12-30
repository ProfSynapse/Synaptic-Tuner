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
    epoch: int = 0
    total_epochs: int = 1
    step: int = 0
    total_steps: int = 1
    loss: float = 0.0
    learning_rate: float = 0.0
    # KTO-specific
    kl: float = 0.0
    margin: float = 0.0
    # GPU memory (current usage only - can overflow to RAM with offloading)
    gpu_memory_gb: float = 0.0
    # History for sparklines
    loss_history: List[float] = field(default_factory=list)
    # Timing
    start_time: float = field(default_factory=time.time)
    steps_per_second: float = 0.0

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
        if not self.loss_history:
            return self.loss
        return min(self.loss_history)


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

    def update(
        self,
        step: int = None,
        epoch: int = None,
        loss: float = None,
        learning_rate: float = None,
        kl: float = None,
        margin: float = None,
        gpu_memory_gb: float = None,
        log_message: str = None,
    ):
        """Update dashboard metrics."""
        if step is not None:
            # Calculate steps per second
            elapsed = time.time() - self.metrics.start_time
            if elapsed > 0:
                self.metrics.steps_per_second = step / elapsed
            self.metrics.step = step

        if epoch is not None:
            self.metrics.epoch = epoch

        if loss is not None:
            self.metrics.loss = loss
            self.metrics.loss_history.append(loss)
            # Keep last 100 values for sparkline
            if len(self.metrics.loss_history) > 100:
                self.metrics.loss_history = self.metrics.loss_history[-100:]

        if learning_rate is not None:
            self.metrics.learning_rate = learning_rate

        if kl is not None:
            self.metrics.kl = kl

        if margin is not None:
            self.metrics.margin = margin

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

        # LEFT COLUMN: Metrics table
        metrics_table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        metrics_table.add_column("Label", style=COLORS['sky'], width=10)
        metrics_table.add_column("Value", style="white")

        metrics_table.add_row("Progress", progress_bar)
        metrics_table.add_row("Epoch", f"{m.epoch + 1}/{m.total_epochs}")
        metrics_table.add_row("Step", f"{m.step:,}/{m.total_steps:,}")
        metrics_table.add_row("Loss", f"[bold]{m.loss:.4f}[/] (best: {m.best_loss:.4f})")

        if self.training_type == "kto":
            metrics_table.add_row("KL", f"{m.kl:.4f}")
            metrics_table.add_row("Margin", f"{m.margin:.4f}")
            if m.kl > 0:
                score = m.margin / m.kl
                metrics_table.add_row("Score", f"[bold {COLORS['aqua']}]{score:.2f}[/]")

        metrics_table.add_row("LR", f"{m.learning_rate:.2e}")

        if m.gpu_memory_gb > 0:
            metrics_table.add_row("GPU Mem", f"{m.gpu_memory_gb:.1f} GB")

        metrics_table.add_row("Elapsed", m.elapsed_str)
        metrics_table.add_row("ETA", m.eta_str)
        metrics_table.add_row("Speed", f"{m.steps_per_second:.1f} steps/s")

        # RIGHT COLUMN: Big sparkline chart
        chart_lines = []
        if self.show_sparklines and m.loss_history:
            # Build a bigger sparkline (multiple rows for height)
            spark_width = 40
            spark = sparkline(m.loss_history, width=spark_width)

            chart_lines.append(Text("Loss Trend", style=f"bold {COLORS['sky']}"))
            chart_lines.append(Text(""))

            # Show the sparkline
            spark_text = Text()
            spark_text.append(spark, style=COLORS['aqua'])
            chart_lines.append(spark_text)

            # Show range
            range_text = Text()
            range_text.append(f"\n{m.loss_history[0]:.3f}", style=COLORS['orange'])
            range_text.append(" → ", style="dim")
            range_text.append(f"{m.loss_history[-1]:.3f}", style=COLORS['aqua'])
            if len(m.loss_history) > 1:
                delta = m.loss_history[-1] - m.loss_history[0]
                delta_style = COLORS['aqua'] if delta < 0 else COLORS['orange']
                range_text.append(f" ({delta:+.3f})", style=delta_style)
            chart_lines.append(range_text)

            # Show min/max
            stats_text = Text()
            stats_text.append(f"\nMin: {min(m.loss_history):.4f}", style="dim")
            stats_text.append(f"  Max: {max(m.loss_history):.4f}", style="dim")
            chart_lines.append(stats_text)
        else:
            chart_lines.append(Text("Loss Trend", style=f"bold {COLORS['sky']}"))
            chart_lines.append(Text(""))
            chart_lines.append(Text("[dim]Waiting for data...[/dim]"))

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
            f"  Epoch: {m.epoch + 1}/{m.total_epochs}",
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
                vertical_overflow="visible",
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

            # Update metrics from log data
            self.update(
                step=data.get('step'),
                epoch=data.get('epoch'),
                loss=data.get('loss'),
                learning_rate=data.get('learning_rate'),
                kl=data.get('kl'),
                margin=data.get('rewards/margins'),
            )

            # Update totals if provided
            if 'total_steps' in data:
                self.metrics.total_steps = data['total_steps']
            if 'total_epochs' in data:
                self.metrics.total_epochs = data['total_epochs']

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
