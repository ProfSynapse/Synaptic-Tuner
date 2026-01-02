"""
Live SynthChat Dashboard for Synaptic Tuner

Real-time display of synthetic data generation progress, results, and statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List
from threading import Event

from .theme import COLORS, BOX
from .console import console, RICH_AVAILABLE, clear_screen


@dataclass
class SynthChatMetrics:
    """Container for generation metrics."""
    total_examples: int = 0
    completed: int = 0
    valid: int = 0
    invalid: int = 0

    # Current generation info
    current_category: str = ""
    current_example: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)

    @property
    def progress_pct(self) -> float:
        if self.total_examples == 0:
            return 0
        return (self.completed / self.total_examples) * 100

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
        if self.completed == 0:
            return "calculating..."
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.completed
        remaining = (self.total_examples - self.completed) * avg_time
        hours, remainder = divmod(int(remaining), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def success_rate(self) -> float:
        if self.completed == 0:
            return 0
        return (self.valid / self.completed) * 100

    @property
    def examples_per_min(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed < 1:
            return 0
        return (self.completed / elapsed) * 60


@dataclass
class ResultEntry:
    """A single result for the scrolling log."""
    status: str  # "valid" or "invalid"
    category: str
    reason: Optional[str] = None


class LiveSynthChatDashboard:
    """
    Real-time synthetic data generation dashboard.

    Usage:
        dashboard = LiveSynthChatDashboard(
            title="SynthChat Generation",
            total_examples=100,
        )

        with dashboard:
            for result in generation_results:
                dashboard.update(result)
    """

    def __init__(
        self,
        title: str = "Synthetic Data Generation",
        total_examples: int = 100,
        log_lines: int = 5,
    ):
        self.title = title
        self.log_lines = log_lines

        self.metrics = SynthChatMetrics(total_examples=total_examples)
        self.results_log: List[ResultEntry] = []

        self._stop_event = Event()
        self._live = None
        self._last_update_time = 0.0
        self._min_update_interval = 0.5  # Match Live refresh rate

    def update(
        self,
        status: str = None,
        category: str = None,
        reason: str = None,
        is_current: bool = False,
    ):
        """Update dashboard with a new result or current generation info.

        Args:
            status: Result status ("valid" or "invalid")
            category: Tool/behavior category name
            reason: Brief failure reason (for invalid)
            is_current: If True, just update current display (no result logged)
        """
        if is_current and category:
            self.metrics.current_category = category
            self.metrics.current_example = self.metrics.completed + 1
        elif status:
            # Update counts
            self.metrics.completed += 1
            if status == "valid":
                self.metrics.valid += 1
            else:
                self.metrics.invalid += 1

            # Add to results log
            entry = ResultEntry(
                status=status,
                category=category or "unknown",
                reason=reason,
            )
            self.results_log.append(entry)
            if len(self.results_log) > self.log_lines:
                self.results_log = self.results_log[-self.log_lines:]

            # Clear current
            self.metrics.current_category = ""

        # Refresh display
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

        # Progress bar
        prog_width = 35
        filled = int(m.progress_pct / 100 * prog_width)
        progress_bar = f"[{COLORS['aqua']}]{'█' * filled}[/][{COLORS['cello']}]{'░' * (prog_width - filled)}[/] {m.completed}/{m.total_examples} ({m.progress_pct:.1f}%)"

        # LEFT COLUMN: Metrics
        metrics_table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        metrics_table.add_column("Label", style=COLORS['sky'], width=12)
        metrics_table.add_column("Value", style="white")

        metrics_table.add_row("Progress", progress_bar)
        metrics_table.add_row("Elapsed", m.elapsed_str)
        metrics_table.add_row("ETA", m.eta_str)
        metrics_table.add_row("Speed", f"{m.examples_per_min:.1f} ex/min")
        metrics_table.add_row("", "")  # Spacer

        # Results summary
        valid_style = COLORS['aqua'] if m.valid > 0 else "dim"
        invalid_style = COLORS['orange'] if m.invalid > 0 else "dim"

        results_text = Text()
        results_text.append(f"{m.valid}", style=valid_style)
        results_text.append(" valid  ", style="dim")
        results_text.append(f"{m.invalid}", style=invalid_style)
        results_text.append(" invalid", style="dim")

        metrics_table.add_row("Results", results_text)

        # Success rate
        if m.completed > 0:
            rate_text = Text()
            rate = m.success_rate
            rate_style = COLORS['aqua'] if rate >= 90 else ("yellow" if rate >= 70 else COLORS['orange'])
            rate_text.append(f"{rate:.1f}%", style=rate_style)
            metrics_table.add_row("Success", rate_text)

        # Current generation
        if m.current_category:
            current_text = Text()
            current_text.append(f"#{m.current_example} ", style=COLORS['sky'])
            current_text.append(m.current_category, style="white")
            metrics_table.add_row("", "")
            metrics_table.add_row("Generating", current_text)

        left_panel = Panel(
            metrics_table,
            border_style=COLORS['cello'],
            box=box.ROUNDED,
            padding=(0, 1),
        )

        # RIGHT COLUMN: Recent results log
        log_lines = []
        for entry in self.results_log:
            line = Text()
            if entry.status == "valid":
                line.append("  ✓ VALID  ", style="bold green")
            else:
                line.append("  ✗ INVALID", style="bold red")

            # Truncate long names
            name = entry.category
            if len(name) > 25:
                name = name[:22] + "..."
            line.append(f" {name}", style="white")

            if entry.reason and entry.status == "invalid":
                # Truncate reason
                reason = entry.reason
                if len(reason) > 30:
                    reason = reason[:27] + "..."
                line.append(f"\n           {reason}", style="dim")

            log_lines.append(line)

        # Pad to fixed height
        while len(log_lines) < self.log_lines:
            log_lines.append(Text(""))

        log_content = Group(*log_lines)
        log_panel = Panel(
            log_content,
            title="Recent Results",
            title_align="left",
            border_style=COLORS['cello'],
            box=box.ROUNDED,
            padding=(0, 1),
        )

        # Two-column layout
        columns = Columns([left_panel, log_panel], equal=True, expand=True)

        # Main container
        main_panel = Panel(
            columns,
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
        bar_width = 20
        filled = int(m.progress_pct / 100 * bar_width)
        bar = '█' * filled + '░' * (bar_width - filled)

        lines = [
            f"\n{'=' * 60}",
            f"  {self.title.upper()}",
            f"{'=' * 60}",
            f"  Progress: [{bar}] {m.completed}/{m.total_examples} ({m.progress_pct:.1f}%)",
            f"  Elapsed: {m.elapsed_str}  ETA: {m.eta_str}",
            f"  Valid: {m.valid}  Invalid: {m.invalid}  Rate: {m.success_rate:.1f}%",
        ]

        if m.current_category:
            lines.append(f"  Generating: {m.current_category}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def __enter__(self):
        """Start live display."""
        if RICH_AVAILABLE:
            from rich.live import Live
            self._live = Live(
                self._build_display(),
                console=console,
                refresh_per_second=2,
                transient=False,
                vertical_overflow="crop",
                screen=True,  # Alternate screen buffer - no flicker
            )
            self._live.__enter__()
        return self

    def __exit__(self, *args):
        """Stop live display."""
        if self._live:
            self._live.__exit__(*args)
            self._live = None
        self._stop_event.set()

    def stop(self):
        """Signal dashboard to stop."""
        self._stop_event.set()
