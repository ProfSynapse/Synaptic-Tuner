"""
Live Evaluation Dashboard for Synaptic Tuner

Real-time display of evaluation progress, results, and statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from threading import Event

from .theme import COLORS, BOX
from .console import console, RICH_AVAILABLE, clear_screen


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    total_tests: int = 0
    completed: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    errors: int = 0

    # Behavior-specific counts
    behavior_tested: int = 0
    behavior_passed: int = 0

    # Current test info
    current_test: str = ""
    current_latency: float = 0.0

    # Timing
    start_time: float = field(default_factory=time.time)

    @property
    def progress_pct(self) -> float:
        if self.total_tests == 0:
            return 0
        return (self.completed / self.total_tests) * 100

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
        remaining = (self.total_tests - self.completed) * avg_time
        hours, remainder = divmod(int(remaining), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def pass_rate(self) -> float:
        if self.completed == 0:
            return 0
        return (self.passed / self.completed) * 100

    @property
    def schema_pass_rate(self) -> float:
        schema_passed = self.passed + self.warned  # Warn means schema passed but behavior issue
        if self.completed == 0:
            return 0
        return (schema_passed / self.completed) * 100


@dataclass
class ResultEntry:
    """A single result for the scrolling log."""
    status: str  # "pass", "warn", "fail", "error"
    name: str
    latency: float
    reason: Optional[str] = None


class LiveEvaluationDashboard:
    """
    Real-time evaluation dashboard with progress, stats, and results log.

    Usage:
        dashboard = LiveEvaluationDashboard(
            title="Model Evaluation",
            total_tests=91,
        )

        with dashboard:
            for record in evaluation_results:
                dashboard.update(record)
    """

    def __init__(
        self,
        title: str = "Evaluation",
        total_tests: int = 100,
        log_lines: int = 5,
    ):
        self.title = title
        self.log_lines = log_lines

        self.metrics = EvaluationMetrics(total_tests=total_tests)
        self.results_log: List[ResultEntry] = []

        self._stop_event = Event()
        self._live = None
        self._last_update_time = 0.0
        self._min_update_interval = 0.5  # Match Live refresh_per_second=2

    def update(
        self,
        status: str = None,
        name: str = None,
        latency: float = None,
        reason: str = None,
        behavior_tested: bool = False,
        behavior_passed: bool = False,
        is_current: bool = False,
        **_kwargs,
    ):
        """Update dashboard with a new result or current test info.

        Args:
            status: Result status ("pass", "warn", "fail", "error")
            name: Test case name/ID
            latency: Response time in seconds
            reason: Brief failure reason (for log display)
            behavior_tested: Whether behavior was tested
            behavior_passed: Whether behavior passed
            is_current: If True, just update current test display (no result logged)
        """
        if is_current and name:
            self.metrics.current_test = name
            self.metrics.current_latency = latency or 0.0
        elif status:
            # Update counts
            self.metrics.completed += 1
            if status == "pass":
                self.metrics.passed += 1
            elif status == "warn":
                self.metrics.warned += 1
            elif status == "fail":
                self.metrics.failed += 1
            elif status == "error":
                self.metrics.errors += 1

            # Update behavior counts
            if behavior_tested:
                self.metrics.behavior_tested += 1
                if behavior_passed:
                    self.metrics.behavior_passed += 1

            # Add to results log
            entry = ResultEntry(
                status=status,
                name=name or "unknown",
                latency=latency or 0.0,
                reason=reason,
            )
            self.results_log.append(entry)
            if len(self.results_log) > self.log_lines:
                self.results_log = self.results_log[-self.log_lines:]

            # Clear current test
            self.metrics.current_test = ""
            self.metrics.current_latency = 0.0

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
        progress_bar = f"[{COLORS['aqua']}]{'█' * filled}[/][{COLORS['cello']}]{'░' * (prog_width - filled)}[/] {m.completed}/{m.total_tests} ({m.progress_pct:.1f}%)"

        # LEFT COLUMN: Metrics
        metrics_table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        metrics_table.add_column("Label", style=COLORS['sky'], width=10)
        metrics_table.add_column("Value", style="white")

        metrics_table.add_row("Progress", progress_bar)
        metrics_table.add_row("Elapsed", m.elapsed_str)
        metrics_table.add_row("ETA", m.eta_str)
        metrics_table.add_row("", "")  # Spacer

        # Results summary
        pass_style = COLORS['aqua'] if m.passed > 0 else "dim"
        warn_style = "yellow" if m.warned > 0 else "dim"
        fail_style = COLORS['orange'] if m.failed > 0 else "dim"
        err_style = "red" if m.errors > 0 else "dim"

        results_text = Text()
        results_text.append(f"{m.passed}", style=pass_style)
        results_text.append(" pass  ", style="dim")
        results_text.append(f"{m.warned}", style=warn_style)
        results_text.append(" warn  ", style="dim")
        results_text.append(f"{m.failed}", style=fail_style)
        results_text.append(" fail  ", style="dim")
        if m.errors > 0:
            results_text.append(f"{m.errors}", style=err_style)
            results_text.append(" err", style="dim")

        metrics_table.add_row("Results", results_text)

        # Pass rates
        if m.completed > 0:
            rate_text = Text()
            schema_rate = m.schema_pass_rate
            rate_style = COLORS['aqua'] if schema_rate >= 90 else ("yellow" if schema_rate >= 70 else COLORS['orange'])
            rate_text.append(f"{schema_rate:.1f}%", style=rate_style)
            metrics_table.add_row("Schema", rate_text)

            if m.behavior_tested > 0:
                beh_rate = (m.behavior_passed / m.behavior_tested) * 100
                beh_style = COLORS['aqua'] if beh_rate >= 90 else ("yellow" if beh_rate >= 70 else COLORS['orange'])
                beh_text = Text()
                beh_text.append(f"{beh_rate:.1f}%", style=beh_style)
                beh_text.append(f" ({m.behavior_passed}/{m.behavior_tested})", style="dim")
                metrics_table.add_row("Behavior", beh_text)

        # Current test
        if m.current_test:
            current_text = Text()
            current_text.append(m.current_test, style=COLORS['sky'])
            if m.current_latency > 0:
                current_text.append(f" ({m.current_latency:.1f}s)", style="dim")
            metrics_table.add_row("", "")
            metrics_table.add_row("Current", current_text)

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
            if entry.status == "pass":
                line.append("  ✓ PASS  ", style="bold green")
            elif entry.status == "warn":
                line.append("  ⚠ WARN  ", style="bold yellow")
            elif entry.status == "fail":
                line.append("  ✗ FAIL  ", style="bold red")
            else:
                line.append("  ⚡ ERR   ", style="bold red")

            # Truncate long names
            name = entry.name
            if len(name) > 25:
                name = name[:22] + "..."
            line.append(name, style="white")
            line.append(f" ({entry.latency:.1f}s)", style="dim")

            if entry.reason and entry.status in ("warn", "fail"):
                # Truncate reason
                reason = entry.reason
                if len(reason) > 30:
                    reason = reason[:27] + "..."
                line.append(f"\n         {reason}", style="dim")

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
            f"  Progress: [{bar}] {m.completed}/{m.total_tests} ({m.progress_pct:.1f}%)",
            f"  Elapsed: {m.elapsed_str}  ETA: {m.eta_str}",
            f"  Pass: {m.passed}  Warn: {m.warned}  Fail: {m.failed}  Err: {m.errors}",
        ]

        if m.current_test:
            lines.append(f"  Current: {m.current_test}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def __enter__(self):
        """Start live display."""
        if RICH_AVAILABLE:
            from rich.live import Live
            # Don't clear_screen() - let Live handle it with screen=True
            self._live = Live(
                self._build_display(),
                console=console,
                refresh_per_second=2,  # Slower refresh reduces flicker
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

    def stop(self):
        """Signal dashboard to stop."""
        self._stop_event.set()


def rich_evaluation_summary(
    records: List[Any],
    aggregate_stats_fn=None,
) -> None:
    """Display a rich evaluation summary with tables and panels.

    Args:
        records: List of EvaluationRecord objects
        aggregate_stats_fn: Optional function to aggregate stats (from reporting.py)
    """
    if not RICH_AVAILABLE:
        # Fall back to text display
        print("\n=== EVALUATION SUMMARY ===")
        return

    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Group
    from rich import box

    # Get stats - either use provided function or compute basic stats
    if aggregate_stats_fn:
        stats = aggregate_stats_fn(records)
    else:
        # Basic stats computation
        total = len(records)
        passed = sum(1 for r in records if getattr(r, 'passed', False))
        warned = sum(1 for r in records if getattr(r, 'status', '') == 'warn')
        failed = total - passed - warned
        stats = {
            'total': total,
            'passed': passed,
            'warned': warned,
            'failed': failed,
            'by_tag': {},
        }

    # Overall summary panel
    summary_text = Text()
    summary_text.append(f"\nEvaluated ", style="white")
    summary_text.append(f"{stats['total']}", style=f"bold {COLORS['aqua']}")
    summary_text.append(" tests: ", style="white")
    summary_text.append(f"{stats['passed']} passed", style="bold green")
    if stats.get('warned', 0) > 0:
        summary_text.append(f", {stats['warned']} warned", style="bold yellow")
    summary_text.append(f", {stats['failed']} failed", style=f"bold {COLORS['orange']}")
    summary_text.append("\n")

    # Pass rate
    pass_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
    rate_style = COLORS['aqua'] if pass_rate >= 90 else ("yellow" if pass_rate >= 70 else COLORS['orange'])
    summary_text.append(f"Pass Rate: ", style="white")
    summary_text.append(f"{pass_rate:.1f}%", style=f"bold {rate_style}")

    console.print(Panel(
        summary_text,
        title="Summary",
        title_align="left",
        border_style=COLORS['aqua'],
        box=box.ROUNDED,
    ))

    # Results by tag table
    if stats.get('by_tag'):
        tag_table = Table(
            title="Results by Category",
            box=box.ROUNDED,
            border_style=COLORS['cello'],
            header_style=f"bold {COLORS['sky']}",
        )
        tag_table.add_column("Category", style=COLORS['sky'])
        tag_table.add_column("Passed", justify="right", style="green")
        tag_table.add_column("Warned", justify="right", style="yellow")
        tag_table.add_column("Failed", justify="right", style=COLORS['orange'])
        tag_table.add_column("Total", justify="right")
        tag_table.add_column("Pass Rate", justify="right")

        for tag, bucket in stats['by_tag'].items():
            rate = bucket.get('pass_rate', 0) * 100
            rate_style = "green" if rate >= 90 else ("yellow" if rate >= 70 else COLORS['orange'])
            tag_table.add_row(
                tag,
                str(bucket.get('passed', 0)),
                str(bucket.get('warned', 0)),
                str(bucket.get('failed', 0)),
                str(bucket.get('total', 0)),
                Text(f"{rate:.1f}%", style=rate_style),
            )

        console.print(tag_table)

    # Top failure reasons
    if stats.get('top_failure_reasons'):
        failure_lines = []
        for reason, count in stats['top_failure_reasons'][:5]:
            line = Text()
            line.append(f"  {count}× ", style=COLORS['orange'])
            line.append(reason, style="white")
            failure_lines.append(line)

        if failure_lines:
            console.print(Panel(
                Group(*failure_lines),
                title="Top Failure Reasons",
                title_align="left",
                border_style=COLORS['orange'],
                box=box.ROUNDED,
            ))

    console.print()
