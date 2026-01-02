"""
Unified Training Progress Display

A beautiful Rich-based progress display that works across training backends
(MLX on Mac, Unsloth on NVIDIA). Parses training output and renders a live
dashboard with progress bars, stats, and ETA.

Usage:
    with TrainingProgressDisplay(total_steps=1000, model_name="Qwen3-0.6B") as display:
        for line in process.stdout:
            display.update_from_line(line)
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.table import Table
from rich.text import Text
from rich.align import Align


@dataclass
class TrainingStats:
    """Current training statistics."""
    current_step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_sec: float = 0.0
    iter_per_sec: float = 0.0
    peak_memory_gb: float = 0.0
    trained_tokens: int = 0
    epoch: int = 1
    total_epochs: int = 1
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.current_step == 0 or self.iter_per_sec == 0:
            return None
        remaining = self.total_steps - self.current_step
        return remaining / self.iter_per_sec


class TrainingProgressDisplay:
    """
    Rich-based training progress display.

    Parses output from mlx_lm and Unsloth trainers and renders
    a beautiful live dashboard.
    """

    # Regex patterns for parsing different trainer outputs
    PATTERNS = {
        # mlx_lm format: Iter 50: Train loss 0.809, Learning Rate 2.000e-04, It/sec 0.258, ...
        'mlx_train': re.compile(
            r'Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+-]+), '
            r'It/sec ([\d.]+), Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB'
        ),
        'mlx_val': re.compile(r'Iter (\d+): Val loss ([\d.]+)'),
        'mlx_start': re.compile(r'Starting training\.\.\., iters: (\d+)'),

        # Unsloth/TRL format
        'unsloth_loss': re.compile(r"'loss': ([\d.]+)"),
        'unsloth_step': re.compile(r"'step': (\d+)"),
        'unsloth_lr': re.compile(r"'learning_rate': ([\d.e+-]+)"),
        'unsloth_epoch': re.compile(r"'epoch': ([\d.]+)"),
        'trl_progress': re.compile(r'(\d+)/(\d+) \['),  # tqdm style
    }

    # Colors from the theme
    PURPLE = "#93278F"
    AQUA = "#00A99D"
    GOLD = "#FFD700"

    def __init__(
        self,
        total_steps: int = 0,
        model_name: str = "Model",
        platform: str = "MLX",
        console: Optional[Console] = None
    ):
        self.console = console or Console()
        self.stats = TrainingStats(total_steps=total_steps)
        self.model_name = model_name
        self.platform = platform
        self.live: Optional[Live] = None
        self._raw_lines: list[str] = []  # Store raw output for passthrough

    def __enter__(self):
        self.stats.start_time = time.time()
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=2,  # Slower refresh reduces flicker
            transient=False,
            vertical_overflow="crop",
            screen=True,  # Alternate screen buffer - much less flicker
        )
        self.live.__enter__()
        return self

    def __exit__(self, *args):
        if self.live:
            self.live.__exit__(*args)

    def update_from_line(self, line: str) -> bool:
        """
        Parse a line of training output and update stats.

        Returns True if the line was parsed, False otherwise.
        """
        line = line.strip()
        if not line:
            return False

        # Try mlx_lm patterns
        if match := self.PATTERNS['mlx_start'].search(line):
            self.stats.total_steps = int(match.group(1))
            return True

        if match := self.PATTERNS['mlx_train'].search(line):
            self.stats.current_step = int(match.group(1))
            self.stats.train_loss = float(match.group(2))
            self.stats.learning_rate = float(match.group(3))
            self.stats.iter_per_sec = float(match.group(4))
            self.stats.tokens_per_sec = float(match.group(5))
            self.stats.trained_tokens = int(match.group(6))
            self.stats.peak_memory_gb = float(match.group(7))
            self.stats.last_update = time.time()
            self._update_display()
            return True

        if match := self.PATTERNS['mlx_val'].search(line):
            self.stats.current_step = int(match.group(1))
            self.stats.val_loss = float(match.group(2))
            self._update_display()
            return True

        # Try Unsloth/TRL patterns
        if match := self.PATTERNS['trl_progress'].search(line):
            self.stats.current_step = int(match.group(1))
            self.stats.total_steps = int(match.group(2))
            self._update_display()
            return True

        if match := self.PATTERNS['unsloth_loss'].search(line):
            self.stats.train_loss = float(match.group(1))

        if match := self.PATTERNS['unsloth_step'].search(line):
            self.stats.current_step = int(match.group(1))
            self._update_display()
            return True

        if match := self.PATTERNS['unsloth_lr'].search(line):
            self.stats.learning_rate = float(match.group(1))

        if match := self.PATTERNS['unsloth_epoch'].search(line):
            self.stats.epoch = int(float(match.group(1))) + 1

        return False

    def _update_display(self):
        """Refresh the live display."""
        if self.live:
            self.live.update(self._render())

    def _format_time(self, seconds: Optional[float]) -> str:
        """Format seconds as human-readable time."""
        if seconds is None:
            return "--:--"
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m {secs}s"

    def _render(self) -> Panel:
        """Render the training dashboard."""
        stats = self.stats

        # Build progress bar
        progress = Progress(
            SpinnerColumn(style=f"bold {self.PURPLE}"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40, style=self.PURPLE, complete_style=self.AQUA),
            TextColumn("[bold]{task.percentage:>5.1f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.completed}/{task.total}"),
            console=self.console,
            expand=True
        )
        task_id = progress.add_task("Training", total=stats.total_steps, completed=stats.current_step)

        # Build stats table
        table = Table.grid(padding=(0, 3))
        table.add_column(justify="right", style="dim")
        table.add_column(justify="left", style=f"bold {self.AQUA}")
        table.add_column(justify="right", style="dim")
        table.add_column(justify="left", style=f"bold {self.AQUA}")

        # Row 1: Losses
        train_loss_str = f"{stats.train_loss:.4f}" if stats.train_loss > 0 else "--"
        val_loss_str = f"{stats.val_loss:.4f}" if stats.val_loss > 0 else "--"
        table.add_row("Train Loss:", train_loss_str, "Val Loss:", val_loss_str)

        # Row 2: Speed metrics
        lr_str = f"{stats.learning_rate:.2e}" if stats.learning_rate > 0 else "--"
        tokens_str = f"{stats.tokens_per_sec:.0f}" if stats.tokens_per_sec > 0 else "--"
        table.add_row("Learning Rate:", lr_str, "Tokens/sec:", tokens_str)

        # Row 3: Memory and time
        mem_str = f"{stats.peak_memory_gb:.1f} GB" if stats.peak_memory_gb > 0 else "--"
        eta_str = self._format_time(stats.eta_seconds)
        table.add_row("Peak Memory:", mem_str, "ETA:", eta_str)

        # Row 4: Elapsed and epoch
        elapsed_str = self._format_time(stats.elapsed_seconds)
        epoch_str = f"{stats.epoch}/{stats.total_epochs}" if stats.total_epochs > 1 else "--"
        table.add_row("Elapsed:", elapsed_str, "Epoch:", epoch_str)

        # Combine into panel
        content = Group(
            Text(f"  Model: {self.model_name}", style="bold white"),
            Text(f"  Platform: {self.platform}", style="dim"),
            Text(""),
            progress,
            Text(""),
            Align.center(table),
        )

        return Panel(
            content,
            title=f"[bold {self.AQUA}]Training Progress[/]",
            border_style=self.PURPLE,
            padding=(1, 2),
        )

    def finish(self, success: bool = True):
        """Show completion message."""
        if self.live:
            self.live.stop()

        elapsed = self._format_time(self.stats.elapsed_seconds)

        if success:
            self.console.print(f"\n[bold {self.AQUA}]✓ Training complete![/] ({elapsed})")
            self.console.print(f"  Final train loss: {self.stats.train_loss:.4f}")
            if self.stats.val_loss > 0:
                self.console.print(f"  Final val loss: {self.stats.val_loss:.4f}")
        else:
            self.console.print(f"\n[bold red]✗ Training failed[/] ({elapsed})")


def run_with_progress(
    cmd: list[str],
    cwd: str,
    total_steps: int = 0,
    model_name: str = "Model",
    platform: str = "MLX"
) -> int:
    """
    Run a training command with live progress display.

    Args:
        cmd: Command to run
        cwd: Working directory
        total_steps: Total training steps (0 for auto-detect)
        model_name: Model name to display
        platform: Platform name (MLX, CUDA, etc.)

    Returns:
        Exit code from the training process
    """
    import subprocess

    console = Console()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )

        with TrainingProgressDisplay(
            total_steps=total_steps,
            model_name=model_name,
            platform=platform,
            console=console
        ) as display:
            for line in process.stdout:
                parsed = display.update_from_line(line)
                # If we couldn't parse it, it might be important - print it
                if not parsed and line.strip():
                    # Check if it's not just a progress bar artifact
                    if not line.startswith('\r') and '[' not in line[:5]:
                        console.print(f"[dim]{line.strip()}[/]")

            return_code = process.wait()
            display.finish(success=(return_code == 0))
            return return_code

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/]")
        if 'process' in locals():
            process.terminate()
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        return 1
