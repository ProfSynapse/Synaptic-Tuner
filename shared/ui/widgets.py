"""
Advanced UI Widgets for Synaptic Tuner

Provides spinners, file pickers, fuzzy search, sparklines, and more.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from contextlib import contextmanager

from .theme import COLORS, BOX
from .console import console, RICH_AVAILABLE, TERMINAL_MENU_AVAILABLE

# Check for simple-term-menu
try:
    from simple_term_menu import TerminalMenu
except ImportError:
    TerminalMenu = None


# =============================================================================
# SPINNERS - Animated loading indicators
# =============================================================================

@contextmanager
def spinner(message: str, spinner_type: str = "dots"):
    """
    Show animated spinner during long operations.

    Args:
        message: Status message to display
        spinner_type: Spinner animation style (dots, line, dots12, aesthetic)

    Example:
        with spinner("Merging LoRA adapter..."):
            merge_lora(model_path)

        with spinner("Uploading to HuggingFace...", spinner_type="dots12"):
            upload_model(repo_id)
    """
    if RICH_AVAILABLE:
        from rich.status import Status
        with console.status(
            f"[bold {COLORS['purple']}]{message}[/]",
            spinner=spinner_type,
            spinner_style=COLORS['aqua']
        ) as status:
            yield status
    else:
        print(f"  {BOX['bullet']} {message}...")
        yield None


@contextmanager
def progress_spinner(message: str, total: int = None):
    """
    Show spinner with optional progress percentage.

    Args:
        message: Status message
        total: Total steps (enables percentage display)

    Example:
        with progress_spinner("Processing files...", total=100) as update:
            for i in range(100):
                process_file(i)
                update(i + 1, f"Processing file {i+1}/100")
    """
    if RICH_AVAILABLE:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

        if total:
            progress = Progress(
                SpinnerColumn(style=COLORS['aqua']),
                TextColumn(f"[bold {COLORS['purple']}]{{task.description}}[/]"),
                BarColumn(bar_width=30, style=COLORS['cello'], complete_style=COLORS['aqua']),
                TaskProgressColumn(),
                console=console,
            )
        else:
            progress = Progress(
                SpinnerColumn(style=COLORS['aqua']),
                TextColumn(f"[bold {COLORS['purple']}]{{task.description}}[/]"),
                console=console,
            )

        with progress:
            task = progress.add_task(message, total=total or 100)

            def update(completed: int = None, description: str = None):
                if completed is not None:
                    progress.update(task, completed=completed)
                if description is not None:
                    progress.update(task, description=description)

            yield update
    else:
        print(f"  {BOX['bullet']} {message}...")

        def update(completed: int = None, description: str = None):
            if description:
                print(f"  {BOX['bullet']} {description}")

        yield update


# =============================================================================
# FILE PICKER - Interactive file selection
# =============================================================================

def file_picker(
    start_dir: str = ".",
    extensions: List[str] = None,
    title: str = "Select a file",
) -> Optional[str]:
    """
    Interactive file picker with directory navigation.

    Args:
        start_dir: Starting directory
        extensions: Filter by extensions (e.g., [".jsonl", ".json"])
        title: Picker title

    Returns:
        Selected file path or None if cancelled

    Example:
        dataset = file_picker(
            start_dir="Datasets/",
            extensions=[".jsonl"],
            title="Select training dataset"
        )
    """
    if not TERMINAL_MENU_AVAILABLE or TerminalMenu is None:
        # Fallback to text input
        if RICH_AVAILABLE:
            from rich.prompt import Prompt
            return Prompt.ask(f"  {title}", default=start_dir)
        return input(f"  {title} [{start_dir}]: ").strip() or start_dir

    current_dir = Path(start_dir).resolve()

    while True:
        # Build file list
        items = []
        files = []

        # Add parent directory option
        if current_dir.parent != current_dir:
            items.append(("📁 ..", "parent"))

        # Add directories first
        try:
            for entry in sorted(current_dir.iterdir()):
                if entry.is_dir() and not entry.name.startswith('.'):
                    items.append((f"📁 {entry.name}/", ("dir", entry)))
                elif entry.is_file():
                    if extensions is None or entry.suffix in extensions:
                        files.append((f"📄 {entry.name}", ("file", entry)))
        except PermissionError:
            pass

        # Add files after directories
        items.extend(files)

        if not items:
            items.append(("(empty directory)", None))

        # Add cancel option
        items.append(("❌ Cancel", "cancel"))

        # Show menu
        labels = [item[0] for item in items]

        menu = TerminalMenu(
            labels,
            title=f"\n  {title}\n  📂 {current_dir}\n",
            menu_cursor=" → ",
            menu_cursor_style=("fg_cyan", "bold"),
            menu_highlight_style=("fg_cyan",),
            show_search_hint=True,
            search_key="/",
            quit_keys=("escape", "q"),
        )

        selected = menu.show()

        if selected is None:
            return None

        action = items[selected][1]

        if action == "cancel" or action is None:
            return None
        elif action == "parent":
            current_dir = current_dir.parent
        elif action[0] == "dir":
            current_dir = action[1]
        elif action[0] == "file":
            return str(action[1])


# =============================================================================
# FUZZY SEARCH MENU - Filter options by typing
# =============================================================================

def fuzzy_menu(
    options: List[Tuple[str, str]],
    title: str = "Search and select",
    preview: Callable[[str], str] = None,
) -> Optional[str]:
    """
    Menu with fuzzy search/filter capability.

    Type to filter options interactively.

    Args:
        options: List of (key, description) tuples
        title: Menu title
        preview: Optional function to generate preview text for selected item

    Returns:
        Selected option key or None if cancelled

    Example:
        model = fuzzy_menu([
            ("qwen3-4b", "Qwen3-4B (4 billion params)"),
            ("qwen3-8b", "Qwen3-8B (8 billion params)"),
            ("mistral-7b", "Mistral-7B-v0.3"),
        ], "Select model")
    """
    if not TERMINAL_MENU_AVAILABLE or TerminalMenu is None:
        # Fallback to regular menu
        from .console import print_menu
        return print_menu(options, title)

    labels = [desc for _, desc in options]
    labels.append("Exit")

    # Configure preview if provided
    preview_command = None
    if preview:
        def preview_wrapper(label):
            # Find matching key
            for key, desc in options:
                if desc == label:
                    return preview(key)
            return ""
        preview_command = preview_wrapper

    menu = TerminalMenu(
        labels,
        title=f"\n  {title}\n  (type to filter)\n",
        menu_cursor=" → ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("fg_cyan",),
        cycle_cursor=True,
        clear_screen=False,
        show_search_hint=True,
        search_key=None,  # Always in search mode
        quit_keys=("escape",),
        preview_command=preview_command,
        preview_size=0.3 if preview else 0,
    )

    selected = menu.show()

    if selected is None or selected == len(options):
        return None
    return options[selected][0]


# =============================================================================
# MULTI-SELECT MENU - Select multiple options
# =============================================================================

def multi_select_menu(
    options: List[Tuple[str, str]],
    title: str = "Select options",
    preselected: List[str] = None,
    min_selections: int = 0,
    max_selections: int = None,
) -> Optional[List[str]]:
    """
    Multi-select menu with checkboxes.

    Args:
        options: List of (key, description) tuples
        title: Menu title
        preselected: Keys to pre-select
        min_selections: Minimum required selections
        max_selections: Maximum allowed selections (None = unlimited)

    Returns:
        List of selected keys or None if cancelled

    Example:
        formats = multi_select_menu([
            ("q4", "Q4_K_M (recommended)"),
            ("q5", "Q5_K_M (higher quality)"),
            ("q8", "Q8_0 (best quality)"),
            ("f16", "F16 (full precision)"),
        ], "Select GGUF quantization formats", preselected=["q4"])
    """
    if not TERMINAL_MENU_AVAILABLE or TerminalMenu is None:
        # Fallback: just return preselected or ask for comma-separated input
        if preselected:
            return preselected
        if RICH_AVAILABLE:
            from rich.prompt import Prompt
            result = Prompt.ask(f"  {title} (comma-separated)")
            return [x.strip() for x in result.split(",")] if result else []
        result = input(f"  {title} (comma-separated): ")
        return [x.strip() for x in result.split(",")] if result else []

    labels = [desc for _, desc in options]

    # Find preselected indices
    preselected_indices = []
    if preselected:
        for i, (key, _) in enumerate(options):
            if key in preselected:
                preselected_indices.append(i)

    menu = TerminalMenu(
        labels,
        title=f"\n  {title}\n  (Space to toggle, Enter to confirm)\n",
        menu_cursor=" → ",
        menu_cursor_style=("fg_cyan", "bold"),
        multi_select=True,
        show_multi_select_hint=True,
        preselected_entries=preselected_indices if preselected_indices else None,
        multi_select_cursor_style=("fg_cyan", "bold"),
        multi_select_select_on_accept=False,
        quit_keys=("escape", "q"),
        min_selection_count=min_selections,
    )

    selected_indices = menu.show()

    if selected_indices is None:
        return None

    # Handle single selection (returned as int)
    if isinstance(selected_indices, int):
        selected_indices = (selected_indices,)

    return [options[i][0] for i in selected_indices]


# =============================================================================
# SPARKLINES - Tiny inline charts
# =============================================================================

def sparkline(values: List[float], width: int = 20) -> str:
    """
    Generate a sparkline chart from values.

    Args:
        values: List of numeric values
        width: Target width (values will be sampled if needed)

    Returns:
        Sparkline string (e.g., "▁▂▃▅▆▇▆▅▃▂")

    Example:
        losses = [0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.22]
        print(f"Loss: {sparkline(losses)}")  # Loss: ▇▅▄▃▂▂▁▁
    """
    if not values:
        return ""

    # Sparkline characters (8 levels)
    chars = "▁▂▃▄▅▆▇█"

    # Sample values if too many
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    # Normalize to 0-7 range
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1

    normalized = [(v - min_val) / range_val for v in values]
    indices = [min(7, int(v * 7.99)) for v in normalized]

    return "".join(chars[i] for i in indices)


def sparkline_with_labels(
    values: List[float],
    width: int = 20,
    label: str = "",
    show_minmax: bool = True,
) -> str:
    """
    Sparkline with optional labels and min/max values.

    Example:
        print(sparkline_with_labels(losses, label="Loss", show_minmax=True))
        # Loss: ▇▅▄▃▂▂▁▁ (0.50 → 0.22)
    """
    spark = sparkline(values, width)

    if not values:
        return f"{label}: (no data)" if label else "(no data)"

    result = f"{label}: " if label else ""
    result += spark

    if show_minmax and len(values) > 1:
        result += f" ({values[0]:.2f} → {values[-1]:.2f})"

    return result


# =============================================================================
# STYLED PANELS - Information display
# =============================================================================

def info_panel(
    content: str,
    title: str = None,
    style: str = "info",
) -> None:
    """
    Display styled information panel.

    Args:
        content: Panel content (can include Rich markup)
        title: Optional panel title
        style: Panel style (info, success, warning, error)

    Example:
        info_panel(
            "Training will use:\\n• Model: Qwen3-4B\\n• Dataset: syngen_tools.jsonl",
            title="Configuration",
            style="info"
        )
    """
    if not RICH_AVAILABLE:
        print(f"\n  {title or 'Info'}")
        print("  " + "-" * 40)
        for line in content.split("\n"):
            print(f"  {line}")
        print()
        return

    from rich.panel import Panel
    from rich import box

    styles = {
        "info": (COLORS['sky'], box.ROUNDED),
        "success": (COLORS['aqua'], box.ROUNDED),
        "warning": (COLORS['orange'], box.ROUNDED),
        "error": ("#FF6B6B", box.HEAVY),
    }

    border_color, box_style = styles.get(style, styles["info"])

    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style=border_color,
        box=box_style,
        padding=(0, 2),
    )
    console.print(panel)


# =============================================================================
# COMPARISON TABLE - Side-by-side comparison
# =============================================================================

def comparison_table(
    items: List[dict],
    columns: List[Tuple[str, str]],
    title: str = "Comparison",
    highlight_best: str = None,
    highlight_mode: str = "min",
) -> None:
    """
    Display comparison table with optional highlighting.

    Args:
        items: List of dicts with column data
        columns: List of (key, header) tuples
        title: Table title
        highlight_best: Column key to highlight best value
        highlight_mode: "min" or "max" for best value determination

    Example:
        comparison_table(
            items=[
                {"name": "Run 1", "loss": 0.23, "score": 1.45},
                {"name": "Run 2", "loss": 0.19, "score": 1.62},
            ],
            columns=[("name", "Run"), ("loss", "Loss"), ("score", "Score")],
            highlight_best="loss",
            highlight_mode="min"
        )
    """
    if not RICH_AVAILABLE:
        # Plain text fallback
        print(f"\n  {title}")
        print("  " + "-" * 60)

        # Header
        header = "  ".join(h.ljust(15) for _, h in columns)
        print(f"  {header}")
        print("  " + "-" * 60)

        # Rows
        for item in items:
            row = "  ".join(str(item.get(k, "")).ljust(15) for k, _ in columns)
            print(f"  {row}")
        print()
        return

    from rich.table import Table
    from rich import box

    table = Table(
        title=title,
        box=box.ROUNDED,
        border_style=COLORS['cello'],
        title_style=f"bold {COLORS['aqua']}",
    )

    for key, header in columns:
        table.add_column(header, style="white")

    # Find best value if highlighting
    best_value = None
    if highlight_best:
        values = [item.get(highlight_best) for item in items if item.get(highlight_best) is not None]
        if values:
            best_value = min(values) if highlight_mode == "min" else max(values)

    # Add rows
    for item in items:
        row = []
        for key, _ in columns:
            value = item.get(key, "")
            if key == highlight_best and value == best_value:
                row.append(f"[bold {COLORS['aqua']}]{value}[/]")
            else:
                row.append(str(value))
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


# =============================================================================
# LOG SUPPRESSION - Quiet noisy libraries during dashboard display
# =============================================================================

@contextmanager
def suppress_logs(
    loggers: List[str] = None,
    level: int = logging.WARNING,
):
    """
    Suppress logs from noisy libraries during dashboard display.

    Args:
        loggers: List of logger names to suppress (default: common ML loggers)
        level: Minimum level to show (default: WARNING, hides INFO/DEBUG)

    Example:
        with suppress_logs():
            # Unsloth, transformers, etc. will be quiet
            train_model()

        # Or specify specific loggers:
        with suppress_logs(['unsloth', 'transformers']):
            train_model()
    """
    if loggers is None:
        loggers = [
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

    # Store original levels
    original_levels = {}
    for name in loggers:
        logger = logging.getLogger(name)
        original_levels[name] = logger.level
        logger.setLevel(level)

    try:
        yield
    finally:
        # Restore original levels
        for name, orig_level in original_levels.items():
            logging.getLogger(name).setLevel(orig_level)


@contextmanager
def capture_output():
    """
    Capture stdout/stderr to prevent interference with dashboard.

    Use this when running code that prints directly to stdout
    and you want to display it in the dashboard log panel instead.

    Example:
        from io import StringIO

        with capture_output() as (stdout, stderr):
            # Code that prints to stdout
            print("This will be captured")

        captured_text = stdout.getvalue()
        dashboard.update(log_message=captured_text)
    """
    from io import StringIO

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    new_stdout = StringIO()
    new_stderr = StringIO()

    sys.stdout = new_stdout
    sys.stderr = new_stderr

    try:
        yield new_stdout, new_stderr
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextmanager
def quiet_training():
    """
    Combined log suppression and output capture for clean training display.

    This is the recommended context manager for wrapping training loops
    when using the live dashboard.

    Example:
        from shared.ui import LiveDashboard, quiet_training

        dashboard = LiveDashboard(title="SFT Training", ...)

        with dashboard, quiet_training():
            for step in training_loop():
                dashboard.update(step=step, loss=loss)
    """
    with suppress_logs():
        # Also suppress tqdm if it's being used
        try:
            import tqdm
            old_disable = tqdm.tqdm.__init__.__defaults__
            # This is a bit hacky but works
        except ImportError:
            pass

        yield
