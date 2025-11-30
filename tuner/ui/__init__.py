"""
UI wrapper module for tuner CLI.

Location: /mnt/f/Code/Toolset-Training/tuner/ui/__init__.py
Purpose: Re-export UI functions from Trainers/shared/ui/ for handler use
Used by: All handlers for menu rendering, user prompts, and output

This module serves as a thin wrapper that delegates to the shared UI components
in Trainers/shared/ui/. It provides a consistent import path for handlers while
maintaining graceful fallbacks if rich is not available.
"""

import sys
from pathlib import Path

# Add Trainers directory to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from Trainers.shared.ui import (
    console,
    RICH_AVAILABLE,
    COLORS,
    BOX,
    STYLES,
    clear_screen,
    print_logo,
    print_header,
    print_menu,
    print_config,
    print_success,
    print_error,
    print_info,
    confirm,
    prompt,
)

# Import tuner-specific UI functions
from .table import print_checkpoint_table

__all__ = [
    "console",
    "RICH_AVAILABLE",
    "COLORS",
    "BOX",
    "STYLES",
    "clear_screen",
    "print_logo",
    "print_header",
    "print_menu",
    "print_config",
    "print_success",
    "print_error",
    "print_info",
    "confirm",
    "prompt",
    "print_table",
    "print_checkpoint_table",
]


def print_table(data: list, headers: list, title: str = None):
    """
    Print a table with optional title.

    This is a convenience function for handlers that need to display
    tabular data. Uses rich.Table if available, falls back to plain text.

    Args:
        data: List of rows (each row is a list of column values)
        headers: List of column headers
        title: Optional table title

    Example:
        print_table(
            data=[
                ["1", "20251130_143000", "final_model"],
                ["2", "20251129_120000", "checkpoint-500"],
            ],
            headers=["#", "Run", "Model"],
            title="Available Training Runs"
        )
    """
    if RICH_AVAILABLE:
        from rich.table import Table
        from rich import box as rich_box

        table = Table(
            title=title,
            box=rich_box.ROUNDED,
            border_style=COLORS["cello"],
            show_header=True,
            header_style=f"bold {COLORS['aqua']}"
        )

        # Add columns
        for i, header in enumerate(headers):
            if i == 0:
                # First column (usually index) - orange, center aligned
                table.add_column(header, style=COLORS["orange"], width=4, justify="center")
            else:
                table.add_column(header, style="white")

        # Add rows
        for row in data:
            table.add_row(*[str(cell) for cell in row])

        console.print()
        console.print(table)
        console.print()
    else:
        # Fallback text display
        if title:
            print(f"\n{title}")
        print("-" * 80)
        # Print headers
        header_row = "  ".join(f"{h:<20}" for h in headers)
        print(header_row)
        print("-" * 80)
        # Print data
        for row in data:
            data_row = "  ".join(f"{str(cell):<20}" for cell in row)
            print(data_row)
        print()
