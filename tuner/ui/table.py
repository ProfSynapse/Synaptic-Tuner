"""
Table rendering utilities for checkpoint display.

Location: /mnt/f/Code/Toolset-Training/tuner/ui/table.py
Purpose: Specialized table functions for displaying checkpoints with metrics
Used by: UploadHandler for checkpoint selection
"""

from typing import List
from tuner.core.config import CheckpointInfo

# Import from shared UI
from Trainers.shared.ui import (
    RICH_AVAILABLE,
    console,
    COLORS,
    BOX,
)


def print_checkpoint_table(checkpoints: List[CheckpointInfo], training_type: str):
    """
    Print checkpoint table with metrics based on training type.

    For KTO training, displays: Step, Loss, KL, Margin, Score (Margin/KL)
    For SFT training, displays: Step, Loss, Learning Rate

    Args:
        checkpoints: List of CheckpointInfo objects
        training_type: 'sft' or 'kto'

    Example:
        checkpoints = discovery.discover(run_dir=Path('/path/to/run'))
        print_checkpoint_table(checkpoints, 'kto')
    """
    if RICH_AVAILABLE:
        _print_rich_checkpoint_table(checkpoints, training_type)
    else:
        _print_text_checkpoint_table(checkpoints, training_type)


def _print_rich_checkpoint_table(checkpoints: List[CheckpointInfo], training_type: str):
    """Print checkpoint table using rich library."""
    from rich.table import Table
    from rich import box as rich_box

    table = Table(
        title="Available Checkpoints",
        box=rich_box.ROUNDED,
        border_style=COLORS["cello"],
        show_header=True,
        header_style=f"bold {COLORS['aqua']}"
    )

    # Add columns
    table.add_column("#", style=COLORS["orange"], width=4, justify="center")
    table.add_column("Checkpoint", style="white")
    table.add_column("Step", style=COLORS["sky"], justify="right")
    table.add_column("Loss", justify="right")

    if training_type == "kto":
        table.add_column("KL", justify="right")
        table.add_column("Margin", justify="right")
        table.add_column("Score", justify="right", style=COLORS["aqua"])
    else:
        table.add_column("LR", justify="right")

    # Add rows
    for idx, cp in enumerate(checkpoints, 1):
        if cp.is_final:
            # Final model row
            row = [str(idx), f"{BOX['star']} final_model", "-", "-"]
            if training_type == "kto":
                row.extend(["-", "-", "-"])
            else:
                row.append("-")
        else:
            # Checkpoint row
            loss = f"{cp.metrics.get('loss', 0):.4f}" if cp.metrics else "-"
            row = [str(idx), cp.path.name, str(cp.step), loss]

            if training_type == "kto":
                kl = cp.metrics.get('kl', 0) if cp.metrics else 0
                margin = cp.metrics.get('rewards/margins', 0) if cp.metrics else 0
                score = margin / kl if kl > 0 else 0

                row.extend([
                    f"{kl:.2f}" if cp.metrics else "-",
                    f"{margin:.2f}" if cp.metrics else "-",
                    f"{score:.2f}" if cp.metrics else "-"
                ])
            else:
                lr = cp.metrics.get('learning_rate', 0) if cp.metrics else 0
                row.append(f"{lr:.2e}" if cp.metrics else "-")

        table.add_row(*row)

    console.print()
    console.print(table)

    if training_type == "kto":
        console.print(f"\n  [dim]Score = Margin/KL (higher is better: high margin, low KL)[/dim]")
    console.print()


def _print_text_checkpoint_table(checkpoints: List[CheckpointInfo], training_type: str):
    """Print checkpoint table using plain text."""
    print("\nAvailable Checkpoints:")
    print("-" * 80)

    # Header
    if training_type == "kto":
        print(f"{'#':<4} {'Checkpoint':<20} {'Step':<8} {'Loss':<10} {'KL':<8} {'Margin':<8} {'Score':<8}")
    else:
        print(f"{'#':<4} {'Checkpoint':<20} {'Step':<8} {'Loss':<10} {'LR':<12}")
    print("-" * 80)

    # Rows
    for idx, cp in enumerate(checkpoints, 1):
        if cp.is_final:
            # Final model row
            if training_type == "kto":
                print(f"{idx:<4} {'final_model':<20} {'-':<8} {'-':<10} {'-':<8} {'-':<8} {'-':<8}")
            else:
                print(f"{idx:<4} {'final_model':<20} {'-':<8} {'-':<10} {'-':<12}")
        else:
            # Checkpoint row
            loss = f"{cp.metrics.get('loss', 0):.4f}" if cp.metrics else "-"

            if training_type == "kto":
                kl = cp.metrics.get('kl', 0) if cp.metrics else 0
                margin = cp.metrics.get('rewards/margins', 0) if cp.metrics else 0
                score = margin / kl if kl > 0 else 0

                print(f"{idx:<4} {cp.path.name:<20} {cp.step:<8} {loss:<10} {kl:<8.2f} {margin:<8.2f} {score:<8.2f}")
            else:
                lr = cp.metrics.get('learning_rate', 0) if cp.metrics else 0
                print(f"{idx:<4} {cp.path.name:<20} {cp.step:<8} {loss:<10} {lr:<12.2e}")

    if training_type == "kto":
        print("\nScore = Margin/KL (higher is better: high margin, low KL)")
    print()
