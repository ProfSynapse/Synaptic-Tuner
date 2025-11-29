"""
Path utilities.
"""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root (Toolset-Training/)
    """
    # Navigate up from shared/utilities/paths.py
    return Path(__file__).parent.parent.parent.parent


def get_trainer_root(trainer_name: str = None) -> Path:
    """
    Get the trainer root directory.

    Args:
        trainer_name: Name of trainer (e.g., "rtx3090_sft")

    Returns:
        Path to trainer root
    """
    trainers_dir = get_project_root() / "Trainers"

    if trainer_name:
        return trainers_dir / trainer_name

    return trainers_dir


def get_shared_root() -> Path:
    """
    Get the shared module root directory.

    Returns:
        Path to Trainers/shared/
    """
    return Path(__file__).parent.parent


def find_training_run(
    trainer_name: str,
    run_id: str = None
) -> Optional[Path]:
    """
    Find a training run directory.

    Args:
        trainer_name: Name of trainer (e.g., "rtx3090_sft")
        run_id: Specific run ID (e.g., "20251122_143000") or None for latest

    Returns:
        Path to training run directory or None
    """
    trainer_root = get_trainer_root(trainer_name)

    # Determine output directory name
    if "sft" in trainer_name:
        output_dir = trainer_root / "sft_output_rtx3090"
    elif "kto" in trainer_name:
        output_dir = trainer_root / "kto_output_rtx3090"
    else:
        return None

    if not output_dir.exists():
        return None

    if run_id:
        run_path = output_dir / run_id
        return run_path if run_path.exists() else None

    # Find latest run
    runs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )

    return runs[0] if runs else None
