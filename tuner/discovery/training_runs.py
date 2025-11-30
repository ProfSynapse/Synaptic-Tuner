"""
Training run discovery service.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/training_runs.py
Purpose: Discover and enumerate available training runs
Used by: Upload handler to list training runs for model selection

This module implements the TrainingRunDiscovery service which scans the
filesystem for training run directories. It filters runs to include only
those with final_model or checkpoint directories, and sorts them by
modification time (newest first).

Pattern migrated from: tuner.py lines 220-234 (list_training_runs function)
"""

from pathlib import Path
from typing import List


class TrainingRunDiscovery:
    """
    Discover available training runs.

    This service scans the output directories for SFT and KTO training runs,
    identifying runs that contain either a final_model or checkpoints directory.
    Results are sorted by modification time with newest runs first.

    Example:
        from tuner.discovery import TrainingRunDiscovery

        # Discover SFT runs
        discovery = TrainingRunDiscovery()
        sft_runs = discovery.discover('sft', limit=10)

        # Discover KTO runs
        kto_runs = discovery.discover('kto', limit=5)

        # Get all runs (no limit)
        all_runs = discovery.discover('sft', limit=None)
    """

    def __init__(self, repo_root: Path = None):
        """
        Initialize the training run discovery service.

        Args:
            repo_root: Repository root path. If None, uses current working directory's parent.
        """
        if repo_root is None:
            # Default to repo root (assumes we're in tuner/ or subdirectory)
            self.repo_root = Path(__file__).parent.parent.parent
        else:
            self.repo_root = repo_root

    def discover(self, trainer_type: str, limit: int = 10) -> List[Path]:
        """
        Discover training runs for a specific trainer type.

        Scans the output directory for the specified trainer type and returns
        runs that contain either a final_model or checkpoints directory.
        Results are sorted by modification time (newest first).

        Args:
            trainer_type: Type of trainer ('sft' or 'kto')
            limit: Maximum number of runs to return (default: 10). Use None for no limit.

        Returns:
            List of Path objects to training run directories, sorted newest first.
            Returns empty list if output directory doesn't exist or contains no valid runs.

        Example:
            discovery = TrainingRunDiscovery(repo_root=Path('/path/to/repo'))

            # Get most recent 10 SFT runs
            sft_runs = discovery.discover('sft', limit=10)

            # Get all KTO runs
            all_kto_runs = discovery.discover('kto', limit=None)

            # Check if we have any runs
            if sft_runs:
                latest_run = sft_runs[0]
                print(f"Latest run: {latest_run.name}")

        Directory structure expected:
            Trainers/rtx3090_{trainer_type}/{trainer_type}_output_rtx3090/
            ├── 20251122_143000/
            │   ├── final_model/       <- Valid (has final_model)
            │   ├── checkpoints/
            │   └── logs/
            ├── 20251122_120000/
            │   ├── checkpoints/       <- Valid (has checkpoints)
            │   │   ├── checkpoint-50/
            │   │   └── checkpoint-100/
            │   └── logs/
            └── 20251122_100000/       <- Invalid (no final_model or checkpoints)
                └── logs/
        """
        # Build output directory path
        output_dir = (
            self.repo_root
            / "Trainers"
            / f"rtx3090_{trainer_type}"
            / f"{trainer_type}_output_rtx3090"
        )

        # Return empty list if directory doesn't exist
        if not output_dir.exists():
            return []

        runs = []

        # Iterate through directories, sorted by modification time (newest first)
        for d in sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue

            # Include runs with final_model OR checkpoints
            has_final = (d / "final_model").exists()
            has_checkpoints = (d / "checkpoints").exists() and any((d / "checkpoints").iterdir())

            if has_final or has_checkpoints:
                runs.append(d)

                # Apply limit if specified
                if limit is not None and len(runs) >= limit:
                    break

        return runs
