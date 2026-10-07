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

from shared.utilities.paths import iter_training_output_dirs
from tuner.project import ProjectContext


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

    def __init__(
        self,
        repo_root: Path = None,
        *,
        context: ProjectContext | None = None,
    ):
        """
        Initialize the training run discovery service.

        Args:
            repo_root: Repository root path. If None, uses current working directory's parent.
        """
        self.context = context
        self.repo_root = context.engine_root if context else (repo_root or Path(__file__).parent.parent.parent)

    def output_roots(self, trainer_type: str) -> list[Path]:
        roots: list[Path] = []
        if self.context is not None and self.context.mode == "host":
            roots.extend(
                [
                    self.context.artifact_root / "runs" / "local_docker" / trainer_type,
                    self.context.artifact_root / "runs" / trainer_type,
                    self.context.artifact_root / trainer_type,
                ]
            )
        roots.extend(iter_training_output_dirs(trainer_type, self.repo_root))
        return list(dict.fromkeys(root.resolve(strict=False) for root in roots))

    def discover(self, trainer_type: str, limit: int = 10) -> List[Path]:
        """
        Discover training runs for a specific trainer type.

        Scans the output directory for the specified trainer type and returns
        runs that contain either a final_model or checkpoints directory.
        Results are sorted by modification time (newest first).

        Args:
            trainer_type: Type of trainer ('sft', 'kto', or 'grpo')
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
            Trainers/{trainer_type}/{trainer_type}_output/
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
        runs = []

        seen: set[Path] = set()
        for output_dir in self.output_roots(trainer_type):
            if not output_dir.exists():
                continue
            candidates = [path.parent for path in output_dir.rglob("final_model") if path.is_dir()]
            candidates.extend(
                path.parent
                for path in output_dir.rglob("checkpoints")
                if path.is_dir() and any(path.iterdir())
            )
            for candidate in sorted(
                set(candidates), key=lambda p: (-p.stat().st_mtime, p.as_posix())
            ):
                resolved = candidate.resolve(strict=False)
                if resolved not in seen:
                    runs.append(candidate)
                    seen.add(resolved)

        # Host mode preserves declaring-root precedence: project artifacts are
        # always listed before engine-local legacy defaults. Standalone keeps
        # the historical global newest-first ordering.
        if self.context is None or self.context.mode == "standalone":
            runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if limit is not None:
            runs = runs[:limit]

        return runs
