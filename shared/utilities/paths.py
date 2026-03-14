"""
Path utilities for trainer directories and training outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


TRAINING_METHODS = ("sft", "kto", "grpo")

CANONICAL_TRAINER_DIRS = {method: method for method in TRAINING_METHODS}
LEGACY_TRAINER_DIRS = {method: f"rtx3090_{method}" for method in TRAINING_METHODS}

CANONICAL_OUTPUT_DIRS = {method: f"{method}_output" for method in TRAINING_METHODS}
LEGACY_OUTPUT_DIRS = {method: f"{method}_output_rtx3090" for method in TRAINING_METHODS}


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root.
    """
    return Path(__file__).resolve().parents[2]


def get_trainers_dir(repo_root: Optional[Path] = None) -> Path:
    """Get the Trainers directory."""
    return (repo_root or get_project_root()) / "Trainers"


def normalize_trainer_method(trainer_name: str) -> str:
    """
    Normalize a trainer directory name or method to a training method.

    Args:
        trainer_name: Method name like ``sft`` or directory name like ``rtx3090_sft``

    Returns:
        Canonical method name.

    Raises:
        ValueError: If the trainer name cannot be mapped to a known method.
    """
    if trainer_name in TRAINING_METHODS:
        return trainer_name

    for method, legacy_name in LEGACY_TRAINER_DIRS.items():
        if trainer_name == legacy_name:
            return method

    raise ValueError(f"Unknown trainer name: {trainer_name}")


def get_canonical_trainer_dir_name(method: str) -> str:
    """Get the canonical trainer directory name for a method."""
    return CANONICAL_TRAINER_DIRS[normalize_trainer_method(method)]


def get_legacy_trainer_dir_name(method: str) -> str:
    """Get the legacy trainer directory name for a method."""
    return LEGACY_TRAINER_DIRS[normalize_trainer_method(method)]


def get_canonical_output_dir_name(method: str) -> str:
    """Get the canonical output directory name for a method."""
    return CANONICAL_OUTPUT_DIRS[normalize_trainer_method(method)]


def get_legacy_output_dir_name(method: str) -> str:
    """Get the legacy output directory name for a method."""
    return LEGACY_OUTPUT_DIRS[normalize_trainer_method(method)]


def get_trainer_dir_candidates(method: str, repo_root: Optional[Path] = None) -> list[Path]:
    """Return canonical and legacy trainer directory candidates."""
    normalized = normalize_trainer_method(method)
    trainers_dir = get_trainers_dir(repo_root)
    return [
        trainers_dir / get_canonical_trainer_dir_name(normalized),
        trainers_dir / get_legacy_trainer_dir_name(normalized),
    ]


def get_trainer_root(trainer_name: str = None, repo_root: Optional[Path] = None) -> Path:
    """
    Get the trainer root directory.

    Args:
        trainer_name: Name of trainer (e.g., ``sft`` or ``rtx3090_sft``)
        repo_root: Optional explicit repo root

    Returns:
        Path to trainer root. Prefers the canonical directory when present.
    """
    trainers_dir = get_trainers_dir(repo_root)
    if trainer_name is None:
        return trainers_dir

    candidates = get_trainer_dir_candidates(trainer_name, repo_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def iter_training_output_dirs(method: str, repo_root: Optional[Path] = None) -> list[Path]:
    """
    Return existing training output directories for a method.

    Prefers canonical output names but includes legacy output directories so
    existing runs remain discoverable after the rename.
    """
    normalized = normalize_trainer_method(method)
    output_names = [
        get_canonical_output_dir_name(normalized),
        get_legacy_output_dir_name(normalized),
    ]
    candidates: list[Path] = []

    for trainer_dir in get_trainer_dir_candidates(normalized, repo_root):
        for output_name in output_names:
            candidate = trainer_dir / output_name
            if candidate.exists() and candidate not in candidates:
                candidates.append(candidate)

    if candidates:
        return candidates

    preferred_trainer_dir = get_trainer_root(normalized, repo_root)
    return [preferred_trainer_dir / get_canonical_output_dir_name(normalized)]


def get_primary_training_output_dir(method: str, repo_root: Optional[Path] = None) -> Path:
    """
    Get the preferred output directory for new runs of a method.
    """
    normalized = normalize_trainer_method(method)
    return get_trainer_root(normalized, repo_root) / get_canonical_output_dir_name(normalized)


def is_training_output_dir(name: str) -> bool:
    """Return True if the directory name matches a canonical or legacy output root."""
    return name in set(CANONICAL_OUTPUT_DIRS.values()) | set(LEGACY_OUTPUT_DIRS.values())


def find_training_run(trainer_name: str, run_id: str = None, repo_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find a training run directory for a trainer.

    Args:
        trainer_name: Name of trainer (e.g., ``sft`` or ``rtx3090_sft``)
        run_id: Specific run ID or None for latest
        repo_root: Optional explicit repo root

    Returns:
        Path to training run directory or None.
    """
    method = normalize_trainer_method(trainer_name)
    output_dirs = [path for path in iter_training_output_dirs(method, repo_root) if path.exists()]

    if run_id:
        for output_dir in output_dirs:
            run_path = output_dir / run_id
            if run_path.exists():
                return run_path
        return None

    runs = []
    for output_dir in output_dirs:
        runs.extend(d for d in output_dir.iterdir() if d.is_dir())

    runs = sorted(runs, key=lambda path: path.name, reverse=True)
    return runs[0] if runs else None
