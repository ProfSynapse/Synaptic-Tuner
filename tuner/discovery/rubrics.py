"""
Rubrics discovery service.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/rubrics.py
Purpose: Discover and enumerate available rubric YAML files for data improvement
Used by: List handler to display rubrics from SynthChat and shared directories

This module implements the RubricDiscovery service which scans rubric directories
for YAML files and extracts metadata about each rubric.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from tuner.project import ProjectContext


@dataclass
class RubricInfo:
    """Information about a discovered rubric."""
    path: Path
    name: str
    description: str
    scope: Optional[str]
    source: str  # Which directory (SynthChat, improvement_engine, shared)
    declaring_root: Optional[Path] = None


class RubricDiscovery:
    """
    Discover available rubric YAML files.

    This service scans multiple directories for rubric definitions:
    - SynthChat/rubrics/
    - improvement_engine/rubrics/ (if exists)
    - shared/validation/rubrics/ (if exists)

    Example:
        from tuner.discovery import RubricDiscovery

        discovery = RubricDiscovery()
        rubrics = discovery.discover_all()

        for rubric in rubrics:
            print(f"{rubric.name}: {rubric.description}")
    """

    # Directories to search for rubrics (relative to repo root)
    RUBRIC_DIRS = [
        ("SynthChat/rubrics", "SynthChat"),
        ("improvement_engine/rubrics", "improvement_engine"),
        ("shared/validation/rubrics", "shared"),
    ]

    def __init__(
        self,
        repo_root: Path = None,
        *,
        context: ProjectContext | None = None,
    ):
        """
        Initialize the rubric discovery service.

        Args:
            repo_root: Repository root path. If None, uses module location to find repo root.
        """
        self.context = context
        self.repo_root = context.engine_root if context else (repo_root or Path(__file__).parent.parent.parent)

    def _rubric_dirs(self) -> list[tuple[Path, str]]:
        roots: list[tuple[Path, str]] = []
        if self.context is not None and self.context.mode == "host":
            roots.extend(
                [
                    (self.context.project_root / "rubrics", "project"),
                    (self.context.project_root / "SynthChat" / "rubrics", "project:SynthChat"),
                    (self.context.config_root / "rubrics", "project:config"),
                ]
            )
        roots.extend((self.repo_root / path, source) for path, source in self.RUBRIC_DIRS)
        seen: set[Path] = set()
        unique: list[tuple[Path, str]] = []
        for path, source in roots:
            resolved = path.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append((path, source))
        return unique

    def discover_all(self) -> List[RubricInfo]:
        """
        Discover all rubric YAML files from all rubric directories.

        Returns:
            List of RubricInfo objects sorted by name.
        """
        results: List[RubricInfo] = []
        seen_names = set()

        for rubrics_dir, source in self._rubric_dirs():

            if not rubrics_dir.exists():
                continue

            for filepath in sorted(rubrics_dir.glob("*.yaml")):
                name = filepath.stem

                # Skip duplicates (first occurrence wins)
                if name in seen_names:
                    continue

                try:
                    info = self._analyze_rubric(filepath, source, rubrics_dir)
                    if info:
                        results.append(info)
                        seen_names.add(name)
                except Exception:
                    # Skip files that can't be parsed
                    continue

        # Sort by name
        results.sort(key=lambda r: r.name)
        return results

    def _analyze_rubric(
        self, filepath: Path, source: str, declaring_root: Path
    ) -> Optional[RubricInfo]:
        """
        Analyze a single rubric file and extract metadata.

        Args:
            filepath: Path to the YAML rubric file
            source: Source directory name

        Returns:
            RubricInfo object or None if file is invalid
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            return None

        name = data.get('name', filepath.stem)
        description = data.get('description', '')
        scope = data.get('scope')

        return RubricInfo(
            path=filepath,
            name=filepath.stem,  # Use filename as identifier
            description=description if description else name,
            scope=scope,
            source=source,
            declaring_root=declaring_root,
        )
