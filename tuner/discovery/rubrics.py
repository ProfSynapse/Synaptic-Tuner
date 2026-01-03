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


@dataclass
class RubricInfo:
    """Information about a discovered rubric."""
    path: Path
    name: str
    description: str
    scope: Optional[str]
    source: str  # Which directory (SynthChat, improvement_engine, shared)


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

    def __init__(self, repo_root: Path = None):
        """
        Initialize the rubric discovery service.

        Args:
            repo_root: Repository root path. If None, uses module location to find repo root.
        """
        if repo_root is None:
            self.repo_root = Path(__file__).parent.parent.parent
        else:
            self.repo_root = repo_root

    def discover_all(self) -> List[RubricInfo]:
        """
        Discover all rubric YAML files from all rubric directories.

        Returns:
            List of RubricInfo objects sorted by name.
        """
        results: List[RubricInfo] = []
        seen_names = set()

        for dir_path, source in self.RUBRIC_DIRS:
            rubrics_dir = self.repo_root / dir_path

            if not rubrics_dir.exists():
                continue

            for filepath in sorted(rubrics_dir.glob("*.yaml")):
                name = filepath.stem

                # Skip duplicates (first occurrence wins)
                if name in seen_names:
                    continue

                try:
                    info = self._analyze_rubric(filepath, source)
                    if info:
                        results.append(info)
                        seen_names.add(name)
                except Exception:
                    # Skip files that can't be parsed
                    continue

        # Sort by name
        results.sort(key=lambda r: r.name)
        return results

    def _analyze_rubric(self, filepath: Path, source: str) -> Optional[RubricInfo]:
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
        )
