"""Rubric repository - facade for rubric data access.

Single Responsibility: Coordinate rubric loading and provide query interface.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .rubric_loader import RubricLoader
from .rubric_cache import RubricCache
from shared.utilities import load_yaml


@dataclass
class RubricMetadata:
    """Metadata about a rubric (from initial scan)."""
    key: str
    name: str
    description: str
    filename: str
    scope: str
    pass_threshold: float


class RubricRepository:
    """
    Rubric repository - facade for rubric data access.

    Responsibility: Coordinate loader + cache, provide query interface (SRP).
    Delegates actual loading to RubricLoader, caching to RubricCache.
    """

    def __init__(
        self,
        rubrics_dir: Path,
        skip_files: Optional[List[str]] = None,
        cache_max_size: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize rubric repository.

        Args:
            rubrics_dir: Directory containing rubric YAML files
            skip_files: List of filenames to skip (e.g., ["quality_labels.yaml"])
            cache_max_size: Maximum cache size
            logger: Logger instance
        """
        self.rubrics_dir = Path(rubrics_dir)
        self.skip_files = set(skip_files or [])
        self.logger = logger or logging.getLogger(__name__)

        # Delegate to focused services
        self.loader = RubricLoader(rubrics_dir, logger)
        self.cache = RubricCache(max_size=cache_max_size)

        # Metadata store (populated on init)
        self._metadata: Dict[str, RubricMetadata] = {}
        self._discover_rubrics()

    def get_rubric(self, key: str) -> Dict:
        """
        Get rubric by key (with caching).

        Args:
            key: Rubric key (filename stem)

        Returns:
            Rubric dict

        Raises:
            ValueError: If rubric not found
            FileNotFoundError: If rubric file doesn't exist
        """
        if key not in self._metadata:
            raise ValueError(
                f"Rubric '{key}' not found. Available: {self.list_keys()}"
            )

        # Try cache first
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # Load from disk
        rubric = self.loader.load_from_file(key)

        # Cache it
        self.cache.put(key, rubric)

        return rubric

    def get_metadata(self, key: str) -> Optional[RubricMetadata]:
        """
        Get rubric metadata (without loading full rubric).

        Args:
            key: Rubric key

        Returns:
            RubricMetadata or None if not found
        """
        return self._metadata.get(key)

    def list_keys(self) -> List[str]:
        """
        List all available rubric keys.

        Returns:
            List of rubric keys
        """
        return list(self._metadata.keys())

    def list_metadata(self) -> List[RubricMetadata]:
        """
        List metadata for all available rubrics.

        Returns:
            List of RubricMetadata
        """
        return list(self._metadata.values())

    def get_by_scope(self, scope: str) -> List[str]:
        """
        Get rubric keys for a specific scope.

        Args:
            scope: Scope name (e.g., "thinking", "system_prompt")

        Returns:
            List of rubric keys for that scope
        """
        return [
            key for key, meta in self._metadata.items()
            if meta.scope == scope
        ]

    def exists(self, key: str) -> bool:
        """
        Check if rubric exists.

        Args:
            key: Rubric key

        Returns:
            True if rubric exists
        """
        return key in self._metadata

    def reload(self) -> None:
        """
        Reload rubrics from disk (clears cache and rediscovers).

        Useful if rubric files have been modified.
        """
        self.cache.clear()
        self._metadata.clear()
        self._discover_rubrics()

    def add_skip_file(self, filename: str) -> None:
        """
        Add file to skip list and reload.

        Args:
            filename: Filename to skip (e.g., "my_rubric.yaml")
        """
        self.skip_files.add(filename)
        self.reload()

    def remove_skip_file(self, filename: str) -> None:
        """
        Remove file from skip list and reload.

        Args:
            filename: Filename to stop skipping
        """
        self.skip_files.discard(filename)
        self.reload()

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics
        """
        return {
            "total_rubrics": len(self._metadata),
            **self.cache.get_stats()
        }

    def _discover_rubrics(self) -> None:
        """
        Scan rubrics directory and extract metadata.

        Populates _metadata dict with RubricMetadata for each rubric.
        """
        for yaml_file in self.rubrics_dir.glob("*.yaml"):
            if yaml_file.name in self.skip_files:
                continue

            try:
                # Load just to extract metadata (not cached yet)
                data = load_yaml(yaml_file)
                key = yaml_file.stem

                self._metadata[key] = RubricMetadata(
                    key=key,
                    name=data.get("name", key),
                    description=data.get("description", ""),
                    filename=yaml_file.name,
                    scope=data.get("scope", "response"),
                    pass_threshold=data.get("pass_threshold", 0.8)
                )
            except Exception as e:
                self.logger.warning(f"Could not load rubric metadata from {yaml_file.name}: {e}")
