"""Rubric loader - handles ONLY file I/O operations.

Single Responsibility: Load rubric YAML files from disk.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from shared.utilities import load_yaml


class RubricLoader:
    """
    Loads rubric YAML files from disk.

    Responsibility: ONLY file I/O operations (SRP).
    Does NOT handle caching, discovery, or querying.
    """

    def __init__(self, rubrics_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize rubric loader.

        Args:
            rubrics_dir: Directory containing rubric YAML files
            logger: Logger instance
        """
        self.rubrics_dir = Path(rubrics_dir)
        self.logger = logger or logging.getLogger(__name__)

    def load_from_file(self, rubric_key: str) -> Dict:
        """
        Load rubric YAML file from disk.

        Args:
            rubric_key: Rubric key (filename stem)

        Returns:
            Rubric dict with 'key' field injected

        Raises:
            FileNotFoundError: If rubric file doesn't exist
            ValueError: If YAML parsing fails
        """
        yaml_file = self.rubrics_dir / f"{rubric_key}.yaml"

        if not yaml_file.exists():
            raise FileNotFoundError(f"Rubric file not found: {yaml_file}")

        try:
            rubric = load_yaml(yaml_file)
            # Inject the key into the rubric dict so downstream services can access it
            rubric['key'] = rubric_key
            return rubric
        except Exception as e:
            raise ValueError(f"Failed to load rubric {rubric_key}: {e}")

    def exists(self, rubric_key: str) -> bool:
        """
        Check if rubric file exists.

        Args:
            rubric_key: Rubric key (filename stem)

        Returns:
            True if file exists
        """
        yaml_file = self.rubrics_dir / f"{rubric_key}.yaml"
        return yaml_file.exists()
