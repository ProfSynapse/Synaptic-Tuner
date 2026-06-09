"""Rubric loader -- load YAML rubric files into typed RubricDef instances.

Location: shared/judge/rubric_loader.py
Summary: Loads rubric YAML files from a directory and returns RubricDef
         dataclass instances. No caching -- consumers load once at startup.
         Used by both Evaluator (via RubricLoader directly) and SynthChat.
"""

import logging
from pathlib import Path
from typing import List

import yaml

from .models import RubricDef

logger = logging.getLogger(__name__)

# Fields required in every rubric YAML
_REQUIRED_FIELDS = ("name", "description", "scope", "pass_threshold", "judge_prompt", "output_schema")


class RubricLoader:
    """Load rubric YAML files from a directory into RubricDef instances.

    Args:
        rubrics_dir: Path to the directory containing rubric YAML files.
    """

    def __init__(self, rubrics_dir: Path):
        self.rubrics_dir = Path(rubrics_dir)

    def load(self, rubric_key: str) -> RubricDef:
        """Load a single rubric by key (filename stem).

        Args:
            rubric_key: Rubric identifier, e.g., "tool_call_quality".

        Returns:
            RubricDef instance populated from the YAML file.

        Raises:
            FileNotFoundError: If the rubric YAML file does not exist.
            ValueError: If the YAML is malformed, missing required fields,
                        or contains path traversal characters.
        """
        # Reject path traversal attempts
        if ".." in rubric_key or "/" in rubric_key or "\\" in rubric_key:
            raise ValueError(
                f"Invalid rubric key '{rubric_key}': must not contain '..', '/', or '\\'"
            )

        yaml_path = self.rubrics_dir / f"{rubric_key}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Rubric file not found: {yaml_path}")

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse rubric YAML '{rubric_key}': {exc}")

        if not isinstance(data, dict):
            raise ValueError(f"Rubric '{rubric_key}' YAML did not produce a dict")

        # Validate required fields
        missing = [field for field in _REQUIRED_FIELDS if field not in data]
        if missing:
            raise ValueError(
                f"Rubric '{rubric_key}' missing required fields: {', '.join(missing)}"
            )

        return RubricDef(
            key=rubric_key,
            name=data["name"],
            description=data["description"],
            scope=data["scope"],
            pass_threshold=float(data["pass_threshold"]),
            judge_prompt=data["judge_prompt"],
            output_schema=data["output_schema"],
            improver_prompt=data.get("improver_prompt"),
            dimensions=data.get("dimensions"),
            weights_ratified=bool(data.get("weights_ratified", False)),
        )

    def load_many(self, rubric_keys: List[str]) -> List[RubricDef]:
        """Load multiple rubrics by key. Raises on first failure.

        Args:
            rubric_keys: List of rubric identifiers to load.

        Returns:
            List of RubricDef instances in the same order as the input keys.
        """
        return [self.load(key) for key in rubric_keys]

    def list_available(self) -> List[str]:
        """Return available rubric keys (YAML filenames without extension).

        Returns:
            Sorted list of rubric key strings found in the rubrics directory.
        """
        if not self.rubrics_dir.is_dir():
            logger.warning("Rubrics directory does not exist: %s", self.rubrics_dir)
            return []

        return sorted(p.stem for p in self.rubrics_dir.glob("*.yaml"))

    def exists(self, rubric_key: str) -> bool:
        """Check if a rubric YAML file exists.

        Args:
            rubric_key: Rubric identifier to check.

        Returns:
            True if the corresponding YAML file exists on disk.
        """
        yaml_path = self.rubrics_dir / f"{rubric_key}.yaml"
        return yaml_path.exists()
