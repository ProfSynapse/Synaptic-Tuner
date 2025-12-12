"""Shared version management for dataset files."""

import re
from pathlib import Path
from typing import Optional

from .dataset_scanner import DatasetScanner


class VersionManager:
    """
    Manages semantic versioning for dataset files.

    Handles version extraction and incrementation:
    - v1.5 → v1.5.1 (adds patch)
    - v1.5.1 → v1.5.2 (increments patch)
    """

    def __init__(self):
        self.scanner = DatasetScanner()

    def extract_version(self, filename: str) -> Optional[str]:
        """
        Extract version from filename.

        Args:
            filename: Filename to parse (e.g., "tools_v1.5.jsonl")

        Returns:
            Version string (e.g., "v1.5") or None if not found

        Examples:
            >>> vm = VersionManager()
            >>> vm.extract_version("tools_v1.5.jsonl")
            'v1.5'
            >>> vm.extract_version("tools_v1.5.1.jsonl")
            'v1.5.1'
        """
        # Remove .jsonl extension if present
        if filename.endswith('.jsonl'):
            filename = filename[:-6]

        # Extract version using regex (v1.5 or v1.5.1 format)
        version_match = re.search(r'_v(\d+)\.(\d+)(?:\.(\d+))?$', filename)

        if not version_match:
            return None

        major = version_match.group(1)
        minor = version_match.group(2)
        patch = version_match.group(3)

        version = f"v{major}.{minor}"
        if patch:
            version += f".{patch}"

        return version

    def get_auto_versioned_path(self, input_path: str) -> Path:
        """
        Generate output path with auto-incremented version.

        Args:
            input_path: Input file path

        Returns:
            Output path with incremented version

        Raises:
            ValueError: If version cannot be detected in filename

        Examples:
            tools_v1.5.jsonl → tools_v1.5.1.jsonl
            tools_v1.5.1.jsonl → tools_v1.5.2.jsonl
        """
        input_path = Path(input_path)
        filename = input_path.stem  # Without .jsonl

        # Extract current version
        current_version = self.extract_version(filename)

        if not current_version:
            raise ValueError(
                f"Could not detect version in filename: {filename}\n"
                "Expected format: *_v1.5.jsonl or *_v1.5.1.jsonl"
            )

        # Get next version
        next_version = self.scanner.get_next_version(current_version)

        # Build new filename
        version_match = re.search(r'_v(\d+)\.(\d+)(?:\.(\d+))?$', filename)
        base_name = filename[:version_match.start()]  # Everything before _v1.5
        new_filename = f"{base_name}_{next_version}.jsonl"
        output_path = input_path.parent / new_filename

        return output_path

    def get_version_info(self, input_path: str) -> dict:
        """
        Get version information for a file.

        Args:
            input_path: Input file path

        Returns:
            Dictionary with current_version, next_version, output_path

        Raises:
            ValueError: If version cannot be detected
        """
        input_path = Path(input_path)
        filename = input_path.stem

        current_version = self.extract_version(filename)

        if not current_version:
            raise ValueError(
                f"Could not detect version in filename: {filename}\n"
                "Expected format: *_v1.5.jsonl or *_v1.5.1.jsonl"
            )

        next_version = self.scanner.get_next_version(current_version)
        output_path = self.get_auto_versioned_path(str(input_path))

        return {
            "current_version": current_version,
            "next_version": next_version,
            "input_path": str(input_path),
            "output_path": str(output_path)
        }
