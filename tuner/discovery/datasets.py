"""
Dataset discovery service.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/datasets.py
Purpose: Discover and enumerate available JSONL datasets for training
Used by: List handler to display datasets with metadata (type, examples, size)

This module implements the DatasetDiscovery service which scans the Datasets/
folder for JSONL files and extracts metadata about each dataset.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    path: Path
    name: str
    dataset_type: str  # 'SFT' or 'KTO'
    example_count: int
    size_bytes: int
    size_human: str
    relative_path: str


class DatasetDiscovery:
    """
    Discover available JSONL datasets for training.

    This service scans the Datasets/ folder for JSONL files and extracts
    metadata including example count, file size, and inferred dataset type.

    Example:
        from tuner.discovery import DatasetDiscovery

        discovery = DatasetDiscovery()
        datasets = discovery.discover_all()

        for ds in datasets:
            print(f"{ds.name}: {ds.dataset_type} ({ds.example_count} examples, {ds.size_human})")
    """

    def __init__(self, repo_root: Path = None):
        """
        Initialize the dataset discovery service.

        Args:
            repo_root: Repository root path. If None, uses module location to find repo root.
        """
        if repo_root is None:
            self.repo_root = Path(__file__).parent.parent.parent
        else:
            self.repo_root = repo_root

    def discover_all(self) -> List[DatasetInfo]:
        """
        Discover all JSONL datasets in the Datasets/ folder.

        Recursively scans Datasets/ for .jsonl files and extracts metadata
        for each file including example count, size, and dataset type.

        Returns:
            List of DatasetInfo objects sorted by modification time (newest first).
        """
        datasets_dir = self.repo_root / "Datasets"

        if not datasets_dir.exists():
            return []

        results: List[DatasetInfo] = []

        # Find all JSONL files
        jsonl_files = list(datasets_dir.rglob("*.jsonl"))

        # Sort by modification time (newest first)
        jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for filepath in jsonl_files:
            try:
                info = self._analyze_dataset(filepath, datasets_dir)
                if info:
                    results.append(info)
            except Exception:
                # Skip files that can't be analyzed
                continue

        return results

    def _analyze_dataset(self, filepath: Path, datasets_dir: Path) -> Optional[DatasetInfo]:
        """
        Analyze a single dataset file and extract metadata.

        Args:
            filepath: Path to the JSONL file
            datasets_dir: Base Datasets directory for relative path calculation

        Returns:
            DatasetInfo object or None if file is empty/invalid
        """
        # Get file size
        stat = filepath.stat()
        size_bytes = stat.st_size

        if size_bytes == 0:
            return None

        # Count examples and determine type
        example_count = 0
        has_label_field = False

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                example_count += 1

                # Check first few lines to determine type
                if example_count <= 3:
                    try:
                        data = json.loads(line)
                        if 'label' in data:
                            has_label_field = True
                    except json.JSONDecodeError:
                        continue

        if example_count == 0:
            return None

        # Determine dataset type
        # KTO datasets have interleaved True/False labels
        # SFT datasets are positive examples only (no label field or all True)
        dataset_type = 'KTO' if has_label_field else 'SFT'

        # Human-readable size
        size_human = self._format_size(size_bytes)

        # Calculate relative path
        try:
            relative_path = str(filepath.relative_to(self.repo_root))
        except ValueError:
            relative_path = str(filepath)

        return DatasetInfo(
            path=filepath,
            name=filepath.name,
            dataset_type=dataset_type,
            example_count=example_count,
            size_bytes=size_bytes,
            size_human=size_human,
            relative_path=relative_path,
        )

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable size string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
