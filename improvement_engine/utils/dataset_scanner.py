"""Dataset scanner for auto-discovering available datasets."""

import re
from pathlib import Path
from typing import Dict, List


class DatasetScanner:
    """Scans and categorizes available datasets."""

    def __init__(self, datasets_root: str = "Datasets"):
        """
        Initialize dataset scanner.

        Args:
            datasets_root: Root directory for datasets
        """
        self.root = Path(datasets_root)

    def scan(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Scan for all available datasets.

        Returns:
            Nested dict: {category: {agent_name: [versions]}}
        """
        datasets = {
            "behavior": {},
            "tools_thinking": {},
            "tools_non_thinking": {}
        }

        # Scan behavior datasets
        behavior_dir = self.root / "behavior_datasets"
        if behavior_dir.exists():
            datasets["behavior"] = self._scan_directory(behavior_dir, pattern=r"pairs_v([\d\.]+)\.jsonl")

        # Scan tools datasets (thinking)
        tools_thinking_dir = self.root / "tools_datasets" / "thinking"
        if tools_thinking_dir.exists():
            datasets["tools_thinking"] = self._scan_directory(tools_thinking_dir, pattern=r"tools_v([\d\.]+)\.jsonl")

        # Scan tools datasets (non-thinking)
        tools_non_thinking_dir = self.root / "tools_datasets" / "non_thinking"
        if tools_non_thinking_dir.exists():
            datasets["tools_non_thinking"] = self._scan_directory(tools_non_thinking_dir, pattern=r"tools_v([\d\.]+)\.jsonl")

        return datasets

    def _scan_directory(self, directory: Path, pattern: str) -> Dict[str, List[str]]:
        """
        Scan a directory for dataset files.

        Args:
            directory: Directory to scan
            pattern: Regex pattern to match version numbers

        Returns:
            Dict of {agent_name: [versions]}
        """
        agents = {}

        for agent_dir in directory.iterdir():
            if not agent_dir.is_dir():
                continue

            versions = []
            for file in agent_dir.glob("*.jsonl"):
                match = re.search(pattern, file.name)
                if match:
                    version = match.group(1)
                    versions.append(version)

            if versions:
                # Sort versions
                versions.sort(key=lambda v: [int(x) for x in v.split('.')])
                agents[agent_dir.name] = versions

        return agents

    def get_file_path(self, category: str, agent: str, version: str) -> str:
        """
        Get full file path for a dataset.

        Args:
            category: Dataset category
            agent: Agent name
            version: Version number

        Returns:
            Full file path
        """
        if category == "behavior":
            return str(self.root / "behavior_datasets" / agent / f"pairs_v{version}.jsonl")
        elif category == "tools_thinking":
            return str(self.root / "tools_datasets" / "thinking" / agent / f"tools_v{version}.jsonl")
        elif category == "tools_non_thinking":
            return str(self.root / "tools_datasets" / "non_thinking" / agent / f"tools_v{version}.jsonl")
        else:
            raise ValueError(f"Unknown category: {category}")

    def get_next_version(self, version: str) -> str:
        """
        Get next version number.

        Args:
            version: Current version (e.g., "1.5")

        Returns:
            Next version (e.g., "1.6")
        """
        parts = version.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid version format: {version}")

        major, minor = int(parts[0]), int(parts[1])
        return f"{major}.{minor + 1}"

    def count_examples(self, file_path: str) -> int:
        """
        Count number of examples in a file.

        Args:
            file_path: Path to file

        Returns:
            Number of lines/examples
        """
        path = Path(file_path)
        if not path.exists():
            return 0

        with open(path, 'r') as f:
            return sum(1 for _ in f)
