"""
Prompt set discovery service.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/prompt_sets.py
Purpose: Discover and parse available YAML scenarios for evaluation
Used by: Evaluation handler to list scenarios with descriptions and counts

This module implements the PromptSetDiscovery service which scans
Evaluator/config/scenarios for YAML scenario files.
"""

from pathlib import Path
from typing import List, Tuple, NamedTuple

import yaml


class PromptSetInfo(NamedTuple):
    """Information about a scenario."""
    name: str
    description: str
    count: int
    path: Path


class PromptSetDiscovery:
    """
    Discover available YAML scenarios for evaluation.

    This service scans Evaluator/config/scenarios for YAML scenario files.

    Example:
        from tuner.discovery import PromptSetDiscovery

        discovery = PromptSetDiscovery()
        scenarios = discovery.discover_all()

        for info in scenarios:
            print(f"{info.name}: {info.description} ({info.count} tests)")
    """

    # Known scenarios with descriptions (in display order)
    KNOWN_SCENARIOS = [
        ("tool_prompts", "Tool Prompts - Comprehensive tool calling tests"),
        ("behavior_prompts", "Behavior Prompts - Behavioral pattern evaluation"),
    ]

    def __init__(self, repo_root: Path = None):
        """
        Initialize the prompt set discovery service.

        Args:
            repo_root: Repository root path. If None, uses current working directory's parent.
        """
        if repo_root is None:
            self.repo_root = Path(__file__).parent.parent.parent
        else:
            self.repo_root = repo_root

    def discover(self) -> List[Tuple[str, str, int]]:
        """
        Discover available scenarios (legacy tuple format).

        Returns:
            List of tuples (name, description, count) for backwards compatibility.
        """
        results = self.discover_all()
        return [(r.name, r.description, r.count) for r in results]

    def discover_all(self) -> List[PromptSetInfo]:
        """
        Discover all available YAML scenarios.

        Returns:
            List of PromptSetInfo objects for all discovered scenarios.
        """
        scenarios_dir = self.repo_root / "Evaluator" / "config" / "scenarios"

        if not scenarios_dir.exists():
            return []

        results: List[PromptSetInfo] = []
        seen_names = set()

        # First, iterate through known scenarios (maintains preferred order)
        for name, description in self.KNOWN_SCENARIOS:
            filepath = scenarios_dir / f"{name}.yaml"

            if not filepath.exists():
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                count = self._count_tests(data)
                desc = data.get("description", description)

                results.append(PromptSetInfo(
                    name=name,
                    description=desc,
                    count=count,
                    path=filepath,
                ))
                seen_names.add(name)

            except Exception:
                continue

        # Then, discover any additional YAML files
        for filepath in sorted(scenarios_dir.glob("*.yaml")):
            name = filepath.stem
            if name in seen_names:
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                count = self._count_tests(data)
                if count > 0:
                    desc = data.get("description", name.replace("_", " ").title())
                    results.append(PromptSetInfo(
                        name=name,
                        description=desc,
                        count=count,
                        path=filepath,
                    ))

            except Exception:
                continue

        return results

    @staticmethod
    def _count_tests(data) -> int:
        """Count tests in parsed YAML scenario data."""
        if isinstance(data, dict) and "tests" in data:
            return len(data["tests"])
        return 0
