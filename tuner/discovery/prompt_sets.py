"""
Prompt set discovery service.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/prompt_sets.py
Purpose: Discover and parse available prompt sets for evaluation
Used by: Evaluation handler to list prompt sets with descriptions and counts

This module implements the PromptSetDiscovery service which scans the Evaluator/prompts
directory for JSON prompt set files, parses them to extract metadata, and returns
structured information about each prompt set.

Pattern migrated from: tuner.py lines 749-780 (_list_prompt_sets function)
"""

import json
from pathlib import Path
from typing import List, Tuple


class PromptSetDiscovery:
    """
    Discover available prompt sets for evaluation.

    This service scans the Evaluator/prompts directory for JSON files containing
    prompt sets, parses each file to count prompts, and returns metadata including
    name, description, and prompt count.

    Example:
        from tuner.discovery import PromptSetDiscovery

        # Discover prompt sets
        discovery = PromptSetDiscovery()
        prompt_sets = discovery.discover()

        # Display prompt sets
        for name, description, count in prompt_sets:
            print(f"{name}: {description} ({count} prompts)")

        # Use custom prompts directory
        custom_dir = Path('/path/to/custom/prompts')
        custom_sets = discovery.discover(prompts_dir=custom_dir)
    """

    # Known prompt sets with descriptions (in display order)
    KNOWN_PROMPT_SETS = [
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
            # Default to repo root (assumes we're in tuner/ or subdirectory)
            self.repo_root = Path(__file__).parent.parent.parent
        else:
            self.repo_root = repo_root

    def discover(self, prompts_dir: Path = None) -> List[Tuple[str, str, int]]:
        """
        Discover available prompt sets.

        Scans the prompts directory for known prompt set JSON files, parses them
        to count prompts, and returns metadata for each set found.

        Args:
            prompts_dir: Path to prompts directory. If None, uses Evaluator/prompts.

        Returns:
            List of tuples (name, description, count) for each prompt set found.
            Results are ordered according to KNOWN_PROMPT_SETS.
            Returns empty list if prompts directory doesn't exist.

        Example:
            # Use default directory (Evaluator/prompts)
            discovery = PromptSetDiscovery()
            prompt_sets = discovery.discover()

            # Display results
            for name, description, count in prompt_sets:
                print(f"{name}:")
                print(f"  Description: {description}")
                print(f"  Prompts: {count}")

            # Use custom directory
            custom_sets = discovery.discover(prompts_dir=Path('/custom/prompts'))

        Prompt set file formats supported:
            - List format: ["prompt1", "prompt2", ...]
            - Dict with "prompts" key: {"prompts": [...], "metadata": {...}}
            - Dict with "cases" key: {"cases": [...], "config": {...}}

        Error handling:
            - File doesn't exist: Skipped (not included in results)
            - File parse error: Skipped (not included in results)
            - Invalid format: Count = 0 (included with 0 count)
        """
        # Determine prompts directory
        if prompts_dir is None:
            prompts_dir = self.repo_root / "Evaluator" / "prompts"

        # Return empty list if directory doesn't exist
        if not prompts_dir.exists():
            return []

        prompt_sets = []
        seen_names = set()

        # First, iterate through known prompt sets (maintains preferred order)
        for name, description in self.KNOWN_PROMPT_SETS:
            filepath = prompts_dir / f"{name}.json"

            # Skip if file doesn't exist
            if not filepath.exists():
                continue

            try:
                # Parse JSON file
                with open(filepath) as f:
                    data = json.load(f)

                # Count prompts based on file format
                count = self._count_prompts(data)

                # Add to results
                prompt_sets.append((name, description, count))
                seen_names.add(name)

            except Exception:
                # Skip files that fail to parse
                continue

        # Then, discover any additional JSON files not in KNOWN_PROMPT_SETS
        for filepath in sorted(prompts_dir.glob("*.json")):
            name = filepath.stem
            if name in seen_names:
                continue

            try:
                with open(filepath) as f:
                    data = json.load(f)

                count = self._count_prompts(data)
                if count > 0:
                    # Generate description from filename
                    description = name.replace("_", " ").title()
                    prompt_sets.append((name, description, count))

            except Exception:
                continue

        return prompt_sets

    @staticmethod
    def _count_prompts(data) -> int:
        """
        Count prompts in parsed JSON data.

        Handles multiple JSON formats:
        - List format: ["prompt1", "prompt2", ...]
        - Dict with "prompts" key: {"prompts": [...], ...}
        - Dict with "cases" key: {"cases": [...], ...}

        Args:
            data: Parsed JSON data (list or dict)

        Returns:
            Number of prompts found, or 0 if format not recognized

        Example:
            # List format
            data = ["prompt1", "prompt2", "prompt3"]
            count = PromptSetDiscovery._count_prompts(data)  # 3

            # Dict with "prompts" key
            data = {"prompts": ["p1", "p2"], "metadata": {...}}
            count = PromptSetDiscovery._count_prompts(data)  # 2

            # Dict with "cases" key
            data = {"cases": ["c1", "c2", "c3"], "config": {...}}
            count = PromptSetDiscovery._count_prompts(data)  # 3
        """
        # Handle list format
        if isinstance(data, list):
            return len(data)

        # Handle dict format
        if isinstance(data, dict):
            # Try "prompts" key first
            if "prompts" in data:
                return len(data["prompts"])
            # Try "cases" key as fallback
            if "cases" in data:
                return len(data["cases"])

        # Unknown format
        return 0
