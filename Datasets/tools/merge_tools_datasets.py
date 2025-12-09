#!/usr/bin/env python3
"""
Merge *thinking* tools datasets from agent folders into a single SFT dataset.

This script now:
1. Reads the latest thinking tool datasets for each agent (auto-picks newest version)
2. Combines them with shuffling for training diversity
3. Emits the reasoning-focused merged dataset + metadata
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def parse_version(version_str: str) -> tuple:
    """Convert a version string like '1.5.1' into a comparable tuple."""
    try:
        return tuple(int(x) for x in version_str.split("."))
    except ValueError:
        return (0, 0, 0)


def find_latest_version(agent_dir: Path) -> str:
    """Return the latest tools_v*.jsonl version for the given agent directory."""
    if not agent_dir.exists():
        return None
    latest_version = None
    for file in agent_dir.glob("tools_v*.jsonl"):
        # Extract version between "tools_v" and ".jsonl"
        name = file.name
        if not name.startswith("tools_v") or not name.endswith(".jsonl"):
            continue
        version_str = name[len("tools_v") : -len(".jsonl")]
        if latest_version is None or parse_version(version_str) > parse_version(latest_version):
            latest_version = version_str
    return latest_version


def load_dataset(file_path: Path) -> List[Dict]:
    """Load a JSONL dataset file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'conversations' not in data:
                        print(f"    WARNING: Line {line_num} missing 'conversations' field, skipping")
                        continue
                    examples.append(data)
                except json.JSONDecodeError as e:
                    print(f"    WARNING: Line {line_num} invalid JSON: {e}, skipping")
                    continue
    return examples


def main():
    # Configuration (thinking datasets)
    tools_datasets_dir = Path(__file__).parent.parent / "tools_datasets" / "thinking"
    output_dir = Path(__file__).parent.parent
    output_file = output_dir / "reasoning_tools_sft_12.7.25.jsonl"

    # Agent categories
    agents = [
        "vaultManager",
        "contentManager",
        "memoryManager",
        "vaultLibrarian",
        "agentManager",
    ]

    # Load all datasets
    print("Loading latest thinking tools datasets...")
    all_examples = []
    agent_stats = {}

    agent_versions = {}
    for agent in agents:
        agent_dir = tools_datasets_dir / agent
        latest_version = find_latest_version(agent_dir)
        if not latest_version:
            print(f"  WARNING: No dataset found for {agent} in {agent_dir}")
            continue

        version_tag = f"v{latest_version}"
        file_path = agent_dir / f"tools_{version_tag}.jsonl"
        if not file_path.exists():
            print(f"  WARNING: Expected file not found: {file_path}")
            continue

        examples = load_dataset(file_path)
        all_examples.extend(examples)

        # Count labels for stats (some examples may have label field)
        positive = sum(1 for ex in examples if ex.get('label') is True)
        negative = sum(1 for ex in examples if ex.get('label') is False)
        no_label = sum(1 for ex in examples if 'label' not in ex)

        agent_versions[agent] = version_tag
        agent_stats[agent] = {
            "total": len(examples),
            "positive": positive,
            "negative": negative,
            "no_label": no_label,
            "version": version_tag
        }
        print(f"  {agent} ({version_tag}): {len(examples)} examples")

    # Overall stats
    total_positive = sum(1 for ex in all_examples if ex.get('label') is True)
    total_negative = sum(1 for ex in all_examples if ex.get('label') is False)
    total_no_label = sum(1 for ex in all_examples if 'label' not in ex)

    print(f"\nTotal loaded: {len(all_examples)} examples")
    print(f"  With label=true: {total_positive}")
    print(f"  With label=false: {total_negative}")
    print(f"  Without label: {total_no_label}")

    # Shuffle for training diversity
    print("\nShuffling dataset...")
    random.shuffle(all_examples)

    # Write merged dataset
    print(f"\nWriting merged dataset to {output_file.name}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Write metadata file
    metadata = {
        "created": datetime.now().isoformat(),
        "source": "tools_datasets thinking merge 12.7.25",
        "version": "thinking-latest",
        "agents": agents,
        "agent_versions": agent_versions,
        "agent_stats": agent_stats,
        "total_examples": len(all_examples),
        "positive_examples": total_positive,
        "negative_examples": total_negative,
        "no_label_examples": total_no_label,
        "shuffled": True,
        "format": "SFT-compatible ChatML",
        "notes": "Merged latest thinking tool datasets; auto-selects newest version per agent"
    }

    metadata_file = output_file.with_suffix('.metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata written to {metadata_file.name}")
    print(f"\nMerge complete!")
    print(f"\nOutput file: {output_file}")
    print(f"Total examples: {len(all_examples)}")
    print("Ready for SFT training")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
