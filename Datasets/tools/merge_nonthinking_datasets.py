#!/usr/bin/env python3
"""
Merge *non-thinking* tools datasets from agent folders into a single SFT dataset.

This script:
1. Reads the latest non-thinking tool datasets for each agent
2. Combines them with shuffling for training diversity
3. Emits the merged dataset + metadata
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import argparse


def parse_version(version_str: str) -> tuple:
    """Convert a version string like '1.5' into a comparable tuple."""
    try:
        # Handle versions like "1.4_passed_only" -> extract "1.4"
        clean_version = version_str.split("_")[0]
        return tuple(int(x) for x in clean_version.split("."))
    except ValueError:
        return (0, 0, 0)


def find_latest_version(agent_dir: Path, prefer_passed_only: bool = True) -> str:
    """Return the latest tools_v*.jsonl version for the given agent directory."""
    if not agent_dir.exists():
        return None

    candidates = []
    for file in agent_dir.glob("tools_v*.jsonl"):
        name = file.name
        # Skip failed, review, test, and experimental files
        if "failed" in name or "review" in name or "test" in name or "_full" in name:
            continue
        if not name.startswith("tools_v") or not name.endswith(".jsonl"):
            continue

        # Extract version string
        version_str = name[len("tools_v"): -len(".jsonl")]
        candidates.append((version_str, file))

    if not candidates:
        return None

    # First, check for _passed_only files (these are the cleaned versions)
    passed_only = [c for c in candidates if "passed_only" in c[0]]
    if passed_only and prefer_passed_only:
        # Sort passed_only by version and return highest
        passed_only.sort(key=lambda x: parse_version(x[0]), reverse=True)
        return passed_only[0][0]

    # Otherwise, filter to just clean version numbers (no suffix)
    clean_versions = [c for c in candidates if "_" not in c[0]]
    if clean_versions:
        clean_versions.sort(key=lambda x: parse_version(x[0]), reverse=True)
        return clean_versions[0][0]

    # Fallback to any candidate
    candidates.sort(key=lambda x: parse_version(x[0]), reverse=True)
    return candidates[0][0]


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
    parser = argparse.ArgumentParser(description="Merge non-thinking tools datasets")
    parser.add_argument("--date", default="12.18.25", help="Date suffix for output file (default: 12.18.25)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be merged without writing")
    args = parser.parse_args()

    # Configuration (non-thinking datasets)
    tools_datasets_dir = Path(__file__).parent.parent / "tools_datasets" / "non_thinking"
    output_dir = Path(__file__).parent.parent
    output_file = output_dir / f"nonthinking_tools_sft_{args.date}.jsonl"

    # Agent categories
    agents = [
        "vaultManager",
        "contentManager",
        "memoryManager",
        "vaultLibrarian",
        "agentManager",
    ]

    # Load all datasets
    print("Loading latest non-thinking tools datasets...")
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

        # Count labels for stats
        positive = sum(1 for ex in examples if ex.get('label') is True)
        negative = sum(1 for ex in examples if ex.get('label') is False)
        no_label = sum(1 for ex in examples if 'label' not in ex)

        agent_versions[agent] = version_tag
        agent_stats[agent] = {
            "total": len(examples),
            "positive": positive,
            "negative": negative,
            "no_label": no_label,
            "version": version_tag,
            "file": file_path.name
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

    if args.dry_run:
        print("\n[DRY RUN] Would write to:", output_file)
        return

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
        "source": f"non-thinking tools_datasets merge {args.date}",
        "version": "nonthinking-latest",
        "agents": agents,
        "agent_versions": agent_versions,
        "agent_stats": agent_stats,
        "total_examples": len(all_examples),
        "positive_examples": total_positive,
        "negative_examples": total_negative,
        "no_label_examples": total_no_label,
        "shuffled": True,
        "format": "SFT-compatible ChatML",
        "notes": "Merged latest non-thinking tool datasets; auto-selects newest version per agent"
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
