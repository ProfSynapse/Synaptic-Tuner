#!/usr/bin/env python3
"""
Merge tools datasets from agent folders into a single SFT dataset.

This script:
1. Reads all individual agent dataset files (tools_v1.0.jsonl)
2. Combines them with shuffling for training diversity
3. Creates a merged dataset file with metadata
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict


# Agent categories
AGENTS = [
    "vaultManager",
    "contentManager",
    "memoryManager",
    "vaultLibrarian",
    "agentManager"
]


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
    # Configuration
    tools_datasets_dir = Path(__file__).parent.parent / "tools_datasets"
    output_dir = Path(__file__).parent.parent
    output_file = output_dir / "tools_sft_v1.5_11.29.25.jsonl"

    # Agent versions - use v1.4 where available (corrupted data removed)
    agent_versions = {
        "agentManager": "v1.3",
        "contentManager": "v1.4",   # 4 corrupted examples removed
        "memoryManager": "v1.4",    # 6 corrupted examples removed
        "vaultLibrarian": "v1.3",
        "vaultManager": "v1.3"
    }

    # Load all datasets
    print("Loading tools datasets (v1.4 with corrupted data removed)...")
    all_examples = []
    agent_stats = {}

    for agent in AGENTS:
        version = agent_versions[agent]
        file_path = tools_datasets_dir / agent / f"tools_{version}.jsonl"
        if file_path.exists():
            examples = load_dataset(file_path)
            all_examples.extend(examples)

            # Count labels for stats (some examples may have label field)
            positive = sum(1 for ex in examples if ex.get('label') is True)
            negative = sum(1 for ex in examples if ex.get('label') is False)
            no_label = sum(1 for ex in examples if 'label' not in ex)

            agent_stats[agent] = {
                "total": len(examples),
                "positive": positive,
                "negative": negative,
                "no_label": no_label,
                "version": version
            }
            print(f"  {agent} ({version}): {len(examples)} examples")
        else:
            print(f"  WARNING: {file_path} not found")

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
        "source": "tools_datasets merge v1.5 11.29.25",
        "version": "1.5",
        "agents": AGENTS,
        "agent_versions": agent_versions,
        "agent_stats": agent_stats,
        "total_examples": len(all_examples),
        "positive_examples": total_positive,
        "negative_examples": total_negative,
        "no_label_examples": total_no_label,
        "shuffled": True,
        "format": "SFT-compatible ChatML",
        "notes": "v1.5 fixes: removed 10 corrupted examples (malformed responses) from contentManager and memoryManager"
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
