#!/usr/bin/env python3
"""
Merge behavior datasets into a single KTO-compatible dataset.

This script:
1. Reads all individual behavior dataset files
2. Combines them with proper KTO interleaving (True/False/True/False)
3. Shuffles within behavior categories to prevent overfitting
4. Creates a merged dataset file with metadata
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def load_dataset(file_path: Path) -> List[Dict]:
    """Load a JSONL dataset file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    # Validate required fields
                    if 'conversations' not in data:
                        print(f"    WARNING: Line {line_num} missing 'conversations' field, skipping")
                        continue
                    if 'label' not in data:
                        print(f"    WARNING: Line {line_num} missing 'label' field, skipping")
                        continue
                    examples.append(data)
                except json.JSONDecodeError as e:
                    print(f"    WARNING: Line {line_num} invalid JSON: {e}, skipping")
                    continue
    return examples


def create_interleaved_dataset(examples: List[Dict]) -> List[Dict]:
    """
    Create interleaved dataset (True/False/True/False pattern) for KTO.

    This is required to avoid TRL KTOTrainer bug with homogeneous batches.
    """
    # Separate by label
    positive = [ex for ex in examples if ex['label'] is True]
    negative = [ex for ex in examples if ex['label'] is False]

    # Shuffle each group to mix behaviors
    random.shuffle(positive)
    random.shuffle(negative)

    # Interleave
    interleaved = []
    max_len = max(len(positive), len(negative))

    for i in range(max_len):
        if i < len(positive):
            interleaved.append(positive[i])
        if i < len(negative):
            interleaved.append(negative[i])

    return interleaved


def main():
    # Configuration
    behavior_datasets_dir = Path(__file__).parent.parent / "behavior_datasets"
    output_file = Path(__file__).parent.parent / "behavior_merged_kto_11.28.25.jsonl"

    # All behavior categories now use v1.3 (with text-only response examples added)
    behaviors_v1_3 = [
        "context_continuity",
        "context_efficiency",
        "error_recovery",
        "execute_prompt_usage",
        "intellectual_humility",
        "strategic_tool_selection",
        "verification_before_action",
        "workspace_awareness",
    ]

    # Legacy lists kept empty for backwards compatibility
    behaviors_v1_2 = []
    behaviors_v1_1 = []

    # Response pattern datasets
    response_patterns = [
        ("response_patterns/text_only", "v1.2"),  # Fixed ID mismatches
        ("response_patterns/tool_only", "v1.1"),
        ("response_patterns/tool_text", "v1.1")
    ]

    # New ask_first dataset
    ask_first = [
        ("ask_first", "v1.0")
    ]

    # Load all datasets
    print("Loading behavior datasets (v1.3 with text-only examples)...")
    all_examples = []
    behavior_stats = {}

    # Load v1.3 datasets (with text-only response examples added)
    for behavior in behaviors_v1_3:
        file_path = behavior_datasets_dir / behavior / "pairs_v1.3.jsonl"
        if file_path.exists():
            examples = load_dataset(file_path)
            all_examples.extend(examples)

            positive = sum(1 for ex in examples if ex.get('label') is True)
            negative = sum(1 for ex in examples if ex.get('label') is False)
            behavior_stats[behavior] = {
                "total": len(examples),
                "positive": positive,
                "negative": negative,
                "version": "v1.3"
            }
            print(f"  {behavior} (v1.3): {len(examples)} examples ({positive} positive, {negative} negative)")
        else:
            print(f"  WARNING: {file_path} not found")

    # Load response pattern datasets
    print("\nLoading response pattern datasets...")
    for pattern, version in response_patterns:
        pattern_name = pattern.split('/')[-1]
        file_path = behavior_datasets_dir / "response_patterns" / f"{pattern_name}_pairs_{version}.jsonl"
        if file_path.exists():
            examples = load_dataset(file_path)
            all_examples.extend(examples)

            positive = sum(1 for ex in examples if ex.get('label') is True)
            negative = sum(1 for ex in examples if ex.get('label') is False)
            behavior_stats[pattern] = {
                "total": len(examples),
                "positive": positive,
                "negative": negative,
                "version": version
            }
            print(f"  {pattern_name} ({version}): {len(examples)} examples ({positive} positive, {negative} negative)")
        else:
            print(f"  WARNING: {file_path} not found")

    # Load ask_first dataset (NEW)
    print("\nLoading ask_first dataset...")
    for dataset, version in ask_first:
        file_path = behavior_datasets_dir / dataset / f"pairs_{version}.jsonl"
        if file_path.exists():
            examples = load_dataset(file_path)
            all_examples.extend(examples)

            positive = sum(1 for ex in examples if ex.get('label') is True)
            negative = sum(1 for ex in examples if ex.get('label') is False)
            behavior_stats[dataset] = {
                "total": len(examples),
                "positive": positive,
                "negative": negative,
                "version": version
            }
            print(f"  {dataset} ({version}): {len(examples)} examples ({positive} positive, {negative} negative)")
        else:
            print(f"  WARNING: {file_path} not found")

    # Overall stats
    total_positive = sum(1 for ex in all_examples if ex['label'] is True)
    total_negative = sum(1 for ex in all_examples if ex['label'] is False)

    print(f"\nTotal loaded: {len(all_examples)} examples")
    print(f"  Positive: {total_positive}")
    print(f"  Negative: {total_negative}")

    # Create interleaved dataset
    print("\nCreating interleaved dataset for KTO...")
    interleaved = create_interleaved_dataset(all_examples)

    # Verify interleaving pattern
    labels = [ex['label'] for ex in interleaved]
    consecutive_same = 0
    for i in range(1, len(labels)):
        if labels[i] == labels[i-1]:
            consecutive_same += 1

    print(f"Interleaved dataset: {len(interleaved)} examples")
    print(f"  Consecutive same labels: {consecutive_same} (should be minimal)")
    print(f"  First 20 labels: {labels[:20]}")

    # Write merged dataset
    print(f"\nWriting merged dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in interleaved:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Write metadata file
    all_patterns = [p[0] for p in response_patterns]
    all_ask_first = [a[0] for a in ask_first]
    metadata = {
        "created": datetime.now().isoformat(),
        "source": "behavior_datasets merge 11.28.25 (v1.3 with text-only response examples)",
        "behaviors_v1_3": behaviors_v1_3,
        "response_patterns": all_patterns,
        "ask_first": all_ask_first,
        "all_datasets": behaviors_v1_3 + all_patterns + all_ask_first,
        "behavior_stats": behavior_stats,
        "total_examples": len(interleaved),
        "positive_examples": total_positive,
        "negative_examples": total_negative,
        "interleaved": True,
        "consecutive_same_labels": consecutive_same,
        "format": "KTO-compatible ChatML with interleaved labels",
        "notes": "v1.3 includes ~30% text-only response examples teaching when NOT to use tools"
    }

    metadata_file = output_file.with_suffix('.metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata written to {metadata_file}")
    print("\n✓ Merge complete!")
    print(f"\nOutput file: {output_file}")
    print(f"Total examples: {len(interleaved)}")
    print("Ready for KTO training")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
