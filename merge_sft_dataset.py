#!/usr/bin/env python3
"""
Merge high-quality original examples with enhanced examples into final SFT dataset.

This script:
1. Extracts high-quality examples (label=true) from scored_complete_relabeled.jsonl
2. Merges all 74 enhanced batch files
3. Combines them into a single SFT dataset
4. Shuffles for training diversity
5. Validates the output
"""

import json
import random
from pathlib import Path
from typing import Dict, List

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file and return list of examples."""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num} in {filepath.name}: {e}")
    return examples

def save_jsonl(examples: List[Dict], filepath: Path):
    """Save examples to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(examples)} examples to {filepath}")

def main():
    # Paths
    base_dir = Path(__file__).parent
    quality_dir = base_dir / 'Datasets' / 'quality_review'
    enhancement_dir = quality_dir / 'enhancement_batches'
    output_file = base_dir / 'Datasets' / 'syngen_tools_sft_11.22.25.jsonl'

    print("=" * 70)
    print("SFT DATASET MERGE - Creating Final Training Dataset")
    print("=" * 70)

    # Step 1: Load original high-quality examples
    print("\n[1/5] Loading original high-quality examples...")
    scored_file = quality_dir / 'scored_complete_relabeled.jsonl'
    all_scored = load_jsonl(scored_file)
    high_quality = [ex for ex in all_scored if ex.get('label') == True]
    print(f"  ✓ Loaded {len(high_quality):,} high-quality examples (label=true)")
    print(f"  ✓ Filtered out {len(all_scored) - len(high_quality):,} low-quality examples")

    # Step 2: Load all enhanced batches
    print("\n[2/5] Loading enhanced batches...")
    enhanced_files = sorted(enhancement_dir.glob('enhanced_batch_*.jsonl'))
    print(f"  ✓ Found {len(enhanced_files)} enhanced batch files")

    enhanced_examples = []
    for filepath in enhanced_files:
        batch_examples = load_jsonl(filepath)
        enhanced_examples.extend(batch_examples)
        print(f"  ✓ Loaded {filepath.name}: {len(batch_examples)} examples")

    print(f"\n  ✓ Total enhanced examples: {len(enhanced_examples):,}")

    # Step 3: Clean high-quality examples (remove metadata if present)
    print("\n[3/5] Cleaning original high-quality examples...")
    cleaned_high_quality = []
    for ex in high_quality:
        cleaned = {
            'conversations': ex['conversations'],
            'label': True
        }
        cleaned_high_quality.append(cleaned)
    print(f"  ✓ Cleaned {len(cleaned_high_quality):,} examples")

    # Step 4: Combine and shuffle
    print("\n[4/5] Combining and shuffling...")
    all_examples = cleaned_high_quality + enhanced_examples
    print(f"  ✓ Combined total: {len(all_examples):,} examples")
    print(f"    - Original high-quality: {len(cleaned_high_quality):,}")
    print(f"    - Enhanced low-quality: {len(enhanced_examples):,}")

    # Shuffle for training diversity
    random.seed(42)  # Reproducible shuffle
    random.shuffle(all_examples)
    print(f"  ✓ Shuffled for training diversity (seed=42)")

    # Verify all are label=true
    all_true = all(ex.get('label') == True for ex in all_examples)
    print(f"  ✓ All examples labeled as true: {all_true}")

    # Step 5: Save final dataset
    print("\n[5/5] Saving final SFT dataset...")
    save_jsonl(all_examples, output_file)

    # Summary
    print("\n" + "=" * 70)
    print("MERGE COMPLETE - Summary")
    print("=" * 70)
    print(f"Output file: {output_file}")
    print(f"Total examples: {len(all_examples):,}")
    print(f"  - Original high-quality: {len(cleaned_high_quality):,} ({len(cleaned_high_quality)/len(all_examples)*100:.1f}%)")
    print(f"  - Enhanced examples: {len(enhanced_examples):,} ({len(enhanced_examples)/len(all_examples)*100:.1f}%)")
    print(f"All labels: true")
    print(f"Format: Single-turn, ready for SFT training")
    print("=" * 70)

if __name__ == '__main__':
    main()
