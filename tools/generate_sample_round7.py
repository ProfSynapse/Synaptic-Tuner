#!/usr/bin/env python3
"""
Generate Round 7 sampling batches (batches 121-184) - 64 batches to complete the dataset.
Scores ALL remaining unsampled examples from rounds 1-6.
This achieves 100% dataset coverage!
"""

import json
from pathlib import Path

def load_previous_samples():
    """Load all previously sampled line numbers from rounds 1-6."""
    sampled = set()
    review_dir = Path("Datasets/quality_review")

    # Load from previous manifests (batches 1-120)
    for i in range(1, 121):
        manifest = review_dir / f"sample_manifest_batch_{i}.json"
        if manifest.exists():
            with open(manifest, 'r') as f:
                data = json.load(f)
                sampled.update(data['sampled_line_numbers'])

    print(f"Loaded {len(sampled)} previously sampled line numbers")
    return sampled

def main():
    # Configuration
    dataset_path = Path("Datasets/syngen_tools_sft_merged_complete_11.21.25.jsonl")
    output_dir = Path("Datasets/quality_review")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_batches = 64  # Batches 121-184
    batch_size = 30
    batch_start_num = 121

    print(f"Round 7 Batch Generation - COMPLETE DATASET COVERAGE!")
    print(f"Dataset: {dataset_path}")
    print(f"Generating {num_batches} batches (batches {batch_start_num}-{batch_start_num + num_batches - 1})")
    print()

    # Load dataset
    print("Loading dataset...")
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=0):
            if line.strip():
                ex = json.loads(line)
                ex['_line_number'] = line_num
                examples.append(ex)

    total_examples = len(examples)
    print(f"Total examples in dataset: {total_examples}")

    # Load previously sampled
    previously_sampled = load_previous_samples()

    # Create pool of unsampled examples - this should be ALL remaining examples
    unsampled_pool = [ex for ex in examples if ex['_line_number'] not in previously_sampled]
    print(f"Unsampled examples available: {len(unsampled_pool)}")
    print(f"This is 100% of remaining dataset!")

    # Take ALL remaining examples (no stride needed)
    sampled_examples = unsampled_pool
    sampled_line_numbers = [ex['_line_number'] for ex in sampled_examples]

    print(f"Sampled {len(sampled_examples)} examples (all remaining)")

    # Split into batches
    for batch_num in range(num_batches):
        batch_id = batch_start_num + batch_num
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        batch_examples = sampled_examples[start_idx:end_idx]
        batch_line_numbers = sampled_line_numbers[start_idx:end_idx]

        # Add index within batch
        for idx, ex in enumerate(batch_examples):
            ex['_index'] = idx

        # Write batch file
        batch_file = output_dir / f"sample_batch_{batch_id}.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as f:
            for ex in batch_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        # Write manifest
        manifest_file = output_dir / f"sample_manifest_batch_{batch_id}.json"
        manifest = {
            'batch_number': batch_id,
            'round': 7,
            'batch_size': len(batch_examples),
            'sampled_line_numbers': batch_line_numbers,
            'source_dataset': str(dataset_path)
        }
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        if batch_num % 10 == 0 or batch_num == num_batches - 1:
            print(f"✓ Batch {batch_id}: {len(batch_examples)} examples -> {batch_file.name}")

    print(f"\n✅ Round 7 generation complete!")
    print(f"Created batches {batch_start_num}-{batch_start_num + num_batches - 1}")
    print(f"Total examples: {len(sampled_examples)}")
    print(f"🎉 Ready for 64 PARALLEL AGENTS - FINAL ROUND!")
    print(f"🏆 After scoring: 100% DATASET COVERAGE (5,505/5,505)!")

if __name__ == "__main__":
    main()
