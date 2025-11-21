#!/usr/bin/env python3
"""
Generate batches of poor examples for parallel improvement by agents.
"""

import json
from pathlib import Path
import sys

def load_poor_examples():
    """Load poor-quality examples."""
    poor_path = Path("Datasets/quality_review/poor_examples.jsonl")

    if not poor_path.exists():
        print("Error: poor_examples.jsonl not found. Run analyze_poor_examples.py first.")
        sys.exit(1)

    examples = []
    with open(poor_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples

def generate_batches(examples, batch_size=50):
    """
    Generate batches for parallel improvement.

    Args:
        examples: List of poor examples to improve
        batch_size: Number of examples per batch (default: 50)
    """
    output_dir = Path("Datasets/quality_review/improvement_batches")
    output_dir.mkdir(exist_ok=True)

    num_batches = (len(examples) + batch_size - 1) // batch_size

    print(f"Generating {num_batches} batches of {batch_size} examples each...")
    print(f"Total examples to improve: {len(examples)}")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(examples))
        batch = examples[start_idx:end_idx]

        # Create batch file
        batch_file = output_dir / f"batch_{batch_idx + 1}.jsonl"
        with open(batch_file, 'w') as f:
            for ex in batch:
                f.write(json.dumps(ex) + '\n')

        # Create manifest
        manifest = {
            "batch_number": batch_idx + 1,
            "total_batches": num_batches,
            "batch_size": len(batch),
            "start_index": start_idx,
            "end_index": end_idx,
            "input_file": str(batch_file)
        }

        manifest_file = output_dir / f"batch_{batch_idx + 1}_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"  Batch {batch_idx + 1}: {len(batch)} examples ({start_idx}-{end_idx-1})")

    print(f"\nBatches saved to {output_dir}/")
    print(f"\nNext step: Launch {num_batches} parallel improvement agents")

    return num_batches

def main():
    examples = load_poor_examples()
    num_batches = generate_batches(examples, batch_size=50)

    print(f"\n=== Improvement Batch Summary ===")
    print(f"Total poor examples: {len(examples)}")
    print(f"Batch size: 50")
    print(f"Total batches: {num_batches}")
    print(f"\nTo improve all batches, launch {num_batches} parallel agents")

if __name__ == "__main__":
    main()
