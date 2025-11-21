#!/usr/bin/env python3
"""
Generate third round of 150 samples from the remaining unscored examples.
Avoids duplicating examples from rounds 1 and 2.
"""
import json
import sys
from pathlib import Path

def main():
    dataset_path = Path(__file__).parent.parent / "Datasets" / "syngen_tools_sft_merged_complete_11.21.25.jsonl"
    review_dir = Path(__file__).parent.parent / "Datasets" / "quality_review"

    # Load all previous round manifests to get already-sampled line numbers
    already_sampled = set()

    for manifest_file in ['sample_manifest.json', 'sample_manifest_round2.json']:
        manifest_path = review_dir / manifest_file
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                for batch in manifest.get('batches', []):
                    already_sampled.update(batch['line_numbers'])

    print(f"Previously sampled (Rounds 1+2): {len(already_sampled)} examples")

    # Read all examples
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append((line_num, example))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                continue

    total_examples = len(examples)
    remaining_examples = [(ln, ex) for ln, ex in examples if ln not in already_sampled]

    print(f"Total examples: {total_examples}")
    print(f"Remaining unscored: {len(remaining_examples)}")

    # Sample 150 from remaining using systematic sampling
    sample_size = 150
    stride = len(remaining_examples) // sample_size

    sampled_indices = []
    sampled_examples = []

    # Systematic sampling with stride
    for i in range(0, len(remaining_examples), stride):
        if len(sampled_indices) >= sample_size:
            break
        sampled_indices.append(i)
        sampled_examples.append(remaining_examples[i])

    # Fill any remaining slots
    while len(sampled_indices) < sample_size:
        offset = len(sampled_indices)
        idx = (offset * stride + stride // 2) % len(remaining_examples)
        if idx not in sampled_indices:
            sampled_indices.append(idx)
            sampled_examples.append(remaining_examples[idx])

    # Sort by original line number
    sampled_examples.sort(key=lambda x: x[0])

    # Create 5 batches of 30 examples each (batches 11-15)
    batch_size = 30
    batches = []
    for i in range(5):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = sampled_examples[start_idx:end_idx]
        batches.append(batch)

    # Create round 3 manifest
    manifest_round3 = {
        "round": 3,
        "total_dataset_size": total_examples,
        "already_sampled_count": len(already_sampled),
        "remaining_pool": len(remaining_examples),
        "sample_size": sample_size,
        "sampling_strategy": "systematic_stride_from_remaining",
        "stride": stride,
        "batches": []
    }

    # Write each batch to separate file
    for batch_num, batch in enumerate(batches, start=11):  # Start at batch 11
        batch_file = review_dir / f"sample_batch_{batch_num}.jsonl"

        batch_info = {
            "batch_num": batch_num,
            "size": len(batch),
            "line_numbers": [ex[0] for ex in batch],
            "file": str(batch_file.name)
        }
        manifest_round3["batches"].append(batch_info)

        # Write batch examples
        with open(batch_file, 'w', encoding='utf-8') as f:
            for line_num, example in batch:
                f.write(json.dumps(example) + '\n')

        print(f"Batch {batch_num}: {len(batch)} examples (lines {batch[0][0]}-{batch[-1][0]})")

    # Write round 3 manifest
    manifest_file = review_dir / "sample_manifest_round3.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_round3, f, indent=2)

    print(f"\n✓ Generated {sample_size} new samples across 5 batches (11-15)")
    print(f"✓ Output directory: {review_dir}")
    print(f"✓ Manifest: {manifest_file.name}")

    # Print cumulative coverage
    total_sampled = len(already_sampled) + sample_size
    print(f"\nCumulative coverage:")
    print(f"  Total sampled: {total_sampled} / {total_examples} ({total_sampled/total_examples*100:.1f}%)")
    print(f"  Round 1: 150")
    print(f"  Round 2: 150")
    print(f"  Round 3: {sample_size}")
    print(f"  \n  🎯 Target (statistical): 408 examples")
    print(f"  📊 Achievement: {total_sampled/408*100:.1f}% of target")

if __name__ == "__main__":
    main()
