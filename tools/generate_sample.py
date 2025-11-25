#!/usr/bin/env python3
"""
Generate stratified sample of 150 examples from the full dataset.
Ensures even distribution across the dataset and diverse tool coverage.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def main():
    dataset_path = Path(__file__).parent.parent / "Datasets" / "syngen_tools_sft_merged_complete_11.21.25.jsonl"

    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")

    # Read all examples and track tool usage
    examples = []
    tool_counts = defaultdict(int)

    print("Reading dataset...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                examples.append((line_num, example))

                # Extract tool name from assistant content
                if len(example.get('conversations', [])) > 1:
                    content = example['conversations'][1].get('content', '')
                    if 'tool_call:' in content:
                        # Extract tool name
                        tool_line = content.split('tool_call:')[1].split('\n')[0].strip()
                        tool_counts[tool_line] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                continue

    total_examples = len(examples)
    print(f"Total examples: {total_examples}")
    print(f"Unique tools found: {len(tool_counts)}")

    # Strategy: Systematic sampling with stride
    # 5505 / 150 = ~37, so sample every 37th example
    sample_size = 150
    stride = total_examples // sample_size

    sampled_indices = []
    sampled_examples = []

    # Start at different offsets to ensure we don't miss tool diversity
    # Take every stride-th example
    for i in range(0, total_examples, stride):
        if len(sampled_indices) >= sample_size:
            break
        sampled_indices.append(i)
        sampled_examples.append(examples[i])

    # If we're short, add remaining evenly
    while len(sampled_indices) < sample_size:
        # Add examples from gaps
        offset = len(sampled_indices)
        idx = (offset * stride + stride // 2) % total_examples
        if idx not in sampled_indices:
            sampled_indices.append(idx)
            sampled_examples.append(examples[idx])

    # Sort sampled examples by line number
    sampled_examples.sort(key=lambda x: x[0])

    # Create 5 batches of 30 examples each
    batch_size = 30
    batches = []
    for i in range(5):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = sampled_examples[start_idx:end_idx]
        batches.append(batch)

    # Write sample manifest
    manifest = {
        "total_dataset_size": total_examples,
        "sample_size": sample_size,
        "sampling_strategy": "systematic_stride",
        "stride": stride,
        "batches": []
    }

    # Write each batch to separate file
    output_dir = Path(__file__).parent.parent / "Datasets" / "quality_review"
    output_dir.mkdir(exist_ok=True)

    for batch_num, batch in enumerate(batches, start=1):
        batch_file = output_dir / f"sample_batch_{batch_num}.jsonl"

        batch_info = {
            "batch_num": batch_num,
            "size": len(batch),
            "line_numbers": [ex[0] for ex in batch],
            "file": str(batch_file.name)
        }
        manifest["batches"].append(batch_info)

        # Write batch examples
        with open(batch_file, 'w', encoding='utf-8') as f:
            for line_num, example in batch:
                f.write(json.dumps(example) + '\n')

        print(f"Batch {batch_num}: {len(batch)} examples (lines {batch[0][0]}-{batch[-1][0]})")

    # Write manifest
    manifest_file = output_dir / "sample_manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Generated {sample_size} samples across 5 batches")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Manifest: {manifest_file.name}")

    # Print line numbers for reference
    print(f"\nSample coverage:")
    print(f"  First example: line {sampled_examples[0][0]}")
    print(f"  Last example: line {sampled_examples[-1][0]}")
    print(f"  Average stride: ~{stride}")

if __name__ == "__main__":
    main()
