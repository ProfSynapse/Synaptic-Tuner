#!/usr/bin/env python3
"""
Generate workspace-aware KTO examples from SFT dataset.

Samples from the SFT dataset and creates paired examples:
- Good (label=true): Uses "default" workspace
- Bad (label=false): Uses hallucinated workspace ID

This teaches the model NOT to hallucinate workspace IDs when none are provided.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict[str, Any]], filepath: Path):
    """Save list of dicts to JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def extract_tool_call_args(example: Dict[str, Any]) -> tuple[Dict[str, Any], int, int]:
    """
    Extract tool call arguments from example.
    Returns (args_dict, conv_idx, tool_idx) or None if not found.
    """
    for conv_idx, conv in enumerate(example['conversations']):
        if conv.get('tool_calls'):
            for tool_idx, tc in enumerate(conv['tool_calls']):
                args = json.loads(tc['function']['arguments'])
                return args, conv_idx, tool_idx
    return None, None, None


def create_default_workspace_variant(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a GOOD variant that uses "default" workspace.
    This teaches: When no workspace info provided, use "default".
    """
    variant = json.loads(json.dumps(example))  # Deep copy

    args, conv_idx, tool_idx = extract_tool_call_args(variant)
    if args is None:
        return None

    # Change workspace to "default"
    if 'context' in args and 'workspaceId' in args['context']:
        args['context']['workspaceId'] = 'default'

        # Update sessionMemory to reflect using default workspace
        original_memory = args['context'].get('sessionMemory', '')
        if 'workspace' not in original_memory.lower():
            args['context']['sessionMemory'] = f"{original_memory} Using default workspace as no specific workspace was provided."

        # Update toolContext
        original_tool_context = args['context'].get('toolContext', '')
        if 'default' not in original_tool_context.lower():
            args['context']['toolContext'] = f"{original_tool_context} Operating in default workspace."

    # Serialize back to JSON string
    variant['conversations'][conv_idx]['tool_calls'][tool_idx]['function']['arguments'] = json.dumps(args)

    # Mark as good example with behavior tag
    variant['label'] = True
    variant['behavior'] = 'workspace_default'

    return variant


def create_hallucinated_workspace_variant(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a BAD variant that hallucinates a workspace ID.
    This teaches: Don't make up workspace IDs.
    """
    variant = json.loads(json.dumps(example))  # Deep copy

    args, conv_idx, tool_idx = extract_tool_call_args(variant)
    if args is None:
        return None

    # Keep or ensure it has a hallucinated workspace ID (already present in original)
    # The original already has synthetic IDs, so we just need to mark it as bad
    if 'context' in args and 'workspaceId' in args['context']:
        workspace_id = args['context']['workspaceId']

        # Make sure it's NOT "default"
        if workspace_id == 'default':
            # Generate a hallucinated one
            timestamp = random.randint(1700000000000, 1732400000000)
            suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=9))
            args['context']['workspaceId'] = f'ws_{timestamp}_{suffix}'

        # Update sessionMemory to show this is wrong
        original_memory = args['context'].get('sessionMemory', '')
        # Don't explicitly say it's wrong in the memory - just keep it as is
        # The contrastive learning will figure it out

    # Serialize back to JSON string
    variant['conversations'][conv_idx]['tool_calls'][tool_idx]['function']['arguments'] = json.dumps(args)

    # Mark as bad example with behavior tag
    variant['label'] = False
    variant['behavior'] = 'workspace_hallucination'

    return variant


def generate_workspace_pairs(
    sft_dataset: List[Dict[str, Any]],
    num_pairs: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate paired examples (good/bad) for workspace behavior.
    Returns interleaved True/False list.
    """
    random.seed(seed)

    # Sample from SFT dataset
    sampled = random.sample(sft_dataset, min(num_pairs, len(sft_dataset)))

    pairs = []
    for example in sampled:
        # Create good variant (uses "default")
        good = create_default_workspace_variant(example)
        if good is None:
            continue

        # Create bad variant (hallucinates workspace)
        bad = create_hallucinated_workspace_variant(example)
        if bad is None:
            continue

        # Add pair in interleaved order (True, False)
        pairs.append(good)
        pairs.append(bad)

    print(f"Generated {len(pairs)} examples ({len(pairs)//2} pairs)")
    print(f"  - {sum(1 for p in pairs if p['label'])} positive (default workspace)")
    print(f"  - {sum(1 for p in pairs if not p['label'])} negative (hallucinated workspace)")

    return pairs


def fix_interleaving(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fix non-interleaved dataset by re-shuffling to alternate True/False.
    Preserves all examples, just reorders them.
    """
    # Separate by label
    true_examples = [ex for ex in examples if ex['label']]
    false_examples = [ex for ex in examples if not ex['label']]

    print(f"  - Separating: {len(true_examples)} True, {len(false_examples)} False")

    # Shuffle each group independently
    random.shuffle(true_examples)
    random.shuffle(false_examples)

    # Interleave
    interleaved = []
    max_len = max(len(true_examples), len(false_examples))

    for i in range(max_len):
        if i < len(true_examples):
            interleaved.append(true_examples[i])
        if i < len(false_examples):
            interleaved.append(false_examples[i])

    print(f"  - Interleaved into {len(interleaved)} examples")
    return interleaved


def merge_with_existing_kto(
    new_examples: List[Dict[str, Any]],
    existing_kto: List[Dict[str, Any]],
    fix_existing: bool = True
) -> List[Dict[str, Any]]:
    """
    Merge new workspace examples with existing KTO dataset.
    Maintains interleaved True/False pattern.
    """
    # Verify new examples are interleaved
    labels = [ex['label'] for ex in new_examples]
    if labels != [True, False] * (len(labels) // 2):
        print("WARNING: New examples are not properly interleaved!")

    # Check existing dataset interleaving
    existing_labels = [ex['label'] for ex in existing_kto]
    print(f"\nExisting KTO dataset:")
    print(f"  - Total: {len(existing_kto)} examples")
    print(f"  - Positive: {sum(existing_labels)}")
    print(f"  - Negative: {len(existing_labels) - sum(existing_labels)}")
    print(f"  - First 10 labels: {existing_labels[:10]}")
    print(f"  - Last 10 labels: {existing_labels[-10:]}")

    # Check for consecutive runs
    consecutive_errors = []
    for i in range(1, len(existing_labels)):
        if existing_labels[i] == existing_labels[i-1]:
            consecutive_errors.append(i)

    if consecutive_errors and fix_existing:
        print(f"\n⚠ Found {len(consecutive_errors)} consecutive same-label pairs in existing dataset")
        print(f"  Re-interleaving existing dataset to fix...")
        existing_kto = fix_interleaving(existing_kto)
        existing_labels = [ex['label'] for ex in existing_kto]
        print(f"  ✓ Fixed! New pattern: first 10 = {existing_labels[:10]}, last 10 = {existing_labels[-10:]}")
    elif consecutive_errors:
        print(f"\n⚠ WARNING: Found {len(consecutive_errors)} consecutive same-label pairs")

    # Determine what label we need to start with for new examples
    # Note: new_examples are in pairs [True, False, True, False, ...]
    # where each pair is (Good/default, Bad/hallucinated)
    if len(existing_kto) > 0:
        last_label = existing_kto[-1]['label']
        first_label = new_examples[0]['label']

        # Debug info
        print(f"\n  Last label of existing: {last_label}")
        print(f"  First label of new: {first_label}")

        # We need to start with opposite of last label
        if last_label == first_label:
            # Need to reverse each pair to flip the order
            # [T, F, T, F, ...] → [F, T, F, T, ...]
            print(f"  ⚠ Need to reverse each pair to maintain interleaving")
            reordered = []
            for i in range(0, len(new_examples), 2):
                if i + 1 < len(new_examples):
                    # Reverse the pair
                    reordered.append(new_examples[i + 1])  # False first
                    reordered.append(new_examples[i])      # True second
                else:
                    # Odd number, just append the last one
                    reordered.append(new_examples[i])
            new_examples[:] = reordered
            print(f"  ✓ Reversed pairs. New pattern: {[ex['label'] for ex in new_examples[:6]]}")

    # Merge
    merged = existing_kto + new_examples

    # Verify merged dataset is interleaved
    merged_labels = [ex['label'] for ex in merged]
    print(f"\nMerged dataset:")
    print(f"  - Total: {len(merged)} examples")
    print(f"  - Positive: {sum(merged_labels)}")
    print(f"  - Negative: {len(merged_labels) - sum(merged_labels)}")

    # Check for consecutive same labels
    consecutive_errors = []
    for i in range(1, len(merged_labels)):
        if merged_labels[i] == merged_labels[i-1]:
            consecutive_errors.append(i)

    if consecutive_errors:
        print(f"\n⚠ WARNING: Found {len(consecutive_errors)} consecutive same-label pairs at indices: {consecutive_errors[:10]}")
    else:
        print("\n✓ Dataset is properly interleaved (no consecutive same labels)")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Generate workspace-aware KTO examples")
    parser.add_argument(
        '--sft-dataset',
        type=Path,
        default=Path('Datasets/syngen_tools_sft_11.23.25_toolcall.jsonl'),
        help='Path to SFT dataset'
    )
    parser.add_argument(
        '--kto-dataset',
        type=Path,
        default=Path('Datasets/behavior_merged_kto_11.23.25.jsonl'),
        help='Path to existing KTO dataset'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path (default: kto-dataset with _workspace suffix)'
    )
    parser.add_argument(
        '--num-pairs',
        type=int,
        default=400,
        help='Number of pairs to generate (default: 400 = 800 examples)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing KTO dataset instead of creating new file'
    )
    parser.add_argument(
        '--no-fix',
        action='store_true',
        help='Do not fix interleaving issues in existing dataset'
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        if args.append:
            args.output = args.kto_dataset
        else:
            # Create new file with _workspace suffix
            stem = args.kto_dataset.stem
            args.output = args.kto_dataset.parent / f"{stem}_workspace.jsonl"

    print(f"Loading SFT dataset from {args.sft_dataset}...")
    sft_data = load_jsonl(args.sft_dataset)
    print(f"Loaded {len(sft_data)} SFT examples")

    print(f"\nGenerating {args.num_pairs} workspace behavior pairs...")
    workspace_pairs = generate_workspace_pairs(sft_data, args.num_pairs, args.seed)

    if args.append:
        print(f"\nLoading existing KTO dataset from {args.kto_dataset}...")
        existing_kto = load_jsonl(args.kto_dataset)

        print(f"\nMerging with existing KTO dataset...")
        merged = merge_with_existing_kto(workspace_pairs, existing_kto, fix_existing=not args.no_fix)

        print(f"\nSaving merged dataset to {args.output}...")
        save_jsonl(merged, args.output)
    else:
        print(f"\nSaving workspace pairs to {args.output}...")
        save_jsonl(workspace_pairs, args.output)

    print(f"\n✓ Done! Saved {len(workspace_pairs) if not args.append else len(merged)} examples to {args.output}")

    # Show examples
    print("\n" + "="*80)
    print("EXAMPLE PAIR (Good vs Bad):")
    print("="*80)
    if len(workspace_pairs) >= 2:
        print("\nGOOD (label=True, uses default workspace):")
        print(json.dumps(workspace_pairs[0], indent=2))
        print("\nBAD (label=False, hallucinates workspace):")
        print(json.dumps(workspace_pairs[1], indent=2))


if __name__ == '__main__':
    main()
