#!/usr/bin/env python3
"""
Convert SFT dataset to GSPO (Group Sequence Policy Optimization) format.

SFT Format (input):
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]}
  ],
  "label": true
}

GSPO Format (output):
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "ground_truth_tool": "toolName",
  "ground_truth_args": {...}
}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import random


def extract_tool_info(assistant_msg: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict]]:
    """Extract tool name and arguments from assistant message."""
    tool_calls = assistant_msg.get("tool_calls", [])

    if not tool_calls:
        return None, None

    # Get first tool call (most examples have single tool call)
    first_call = tool_calls[0]
    function_info = first_call.get("function", {})

    tool_name = function_info.get("name")
    args_str = function_info.get("arguments", "{}")

    try:
        tool_args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError:
        tool_args = {}

    return tool_name, tool_args


def convert_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single SFT example to GSPO format."""
    conversations = example.get("conversations", [])

    if len(conversations) < 2:
        return None

    # Build prompt from system + user messages
    prompt = []
    assistant_msg = None

    for msg in conversations:
        role = msg.get("role")
        if role in ("system", "user"):
            prompt.append({
                "role": role,
                "content": msg.get("content", "")
            })
        elif role == "assistant":
            assistant_msg = msg

    if not assistant_msg:
        return None

    # Extract tool information
    tool_name, tool_args = extract_tool_info(assistant_msg)

    if not tool_name:
        return None

    return {
        "prompt": prompt,
        "ground_truth_tool": tool_name,
        "ground_truth_args": tool_args
    }


def convert_dataset(
    input_path: str,
    output_path: str,
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """Convert entire SFT dataset to GSPO format."""

    input_file = Path(input_path)
    output_file = Path(output_path)

    # Read input
    examples = []
    skipped = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                converted = convert_example(example)

                if converted:
                    examples.append(converted)
                else:
                    skipped += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                skipped += 1

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(examples)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Collect stats
    tool_counts = {}
    for ex in examples:
        tool = ex["ground_truth_tool"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    stats = {
        "total_examples": len(examples),
        "skipped": skipped,
        "unique_tools": len(tool_counts),
        "tool_distribution": dict(sorted(tool_counts.items(), key=lambda x: -x[1])[:20])
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert SFT dataset to GSPO format"
    )
    parser.add_argument(
        "input",
        help="Path to input SFT dataset (JSONL)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output GSPO dataset (default: auto-generate)"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the output dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / "gspo_datasets" / f"{input_path.stem}_gspo.jsonl"

    print(f"Converting: {args.input}")
    print(f"Output: {output_path}")
    print()

    stats = convert_dataset(
        args.input,
        output_path,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )

    print(f"Conversion complete!")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Unique tools: {stats['unique_tools']}")
    print()
    print("Top 10 tools:")
    for tool, count in list(stats['tool_distribution'].items())[:10]:
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()
