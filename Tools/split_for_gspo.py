#!/usr/bin/env python3
"""
Split SFT dataset for GSPO testing.

Takes a SFT dataset and splits it:
- 2/3 for SFT training (keeps full conversations)
- 1/3 for GSPO training (extracts ground truth from tool calls)

GSPO format:
{
  "prompt": [system, user messages],
  "ground_truth_tool": "agent_tool",
  "ground_truth_args": {...}
}
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def extract_ground_truth(conversation: List[Dict]) -> Tuple[str, Dict]:
    """
    Extract ground truth tool and args from assistant message.

    Returns:
        (tool_name, args_dict) or (None, None) if no tool call
    """
    for msg in conversation:
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            continue

        # Get first tool call
        tc = tool_calls[0]
        function = tc.get("function", {})
        tool_name = function.get("name", "")
        args_str = function.get("arguments", "{}")

        # Parse arguments
        try:
            if isinstance(args_str, str):
                args = json.loads(args_str)
            else:
                args = args_str
        except json.JSONDecodeError:
            args = {}

        # Extract actual tool name from calls
        calls = args.get("calls", [])
        if calls:
            first_call = calls[0]
            agent = first_call.get("agent", "")
            tool = first_call.get("tool", "")
            if agent and tool:
                return f"{agent}_{tool}", args

        # Fallback to wrapper name
        if tool_name:
            return tool_name, args

    return None, None


def extract_prompt(conversation: List[Dict]) -> List[Dict]:
    """Extract system + user messages as prompt."""
    prompt = []
    for msg in conversation:
        role = msg.get("role", "")
        if role in ("system", "user"):
            prompt.append({"role": role, "content": msg.get("content", "")})
        else:
            break  # Stop at first non-system/user
    return prompt


def split_dataset(
    input_file: Path,
    sft_ratio: float = 0.67,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into SFT and GSPO portions.

    Args:
        input_file: Path to input JSONL
        sft_ratio: Ratio for SFT (default 2/3)
        seed: Random seed for reproducibility

    Returns:
        (sft_examples, gspo_examples)
    """
    # Load all examples
    examples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Filter to only examples with tool calls (for GSPO)
    tool_examples = []
    text_examples = []

    for ex in examples:
        conv = ex.get("conversations", [])
        tool_name, args = extract_ground_truth(conv)
        if tool_name:
            tool_examples.append(ex)
        else:
            text_examples.append(ex)

    print(f"  With tool calls: {len(tool_examples)}")
    print(f"  Text only: {len(text_examples)}")

    # Shuffle tool examples
    random.seed(seed)
    random.shuffle(tool_examples)

    # Split tool examples
    split_idx = int(len(tool_examples) * sft_ratio)
    sft_tool = tool_examples[:split_idx]
    gspo_tool = tool_examples[split_idx:]

    # SFT gets: sft_tool + all text_examples
    sft_examples = sft_tool + text_examples

    # GSPO gets: gspo_tool (converted format)
    gspo_examples = []
    for ex in gspo_tool:
        conv = ex.get("conversations", [])
        prompt = extract_prompt(conv)
        tool_name, args = extract_ground_truth(conv)

        if prompt and tool_name:
            # Serialize args as JSON string to avoid schema inconsistencies
            gspo_examples.append({
                "prompt": prompt,
                "ground_truth_tool": tool_name,
                "ground_truth_args_json": json.dumps(args, ensure_ascii=False) if args else "{}",
            })

    print(f"\nSplit results:")
    print(f"  SFT: {len(sft_examples)} examples")
    print(f"  GSPO: {len(gspo_examples)} examples")

    return sft_examples, gspo_examples


def save_jsonl(examples: List[Dict], output_file: Path) -> None:
    """Save examples to JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Split SFT dataset for GSPO testing")
    parser.add_argument(
        "--input",
        type=str,
        default="Datasets/nonthinking_tools_sft_12.24.25.jsonl",
        help="Input SFT dataset"
    )
    parser.add_argument(
        "--sft-output",
        type=str,
        default="Datasets/sft_train_67pct.jsonl",
        help="Output SFT dataset (2/3)"
    )
    parser.add_argument(
        "--gspo-output",
        type=str,
        default="Datasets/gspo_test_33pct.jsonl",
        help="Output GSPO dataset (1/3)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.67,
        help="SFT ratio (default 0.67 = 2/3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    sft_examples, gspo_examples = split_dataset(
        input_file,
        sft_ratio=args.ratio,
        seed=args.seed
    )

    save_jsonl(sft_examples, Path(args.sft_output))
    save_jsonl(gspo_examples, Path(args.gspo_output))

    print("\nDone! Next steps:")
    print(f"1. Train SFT on: {args.sft_output}")
    print(f"2. Run GSPO on: {args.gspo_output}")

    return 0


if __name__ == "__main__":
    exit(main())
