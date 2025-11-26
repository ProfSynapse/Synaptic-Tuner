#!/usr/bin/env python3
"""
Convert KTO dataset to GSPO or preference learning formats.

KTO format (interleaved True/False):
{
    "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    "label": true/false
}

Output formats:

1. GSPO format (for GSPO-only training - no chosen/rejected):
{
    "prompt": [{"role": "user", "content": "..."}],
    "ground_truth_tool": "tool_name",
    "ground_truth_args": {...}
}

2. Preference format (for post-SFT preference learning):
{
    "prompt": [{"role": "user", "content": "..."}],
    "chosen": "correct assistant response",
    "rejected": "incorrect assistant response",
    "ground_truth_tool": "tool_name",
    "ground_truth_args": {...}
}

Strategy:
1. Match prompts that have BOTH True and False versions → Direct pairs
2. Optionally create synthetic rejects for True-only examples
"""

import json
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

# Valid tool prefixes (agent families)
VALID_AGENTS = [
    "vaultManager", "contentManager", "memoryManager",
    "vaultLibrarian", "agentManager"
]


def is_valid_tool(tool_name: str) -> bool:
    """Check if tool name has a valid agent prefix."""
    if "_" not in tool_name:
        return False
    agent = tool_name.split("_")[0]
    return agent in VALID_AGENTS


def extract_tool_info_from_text(response: str) -> Tuple[Optional[str], Optional[dict]]:
    """Extract tool name and arguments from a text-based response string.

    Handles format: "tool_call: toolName\narguments: {...}"

    Returns:
        Tuple of (tool_name, arguments_dict) or (None, None) if not found.
    """
    # Match tool_call: toolName pattern
    tool_match = re.search(r'tool_call:\s*(\S+)', response)
    if not tool_match:
        return None, None

    tool_name = tool_match.group(1)

    # Extract arguments JSON - look for the JSON block after "arguments:"
    args_match = re.search(r'arguments:\s*(\{[\s\S]*?\})\s*\n\nResult:', response)
    if not args_match:
        # Try without Result (might be at end)
        args_match = re.search(r'arguments:\s*(\{[\s\S]*?\})\s*(?:\n|$)', response)

    if not args_match:
        return tool_name, None

    try:
        args_str = args_match.group(1)
        args = json.loads(args_str)
        return tool_name, args
    except json.JSONDecodeError:
        return tool_name, None


def extract_tool_info_from_tool_calls(tool_calls: list) -> Tuple[Optional[str], Optional[dict]]:
    """Extract tool name and arguments from structured tool_calls format.

    Handles OpenAI-style format:
    {
        "tool_calls": [{
            "function": {
                "name": "toolName",
                "arguments": "{...}"  # JSON string
            }
        }]
    }

    Returns:
        Tuple of (tool_name, arguments_dict) or (None, None) if not found.
    """
    if not tool_calls or len(tool_calls) == 0:
        return None, None

    # Use the first tool call
    tool_call = tool_calls[0]

    # Handle different structures
    if "function" in tool_call:
        func = tool_call["function"]
        tool_name = func.get("name")
        args_str = func.get("arguments", "{}")
    else:
        # Direct format without function wrapper
        tool_name = tool_call.get("name")
        args_str = tool_call.get("arguments", "{}")

    if not tool_name:
        return None, None

    # Parse arguments (may be string or already dict)
    if isinstance(args_str, dict):
        return tool_name, args_str

    try:
        args = json.loads(args_str)
        return tool_name, args
    except json.JSONDecodeError:
        return tool_name, None


def extract_tool_info(assistant_msg: dict) -> Tuple[Optional[str], Optional[dict]]:
    """Extract tool name and arguments from an assistant message.

    Handles both formats:
    1. Text-based: {"content": "tool_call: toolName\narguments: {...}"}
    2. Structured: {"content": null, "tool_calls": [{...}]}

    Returns:
        Tuple of (tool_name, arguments_dict) or (None, None) if not found.
    """
    # Check for structured tool_calls format first
    if "tool_calls" in assistant_msg and assistant_msg.get("tool_calls"):
        return extract_tool_info_from_tool_calls(assistant_msg["tool_calls"])

    # Fall back to text-based format
    content = assistant_msg.get("content")
    if content and isinstance(content, str):
        return extract_tool_info_from_text(content)

    return None, None


def load_kto_dataset(filepath: str) -> list[dict]:
    """Load KTO dataset from JSONL file."""
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def find_user_and_assistant(conversations: list[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    """Find user and assistant messages in a conversation by role.

    Handles different orderings:
    - [user, assistant]
    - [system, user, assistant]
    - etc.

    Returns:
        Tuple of (user_msg, assistant_msg) or (None, None) if not found.
    """
    user_msg = None
    assistant_msg = None

    for msg in conversations:
        role = msg.get("role", "")
        if role == "user" and user_msg is None:
            user_msg = msg
        elif role == "assistant":
            assistant_msg = msg
            break  # Take first assistant message

    return user_msg, assistant_msg


def group_by_prompt(examples: list[dict]) -> dict[str, dict]:
    """Group examples by their prompt, collecting True and False versions.

    Stores the full assistant message dict to support both text and tool_calls formats.
    """
    groups = defaultdict(lambda: {"true": [], "false": []})
    skipped = 0

    for ex in examples:
        conversations = ex.get("conversations", [])

        # Find user and assistant messages by role
        user_msg, assistant_msg = find_user_and_assistant(conversations)

        if not user_msg or not assistant_msg:
            skipped += 1
            continue

        prompt = user_msg["content"]
        label = ex.get("label", True)

        if label:
            groups[prompt]["true"].append(assistant_msg)
        else:
            groups[prompt]["false"].append(assistant_msg)

    if skipped > 0:
        print(f"  - Skipped {skipped} examples without user/assistant pair")

    return groups


def create_gspo_examples(groups: dict[str, dict], include_preference: bool = False) -> list[dict]:
    """Create GSPO examples from prompts with True versions.

    Handles both tool-call responses and text-only responses.

    Args:
        groups: Grouped examples by prompt
        include_preference: If True, include chosen/rejected for preference learning
    """
    examples = []
    tool_examples = 0
    text_examples = 0

    for prompt, versions in groups.items():
        if not versions["true"]:
            continue  # Skip prompts without a True version

        # Use first True version as ground truth
        chosen = versions["true"][0]
        tool_name, tool_args = extract_tool_info(chosen)

        # Create prompt in ChatML format
        prompt_messages = [{"role": "user", "content": prompt}]

        if tool_name and tool_args:
            # Tool-call response
            example = {
                "prompt": prompt_messages,
                "ground_truth_tool": tool_name,
                "ground_truth_args": tool_args
            }
            tool_examples += 1
        elif chosen.get("content"):
            # Text-only response (no tool call, just text content)
            example = {
                "prompt": prompt_messages,
                "ground_truth_response": chosen["content"]
            }
            text_examples += 1
        else:
            # Neither tool call nor text content - skip
            continue

        # Optionally add preference fields
        if include_preference and versions["false"]:
            example["chosen"] = chosen
            example["rejected"] = versions["false"][0]  # Use first False version

        examples.append(example)

    print(f"  - Tool-call examples: {tool_examples}")
    print(f"  - Text-only examples: {text_examples}")

    return examples


def create_preference_pairs(groups: dict[str, dict]) -> list[dict]:
    """Create preference learning pairs from prompts with both True and False versions."""
    pairs = []

    for prompt, versions in groups.items():
        if not (versions["true"] and versions["false"]):
            continue  # Need both for preference learning

        # Use first of each
        chosen = versions["true"][0]
        rejected = versions["false"][0]
        tool_name, tool_args = extract_tool_info(chosen)

        # Create prompt in ChatML format
        prompt_messages = [{"role": "user", "content": prompt}]

        pair = {
            "prompt": prompt_messages,
            "chosen": chosen,
            "rejected": rejected
        }

        # Add ground truth if available
        if tool_name:
            pair["ground_truth_tool"] = tool_name
        if tool_args:
            pair["ground_truth_args"] = tool_args

        pairs.append(pair)

    return pairs


def validate_example(example: dict) -> bool:
    """Validate a GSPO/preference example has required fields.

    Accepts two types:
    1. Tool-call: ground_truth_tool + ground_truth_args
    2. Text-only: ground_truth_response
    """
    # Must have prompt as list
    if not isinstance(example.get("prompt"), list):
        return False
    if len(example["prompt"]) == 0:
        return False

    # Check for text-only response
    if example.get("ground_truth_response"):
        # Text-only example - must have non-empty string
        return isinstance(example["ground_truth_response"], str) and len(example["ground_truth_response"]) > 0

    # Check for tool-call response
    if not example.get("ground_truth_tool"):
        return False

    # Tool must have valid agent prefix
    if not is_valid_tool(example["ground_truth_tool"]):
        return False

    # Must have ground_truth_args
    if not isinstance(example.get("ground_truth_args"), dict):
        return False

    return True


def convert_kto(
    input_path: str,
    output_path: str,
    output_format: str = "gspo",
    shuffle: bool = True
) -> dict:
    """
    Convert KTO dataset to GSPO or preference format.

    Args:
        input_path: Path to KTO JSONL file
        output_path: Path for output JSONL file
        output_format: "gspo" for ground_truth only, "preference" for chosen/rejected pairs
        shuffle: Whether to shuffle the output

    Returns:
        Statistics about the conversion
    """
    print(f"Loading KTO dataset from: {input_path}")
    examples = load_kto_dataset(input_path)
    print(f"Loaded {len(examples)} examples")

    # Group by prompt
    groups = group_by_prompt(examples)
    print(f"Found {len(groups)} unique prompts")

    # Analyze groups
    matched = sum(1 for g in groups.values() if g["true"] and g["false"])
    true_only = sum(1 for g in groups.values() if g["true"] and not g["false"])
    false_only = sum(1 for g in groups.values() if not g["true"] and g["false"])

    print(f"  - Prompts with both True/False: {matched}")
    print(f"  - Prompts with only True: {true_only}")
    print(f"  - Prompts with only False: {false_only}")

    # Create examples based on format
    if output_format == "gspo":
        print("\nCreating GSPO examples (ground_truth only)...")
        output_examples = create_gspo_examples(groups, include_preference=False)
    elif output_format == "preference":
        print("\nCreating preference pairs (chosen/rejected)...")
        output_examples = create_preference_pairs(groups)
    elif output_format == "gspo_preference":
        print("\nCreating GSPO examples with preference (ground_truth + chosen/rejected)...")
        output_examples = create_gspo_examples(groups, include_preference=True)
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    # Validate and filter
    valid_examples = [ex for ex in output_examples if validate_example(ex)]
    print(f"Valid examples after filtering: {len(valid_examples)}")

    # Shuffle if requested
    if shuffle:
        random.shuffle(valid_examples)

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in valid_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nWrote {len(valid_examples)} examples to: {output_path}")

    return {
        "total_kto_examples": len(examples),
        "unique_prompts": len(groups),
        "matched_prompts": matched,
        "true_only_prompts": true_only,
        "false_only_prompts": false_only,
        "output_format": output_format,
        "output_examples": len(valid_examples)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert KTO dataset to GSPO or preference format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output formats:
  gspo            - Ground truth only (prompt + ground_truth_tool + ground_truth_args)
                    Use for GSPO-only training without SFT
  preference      - Chosen/rejected pairs (prompt + chosen + rejected + ground_truth)
                    Use for post-SFT preference learning
  gspo_preference - Both ground truth and preference fields
                    Use for combined training

Examples:
  # Create GSPO dataset for GSPO-only training
  python convert_kto_to_gspo.py input.jsonl output.jsonl --format gspo

  # Create preference pairs for post-SFT training
  python convert_kto_to_gspo.py input.jsonl output.jsonl --format preference
        """
    )
    parser.add_argument("input", help="Input KTO JSONL file")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument("--format", "-f",
                        choices=["gspo", "preference", "gspo_preference"],
                        default="gspo",
                        help="Output format (default: gspo)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle the output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")

    args = parser.parse_args()

    random.seed(args.seed)

    stats = convert_kto(
        args.input,
        args.output,
        output_format=args.format,
        shuffle=not args.no_shuffle
    )

    print("\n=== Conversion Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
