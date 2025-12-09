#!/usr/bin/env python3
"""
Improve thinking blocks in synthetic dataset JSONL files.

This script processes JSONL files containing tool-calling examples and improves
the quality of their thinking blocks by:
- Making goals specific and actionable
- Adding rich contextual memory
- Creating distinct requirements (verification checks) vs plans (execution steps)
- Setting risk-calibrated confidence scores
"""

import json
import sys
import argparse
from pathlib import Path


def improve_thinking_block(example, line_num):
    """
    Hand-craft improvements to a thinking block based on the tool being called.

    Args:
        example: Dict containing the conversation with thinking block
        line_num: Line number in file (for reference)

    Returns:
        Dict with improved thinking block
    """
    # Extract the assistant's response
    conversations = example.get("conversations", [])
    assistant_msg = None
    for msg in conversations:
        if msg.get("role") == "assistant":
            assistant_msg = msg
            break

    if not assistant_msg:
        print(f"Warning: No assistant message found in line {line_num}")
        return example

    content = assistant_msg.get("content", "")

    # Extract current thinking block
    if "<thinking>" not in content or "</thinking>" not in content:
        print(f"Warning: No thinking block found in line {line_num}")
        return example

    start = content.index("<thinking>") + len("<thinking>")
    end = content.index("</thinking>")
    thinking_json = content[start:end].strip()

    try:
        thinking = json.loads(thinking_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in thinking block at line {line_num}: {e}")
        return example

    # Identify tool being called
    tool_calls = assistant_msg.get("tool_calls", [])
    if not tool_calls:
        print(f"Warning: No tool calls found in line {line_num}")
        return example

    tool_name = tool_calls[0].get("function", {}).get("name", "")

    # Check if improvements are needed
    requirements = thinking.get("requirements", [])
    plan = thinking.get("plan", [])

    # If requirements == plan, they need to be distinct
    if requirements == plan:
        print(f"Line {line_num}: Requirements duplicate plan - needs improvement")
        # This is where we'd apply improvements
        # For now, just flag it
        return example

    # Check for weak memory
    memory = thinking.get("memory", "")
    if len(memory) < 50:  # Arbitrary threshold
        print(f"Line {line_num}: Weak memory - needs improvement")

    # Check confidence calibration
    confidence = thinking.get("confidence", 0.5)
    risky = thinking.get("assessment", {}).get("risky", False)

    if risky and confidence > 0.7:
        print(f"Line {line_num}: Risky operation with high confidence ({confidence}) - needs recalibration")

    return example


def process_file(input_file, output_file, start_line=None, end_line=None, dry_run=False):
    """
    Process a JSONL file and improve thinking blocks.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        start_line: Optional start line (1-indexed)
        end_line: Optional end line (1-indexed)
        dry_run: If True, only analyze without writing
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if start_line and line_num < start_line:
                examples.append(line.rstrip('\n'))
                continue
            if end_line and line_num > end_line:
                examples.append(line.rstrip('\n'))
                continue

            try:
                example = json.loads(line)
                improved = improve_thinking_block(example, line_num)
                examples.append(json.dumps(improved, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON at line {line_num}: {e}")
                examples.append(line.rstrip('\n'))

    if not dry_run:
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(example + '\n')
        print(f"Wrote {len(examples)} examples to {output_file}")
    else:
        print(f"Dry run: Would process {len(examples)} examples")


def main():
    parser = argparse.ArgumentParser(
        description="Improve thinking blocks in synthetic dataset JSONL files"
    )
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("output_file", help="Output JSONL file")
    parser.add_argument("--start-line", type=int, help="Start at line number (1-indexed)")
    parser.add_argument("--end-line", type=int, help="End at line number (1-indexed)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing")

    args = parser.parse_args()

    process_file(
        args.input_file,
        args.output_file,
        start_line=args.start_line,
        end_line=args.end_line,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
