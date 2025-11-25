#!/usr/bin/env python3
"""
Restructure SFT dataset to separate tool calls from results.
Converts single-turn examples into multi-turn ping-pong conversations.

RULES:
- ✅ Text → Tool Call (ALLOWED): Model explains, then calls tool
- ❌ Tool Call → Text (NOT ALLOWED): Split into separate turns
- Each turn is either: tool_call alone, text alone, or text+tool_call

Before (single turn):
  User: "Do something"
  Assistant: "tool_call: X\narguments: {...}\n\nResult: {...}\n\nResponse text"

After (multi-turn):
  User: "Do something"
  Assistant: "tool_call: X\narguments: {...}"
  User: "Result: {...}"
  Assistant: "Response text"

With reasoning (text before tool):
  User: "Result: {...}"
  Assistant: "Found it. Loading now.\n\ntool_call: loadSession\narguments: {...}"
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def parse_assistant_message(content: str) -> List[Dict[str, str]]:
    """
    Parse assistant message into segments: tool_calls, results, and text.

    Returns list of segments with 'type' and 'content':
    - type='tool_call': tool call with arguments (everything up to Result:)
    - type='result': result from tool execution
    - type='text': regular text response
    """
    segments = []

    # Split by "Result:" to find boundaries
    # Pattern: tool_call + arguments, then Result:, then text/next tool_call

    remaining = content

    while remaining:
        remaining = remaining.strip()

        # Look for tool_call
        if remaining.startswith('tool_call:'):
            # Find where this tool call ends (at "Result:" or another "tool_call:" or end)
            result_match = re.search(r'\n\s*Result:\s*', remaining)

            if result_match:
                # Extract tool call (everything before Result:)
                tool_call_text = remaining[:result_match.start()].strip()
                segments.append({'type': 'tool_call', 'content': tool_call_text})

                # Move past "Result:"
                remaining = remaining[result_match.end():]

                # Now extract the result JSON using balanced brace counting
                brace_count = 0
                json_end = -1
                in_json = False

                for i, char in enumerate(remaining):
                    if char == '{':
                        brace_count += 1
                        in_json = True
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and in_json:
                            json_end = i + 1
                            break

                if json_end > 0:
                    result_json = remaining[:json_end]
                    segments.append({'type': 'result', 'content': 'Result: ' + result_json})
                    remaining = remaining[json_end:]
                else:
                    # Couldn't parse JSON, skip this
                    print(f"⚠ Could not parse result JSON from: {remaining[:100]}")
                    break

            else:
                # No result after this tool call (shouldn't happen in training data)
                segments.append({'type': 'tool_call', 'content': remaining})
                break

        else:
            # This is text (response or intermediate text)
            # Find next tool_call or end
            next_tool = remaining.find('tool_call:')

            if next_tool > 0:
                # Text before next tool call
                text = remaining[:next_tool].strip()
                if text:
                    segments.append({'type': 'text', 'content': text})
                remaining = remaining[next_tool:]
            else:
                # Text until end
                if remaining:
                    segments.append({'type': 'text', 'content': remaining})
                break

    return segments


def restructure_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert single-turn example into multi-turn ping-pong examples.

    Returns list of new examples (one per turn pair).
    """
    conversations = example['conversations']

    if len(conversations) != 2:
        # Multi-turn already or invalid - skip
        print(f"⚠ Skipping example with {len(conversations)} turns")
        return []

    user_msg = conversations[0]['content']
    assistant_msg = conversations[1]['content']

    # Parse assistant message into segments
    segments = parse_assistant_message(assistant_msg)

    if not segments:
        print(f"⚠ No segments found in assistant message")
        return []

    # Build new multi-turn examples
    # Strategy: Combine text+tool_call when they're adjacent (intermediate responses)
    new_examples = []
    current_prompt = user_msg

    i = 0
    while i < len(segments):
        segment = segments[i]

        if segment['type'] == 'tool_call':
            # Create turn: prompt -> tool_call
            new_examples.append({
                'conversations': [
                    {'role': 'user', 'content': current_prompt},
                    {'role': 'assistant', 'content': segment['content']}
                ],
                'label': True
            })
            # Next prompt will be the result (if there is one)
            current_prompt = None
            i += 1

        elif segment['type'] == 'result':
            # This becomes the next prompt
            current_prompt = segment['content']
            i += 1

        elif segment['type'] == 'text':
            # Check if next segment is a tool_call
            # If so, combine them (text + tool_call in one turn)
            # RULE: Text → Tool Call is ALLOWED
            if i + 1 < len(segments) and segments[i+1]['type'] == 'tool_call':
                combined_content = segment['content'] + '\n\n' + segments[i+1]['content']
                if current_prompt:
                    new_examples.append({
                        'conversations': [
                            {'role': 'user', 'content': current_prompt},
                            {'role': 'assistant', 'content': combined_content}
                        ],
                        'label': True
                    })
                    current_prompt = None
                    i += 2  # Skip both text and tool_call
                else:
                    print(f"⚠ Text+tool_call without prompt")
                    i += 1
            else:
                # Standalone text (final response)
                if current_prompt:
                    new_examples.append({
                        'conversations': [
                            {'role': 'user', 'content': current_prompt},
                            {'role': 'assistant', 'content': segment['content']}
                        ],
                        'label': True
                    })
                else:
                    print(f"⚠ Text segment without prompt: {segment['content'][:50]}...")
                i += 1

    return new_examples


def main():
    input_file = Path('/mnt/c/Users/Joseph/Documents/Code/Toolset-Training/Datasets/syngen_tools_sft_11.18.25.jsonl')
    output_file = Path('/mnt/c/Users/Joseph/Documents/Code/Toolset-Training/Datasets/syngen_tools_sft_pingpong_11.18.25.jsonl')

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print()

    total_input = 0
    total_output = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            example = json.loads(line)
            total_input += 1

            # Restructure this example
            new_examples = restructure_example(example)

            # Write new examples
            for new_ex in new_examples:
                f_out.write(json.dumps(new_ex) + '\n')
                total_output += 1

            if line_num % 100 == 0:
                print(f"Processed {line_num} examples... ({total_output} output)")

    print()
    print(f"✓ Complete!")
    print(f"  Input examples:  {total_input}")
    print(f"  Output examples: {total_output}")
    print(f"  Expansion ratio: {total_output/total_input:.2f}x")
    print()
    print(f"Output saved to: {output_file}")


if __name__ == '__main__':
    main()
