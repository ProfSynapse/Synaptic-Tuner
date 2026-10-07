#!/usr/bin/env bash
# Convert JSONL dataset to readable Markdown for human review
#
# Usage: ./jsonl_to_markdown.sh <input.jsonl> [output.md] [--start N] [--end N]
#
# Examples:
#   ./jsonl_to_markdown.sh data.jsonl                        # Full file → data_review.md
#   ./jsonl_to_markdown.sh data.jsonl review.md              # Custom output path
#   ./jsonl_to_markdown.sh data.jsonl --start 5 --end 10     # Lines 5-10 only

set -euo pipefail

INPUT="${1:?Usage: $0 <input.jsonl> [output.md] [--start N] [--end N]}"
shift

# Parse args
OUTPUT=""
START=1
END=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end)   END="$2"; shift 2 ;;
    *)       OUTPUT="$1"; shift ;;
  esac
done

# Default output: same name with _review.md suffix
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${INPUT%.jsonl}_review.md"
fi

# Verify input exists
if [[ ! -f "$INPUT" ]]; then
  echo "Error: File not found: $INPUT"
  exit 1
fi

TOTAL_LINES=$(wc -l < "$INPUT" | tr -d ' ')
if [[ -z "$END" ]]; then
  END="$TOTAL_LINES"
fi

echo "Converting: $INPUT → $OUTPUT"
echo "Lines: $START-$END of $TOTAL_LINES"
echo ""

# Generate markdown using Python (handles JSON parsing properly)
python3 -c "
import json
import sys
import textwrap

input_file = '$INPUT'
output_file = '$OUTPUT'
start = int('$START')
end = int('$END')

lines = []
with open(input_file) as f:
    for i, line in enumerate(f, 1):
        if i < start:
            continue
        if i > end:
            break
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            lines.append((i, data))
        except json.JSONDecodeError as e:
            lines.append((i, {'_parse_error': str(e)}))

with open(output_file, 'w') as out:
    out.write(f'# Dataset Review: {input_file}\n\n')
    out.write(f'**Lines {start}-{end}** of {int(\"$TOTAL_LINES\")} total\n\n')
    out.write('---\n\n')

    for line_num, data in lines:
        # Skip metadata lines
        if '_meta' in data:
            out.write(f'## Line {line_num}: Metadata\n\n')
            out.write(f'\`\`\`json\n{json.dumps(data[\"_meta\"], indent=2)}\n\`\`\`\n\n')
            out.write('---\n\n')
            continue

        # Handle parse errors
        if '_parse_error' in data:
            out.write(f'## Line {line_num}: PARSE ERROR\n\n')
            out.write(f'> {data[\"_parse_error\"]}\n\n')
            out.write('---\n\n')
            continue

        # Label
        label = data.get('label', None)
        label_str = ''
        if label is True:
            label_str = ' ✅ positive'
        elif label is False:
            label_str = ' ❌ negative'

        out.write(f'## Example {line_num}{label_str}\n\n')

        conversations = data.get('conversations', [])
        for msg in conversations:
            role = msg.get('role', 'unknown')

            # Role header
            if role == 'system':
                out.write('### 🖥️ System\n\n')
            elif role == 'user':
                out.write('### 👤 User\n\n')
            elif role == 'assistant':
                out.write('### 🤖 Assistant\n\n')
            else:
                out.write(f'### {role}\n\n')

            content = msg.get('content', '')

            if role == 'system' and content:
                # System prompts are long — collapse by default
                out.write('<details>\n<summary>System prompt (click to expand)</summary>\n\n')
                out.write(f'\`\`\`\n{content}\n\`\`\`\n\n')
                out.write('</details>\n\n')

            elif role == 'assistant' and content:
                # Split thinking from response
                import re
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)

                if thinking_match:
                    thinking_raw = thinking_match.group(1).strip()
                    response_text = content[:thinking_match.start()] + content[thinking_match.end():]
                    response_text = response_text.strip()

                    # Try to parse thinking as JSON for pretty-print
                    out.write('**Thinking:**\n\n')
                    try:
                        thinking_json = json.loads(thinking_raw)
                        out.write(f'\`\`\`json\n{json.dumps(thinking_json, indent=2)}\n\`\`\`\n\n')
                    except (json.JSONDecodeError, TypeError):
                        out.write(f'\`\`\`\n{thinking_raw}\n\`\`\`\n\n')

                    if response_text:
                        out.write('**Response:**\n\n')
                        out.write(f'{response_text}\n\n')
                else:
                    out.write(f'{content}\n\n')

            elif content:
                out.write(f'{content}\n\n')

            # Tool calls
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                out.write('**Tool Calls:**\n\n')
                for tc in tool_calls:
                    func = tc.get('function', {})
                    name = func.get('name', 'unknown')
                    args = func.get('arguments', {})

                    out.write(f'> \`{name}\`\n\n')

                    # Parse arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass

                    if isinstance(args, dict):
                        out.write(f'\`\`\`json\n{json.dumps(args, indent=2)}\n\`\`\`\n\n')
                    else:
                        out.write(f'\`\`\`\n{args}\n\`\`\`\n\n')

        out.write('---\n\n')

    out.write(f'*Generated from \`{input_file}\` lines {start}-{end}*\n')

print(f'Done! {len(lines)} examples written to {output_file}')
"
