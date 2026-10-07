#!/usr/bin/env bash
# Combine multiple JSONL datasets into a single shuffled file
# Adds a "source" field to each example with the parent folder name
#
# Usage: ./combine_datasets.sh -o OUTPUT FILE1 FILE2 [FILE3...]
#        ./combine_datasets.sh -o OUTPUT DIR/          (all .jsonl in dir)
#
# Examples:
#   # Combine specific files
#   ./combine_datasets.sh -o Datasets/combined.jsonl \
#     Datasets/tools_datasets/thinking/contentManager/tools_v1.8.jsonl \
#     Datasets/tools_datasets/thinking/storageManager/tools_v1.5.jsonl \
#     Datasets/tools_datasets/thinking/searchManager/tools_v1.3.jsonl
#
#   # Combine all JSONL files in a directory
#   ./combine_datasets.sh -o Datasets/all_managers.jsonl \
#     Datasets/tools_datasets/thinking/
#
#   # Combine with custom seed for reproducible shuffle
#   ./combine_datasets.sh -o Datasets/combined.jsonl --seed 42 FILE1 FILE2

set -euo pipefail

# Parse args
OUTPUT=""
SEED=""
INPUTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output) OUTPUT="$2"; shift 2 ;;
    --seed)      SEED="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 -o OUTPUT [--seed N] FILE1 [FILE2...] [DIR/]"
      echo ""
      echo "Combines multiple JSONL files into a single shuffled dataset."
      echo "Adds 'source' field from parent folder name (e.g., 'contentManager')."
      echo ""
      echo "Options:"
      echo "  -o, --output PATH   Output JSONL file (required)"
      echo "  --seed N            Random seed for reproducible shuffle"
      echo ""
      echo "Arguments:"
      echo "  FILE1 FILE2...      JSONL files to combine"
      echo "  DIR/                Directory — combines all *.jsonl files within"
      exit 0
      ;;
    *) INPUTS+=("$1"); shift ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  echo "Error: -o/--output is required"
  echo "Usage: $0 -o OUTPUT FILE1 [FILE2...]"
  exit 1
fi

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "Error: No input files or directories specified"
  exit 1
fi

# Expand directories to their JSONL files
FILES=()
for input in "${INPUTS[@]}"; do
  if [[ -d "$input" ]]; then
    while IFS= read -r f; do
      FILES+=("$f")
    done < <(find "$input" -name "*.jsonl" -type f | sort)
  elif [[ -f "$input" ]]; then
    FILES+=("$input")
  else
    echo "Warning: Skipping not found: $input"
  fi
done

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Error: No JSONL files found"
  exit 1
fi

echo "=== Combine Datasets ==="
echo "Output: $OUTPUT"
echo "Files:  ${#FILES[@]}"
for f in "${FILES[@]}"; do
  lines=$(wc -l < "$f" | tr -d ' ')
  folder=$(basename "$(dirname "$f")")
  echo "  [$folder] $f ($lines lines)"
done
echo ""

# Combine and shuffle using Python
SEED_ARG="${SEED:-}"

python3 -c "
import json
import random
import sys
from pathlib import Path

files = '''$(printf '%s\n' "${FILES[@]}")'''.strip().split('\n')
output = '$OUTPUT'
seed = '$SEED_ARG'

if seed:
    random.seed(int(seed))

all_examples = []
stats = {}

for filepath in files:
    filepath = filepath.strip()
    if not filepath:
        continue

    folder = Path(filepath).parent.name
    if folder not in stats:
        stats[folder] = 0

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f'Warning: Skipping malformed JSON in {filepath}:{line_num}: {e}')
                continue

            # Skip metadata lines
            if '_meta' in data:
                continue

            # Add source label
            data['source'] = folder

            all_examples.append(data)
            stats[folder] += 1

# Shuffle
random.shuffle(all_examples)

# Write output
Path(output).parent.mkdir(parents=True, exist_ok=True)
with open(output, 'w') as f:
    for example in all_examples:
        f.write(json.dumps(example) + '\n')

# Summary
total = len(all_examples)
print(f'Combined {total} examples from {len(stats)} sources:')
for folder, count in sorted(stats.items()):
    pct = count / total * 100 if total else 0
    print(f'  {folder}: {count} ({pct:.1f}%)')
print(f'\nOutput: {output}')
print(f'Total:  {total} examples (shuffled' + (f', seed={seed}' if seed else '') + ')')
"
