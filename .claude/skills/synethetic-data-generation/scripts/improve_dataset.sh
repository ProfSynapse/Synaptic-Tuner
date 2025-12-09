#!/bin/bash
# improve_dataset.sh - Extract and improve JSONL dataset examples
#
# Usage:
#   ./improve_dataset.sh <file> <start_line> <count>
#
# Example:
#   ./improve_dataset.sh data.jsonl 10 5
#   Reads lines 10-14 and prints them for manual improvement

set -e

FILE="$1"
START_LINE="$2"
COUNT="${3:-10}"

if [ -z "$FILE" ] || [ -z "$START_LINE" ]; then
    echo "Usage: $0 <file> <start_line> [count]"
    echo "  file: JSONL file to process"
    echo "  start_line: Starting line number (1-indexed)"
    echo "  count: Number of lines to extract (default: 10)"
    exit 1
fi

if [ ! -f "$FILE" ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

END_LINE=$((START_LINE + COUNT - 1))

echo "=== Extracting lines $START_LINE-$END_LINE from $FILE ==="
echo ""

# Extract the lines
sed -n "${START_LINE},${END_LINE}p" "$FILE"

echo ""
echo "=== Instructions ==="
echo "1. Review the extracted lines above"
echo "2. Hand-craft improved versions"
echo "3. Use replace_lines.sh to apply changes"
