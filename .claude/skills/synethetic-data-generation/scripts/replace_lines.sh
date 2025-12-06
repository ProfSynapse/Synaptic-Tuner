#!/bin/bash
# replace_lines.sh - Replace specific lines in a JSONL file
#
# Usage:
#   ./replace_lines.sh <file> <line_num> <new_content>
#
# Example:
#   ./replace_lines.sh data.jsonl 10 '{"improved": "content"}'

set -e

FILE="$1"
LINE_NUM="$2"
NEW_CONTENT="$3"

if [ -z "$FILE" ] || [ -z "$LINE_NUM" ] || [ -z "$NEW_CONTENT" ]; then
    echo "Usage: $0 <file> <line_num> <new_content>"
    echo "  file: JSONL file to modify"
    echo "  line_num: Line number to replace (1-indexed)"
    echo "  new_content: New JSON content (properly escaped)"
    exit 1
fi

if [ ! -f "$FILE" ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

# Create backup
cp "$FILE" "${FILE}.backup"

# Create temp file with escaped content
TEMP_FILE=$(mktemp)
echo "$NEW_CONTENT" > "$TEMP_FILE"

# Use sed to replace the line
sed -i "${LINE_NUM}r $TEMP_FILE" "$FILE"
sed -i "${LINE_NUM}d" "$FILE"

rm "$TEMP_FILE"

echo "✓ Replaced line $LINE_NUM in $FILE"
echo "  Backup saved as ${FILE}.backup"
