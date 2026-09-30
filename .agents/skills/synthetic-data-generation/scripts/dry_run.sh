#!/usr/bin/env bash
# Dry-run script for testing new or modified scenarios
#
# Usage: ./dry_run.sh <scenario_key> [count] [extra_args...]
#
# Examples:
#   ./dry_run.sh storageManager_createFolder          # 3 examples (default)
#   ./dry_run.sh contentManager_write_blog 5          # 5 examples
#   ./dry_run.sh essay_outline 2 --docs "essays/"     # 2 with docs

set -euo pipefail

SCENARIO_KEY="${1:?Usage: $0 <scenario_key> [count] [extra_args...]}"
COUNT="${2:-3}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="Datasets/synthchat"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/dryrun_${SCENARIO_KEY}_${TIMESTAMP}.jsonl"

echo "=== Dry Run: ${SCENARIO_KEY} ==="
echo "Count:  ${COUNT}"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Create temporary targets file
TARGETS_FILE=$(mktemp)
echo "{\"${SCENARIO_KEY}\": ${COUNT}}" > "$TARGETS_FILE"

# Run generation with limited iterations
python -m SynthChat.run generate \
  --scenarios "$SCENARIO_KEY" \
  --targets-file "$TARGETS_FILE" \
  --max-iterations 3 \
  --output "$OUTPUT_FILE" \
  "$@"

rm -f "$TARGETS_FILE"

echo ""
echo "=== Dry Run Complete ==="
echo "Output: ${OUTPUT_FILE}"
echo ""

# Show summary
LINES=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
echo "Generated ${LINES} lines (includes metadata header if enabled)"
echo ""
echo "To inspect:"
echo "  python -c \"import json, sys; [print(json.dumps(json.loads(l), indent=2)) for l in open('${OUTPUT_FILE}')]\""
echo ""
echo "To validate:"
echo "  python -m SynthChat.run validate -i ${OUTPUT_FILE}"
