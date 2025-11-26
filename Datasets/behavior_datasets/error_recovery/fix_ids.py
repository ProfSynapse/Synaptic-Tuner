#!/usr/bin/env python3
"""Fix ID format issues in error_recovery dataset (10 chars -> 9 chars)."""

import json
import re

input_file = "pairs_v1.1.jsonl"
output_file = "pairs_v1.2.jsonl"

# Pattern to find 10-char IDs and fix them
session_pattern = re.compile(r'session_(\d{13})_([a-z0-9]{10})')
workspace_pattern = re.compile(r'ws_(\d{13})_([a-z0-9]{10})')

def fix_id(match):
    """Truncate 10-char suffix to 9 chars."""
    timestamp = match.group(1)
    suffix = match.group(2)[:9]  # Truncate to 9 chars
    prefix = "session_" if match.group(0).startswith("session_") else "ws_"
    return f"{prefix}{timestamp}_{suffix}"

fixed_count = 0
examples = []

with open(input_file, 'r') as f:
    for line in f:
        original = line
        # Fix in the JSON string directly
        line = session_pattern.sub(lambda m: f"session_{m.group(1)}_{m.group(2)[:9]}", line)
        line = workspace_pattern.sub(lambda m: f"ws_{m.group(1)}_{m.group(2)[:9]}", line)
        if line != original:
            fixed_count += 1
        examples.append(line.strip())

with open(output_file, 'w') as f:
    for ex in examples:
        f.write(ex + '\n')

print(f"Fixed {fixed_count} examples with 10-char IDs")
print(f"Output: {output_file}")
