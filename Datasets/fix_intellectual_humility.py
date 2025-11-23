#!/usr/bin/env python3
"""
Fix intellectual_humility dataset by replacing "pattern": "text_only" with "label": true.

The text_only entries are positive examples of intellectual humility (asking clarifying
questions instead of making assumptions), so they should all have label=true.
"""

import json
from pathlib import Path

def fix_dataset():
    input_file = Path(__file__).parent / "behavior_datasets/intellectual_humility/pairs_v1.0.jsonl"
    backup_file = input_file.with_suffix('.jsonl.backup')

    # Create backup
    print(f"Creating backup: {backup_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        backup_content = f.read()
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(backup_content)

    # Fix entries
    fixed_count = 0
    total_count = 0
    fixed_entries = []

    print(f"\nProcessing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                total_count += 1
                data = json.loads(line)

                # Check if this has "pattern" instead of "label"
                if 'pattern' in data:
                    pattern_type = data['pattern']
                    # Remove "pattern" and add "label": true
                    # Both text_only and tool_only are positive examples of intellectual_humility
                    del data['pattern']
                    data['label'] = True
                    fixed_count += 1
                    print(f"  Fixed line {line_num}: Added label=true for {pattern_type} response")

                fixed_entries.append(data)

    # Write fixed version
    print(f"\nWriting fixed dataset...")
    with open(input_file, 'w', encoding='utf-8') as f:
        for entry in fixed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n✓ Fix complete!")
    print(f"  Total entries: {total_count}")
    print(f"  Fixed entries: {fixed_count}")
    print(f"  Backup saved to: {backup_file}")

if __name__ == "__main__":
    fix_dataset()
