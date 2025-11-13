#!/usr/bin/env python3
import json

# ============ CONFIGURATION ============
# Set to "TT" to find True-True pairs, "FF" to find False-False pairs
pair_type = "FF"  # Change to "TT" or "FF"
# =======================================

with open("syngen_toolset_v1.0.0_copilot.jsonl", "r") as f:
    lines = f.readlines()

# Find the next pair
target_pair = None
for i in range(len(lines) - 1):
    curr = json.loads(lines[i])
    next_line = json.loads(lines[i+1])
    if pair_type == "TT":
        if curr.get("label") == True and next_line.get("label") == True:
            target_pair = i
            break
    elif pair_type == "FF":
        if curr.get("label") == False and next_line.get("label") == False:
            target_pair = i
            break

if target_pair is None:
    print(f"\n✓ No {pair_type} pairs found!")
    exit(0)

# Show the first line of the pair (the one we'll insert AFTER)
line_to_read = target_pair + 1
example = json.loads(lines[line_to_read - 1])
next_example = json.loads(lines[line_to_read])

print(f"\n=== FOUND {pair_type} PAIR AT LINES {line_to_read}-{line_to_read+1} ===")
print(f"\n--- LINE {line_to_read} (First of pair) ---")
print(f"Label: {example['label']}")
print(f"\nUser: {example['conversations'][0]['content']}")
print(f"\nAssistant response (first 500 chars):\n{example['conversations'][1]['content'][:500]}...")

print(f"\n--- LINE {line_to_read+1} (Second of pair) ---")
print(f"Label: {next_example['label']}")
print(f"\nUser: {next_example['conversations'][0]['content']}")
print(f"\nAssistant response (first 500 chars):\n{next_example['conversations'][1]['content'][:500]}...")

# Show context
print(f"\n=== CONTEXT ===")
for i in range(max(0, line_to_read - 3), min(len(lines), line_to_read + 4)):
    ex = json.loads(lines[i])
    if i == line_to_read - 1:
        marker = " <-- PAIR START"
    elif i == line_to_read:
        marker = " <-- PAIR END (insert after this)"
    else:
        marker = ""
    print(f"Line {i+1}: label={ex['label']}{marker}")

# Count all pairs
true_true_pairs = []
false_false_pairs = []
for i in range(len(lines) - 1):
    curr = json.loads(lines[i])
    next_line = json.loads(lines[i+1])
    if curr.get("label") == True and next_line.get("label") == True:
        true_true_pairs.append(i+1)
    elif curr.get("label") == False and next_line.get("label") == False:
        false_false_pairs.append(i+1)

print(f"\n=== STATUS ===")
print(f"Total lines: {len(lines)}")
print(f"True-True pairs: {len(true_true_pairs)}")
print(f"False-False pairs: {len(false_false_pairs)}")
print(f"Total breaks needed: {len(true_true_pairs) + len(false_false_pairs)}")

print(f"\n=== ACTION NEEDED ===")
if pair_type == "FF":
    print(f"Insert a TRUE example at line {line_to_read+1}")
    print(f"This will break the FF pair at lines {line_to_read}-{line_to_read+1}")
    print(f"\nUpdate insert_true_example.py:")
    print(f"  line_number = {line_to_read+1}")
else:
    print(f"Insert a FALSE example at line {line_to_read+1}")
    print(f"This will break the TT pair at lines {line_to_read}-{line_to_read+1}")
    print(f"\nUpdate insert_at_line.py:")
    print(f"  line_number = {line_to_read+1}")
