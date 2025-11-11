import json

file_path = "/home/user/Toolset-Training/syngen_toolset_v1.0.0_claude.jsonl"

with open(file_path, 'r') as f:
    lines = f.readlines()

consecutive_pairs = []
labels = []

for i, line in enumerate(lines):
    try:
        entry = json.loads(line)
        label = entry.get('label')
        labels.append((i+1, label))  # Store line number and label
    except Exception as e:
        print(f"Error parsing line {i+1}: {e}")
        labels.append((i+1, None))

# Check for consecutive True-True pairs
for i in range(len(labels)-1):
    if labels[i][1] == True and labels[i+1][1] == True:
        consecutive_pairs.append((labels[i][0], labels[i+1][0]))

print(f"Total lines: {len(lines)}")
print(f"Found {len(consecutive_pairs)} consecutive True-True pairs")
print()

if len(consecutive_pairs) > 0:
    print(f"First 20 pairs (line numbers):")
    for pair in consecutive_pairs[:20]:
        print(f"  Lines {pair[0]}-{pair[1]}")

    if len(consecutive_pairs) > 20:
        print(f"\nLast 10 pairs (line numbers):")
        for pair in consecutive_pairs[-10:]:
            print(f"  Lines {pair[0]}-{pair[1]}")
else:
    print("✓ No consecutive True-True pairs found! The dataset is properly interleaved.")

# Also check for consecutive False-False pairs (which shouldn't exist per constraints)
consecutive_false_pairs = []
for i in range(len(labels)-1):
    if labels[i][1] == False and labels[i+1][1] == False:
        consecutive_false_pairs.append((labels[i][0], labels[i+1][0]))

if len(consecutive_false_pairs) > 0:
    print(f"\n⚠ WARNING: Found {len(consecutive_false_pairs)} consecutive False-False pairs:")
    for pair in consecutive_false_pairs[:10]:
        print(f"  Lines {pair[0]}-{pair[1]}")
else:
    print("\n✓ No consecutive False-False pairs found.")

# Show label distribution
true_count = sum(1 for _, label in labels if label == True)
false_count = sum(1 for _, label in labels if label == False)
print(f"\nLabel distribution:")
print(f"  True: {true_count}")
print(f"  False: {false_count}")
print(f"  Total: {true_count + false_count}")
