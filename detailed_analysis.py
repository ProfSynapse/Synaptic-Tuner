import json

file_path = "/home/user/Toolset-Training/syngen_toolset_v1.0.0_claude.jsonl"

with open(file_path, 'r') as f:
    lines = f.readlines()

labels = []
for i, line in enumerate(lines):
    try:
        entry = json.loads(line)
        label = entry.get('label')
        labels.append((i+1, label))
    except Exception as e:
        labels.append((i+1, None))

# Find consecutive sequences
def find_sequences(labels, target_value):
    sequences = []
    current_seq = []

    for line_num, label in labels:
        if label == target_value:
            current_seq.append(line_num)
        else:
            if len(current_seq) >= 2:
                sequences.append(current_seq)
            current_seq = []

    if len(current_seq) >= 2:
        sequences.append(current_seq)

    return sequences

true_sequences = find_sequences(labels, True)
false_sequences = find_sequences(labels, False)

print("=" * 60)
print("CONSECUTIVE TRUE SEQUENCES (2+ in a row)")
print("=" * 60)
print(f"\nTotal sequences: {len(true_sequences)}")
print(f"Total insertions needed: {sum(len(seq)-1 for seq in true_sequences)}")
print()

# Group by length
by_length = {}
for seq in true_sequences:
    length = len(seq)
    if length not in by_length:
        by_length[length] = []
    by_length[length].append(seq)

for length in sorted(by_length.keys(), reverse=True):
    seqs = by_length[length]
    print(f"Length {length}: {len(seqs)} sequence(s)")
    if length > 5:  # Show details for long sequences
        for seq in seqs:
            print(f"  Lines {seq[0]}-{seq[-1]}")

print("\n" + "=" * 60)
print("CONSECUTIVE FALSE SEQUENCES (2+ in a row) - VIOLATIONS!")
print("=" * 60)
print(f"\nTotal sequences: {len(false_sequences)}")
print()

for seq in false_sequences:
    print(f"Lines {seq[0]}-{seq[-1]} ({len(seq)} consecutive False)")

# Calculate work needed
print("\n" + "=" * 60)
print("WORK REQUIRED")
print("=" * 60)
print(f"\nTo fix True-True pairs:")
print(f"  - Need to insert {sum(len(seq)-1 for seq in true_sequences)} False examples")
print(f"\nTo fix False-False pairs:")
print(f"  - Need to insert {sum(len(seq)-1 for seq in false_sequences)} True examples")
print(f"\nTotal insertions needed: {sum(len(seq)-1 for seq in true_sequences) + sum(len(seq)-1 for seq in false_sequences)}")
