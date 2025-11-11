import json

file_path = "/home/user/Toolset-Training/syngen_toolset_v1.0.0_claude.jsonl"

with open(file_path, 'r') as f:
    lines = f.readlines()

consecutive_pairs = []
for i in range(len(lines)-1):
    try:
        entry1 = json.loads(lines[i])
        entry2 = json.loads(lines[i+1])
        if entry1.get('label') == True and entry2.get('label') == True:
            consecutive_pairs.append((i+1, i+2))  # Line numbers (1-indexed)
    except:
        pass

print(f"Total lines in file: {len(lines)}")
print(f"Found {len(consecutive_pairs)} consecutive True-True pairs")
print(f"\nFirst 20 pairs: {consecutive_pairs[:20]}")
print(f"\nLast 10 pairs: {consecutive_pairs[-10:]}")
