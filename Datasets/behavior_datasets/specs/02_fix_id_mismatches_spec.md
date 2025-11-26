# Spec: Fix ID Mismatches in text_only_pairs

## Overview
Fix 75 examples in text_only_pairs where tool call IDs don't match system message IDs.

## Affected Dataset
- `response_patterns/text_only_pairs_v1.1.jsonl` (150 examples, 75 with mismatches)

## Problem Description
The `label: false` examples have system messages with one set of IDs, but tool calls use different IDs:

**Current (broken):**
```json
{
  "conversations": [
    {"role": "system", "content": "...sessionId: \"session_1764094145839_sz0zu5yf0\"\n...workspaceId: \"ws_1764094145839_baewnvzow\"..."},
    {"role": "user", "content": "Result: {...}"},
    {"role": "assistant", "content": null, "tool_calls": [{
      "function": {"arguments": "{\"context\": {\"sessionId\": \"session_1732543200000_a1b2c3d4e\", \"workspaceId\": \"ws_1732543200000_f5g6h7i8j\", ...}}"}
    }]}
  ],
  "label": false
}
```

System says: `session_1764094145839_sz0zu5yf0`, `ws_1764094145839_baewnvzow`
Tool uses: `session_1732543200000_a1b2c3d4e`, `ws_1732543200000_f5g6h7i8j`

## Required Fix
Update the tool call arguments to use the IDs from the system message:

**Fixed:**
```json
{
  "conversations": [
    {"role": "system", "content": "...sessionId: \"session_1764094145839_sz0zu5yf0\"\n...workspaceId: \"ws_1764094145839_baewnvzow\"..."},
    {"role": "user", "content": "Result: {...}"},
    {"role": "assistant", "content": null, "tool_calls": [{
      "function": {"arguments": "{\"context\": {\"sessionId\": \"session_1764094145839_sz0zu5yf0\", \"workspaceId\": \"ws_1764094145839_baewnvzow\", ...}}"}
    }]}
  ],
  "label": false
}
```

## Implementation Steps

### 1. Parse System Message
Extract IDs from system message using regex:
- sessionId: `sessionId:\s*["']([^"']+)["']`
- workspaceId: `workspaceId:\s*["']([^"']+)["']`

### 2. Parse and Update Tool Call Arguments
For each tool_call in the assistant message:
1. Parse `function.arguments` as JSON
2. Navigate to `context.sessionId` and `context.workspaceId`
3. Replace with values from system message
4. Re-serialize arguments back to JSON string

### 3. Only Fix label=false Examples
The `label=true` examples already have matching IDs (text-only responses have no tool calls).
Only the `label=false` examples have tool calls with mismatched IDs.

## Important Notes

### Why IDs Must Match
- The model learns to use IDs from the system context
- Mismatched IDs teach the model to ignore context and make up IDs
- This defeats the purpose of context injection training

### Preserve Other Context Fields
When updating IDs, preserve all other context fields:
- sessionDescription
- sessionMemory
- toolContext
- primaryGoal
- subgoal

Only change sessionId and workspaceId.

## Validation
After processing, run:
```bash
python tools/validate_syngen.py Datasets/behavior_datasets/response_patterns/text_only_pairs_v1.2.jsonl
```

Expected: 0 ID mismatch errors.

## Output Format
- Create new file: `text_only_pairs_v1.2.jsonl`
- Preserve v1.1 as backup
- Maintain all existing fields (label, pattern)
