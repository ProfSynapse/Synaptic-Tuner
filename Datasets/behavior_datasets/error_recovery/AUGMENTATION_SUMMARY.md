# Error Recovery Dataset Augmentation Summary

**Date**: 2025-11-22
**File**: `Datasets/behavior_datasets/error_recovery/pairs_v1.0.jsonl`
**Task**: Add 30 response pattern examples demonstrating error recovery behavior

## Results

### Dataset Growth
- **Original size**: 100 examples (50 positive/negative pairs)
- **New examples added**: 30 examples
- **Final size**: 130 examples
- **New total**: 65 positive examples demonstrating error recovery patterns

### New Examples Breakdown (Examples 101-130)

#### Pattern 1: Text-Only Responses (10 examples: 101-110)
**Purpose**: Explain adaptation strategy to user after receiving error

**Characteristics**:
- Uses `[Previous: ...]` notation to show what failed
- Explains what went wrong and why
- Describes next steps without executing them
- No tool calls in the response
- Pure strategic explanation

**Example**:
```
User: Find the design mockups folder
Assistant: [Previous: searchDirectory for 'design mockups' returned 0 results]

The direct search didn't find any folders with that exact name. Let me try a
broader approach - I'll search for just 'design' to see all design-related
folders in the vault, then we can identify which one contains the mockups.
```

#### Pattern 2: Tool-Only Responses (10 examples: 111-120)
**Purpose**: Immediately try alternative approach after failure

**Characteristics**:
- Uses `[Previous: ...]` notation to show what failed
- Immediately makes alternative tool call
- Complete context objects with all 7 fields
- Rich sessionMemory explaining the failure
- Minimal or no explanatory text before tool call
- Shows action-oriented error recovery

**Example**:
```
User: Find the user research notes
Assistant: [Previous: searchContent for 'user research notes' returned 0 results]

tool_call: vaultLibrarian_searchContent
arguments: {
  "context": {
    "sessionId": "session_1732350000000_a1b2c3d4e5",
    "workspaceId": "ws_1732350000000_f6g7h8i9j0",
    "sessionDescription": "Broadening search after exact phrase failed",
    "sessionMemory": "Initial search for 'user research notes' found nothing...",
    "toolContext": "Using broader search query after exact match failed...",
    "primaryGoal": "Locate user research notes",
    "subgoal": "Search with simplified query"
  },
  "query": "user research",
  "limit": 50,
  "includeContent": true
}
```

#### Pattern 3: Tool+Text Responses (10 examples: 121-130)
**Purpose**: Acknowledge error AND execute new approach simultaneously

**Characteristics**:
- Uses `[Previous: ...]` notation to show what failed
- Explains the problem and adaptation strategy
- Then executes the alternative approach
- Complete context objects with all 7 fields
- Combines user communication with action
- Shows both reasoning and execution

**Example**:
```
User: Find the integration test documentation
Assistant: [Previous: searchContent for 'integration test documentation' in
'docs/' returned 0 results]

The documentation isn't in the docs folder with that exact phrasing. Let me try
a different approach - I'll search the entire vault for just 'integration test'
to see if the documentation is located elsewhere or uses different terminology
like 'integration testing guide' or 'API integration docs'.

tool_call: vaultLibrarian_searchContent
arguments: {
  "context": {
    "sessionId": "session_1732360000000_k1l2m3n4o5",
    "workspaceId": "ws_1732360000000_p6q7r8s9t0",
    ...
  },
  "query": "integration test",
  "limit": 50,
  "includeContent": true
}
```

## Error Recovery Scenarios Covered

The 30 new examples demonstrate recovery from:

1. **Search failures**: No results, wrong scope, too specific queries
2. **File operation failures**: File not found, path doesn't exist
3. **Replace/update failures**: Content mismatch, wrong format
4. **Folder operation failures**: Missing destinations, wrong paths
5. **Session/workspace failures**: Invalid IDs, not found
6. **Partial successes**: Incomplete replacements, missing instances
7. **Assumption failures**: Wrong paths, incorrect formats

## Adaptation Strategies Demonstrated

- Broadening search queries (specific → general)
- Changing search scope (folder → vault-wide)
- Switching tools (searchContent → searchDirectory)
- Reading before replacing (to verify format)
- Creating missing destinations (folders before moving)
- Listing to explore structure (when paths unknown)
- Trying alternative formats (case-insensitive, acronyms)
- Increasing limits (when partial results suggest more exist)

## Quality Validation

### All 30 New Examples Pass:
✓ Use `[Previous: ...]` notation to simulate post-error scenario
✓ Labeled as `true` (demonstrating good error recovery behavior)
✓ Tagged with `behavior: "error_recovery"`

### Tool-Based Examples (20 examples: 111-130) Have:
✓ Complete context objects with all 7 required fields:
  - sessionId (correct timestamp format)
  - workspaceId (correct format)
  - sessionDescription (clear description)
  - sessionMemory (never empty, explains failure)
  - toolContext (explains why this tool/approach)
  - primaryGoal (user's main objective)
  - subgoal (what this call achieves)

### Text-Only Examples (10 examples: 101-110):
✓ Provide strategic explanations without tool calls
✓ Explain what failed and why
✓ Describe planned next steps

## Usage

These examples train models to:

1. **Recognize failures** from error messages and results
2. **Diagnose problems** (why the approach failed)
3. **Adapt strategy** (broader search, different tool, verify first)
4. **Maintain context** (carry forward session understanding)
5. **Communicate effectively** (explain OR act OR both)

## File Location

**Dataset**: `/home/user/Toolset-Training/Datasets/behavior_datasets/error_recovery/pairs_v1.0.jsonl`

**Format**: JSONL (one JSON object per line)

**Structure**:
```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "[Previous: ...]\n\n..."}
  ],
  "label": true,
  "behavior": "error_recovery"
}
```

## Next Steps

The augmented dataset can now be used for:
- Training models on error recovery behavior
- Fine-tuning with KTO (preference learning)
- Evaluating model error recovery capabilities
- Generating additional synthetic examples

---

**Total examples**: 130
**Error recovery demonstrations**: 65 positive examples
**Response patterns**: Text-Only, Tool-Only, Tool+Text
**Ready for training**: ✓
