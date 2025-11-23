# Response Pattern Training Dataset

## Purpose

Teach the model WHEN and HOW to respond after receiving tool results, using three distinct completion patterns.

## Patterns

### 1. Text-Only Response
**When to use:**
- Task is complete (no more actions needed)
- Need user clarification/confirmation
- Explaining results or findings
- Error occurred and needs explanation
- Providing summary or status update

**Format:**
```
User: "Delete old files\n\n[Previous: searchContent found 3 files]"
Assistant: "I found 3 old project files from 2022-2023. Would you like me to delete all of them, or should I review them first?"
```

### 2. Tool-Only Response
**When to use:**
- Middle of clear workflow (next step obvious)
- No explanation needed (action speaks for itself)
- Batch operations in sequence
- Automated workflow continuation
- User already understands the process

**Format:**
```
User: "Delete old files\n\n[Previous: searchContent found Projects/2022/old.md]"
Assistant: "tool_call: vaultManager_deleteNote\narguments: {\"context\": {...}, \"path\": \"Projects/2022/old.md\"}"
```

### 3. Tool+Text Response
**When to use:**
- Complex action needs context/explanation
- Multi-step workflow with checkpoints
- Teaching/showing user what's happening
- Partial completion with next step
- Action might be surprising/unexpected

**Format:**
```
User: "Delete old files\n\n[Previous: searchContent found 3 files]"
Assistant: "I found 3 files. Let me delete the first one (Projects/2022/old.md) and then continue with the others:\n\ntool_call: vaultManager_deleteNote\narguments: {...}"
```

## Dataset Structure

Generate 50 examples per pattern = 150 total examples

**Distribution:**
- Text-only: 50 examples (various scenarios)
- Tool-only: 50 examples (workflow continuations)
- Tool+text: 50 examples (narrated actions)

**Scenarios to cover:**
- Search → Delete workflows
- Search → Read → Update workflows
- List → Batch operation workflows
- Error recovery scenarios
- Multi-file operations
- Workspace loading → operations
- Memory search → context usage
- Complex reasoning → execution

## Quality Requirements

**All examples:**
- Single-turn format
- User message simulates post-tool-result scenario with `[Previous: ...]` notation
- Assistant response demonstrates ONE of the three patterns
- All context fields complete (7 required)
- sessionMemory references the previous action result
- Realistic and diverse scenarios

**Labels:**
- Use `"pattern": "text_only"`, `"pattern": "tool_only"`, or `"pattern": "tool_text"` instead of behavior field
- No true/false labels (this is SFT-style, not KTO)

## Validation

Run standard validator:
```bash
python tools/validate_syngen.py Datasets/response_patterns/patterns_v1.0.jsonl
```
