# Dataset Enhancement Specification

**Version:** 1.1 (Updated 2025-11-22 after Round 1 feedback)

## Mission

Enhance low-quality synthetic tool-calling examples by fixing issues identified in quality scores. Each example has been scored across 5 dimensions and marked as `label: false` (undesirable). Your task is to create improved versions that would score higher and be marked as `label: true` (desirable).

## Input Format

Each example in your batch contains:
```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "tool_call: ...\narguments: {...}"}
  ],
  "label": false,
  "quality_scores": {
    "sessionMemory_quality": 1-5,
    "toolContext_quality": 1-5,
    "goal_coherence": 1-5,
    "prompt_naturalness": 1-5,
    "response_realism": 1-5,
    "overall_quality": 1.0-5.0,
    "notes": "Detailed explanation of issues..."
  }
}
```

## Quality Scoring Rubric

### 1. sessionMemory_quality (1-5)
- **Score 1**: Empty string, empty array, or "Starting new session"
- **Score 2**: Generic placeholder ("Previous context", "User context")
- **Score 3**: Short but relevant (20-40 chars, basic context)
- **Score 4**: Good context (50-80 chars, specific prior actions/results)
- **Score 5**: Excellent context (80+ chars, detailed prior actions with specifics)

**Fix Strategy:**
- Never leave empty
- Reference specific prior tool calls: "Found 23 files via vaultLibrarian_searchContent"
- Include concrete details: file paths, counts, results
- Show workflow continuity: "Created folder structure, now organizing files"

### 2. toolContext_quality (1-5)
- **Score 1**: Generic placeholder ("Tool usage context", "Context for tool")
- **Score 2**: Just restates action ("Appending content", "Moving file")
- **Score 3**: Explains WHAT (30-50 chars, describes operation)
- **Score 4**: Explains WHY (50-80 chars, shows reasoning)
- **Score 5**: Explains workflow reasoning (80+ chars, alternatives considered)

**Fix Strategy:**
- Must be a STRING (not an object)
- Explain WHY tool was chosen: "Using append instead of replace to preserve existing entries"
- Show workflow reasoning: "After confirming path via search, now appending content"
- Consider alternatives mentioned

### 3. goal_coherence (1-5)
- **Score 1**: Completely generic ("Complete user request")
- **Score 2**: Redundant (primaryGoal and subgoal identical)
- **Score 3**: Some hierarchy (subgoal restates primaryGoal differently)
- **Score 4**: Clear hierarchy (subgoal shows current step)
- **Score 5**: Strategic decomposition (subgoal indicates position in workflow)

**Fix Strategy:**
- primaryGoal: User's overall objective
- subgoal: Current specific step/action
- Show decomposition: "Organize research notes" → "Create folder structure"

### 4. prompt_naturalness (1-5)
- **Score 1**: Command-style ("Get file content")
- **Score 2**: "Result:" continuation prompts (not real user input)
- **Score 3**: Basic conversational ("I need to...")
- **Score 4**: Natural with context ("Can you help me find...")
- **Score 5**: Highly natural with temporal/casual language

**Fix Strategy:**
- Remove "Result:" continuation prompts - make single-turn or natural follow-ups
- Add conversational markers: "Can you", "I need", "Please help"
- Include temporal references: "from last week", "for tomorrow"
- Use casual phrasing: "my notes", "that folder we created"

### 5. response_realism (1-5)
- **Score 1**: Malformed tool call structure
- **Score 2**: Basic tool call, minimal arguments
- **Score 3**: Complete tool call with all required fields
- **Score 4**: Well-structured with realistic parameter values
- **Score 5**: Excellent structure with context-appropriate parameters

**Fix Strategy:**
- **DO NOT add Result objects to assistant completions**
- Keep single-turn structure: user request → assistant tool call
- Ensure tool call is properly formatted
- Use realistic parameter values based on context
- If example has "Result:" in USER message, convert to natural request

## Common Issues & Fixes

### Issue 1: Empty sessionMemory
**Before:**
```json
"sessionMemory": ""
```
**After:**
```json
"sessionMemory": "User organized Projects folder. Created 3 subfolders for redesign workflow."
```

### Issue 2: toolContext as Object (Schema Violation)
**Before:**
```json
"toolContext": {"currentPath": "/home/user", "openFiles": []}
```
**After:**
```json
"toolContext": "Listing sessions to find Friday afternoon work after user requested historical context"
```

### Issue 3: Tool Call Structure
**Before:**
```
tool_call: vaultLibrarian_searchContent
arguments: {...}

Result: {"success": true}

Response text here
```
**After (Remove Result, keep single-turn):**
```
tool_call: vaultLibrarian_searchContent
arguments: {...}
```
**Note:** Assistant completion should ONLY contain tool_call and arguments. NO Result objects.

### Issue 4: "Result:" Continuation Prompts
**Before:**
```
USER: Result: {"success": true, "path": "notes.md"}

Move the file to Archive
```
**After:**
```
USER: Please move my completed project notes to the Archive folder
```

### Issue 5: Weak Goal Hierarchy
**Before:**
```json
"primaryGoal": "Create summary",
"subgoal": "Create summary note"
```
**After:**
```json
"primaryGoal": "Document research findings",
"subgoal": "Create summary note in Research folder"
```

## Enhancement Process

### Step 1: Read Quality Notes
Carefully read the `notes` field for each example. It contains specific issues identified.

### Step 2: Identify Issues
Check each dimension score:
- **< 3**: Needs significant improvement
- **3**: Acceptable but could be better
- **> 3**: Keep as-is

### Step 3: Fix Issues Systematically
Work through each low-scoring dimension:
1. Fix sessionMemory (if score < 3)
2. Fix toolContext (if score < 3)
3. Improve goals (if score < 3)
4. Fix prompt if "Result:" continuation (if score < 3)
5. Remove any Result objects from assistant completion (keep single-turn)

### Step 4: Validate Against Tool Schema
- Check tool name exists in tool_schemas.json
- Verify all required parameters are present
- Ensure context object is first parameter
- Confirm all 7 context fields are present
- Validate sessionMemory is never empty

### Step 5: Clean Structure
- Remove any Result objects from assistant completion
- Keep single-turn: one user message, one assistant message
- Assistant message should ONLY be: `tool_call: ...\narguments: {...}`

### Step 6: Mark as Desirable
After enhancement, set:
```json
"label": true
```

## Output Format

For each enhanced example, output:
```json
{
  "conversations": [
    {"role": "user", "content": "enhanced user prompt"},
    {"role": "assistant", "content": "enhanced assistant response with Result"}
  ],
  "label": true
}
```

**DO NOT** include:
- `quality_scores` field
- `_index` field
- Any metadata fields

## Validation Requirements

After enhancing your batch, validate using:
```bash
python tools/validate_syngen.py Datasets/quality_review/enhancement_batches/enhanced_batch_XXX.jsonl
```

**Must pass:**
- No validation errors
- All context objects present
- sessionMemory never empty
- toolContext is string
- All required context fields present

## Example Enhancement

### Before (Score: 2.8, Label: false)
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Load my work from last Friday afternoon."
    },
    {
      "role": "assistant",
      "content": "tool_call: memoryManager_listSessions\narguments: {\"context\": {\"sessionId\": \"session_1731398400000_a1b2c3d4e\", \"workspaceId\": \"ws_1731398400000_f5g6h7i8j\", \"sessionDescription\": \"Finding last Friday's session\", \"sessionMemory\": [], \"toolContext\": {\"currentPath\": \"/home/user/Projects\", \"openFiles\": [], \"recentCommands\": []}, \"primaryGoal\": \"Load Friday afternoon work\", \"subgoal\": \"Find Friday sessions\"}, \"limit\": 20, \"order\": \"desc\"}"
    }
  ],
  "label": false,
  "quality_scores": {
    "sessionMemory_quality": 1,
    "toolContext_quality": 2,
    "goal_coherence": 4,
    "prompt_naturalness": 5,
    "response_realism": 2,
    "overall_quality": 2.8,
    "notes": "sessionMemory is empty array [] earning automatic score=1. toolContext is incorrectly formatted as object instead of string. No Result object shown."
  }
}
```

### After (Enhanced, Label: true)
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Load my work from last Friday afternoon."
    },
    {
      "role": "assistant",
      "content": "tool_call: memoryManager_listSessions\narguments: {\"context\": {\"sessionId\": \"session_1731398400000_a1b2c3d4e\", \"workspaceId\": \"ws_1731398400000_f5g6h7i8j\", \"sessionDescription\": \"Finding last Friday's session\", \"sessionMemory\": \"User previously worked on project redesign last week. Workspace contains multiple active sessions from Monday-Friday. Need to retrieve Friday afternoon work to continue progress.\", \"toolContext\": \"Listing recent sessions in reverse chronological order to locate Friday afternoon work session for resuming project tasks\", \"primaryGoal\": \"Resume Friday afternoon work session\", \"subgoal\": \"Identify and list sessions from last Friday\"}, \"limit\": 20, \"order\": \"desc\"}"
    }
  ],
  "label": true
}
```

**Changes Made:**
1. ✅ Fixed sessionMemory: Added rich context about previous work and workspace state (139 chars with specific details)
2. ✅ Fixed toolContext: Changed from object to string, explained WHY listing sessions and workflow reasoning (117 chars)
3. ✅ Improved goal_coherence: Made subgoal more specific about "from last Friday"
4. ✅ Kept good prompt_naturalness: User prompt was already natural (score 5)
5. ✅ Removed any Result objects: Keep single-turn structure (tool call only)

## Quality Control

Before submitting your enhanced batch:

1. **Every example must have `label: true`**
2. **Every sessionMemory must have content (min 50 chars with specifics)**
3. **Every toolContext must be a string (not object)**
4. **NO Result objects in assistant completions (single-turn only)**
5. **Assistant completion format: `tool_call: toolName\narguments: {...}`**
6. **Run validator and fix any errors**

## Task Assignment

You have been assigned: **batch_XXX.jsonl**

1. Read the batch file from: `Datasets/quality_review/enhancement_batches/batch_XXX.jsonl`
2. Enhance each example following this specification
3. Write enhanced examples to: `Datasets/quality_review/enhancement_batches/enhanced_batch_XXX.jsonl`
4. Validate the output file
5. Report summary:
   - Examples enhanced: XX/50
   - Validation status: PASS/FAIL
   - Common issues fixed: [list]

## sessionMemory Best Practices

**Minimum:** 50 characters
**Target:** 80-120 characters
**Maximum:** 150 characters (avoid excessive length)

**Content Guidelines:**
- Reference specific prior actions with concrete details
- Include numbers, file names, folder paths when relevant
- Show workflow continuity ("Previously X, now Y")
- Avoid generic placeholders ("User context", "Previous session")

**Good Examples:**
- "User organized Projects folder. Created 3 subfolders for redesign workflow. Previously reviewed 47 files for migration."
- "Completed authentication module. UI components 60% done. Backend API endpoints 80% complete."

## toolContext Templates by Tool Category

### Search/List Operations
**Template:** "Using [tool] to locate [target] before [next action]. Searching enables [workflow benefit]."

**Example:** "Using vaultLibrarian_searchContent to locate research notes before organizing files. Search results will guide folder structure decisions."

### Create Operations
**Template:** "Creating [resource] to establish [purpose]. New [item] enables [workflow capability]."

**Example:** "Creating Project folder to establish workspace organization. New structure enables systematic file management."

### Update/Modify Operations
**Template:** "Modifying [target] to [purpose]. Using [method] instead of [alternative] to preserve [what]."

**Example:** "Appending to README to add documentation. Using append instead of replace to preserve existing content and history."

### Load/Retrieve Operations
**Template:** "Loading [resource] to continue [workflow] from [prior state]. Restoration enables [benefit]."

**Example:** "Loading Friday session to continue project work from last session. Restoration provides context and continuity."

## Common Pitfalls to Avoid

### ❌ DON'T: Copy quality_scores into enhanced examples
- Remove ALL metadata: quality_scores, _index, _line_number, relabeled
- Output should ONLY have: conversations and label

### ❌ DON'T: Add Result objects to assistant completions
- Assistant message = tool_call + arguments ONLY
- NO Result objects, NO response text, NO preamble

### ❌ DON'T: Keep empty sessionMemory
- Every example must have sessionMemory ≥ 50 chars
- "Starting new session" is NOT acceptable (too generic)

### ❌ DON'T: Use objects for toolContext
- toolContext MUST be STRING type
- {"currentPath": "..."} → "Listing sessions to locate work context"

### ❌ DON'T: Keep "Result:" in user messages
- Convert to natural requests
- "Result: {...}" → "Please help me with..."

### ✅ DO: Validate your batch
- Run: `python tools/validate_syngen.py [your_file].jsonl`
- Expect "Result:" warnings (this is correct - see below)
- Fix any ERROR messages (warnings are OK)

## Expected Validation Warnings

When validating enhanced batches, you will see warnings about **"Tool calls present but no 'Result:' markers found"**. This is **expected and correct** behavior.

**Why this happens:**
- Single-turn format intentionally omits Result objects
- Validator warns because it expects multi-turn conversations
- These warnings do NOT indicate errors

**Action:** Ignore these warnings - they confirm proper single-turn structure.

## Handling Unknown Tool Parameters

Some original examples may include parameters not in tool_schemas.json (e.g., `folder`, `provider`, `fileTypes`).

**What to do:**
- **Keep parameters if they're contextually relevant** (even if not in schema)
- **Remove parameters if they're clearly wrong** (typos, inappropriate for tool)
- **Validation warnings are OK** - they don't fail validation

The validator will warn about unexpected parameters, but this doesn't affect training quality.

## Pre-Submission Checklist

Before reporting your batch as complete, verify:

- [ ] All 50 examples have `label: true`
- [ ] All sessionMemory ≥ 50 chars with specific details
- [ ] All toolContext are STRING type (no objects)
- [ ] NO Result objects in assistant completions
- [ ] NO metadata fields (quality_scores, _index, etc.)
- [ ] Validation passed with 0 errors (warnings OK)
- [ ] Assistant format: `tool_call: toolName\narguments: {...}` only
- [ ] User prompts are natural (no "Result:" JSON)
- [ ] All 7 context fields present in every example

## Reference Files

- Tool schemas: `/home/user/Toolset-Training/tools/tool_schemas.json`
- Validator: `/home/user/Toolset-Training/tools/validate_syngen.py`
- Quality report: `/home/user/Toolset-Training/Datasets/quality_review/quality_triage_report.md`

Good luck! Create high-quality enhanced examples that will improve our training dataset.
