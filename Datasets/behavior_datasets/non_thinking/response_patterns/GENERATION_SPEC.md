# Response Pattern Dataset Generation Specification

**Version:** 1.0
**Created:** 2025-11-25
**Purpose:** Teach models when to respond with text vs continue with tool calls after receiving tool results

---

## Overview

This dataset teaches **response completion patterns** - helping models decide:
1. When to STOP using tools and respond with text
2. When to CONTINUE with tool-only responses (workflow middle)
3. When to COMBINE tool calls with explanatory text

**Critical Gap:** Current datasets (behavior_merged_kto_v1.1.jsonl, syngen_tools_sft_11.24.25_cleaned.jsonl) contain 100% tool-only responses. Models don't learn when to summarize, ask clarification, or explain results.

---

## Three Response Patterns

### Pattern 1: Text-Only Response
**When to use:**
- Task complete, results ready to present
- Need user clarification/confirmation
- Explaining results from tool execution
- Error needs explanation
- Status update or summary

**Example:**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Result: {\"success\": true, \"results\": [{\"path\": \"Projects/2022/old-prototype.md\", \"modified\": \"2022-03-15\"}, {\"path\": \"Projects/2023/abandoned.md\", \"modified\": \"2023-07-22\"}], \"count\": 2}"
    },
    {
      "role": "assistant",
      "content": "I found 2 old project files:\n\n1. Projects/2022/old-prototype.md (last modified March 2022)\n2. Projects/2023/abandoned.md (last modified July 2023)\n\nWould you like me to delete both of these files?"
    }
  ],
  "label": true,
  "pattern": "text_only"
}
```

**Negative counterpart (label: false):**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Result: {\"success\": true, \"results\": [{\"path\": \"Projects/2022/old-prototype.md\", \"modified\": \"2022-03-15\"}, {\"path\": \"Projects/2023/abandoned.md\", \"modified\": \"2023-07-22\"}], \"count\": 2}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "vaultManager_deleteNote",
          "arguments": "{\"context\": {...}, \"path\": \"Projects/2022/old-prototype.md\"}"
        }
      }]
    }
  ],
  "label": false,
  "pattern": "text_only"
}
```

### Pattern 2: Tool-Only Response
**When to use:**
- Middle of clear workflow (next step obvious)
- Single target identified, action is clear
- Batch operation in progress
- Automated workflow continuation

**Example:**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Result: {\"success\": true, \"results\": [{\"path\": \"Projects/cleanup-target.md\"}], \"count\": 1}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_def456",
        "type": "function",
        "function": {
          "name": "vaultManager_deleteNote",
          "arguments": "{\"context\": {\"sessionId\": \"session_1732300800000_a1b2c3d4e\", \"workspaceId\": \"ws_1732300800000_f5g6h7i8j\", \"sessionDescription\": \"Deleting identified cleanup target\", \"sessionMemory\": \"Search found single cleanup target file. User requested deletion workflow - proceeding with removal.\", \"toolContext\": \"Deleting the identified file as part of cleanup workflow. Single clear target requires no confirmation.\", \"primaryGoal\": \"Clean up old project files\", \"subgoal\": \"Delete identified target file\"}, \"path\": \"Projects/cleanup-target.md\"}"
        }
      }]
    }
  ],
  "label": true,
  "pattern": "tool_only"
}
```

**Negative counterpart (label: false):**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Result: {\"success\": true, \"results\": [{\"path\": \"Projects/cleanup-target.md\"}], \"count\": 1}"
    },
    {
      "role": "assistant",
      "content": "I found the cleanup target at Projects/cleanup-target.md. Let me explain what I'll do next. Since we identified this file in our search and it matches the cleanup criteria, I'll now proceed to delete it. This is a safe operation because we verified the target first."
    }
  ],
  "label": false,
  "pattern": "tool_only"
}
```

### Pattern 3: Tool+Text Response
**When to use:**
- Complex action needs context/explanation
- Multi-step workflow with checkpoints
- Teaching user what's happening
- Action might be surprising
- Starting batch operation

**Example:**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Result: {\"success\": true, \"results\": [{\"path\": \"Config/dev.yaml\"}, {\"path\": \"Config/prod.yaml\"}, {\"path\": \"Settings/app.config\"}], \"count\": 3}"
    },
    {
      "role": "assistant",
      "content": "I found 3 config files. Let me read the production config first since that's most likely where the API endpoint is defined:\n\n",
      "tool_calls": [{
        "id": "call_ghi789",
        "type": "function",
        "function": {
          "name": "contentManager_readContent",
          "arguments": "{\"context\": {...}, \"filePath\": \"Config/prod.yaml\", \"includeLineNumbers\": true}"
        }
      }]
    }
  ],
  "label": true,
  "pattern": "tool_text"
}
```

**Negative counterpart (label: false):**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Result: {\"success\": true, \"results\": [{\"path\": \"Config/dev.yaml\"}, {\"path\": \"Config/prod.yaml\"}, {\"path\": \"Settings/app.config\"}], \"count\": 3}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_ghi789",
        "type": "function",
        "function": {
          "name": "contentManager_readContent",
          "arguments": "{\"context\": {...}, \"filePath\": \"Config/prod.yaml\", \"includeLineNumbers\": true}"
        }
      }]
    }
  ],
  "label": false,
  "pattern": "tool_text"
}
```

---

## Dataset Structure

### Target Size
**150-200 examples per pattern** (matching other behavior datasets)
- 75-100 pairs × 2 (positive + negative) = 150-200 examples
- Interleaved True/False pattern required for KTO

### File Structure
```
behavior_datasets/response_patterns/
├── GENERATION_SPEC.md (this file)
├── text_only_pairs_v1.0.jsonl (150-200 examples)
├── tool_only_pairs_v1.0.jsonl (150-200 examples)
├── tool_text_pairs_v1.0.jsonl (150-200 examples)
└── README.md (summary)
```

---

## User Message Formats

### Format 1: Result Object (PREFERRED)
Simulates receiving a tool result and deciding how to respond:

```json
{
  "role": "user",
  "content": "Result: {\"success\": true, \"results\": [...], \"count\": N}"
}
```

**Use this for:**
- Showing search results (searchContent, searchDirectory, searchMemory)
- List results (listDirectory, listSessions, listWorkspaces)
- Operation confirmations (createFolder, deleteNote, moveNote)
- Read results (readContent)
- Batch operation results

### Format 2: Natural Request with Context
User request that implies a tool was just executed:

```json
{
  "role": "user",
  "content": "Delete old project files"
}
```

**Use this for:**
- Initial requests where tool hasn't been executed yet
- Pattern 2 (tool_only) where we want to show immediate action
- Scenarios where context is in the session rather than prior result

### Format 3: Continuation Request
User asking for next step after seeing results:

```json
{
  "role": "user",
  "content": "Yes, delete them"
}
```

**Use sparingly** - only for multi-turn scenarios

---

## Quality Requirements

### All Examples Must:
1. ✅ **Pass syngen validator** - Complete context objects, schema compliance
2. ✅ **Include complete context** - All 7 fields in tool calls
3. ✅ **sessionMemory never empty** - Must reference prior state/actions
4. ✅ **toolContext is STRING** - Not an object (common mistake)
5. ✅ **Realistic scenarios** - Natural tool results, plausible workflows
6. ✅ **Single-turn format** - One user message → one assistant response
7. ✅ **Label accuracy** - True = good pattern, False = anti-pattern
8. ✅ **Interleaved** - Strict True/False/True/False pattern

### Pattern-Specific Quality

#### Pattern 1 (Text-Only) - POSITIVE Examples:
- ✅ Clear, helpful text response
- ✅ Summarizes results concretely (not vaguely)
- ✅ Asks specific clarifying questions
- ✅ Explains errors constructively
- ✅ NO tool calls in response
- ✅ Appropriate length (not too brief, not too verbose)

#### Pattern 1 (Text-Only) - NEGATIVE Examples:
- ❌ Makes unnecessary tool call when should ask/summarize
- ❌ Proceeds without confirmation when risky
- ❌ Continues workflow when should pause
- ❌ Doesn't explain error/result when should

#### Pattern 2 (Tool-Only) - POSITIVE Examples:
- ✅ Clean tool call, no unnecessary text
- ✅ Obvious next step in workflow
- ✅ Context shows workflow continuity
- ✅ Single clear target identified
- ✅ Appropriate automation (user expects it)

#### Pattern 2 (Tool-Only) - NEGATIVE Examples:
- ❌ Stops to explain when action is obvious
- ❌ Over-explains simple next step
- ❌ Unnecessary status update mid-workflow
- ❌ Asks confirmation on clear next step

#### Pattern 3 (Tool+Text) - POSITIVE Examples:
- ✅ Explanation adds value (not redundant)
- ✅ Clarifies WHY this tool/approach
- ✅ Previews what will happen
- ✅ Helps user understand complex operation
- ✅ Text + tool call both present

#### Pattern 3 (Tool+Text) - NEGATIVE Examples:
- ❌ Text doesn't add value (just restates)
- ❌ Over-explains trivial operation
- ❌ Explanation contradicts tool call
- ❌ Could be tool-only (no explanation needed)

---

## Scenario Coverage

### Tool Distribution (Target all 47+ tools)

**vaultManager (9 tools):**
- createFolder, createNote, deleteNote, deleteFolder
- moveNote, moveFolder, renameNote, renameFolder
- listDirectory

**contentManager (8 tools):**
- readContent, appendContent, replaceContent, replaceByLine
- prependContent, insertContent, deleteContent, batchContent

**memoryManager (11 tools):**
- createSession, loadSession, updateSession, deleteSession, listSessions
- createWorkspace, loadWorkspace, updateWorkspace, deleteWorkspace, listWorkspaces
- searchMemory

**vaultLibrarian (6 tools):**
- searchContent, searchDirectory, searchByTag
- batch, batchSearch, getFileMetadata

**agentManager (7 tools):**
- executePrompt, batchExecutePrompt, streamPrompt
- createAgent, manageAgent, getAgentStatus, cancelAgent

### Result Types to Cover

**Search results:**
- 0 results (no matches)
- 1 result (single match)
- 2-5 results (few matches)
- 10+ results (many matches)
- 50+ results (overwhelming)

**Operation results:**
- Success (simple)
- Success with data (complex)
- Partial success (some failed)
- Failure with error
- Empty/null results

**Content results:**
- Short content (< 100 chars)
- Medium content (100-500 chars)
- Long content (500+ chars)
- Structured data (JSON, YAML)
- Binary/metadata only

### Domain Coverage

**Scenario domains:**
- Research paper organization
- Project management
- Meeting notes / journaling
- Budget tracking
- Code documentation
- Learning / study notes
- Workspace workflows
- Session/memory management
- Configuration management
- Archive/cleanup operations

**User intent types:**
- Information seeking (search, list, explore)
- Organization (create structure, move, categorize)
- Cleanup (delete, archive, prune)
- Creation (new content, folders, workspaces)
- Updates (modify, append, replace)
- Workflows (multi-step, batch operations)

---

## Example Scenarios by Pattern

### Pattern 1: Text-Only Scenarios (75-100 pairs)

#### Clarification Needed (~25 pairs)
- Search returned multiple files, which to delete?
- Multiple configs found, which to update?
- Ambiguous paths, need user selection
- Risk assessment, confirm before proceeding

#### Task Complete (~25 pairs)
- Search complete, showing results
- Operation successful, confirmation
- Workspace loaded, ready for instruction
- Batch operation complete, summary

#### Error Explanation (~15 pairs)
- File not found, explain and suggest
- Replace failed (string not found), why?
- Permission denied, what to do?
- Invalid parameters, clarify issue

#### Result Summary (~10 pairs)
- Complex search results, summarize patterns
- Large dataset, highlight key findings
- Memory search results, relevant context
- Workspace metadata, structure overview

### Pattern 2: Tool-Only Scenarios (75-100 pairs)

#### Workflow Continuation (~35 pairs)
- Search found 1 file → delete it
- Read successful → append content
- List returned clear targets → batch delete
- Create folder successful → create next folder

#### Automated Sequences (~25 pairs)
- Workspace workflow in progress
- Batch operation executing
- Retry after error (auto-adapt)
- Multi-file operation sequence

#### Obvious Next Steps (~15 pairs)
- Found exact match → read it
- Located target → move it
- Identified file → update it
- Verified path → create there

### Pattern 3: Tool+Text Scenarios (75-100 pairs)

#### Complex Decision (~25 pairs)
- Multiple approaches possible, explaining choice
- Uncertain which file to target, trying most likely
- Starting multi-step operation, previewing plan
- Batch operation on many files, explaining scope

#### Teaching Moments (~20 pairs)
- Showing user how tool works
- Explaining workflow logic
- Demonstrating best practice
- Building user understanding

#### Surprising Actions (~15 pairs)
- Action might be unexpected, explaining rationale
- Using non-obvious tool choice, justifying
- Skipping expected step, explaining why
- Trying alternative approach, reasoning

#### Checkpoint Updates (~15 pairs)
- Midway through long operation, status
- Starting new phase, context
- Found partial results, next approach
- Adapting strategy, explaining change

---

## Context Object Guidelines

### sessionMemory Requirements
**For Pattern 1 (text-only):**
- Reference the tool result just received
- Explain what was found/done
- Note any ambiguities or issues
- 80-150 chars target

**Example:**
```
"sessionMemory": "Search found 3 old project files from 2022-2023. User requested deletion but didn't specify which files. Need confirmation before proceeding with destructive operation."
```

**For Pattern 2 (tool-only):**
- Show workflow continuity
- Reference clear next step
- Note why action is obvious
- 60-120 chars target

**Example:**
```
"sessionMemory": "Search identified single cleanup target. User requested automated cleanup workflow. Proceeding with deletion as next step."
```

**For Pattern 3 (tool+text):**
- Explain complexity or reasoning
- Reference multiple options/considerations
- Show decision-making process
- 80-150 chars target

**Example:**
```
"sessionMemory": "Found 3 config files. Need to identify which contains API endpoint before updating. Starting with prod config as most likely location. Will try others if needed."
```

### toolContext Requirements
**MUST be STRING** (not object!)

**For all patterns:**
- Explain WHY this tool chosen
- Show workflow position
- Reference prior actions if relevant
- 60-120 chars target

**Example (Pattern 1):**
```
"toolContext": "N/A - responding with text to ask user which files to delete from search results"
```

**Example (Pattern 2):**
```
"toolContext": "Deleting identified target as next step in automated cleanup workflow. Single file, no confirmation needed."
```

**Example (Pattern 3):**
```
"toolContext": "Reading prod config first to locate API endpoint before replacement. Explaining choice since multiple configs exist."
```

### Goals Requirements
**primaryGoal:** Overall user objective (consistent across sequence)
**subgoal:** Current step or decision point

**Example (Pattern 1 - clarification):**
```
"primaryGoal": "Delete old project files safely",
"subgoal": "Confirm which of the 3 found files to delete"
```

**Example (Pattern 2 - workflow):**
```
"primaryGoal": "Clean up old project files",
"subgoal": "Delete identified target file"
```

**Example (Pattern 3 - complex):**
```
"primaryGoal": "Update API endpoint in production config",
"subgoal": "Identify which config file contains the endpoint"
```

---

## Validation Checklist

### Before Submission
Each pair must pass:

1. ✅ **Schema validation**
   - Run: `python tools/validate_syngen.py <file.jsonl>`
   - All examples must pass

2. ✅ **Pair structure**
   - Identical user messages in both examples
   - Positive labeled `true`, negative labeled `false`
   - Both include `"pattern": "pattern_name"` field
   - Clear behavioral contrast

3. ✅ **Context completeness**
   - All 7 context fields present
   - sessionMemory never empty
   - toolContext is STRING (not object)
   - Realistic IDs (session_NNNNNNNNNNNNN_XXXXXXXXX)

4. ✅ **Interleaving**
   - Perfect True/False/True/False pattern
   - No consecutive True or False labels
   - Equal counts (if odd, one extra True)

5. ✅ **Quality sampling**
   - Review 10% of examples manually
   - Verify pattern contrast is clear
   - Ensure realistic scenarios
   - Check text quality (grammar, clarity)

### Common Mistakes to Avoid

❌ **toolContext as object:**
```json
"toolContext": {"action": "delete", "target": "file.md"}  // WRONG
```
✅ **toolContext as string:**
```json
"toolContext": "Deleting file.md as identified cleanup target"  // CORRECT
```

❌ **Empty sessionMemory:**
```json
"sessionMemory": ""  // WRONG - NEVER EMPTY
```
✅ **Populated sessionMemory:**
```json
"sessionMemory": "Search completed, found 3 files matching criteria"  // CORRECT
```

❌ **Generic context:**
```json
"sessionMemory": "Working on task",
"toolContext": "Using tool"
```
✅ **Specific context:**
```json
"sessionMemory": "Found 3 old project files from 2022-2023, user requested deletion but targets unclear",
"toolContext": "Asking user to confirm which files to delete before proceeding with destructive operation"
```

❌ **Unrealistic negative examples:**
```json
{"content": "asdfghjkl"}  // Gibberish
```
✅ **Realistic negative examples:**
```json
{"tool_calls": [...]}  // Proceeds with action when should ask - realistic mistake
```

---

## Generation Strategy

### Parallel Agent Approach

1. **Spawn 3 agents in parallel:**
   - Agent 1: Pattern 1 (text_only) → 75-100 pairs
   - Agent 2: Pattern 2 (tool_only) → 75-100 pairs
   - Agent 3: Pattern 3 (tool_text) → 75-100 pairs

2. **Each agent produces:**
   - JSONL file with interleaved examples
   - Coverage tracking (tools used, scenarios covered)
   - Quality notes (any edge cases, challenges)

3. **Merge strategy:**
   - Validate each pattern file individually
   - Combine into merged dataset (optional)
   - OR keep separate for targeted training

### Agent Instructions Template

**Your task:** Generate {N} pairs for the `{pattern_name}` response pattern.

**Requirements:**
1. Create {N/2} positive + {N/2} negative examples (total {N})
2. Interleave perfectly: True, False, True, False, ...
3. Cover diverse scenarios from the spec
4. Use variety of tools (aim for 15+ different tools)
5. Ensure all context fields complete
6. Pass syngen validator

**Output:**
- File: `{pattern_name}_pairs_v1.0.jsonl`
- One JSON object per line
- Interleaved labels

**Quality checks:**
- Run validator after every 20 pairs
- Track tool usage to ensure diversity
- Review 5-10 examples for quality
- Verify interleaving pattern

---

## Success Criteria

### Dataset-Level Metrics
- ✅ 150-200 examples per pattern (450-600 total)
- ✅ 100% pass syngen validation
- ✅ Perfect interleaving (True/False/True/False)
- ✅ 50% True, 50% False labels exactly
- ✅ 15+ different tools per pattern
- ✅ All 5 agent families represented
- ✅ 8+ scenario domains covered

### Example-Level Quality
- ✅ Positive examples demonstrate ideal pattern usage
- ✅ Negative examples show realistic mistakes
- ✅ Clear behavioral contrast in each pair
- ✅ Context fields rich and specific
- ✅ Natural user messages
- ✅ Realistic tool results

### Post-Training Goals
After training on this dataset, model should:
1. **Recognize completion** (>85% accuracy) - Stop and summarize when results ready
2. **Ask clarification appropriately** (>80% accuracy) - Identify ambiguous situations
3. **Maintain workflow efficiency** (>90% accuracy) - Continue obvious workflows
4. **Provide helpful context** (>85% accuracy) - Explain complex operations

---

## Quick Reference

### Pattern Decision Tree

```
After receiving tool result, ask:

1. Are results ready to show user?
   YES → Pattern 1 (text-only) - Summarize results
   NO → Continue to 2

2. Is next step obvious and safe?
   YES → Pattern 2 (tool-only) - Execute next step
   NO → Continue to 3

3. Does action need explanation?
   YES → Pattern 3 (tool+text) - Explain + execute
   NO → Pattern 2 (tool-only)
```

### Label Assignment

```
label: true  = Correct pattern choice for situation
label: false = Wrong pattern choice (realistic mistake)
```

### File Naming

```
text_only_pairs_v1.0.jsonl
tool_only_pairs_v1.0.jsonl
tool_text_pairs_v1.0.jsonl
```

### Context Object Template

```json
{
  "sessionId": "session_1732NNNNNNNNN_XXXXXXXXX",
  "workspaceId": "ws_1732NNNNNNNNN_XXXXXXXXX",
  "sessionDescription": "Brief description of task",
  "sessionMemory": "What happened before, why we're at this decision point, relevant context (80-150 chars)",
  "toolContext": "Why this response choice, workflow position, reasoning (60-120 chars) - STRING NOT OBJECT",
  "primaryGoal": "User's overall objective",
  "subgoal": "Current step or decision point"
}
```

---

**Version History:**
- 1.0 (2025-11-25): Initial specification for parallel agent generation
