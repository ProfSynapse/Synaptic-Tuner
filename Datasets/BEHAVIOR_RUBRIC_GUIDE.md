# Behavior-Based KTO Training Rubric Guide

**Version:** 1.0
**Created:** 2025-11-22
**Purpose:** Framework for generating synthetic conversation pairs that teach desirable tool-use behaviors through preference learning (KTO)

---

## Table of Contents

1. [Overview](#overview)
2. [Rubric Framework](#rubric-framework)
3. [Behavior Definitions](#behavior-definitions)
4. [Synthetic Generation Instructions](#synthetic-generation-instructions)
5. [Quality Control](#quality-control)
6. [Examples](#examples)

---

## Overview

### What This Guide Does

This rubric system enables creation of **preference learning datasets** that teach models not just WHICH tools to use, but HOW to use them well. Each behavior represents a best practice or desirable pattern in tool usage.

### Core Concept: Paired Contrastive Examples

For each behavior, we generate **matched pairs**:
- **Positive example (label: true)**: Demonstrates the desirable behavior
- **Negative example (label: false)**: Shows a realistic mistake or anti-pattern

Both examples use the **identical user request** but differ in the assistant's response quality.

### Training Flow

```
SFT Dataset (existing)          →  Model learns basic tool syntax
    ↓
Behavior KTO Dataset (new)      →  Model learns best practices
    ↓
Evaluation (behavior-specific)  →  Verify behaviors are learned
```

---

## Rubric Framework

### Standard Rubric Structure

Each behavior rubric contains:

```yaml
behavior_name: string          # Unique identifier
category: string               # Group (humility, verification, planning, etc.)
description: string            # What this behavior teaches
priority: high|medium|low      # Training importance

positive_indicators:           # What GOOD examples show
  context_patterns:            # What appears in context fields
    sessionMemory: []
    toolContext: []
    goals: []
  tool_patterns:               # Tool selection patterns
    preferred_tools: []
    tool_sequences: []
  parameter_patterns:          # How parameters are used
    []

negative_indicators:           # What BAD examples show
  context_patterns:
    sessionMemory: []
    toolContext: []
    goals: []
  tool_patterns:
    antipatterns: []
    wrong_sequences: []
  parameter_patterns:
    []

trigger_scenarios:             # When this behavior applies
  []

example_pairs:                 # Seed examples
  []

metrics:                       # How to measure if learned
  []
```

---

## Behavior Definitions

### 1. Intellectual Humility (IH)

**Category:** Verification & Caution
**Priority:** HIGH
**Description:** Model recognizes uncertainty, verifies assumptions, and escalates complex tasks rather than proceeding overconfidently.

#### Positive Indicators

**Context Patterns:**
- **sessionMemory:**
  - Explicitly mentions uncertainty: "User request ambiguous - need clarification"
  - Notes what information is missing: "File path not specified, must search first"
  - References prior verification attempts: "Searched 47 files, found 3 candidates, need user to confirm"

- **toolContext:**
  - Explains WHY verification is needed: "Searching to verify file existence before destructive operation"
  - Mentions alternatives considered: "Using search instead of direct access to avoid errors"
  - Shows workflow reasoning: "Will ask user to confirm after locating targets"

- **Goals:**
  - primaryGoal: User's intended outcome
  - subgoal: Current verification/clarification step (e.g., "Identify which files to delete")

**Tool Patterns:**
- **Preferred sequence:**
  1. Search/verify BEFORE action: `vaultLibrarian_searchContent` → confirm → `vaultManager_deleteNote`
  2. Escalate complex reasoning: `agentManager_executePrompt` with stronger model
  3. Check memory for prior context: `memoryManager_searchMemory` before assuming

- **Specific tools:**
  - `vaultLibrarian_searchContent` - Verify files exist
  - `vaultLibrarian_searchDirectory` - Explore structure before organizing
  - `agentManager_executePrompt` - Consult smarter model for complex questions
  - `memoryManager_searchMemory` - Check if user already provided context

**Parameter Patterns:**
- Search with `includeContent: false` when just verifying existence
- `limit: 20-50` to get comprehensive results before filtering
- executePrompt includes `returnContent: true` to capture response

#### Negative Indicators

**Context Patterns:**
- **sessionMemory:**
  - Empty or generic: "", "Starting task", "User made request"
  - Overconfident: "Obviously user wants X", "Clearly means Y"
  - No acknowledgment of ambiguity

- **toolContext:**
  - Just restates action: "Deleting files", "Moving folder"
  - No reasoning: "User requested this"
  - Assumes without verification: "File is at Projects/old"

**Tool Patterns:**
- **Antipatterns:**
  - Direct destructive operations without search: `vaultManager_deleteFolder` immediately
  - Assumes paths exist: `vaultManager_moveNote` with hardcoded path
  - Makes up answers instead of searching or asking
  - Repeats failed approach without trying alternatives

**Parameter Patterns:**
- Hardcoded paths without verification: `"path": "Projects/old"` (assumed)
- `recursive: true` without confirming scope
- No safety parameters when available

#### Trigger Scenarios

Use this behavior when user request contains:
- **Ambiguous references:** "the old files", "that folder", "my notes"
- **Destructive operations:** delete, move, replace with unclear targets
- **Complex reasoning:** "What's the best way to...", "How should I organize..."
- **Requests for external knowledge:** "What are best practices for..."
- **Missing context:** No paths, no dates, no specific identifiers

#### Metrics

- **Verification rate:** % of destructive operations preceded by search (target: >95%)
- **Clarification rate:** % of ambiguous requests that trigger verification (target: 60-80%)
- **Escalation rate:** % of complex reasoning tasks using executePrompt (target: >80%)
- **Error avoidance:** % reduction in "file not found" errors (target: >70%)

---

### 2. Context Continuity (CC)

**Category:** Memory & Workflow
**Priority:** HIGH
**Description:** Model maintains rich, specific context across operations, showing workflow continuity and leveraging prior actions.

#### Positive Indicators

**Context Patterns:**
- **sessionMemory (target: 80-150 chars):**
  - References specific prior actions: "Created 3 folders for project redesign (UI, Backend, Docs)"
  - Includes concrete numbers/details: "Searched 47 files, found 23 matches, filtered to 8 candidates"
  - Shows workflow progression: "Completed folder setup, now organizing files by type"
  - Mentions prior results: "Previous search returned API/REST-Endpoints.md and API/Authentication.md"

- **toolContext (target: 60-120 chars):**
  - Explains position in workflow: "After confirming targets via search, now executing move operation"
  - References workflow logic: "Using append instead of replace to preserve existing changelog entries"
  - Shows strategic reasoning: "Batch operation more efficient than 8 individual moves"

- **Goals:**
  - primaryGoal: Remains consistent across operation sequence
  - subgoal: Shows clear progression (search → verify → act → confirm)

**Tool Patterns:**
- Logical sequences that reference prior steps
- Uses results from one tool to inform next tool's parameters
- Mentions prior tool calls in context fields

**Parameter Patterns:**
- Parameters derived from prior results (not hardcoded)
- Paths verified through search before use
- Limits adjusted based on prior result counts

#### Negative Indicators

**Context Patterns:**
- **sessionMemory:**
  - Empty: "" or `[]`
  - Generic placeholders: "User context", "Previous session", "Working on task"
  - Too short: < 50 characters
  - No specifics: "User working on project" (which project? what phase?)

- **toolContext:**
  - Just restates action: "Appending content", "Moving file"
  - Object instead of string: `{"path": "...", "action": "..."}` ← SCHEMA VIOLATION
  - Generic: "Tool usage context", "Performing operation"

- **Goals:**
  - Identical: primaryGoal and subgoal are the same
  - Too generic: "Complete user request", "Help user"
  - No decomposition: subgoal doesn't show current step

**Tool Patterns:**
- Each operation seems isolated (no workflow continuity)
- Doesn't reference prior tool calls
- Repeats searches unnecessarily

#### Trigger Scenarios

- Multi-step operations (search → read → update)
- Continuation of prior work
- Batch operations
- Workflow-based tasks

#### Metrics

- **sessionMemory quality:** Avg length and specificity score (target: 80+ chars, 4-5/5 quality)
- **toolContext quality:** % that explain WHY vs just WHAT (target: >90%)
- **Goal coherence:** % with clear primaryGoal → subgoal decomposition (target: >85%)
- **Workflow continuity:** % that reference prior actions (target: >70%)

---

### 3. Verification Before Action (VBA)

**Category:** Safety & Reliability
**Priority:** HIGH
**Description:** Model verifies assumptions and previews changes before executing potentially destructive or irreversible operations.

#### Positive Indicators

**Tool Patterns:**
- **Search before destructive ops:**
  - `vaultLibrarian_searchContent` → `vaultManager_deleteNote`
  - `vaultManager_listDirectory` → `vaultManager_deleteFolder`
  - `vaultLibrarian_searchDirectory` → batch delete

- **Read before modify:**
  - `contentManager_readContent` → `contentManager_replaceContent`
  - `contentManager_readContent` → `contentManager_replaceByLine`

- **List before batch:**
  - `memoryManager_listSessions` → batch session operations
  - `vaultManager_listDirectory` → `vaultLibrarian_batch` operations

**Context Patterns:**
- toolContext explains verification rationale
- sessionMemory notes what's being verified
- Goals show verification as explicit subgoal

**Parameter Patterns:**
- Search uses appropriate scope (folder paths, file types)
- Read operations preview full content before replace
- List operations use reasonable limits (not 1, not 1000)

#### Negative Indicators

**Tool Patterns:**
- **Antipatterns:**
  - Direct `vaultManager_deleteNote` without searching
  - `contentManager_replaceContent` without reading first
  - `vaultManager_moveFolder` with `overwrite: true` without checking destination
  - Batch operations without listing targets first

**Context Patterns:**
- No mention of verification
- Assumes paths/content exists
- toolContext doesn't explain safety reasoning

#### Trigger Scenarios

- Delete operations (files, folders, content)
- Replace/overwrite operations
- Move operations with `overwrite: true`
- Batch operations affecting multiple items
- Operations in unfamiliar directories

#### Metrics

- **Verification rate:** % of destructive ops with prior search/list (target: >95%)
- **Preview rate:** % of replace ops with prior read (target: >85%)
- **Safety parameter usage:** % using appropriate safety flags (target: >80%)

---

### 4. Strategic Tool Selection (STS)

**Category:** Efficiency & Planning
**Priority:** MEDIUM
**Description:** Model chooses optimal tools and sequences for the task, preferring batch operations and avoiding unnecessary steps.

#### Positive Indicators

**Tool Patterns:**
- **Batch over iteration:**
  - `vaultLibrarian_batch` for multiple related searches (not 5× `searchContent`)
  - `contentManager_batchContent` for multiple file operations
  - `agentManager_batchExecutePrompt` for multiple prompts

- **Appropriate scope:**
  - `searchContent` for content matching (not `searchDirectory` then manual filtering)
  - `searchDirectory` for file/folder structure (not `searchContent`)
  - `searchMemory` for prior session data (not re-searching files)

- **Efficient sequences:**
  - Single search with specific query (not multiple vague searches)
  - Direct read of known file (not search → list → read for known path)
  - Proper use of `includeContent` flag (true only when needed)

**Context Patterns:**
- toolContext explains efficiency reasoning: "Batch search more efficient than 5 individual calls"
- sessionMemory shows planning: "Need to search 3 different topics, using batch operation"

**Parameter Patterns:**
- `searches` array in batch operations groups related queries
- `mergeResults: true` when consolidated output needed
- Appropriate `limit` values (not always default)

#### Negative Indicators

**Tool Patterns:**
- **Antipatterns:**
  - Multiple individual searches instead of batch
  - Using `searchContent` when `searchDirectory` appropriate (or vice versa)
  - Unnecessary intermediate steps
  - Re-searching for information already obtained
  - Using `includeContent: true` then not using content

**Context Patterns:**
- No mention of efficiency considerations
- Doesn't acknowledge batch alternatives
- No planning for multi-step operations

#### Trigger Scenarios

- Multiple related searches needed
- Multiple file operations on different files
- Need to check multiple conditions
- Large-scale operations (many files/folders)

#### Metrics

- **Batch usage rate:** % of multi-operation tasks using batch tools (target: >70%)
- **Search efficiency:** Avg queries per task (target: minimize)
- **Unnecessary steps:** % of operations with redundant tool calls (target: <10%)

---

### 5. Error Recovery & Adaptation (ERA)

**Category:** Resilience
**Priority:** MEDIUM
**Description:** Model handles failures gracefully, tries alternative approaches, and learns from errors without repeating failed attempts.

#### Positive Indicators

**Context Patterns:**
- **sessionMemory acknowledges failures:**
  - "Previous direct access to Projects/old failed (not found), now searching with pattern"
  - "Replace operation failed due to content mismatch, reading full content to identify exact string"
  - "Workspace load failed, trying alternative workspaceId from memory"

- **toolContext explains adaptation:**
  - "Using search after direct read failed - file may have moved"
  - "Trying case-insensitive search since exact match returned no results"
  - "Escalating to executePrompt since automated approach failed twice"

**Tool Patterns:**
- **Fallback sequences:**
  1. Direct access fails → Search for item
  2. Exact match fails → Fuzzy/pattern matching
  3. Automated solution fails 2× → executePrompt escalation
  4. File not found → searchDirectory to explore actual structure

- **Alternative approaches:**
  - Change search parameters (case-sensitive → insensitive)
  - Broaden scope (specific folder → vault-wide)
  - Switch tool families (vaultManager → vaultLibrarian)

**Parameter Patterns:**
- Adjusted based on prior failure (broader limits, different filters)
- Includes context about what was already tried
- Uses `includeContent: true` to understand why prior match failed

#### Negative Indicators

**Tool Patterns:**
- **Antipatterns:**
  - Repeats identical failed operation
  - No change in parameters after failure
  - Gives up after single failure
  - No acknowledgment of error in context
  - Doesn't try alternative tools

**Context Patterns:**
- sessionMemory omits prior failures
- toolContext doesn't explain why this approach differs
- No learning from errors

#### Trigger Scenarios

- File not found errors
- Content mismatch in replace operations
- Permission/access errors
- Unexpected tool results (empty, wrong format)
- Search returns no results

#### Metrics

- **Recovery rate:** % of failures followed by successful alternative (target: >75%)
- **Repetition rate:** % of failures repeated identically (target: <5%)
- **Escalation appropriateness:** % of 2+ failures escalating to executePrompt (target: >60%)

---

### 6. Workspace Awareness (WA)

**Category:** Context & Structure
**Priority:** LOW-MEDIUM
**Description:** Model leverages workspace metadata, follows workspace workflows, and maintains workspace-specific context.

#### Positive Indicators

**Tool Patterns:**
- **Workspace loading:**
  - `memoryManager_loadWorkspace` before workspace-specific operations
  - Uses workspace data to inform decisions
  - References workspace `keyFiles`, `workflows`, `preferences`

- **Workflow following:**
  - sessionMemory references workspace workflows: "Following budget workspace monthly-review workflow"
  - toolContext mentions workflow steps: "Step 2 of expense-tracking workflow: categorize expense"
  - Goals align with workspace purpose

**Context Patterns:**
- **sessionMemory:**
  - References workspace context: "Budget workspace tracks monthly expenses, currently Nov 30% spent at 33% elapsed"
  - Mentions workspace structure: "Per workspace keyFiles, checking Budget/Current/november-2025.md"

- **toolContext:**
  - Explains workspace rationale: "Using workspace-defined quarterly-review workflow for systematic analysis"
  - References workspace preferences: "Workspace prefers append over replace for changelog"

**Parameter Patterns:**
- Paths align with workspace `directoryStructure`
- Operations follow workspace `workflows`
- workspaceId consistently used in context

#### Negative Indicators

**Tool Patterns:**
- **Antipatterns:**
  - Ignores available workspace data
  - Doesn't load workspace when workspace-specific operation
  - Generic operations when workspace workflows available
  - Incorrect workspaceId in context

**Context Patterns:**
- No reference to workspace structure
- Doesn't acknowledge workspace workflows
- Generic approach ignoring workspace preferences

#### Trigger Scenarios

- Operations within defined workspace folders
- Tasks matching workspace workflows
- Requests referencing workspace purpose
- Workspace-specific file organization

#### Metrics

- **Workspace loading rate:** % of workspace ops preceded by loadWorkspace (target: >70%)
- **Workflow adherence:** % following defined workflows when applicable (target: >80%)
- **Context accuracy:** % using correct workspaceId (target: >95%)

---

## Synthetic Generation Instructions

### Prerequisites

Before generating examples, you need:
1. ✅ Access to tool schemas (`tools/tool_schemas.json`)
2. ✅ Understanding of context object requirements (7 fields, sessionMemory never empty)
3. ✅ Knowledge of ChatML format for this project
4. ✅ Familiarity with the rubric for target behavior

### Step-by-Step Generation Process

#### Step 1: Choose Behavior & Scenario

Select:
- **Target behavior:** e.g., "intellectual_humility"
- **Trigger scenario:** e.g., "User request with ambiguous file reference"
- **Tools involved:** e.g., `vaultLibrarian_searchContent`, `vaultManager_deleteNote`

#### Step 2: Create User Request

Write a **realistic, natural user request** that triggers the behavior:

**Good user requests:**
- ✅ "Can you delete the old project files?" (ambiguous - which files?)
- ✅ "Move my meeting notes to the archive" (which meetings? where?)
- ✅ "What's the best way to organize 200+ research papers?" (complex reasoning)

**Bad user requests:**
- ❌ "Delete /Projects/2023/old-prototype" (too specific, no ambiguity)
- ❌ "Execute vaultManager_deleteNote" (not natural language)
- ❌ "Result: {...}" (continuation format, not initial request)

#### Step 3: Generate POSITIVE Example (label: true)

Create assistant response demonstrating **ALL positive indicators** for the behavior:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Can you delete the old project files?"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultLibrarian_searchContent\narguments: {\"context\": {\"sessionId\": \"session_1731398400000_a1b2c3d4e\", \"workspaceId\": \"ws_1731398400000_f5g6h7i8j\", \"sessionDescription\": \"Searching for old project files before deletion\", \"sessionMemory\": \"User requested deletion of 'old project files' but target unclear. Multiple project folders exist from 2022-2024. Need to search and verify which files are actually old/obsolete before proceeding with potentially destructive operation.\", \"toolContext\": \"Searching for project files to identify deletion candidates before proceeding. Using search instead of direct delete to avoid removing wrong files. Will ask user to confirm specific targets once located.\", \"primaryGoal\": \"Safely delete obsolete project files\", \"subgoal\": \"Identify which files are old and should be deleted\"}, \"query\": \"project\", \"limit\": 50, \"includeContent\": false}"
    }
  ],
  "label": true,
  "behavior": "intellectual_humility"
}
```

**Checklist for positive example:**
- ✅ Demonstrates verification (searches before deleting)
- ✅ sessionMemory explains uncertainty ("target unclear")
- ✅ sessionMemory is specific and detailed (80+ chars)
- ✅ toolContext explains WHY search chosen ("to avoid removing wrong files")
- ✅ toolContext is a STRING (not object)
- ✅ Goals show clear decomposition (overall goal → current step)
- ✅ All 7 context fields present
- ✅ Realistic parameter values

#### Step 4: Generate NEGATIVE Example (label: false)

Create assistant response with **SAME user request** demonstrating **anti-patterns**:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Can you delete the old project files?"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultManager_deleteFolder\narguments: {\"context\": {\"sessionId\": \"session_1731398400000_a1b2c3d4e\", \"workspaceId\": \"ws_1731398400000_f5g6h7i8j\", \"sessionDescription\": \"Deleting old project files\", \"sessionMemory\": \"User wants old files deleted.\", \"toolContext\": \"Removing old project folder as requested.\", \"primaryGoal\": \"Delete old files\", \"subgoal\": \"Remove folder\"}, \"path\": \"Projects/old\", \"recursive\": true}"
    }
  ],
  "label": false,
  "behavior": "intellectual_humility"
}
```

**Checklist for negative example:**
- ✅ Demonstrates anti-pattern (direct delete without verification)
- ✅ sessionMemory is weak/generic ("User wants old files deleted" < 50 chars)
- ✅ toolContext just restates action (no WHY reasoning)
- ✅ Goals are redundant/weak
- ✅ Assumes path without verification ("Projects/old")
- ✅ Uses dangerous parameter (`recursive: true`) without confirmation
- ✅ Still valid JSON/structure (realistic mistake, not nonsense)
- ✅ SAME user request as positive example

#### Step 5: Validate Pair

**Required checks:**
1. ✅ Both examples use identical user request (exact string match)
2. ✅ Positive labeled `true`, negative labeled `false`
3. ✅ Both include `"behavior": "behavior_name"` field
4. ✅ Both pass schema validation (tool exists, required params present)
5. ✅ Context objects have all 7 fields in both
6. ✅ sessionMemory never empty in either
7. ✅ toolContext is STRING in both (not object)
8. ✅ Positive demonstrates ALL key positive indicators
9. ✅ Negative demonstrates realistic anti-pattern (not gibberish)
10. ✅ Single-turn format (no Result objects in assistant content)

### Generation Templates by Behavior

#### Template: Intellectual Humility

```
USER REQUEST: [Ambiguous or complex task]

POSITIVE (search/verify/escalate first):
- Tool: vaultLibrarian_searchContent, agentManager_executePrompt, etc.
- sessionMemory: "User request [SPECIFIC AMBIGUITY]. Need to [VERIFICATION APPROACH]. [WHAT WE KNOW]."
- toolContext: "Searching to [VERIFY WHAT] before [RISKY ACTION]. [SAFETY REASONING]."
- primaryGoal: [User's actual outcome]
- subgoal: "Identify/verify [WHAT NEEDS CLARIFICATION]"

NEGATIVE (assume and act):
- Tool: Direct destructive operation (deleteFolder, moveNote, replaceContent)
- sessionMemory: Generic/short "User wants X"
- toolContext: Just restates action "Deleting files"
- Hardcoded paths without verification
```

#### Template: Context Continuity

```
USER REQUEST: [Multi-step or continuation task]

POSITIVE (rich context):
- sessionMemory: "Previously [PRIOR ACTION with DETAILS]. Now [CURRENT PHASE]. [WORKFLOW STATE]." (80-150 chars)
- toolContext: "After [PRIOR STEP], now [CURRENT STEP] to [PURPOSE]. Using [TOOL] because [REASONING]." (60-120 chars)
- primaryGoal: [Consistent across sequence]
- subgoal: [Shows progression in workflow]

NEGATIVE (weak context):
- sessionMemory: "" or "Working on task" (<50 chars, no specifics)
- toolContext: "Moving file" (no workflow reasoning)
- primaryGoal/subgoal: Identical or too generic
```

#### Template: Verification Before Action

```
USER REQUEST: [Destructive operation with unclear target]

POSITIVE (verify first):
- Tool sequence: Search/List → Verify → Act
- sessionMemory: "Need to [VERIFY WHAT] before [RISKY ACTION]"
- toolContext: "Using [VERIFICATION TOOL] to confirm [WHAT] before proceeding with [ACTION]"

NEGATIVE (act immediately):
- Tool: Direct destructive operation
- No prior search/list/read
- Assumes paths/content exists
- Uses dangerous flags (recursive: true, overwrite: true) without checking
```

### Batch Generation Guidelines

When generating multiple pairs:

1. **Vary user requests** across the same behavior:
   - Different ambiguity types (path, target, method)
   - Different tools involved
   - Different domains (files, memory, agents, workspaces)

2. **Maintain consistent quality:**
   - All positive examples should be 4-5/5 quality
   - All negative examples should be realistic mistakes (not random garbage)
   - sessionMemory in positives: 80-150 chars
   - sessionMemory in negatives: 0-50 chars OR generic

3. **Balance tool coverage:**
   - Don't just use `searchContent` + `deleteNote` for every IH example
   - Vary: searchDirectory, searchMemory, executePrompt, etc.
   - Cover different agent families

4. **Ensure diversity:**
   - Different users' use cases (research, projects, journaling, budget, etc.)
   - Different file types and folder structures
   - Different workflow contexts

### Example: Generating 10 Pairs for Intellectual Humility

**Scenario 1:** Ambiguous deletion
- User: "Delete the old project files"
- Positive: searchContent first
- Negative: deleteFolder with assumed path

**Scenario 2:** Complex reasoning
- User: "What's the best way to organize 200+ research papers?"
- Positive: executePrompt to consult stronger model
- Negative: Creates generic folder structure without research

**Scenario 3:** Unclear move target
- User: "Move my meeting notes to the archive"
- Positive: searchContent to find which meeting notes
- Negative: moveFolder with assumed path

**Scenario 4:** Ambiguous reference
- User: "Can you update the config file with the new API key?"
- Positive: searchContent to find which config file (multiple exist)
- Negative: replaceContent with assumed file path

**Scenario 5:** Risky replace
- User: "Replace the database connection string in my settings"
- Positive: readContent first to verify location, then replace
- Negative: replaceContent with hardcoded search string

...and so on for 10 pairs (20 total examples).

---

## Quality Control

### Automated Validation

After generating pairs, validate with:

```bash
# Basic schema validation
python tools/validate_syngen.py Datasets/behavior_datasets/intellectual_humility_pairs.jsonl

# Behavior-specific validation
python tools/validate_behavior_pairs.py \
  --dataset Datasets/behavior_datasets/intellectual_humility_pairs.jsonl \
  --rubric Datasets/behavior_rubrics/intellectual_humility.yaml
```

### Manual Review Checklist

Sample 10-20% of generated pairs and verify:

**For each pair:**
- [ ] User request is identical in both examples
- [ ] User request is natural (not command-style or "Result:" format)
- [ ] Positive example demonstrates ALL key positive indicators
- [ ] Negative example shows realistic mistake (not nonsense)
- [ ] sessionMemory in positive is specific (80+ chars with details)
- [ ] sessionMemory in negative is weak/generic (<50 chars or placeholder)
- [ ] toolContext in both is STRING (not object)
- [ ] toolContext in positive explains WHY (60+ chars)
- [ ] toolContext in negative just restates WHAT (<40 chars)
- [ ] Tools chosen are appropriate for scenario
- [ ] All 7 context fields present in both
- [ ] Parameters are realistic (not obviously made-up)
- [ ] Single-turn format (no Result objects)
- [ ] Behavior field matches target behavior

**Across full dataset:**
- [ ] Good variety of user requests (not repetitive)
- [ ] Multiple tools covered (not just 2-3 tools)
- [ ] Different domains/use cases represented
- [ ] Balanced True/False labels (should be exactly 1:1)
- [ ] No duplicate examples

### Common Quality Issues

**Issue 1: Unrealistic negative examples**
```json
// BAD - Too obviously wrong
{"content": "tool_call: nonexistentTool\narguments: garbage"}

// GOOD - Realistic mistake
{"content": "tool_call: vaultManager_deleteFolder\narguments: {..., \"path\": \"old\", \"recursive\": true}"}
```

**Issue 2: User requests not identical**
```json
// BAD
Positive user: "Can you delete the old project files?"
Negative user: "Delete old files"

// GOOD
Both: "Can you delete the old project files?"
```

**Issue 3: Positive example not strong enough**
```json
// BAD - sessionMemory still weak
"sessionMemory": "User wants to delete files. Need to search first."

// GOOD - Rich, specific context
"sessionMemory": "User requested deletion of 'old project files' but target unclear. Multiple project folders exist from 2022-2024. Need to search and verify which files are actually old/obsolete before proceeding with potentially destructive operation."
```

---

## Examples

### Complete Example Pair: Intellectual Humility

**Scenario:** User requests file deletion with ambiguous target

#### Positive Example (label: true)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Can you delete the old project files from last year?"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultLibrarian_searchContent\narguments: {\"context\": {\"sessionId\": \"session_1732300800000_h7j2k9m3n\", \"workspaceId\": \"ws_1732300800000_p4q8r1s6t\", \"sessionDescription\": \"Searching for old project files before deletion\", \"sessionMemory\": \"User requested deletion of 'old project files from last year' but specific paths unclear. Workspace contains Projects folder with multiple year-based subfolders (2022, 2023, 2024). Current year is 2025, so 'last year' likely means 2024, but need to verify before destructive operation.\", \"toolContext\": \"Using searchContent to locate files from 2024 before proceeding with deletion. Searching with 'project' query and will filter by date. This verification prevents accidentally deleting wrong files and allows user to confirm targets.\", \"primaryGoal\": \"Safely delete old project files from 2024\", \"subgoal\": \"Identify and verify which project files are from last year\"}, \"query\": \"project\", \"folder\": \"Projects\", \"limit\": 50, \"includeContent\": false}"
    }
  ],
  "label": true,
  "behavior": "intellectual_humility"
}
```

**Why this is good:**
- ✅ Searches before deleting (verification pattern)
- ✅ sessionMemory is detailed (167 chars) and explains ambiguity
- ✅ toolContext explains WHY search chosen (prevention rationale)
- ✅ Goals decompose well (overall safety → current verification)
- ✅ Parameters appropriate (`folder: "Projects"` narrows scope)

#### Negative Example (label: false)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Can you delete the old project files from last year?"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultManager_deleteFolder\narguments: {\"context\": {\"sessionId\": \"session_1732300800000_h7j2k9m3n\", \"workspaceId\": \"ws_1732300800000_p4q8r1s6t\", \"sessionDescription\": \"Deleting old files\", \"sessionMemory\": \"User wants old project files deleted.\", \"toolContext\": \"Removing last year's project folder.\", \"primaryGoal\": \"Delete old files\", \"subgoal\": \"Remove folder\"}, \"path\": \"Projects/2024\", \"recursive\": true}"
    }
  ],
  "label": false,
  "behavior": "intellectual_humility"
}
```

**Why this is bad (intentionally):**
- ❌ Direct deletion without verification
- ❌ sessionMemory is generic (39 chars, no detail)
- ❌ toolContext just restates action (no reasoning)
- ❌ Assumes path "Projects/2024" without confirming
- ❌ Uses `recursive: true` without checking what's inside
- ❌ Goals are weak and redundant

---

### Complete Example Pair: Context Continuity

**Scenario:** Multi-step workflow continuation

#### Positive Example (label: true)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Now organize those files by topic"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultManager_createFolder\narguments: {\"context\": {\"sessionId\": \"session_1732301200000_b3c7d2e9f\", \"workspaceId\": \"ws_1732301200000_g4h1j5k8m\", \"sessionDescription\": \"Organizing research files by topic\", \"sessionMemory\": \"Previously searched and found 47 research files across 3 folders. User reviewed list and confirmed 23 papers are relevant for reorganization. Identified 4 main topics: ML, NLP, Vision, Robotics. Now creating folder structure to organize by these topics.\", \"toolContext\": \"Creating topic-based folders as first step in organization workflow. After folder creation, will batch-move files to appropriate topics based on content analysis. Using createFolder instead of manual creation ensures consistent structure.\", \"primaryGoal\": \"Organize 23 research papers by topic\", \"subgoal\": \"Create topic folder structure (ML, NLP, Vision, Robotics)\"}, \"path\": \"Research/Topics\"}"
    }
  ],
  "label": true,
  "behavior": "context_continuity"
}
```

**Why this is good:**
- ✅ sessionMemory shows workflow progression (158 chars with specifics)
- ✅ References prior action results ("found 47 research files", "confirmed 23 papers")
- ✅ toolContext explains position in workflow ("first step", "After folder creation, will...")
- ✅ Goals show clear progression from overall to current step

#### Negative Example (label: false)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Now organize those files by topic"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultManager_createFolder\narguments: {\"context\": {\"sessionId\": \"session_1732301200000_b3c7d2e9f\", \"workspaceId\": \"ws_1732301200000_g4h1j5k8m\", \"sessionDescription\": \"Creating folders\", \"sessionMemory\": \"Organizing files.\", \"toolContext\": \"Making new folder.\", \"primaryGoal\": \"Organize files\", \"subgoal\": \"Create folder\"}, \"path\": \"Research/Topics\"}"
    }
  ],
  "label": false,
  "behavior": "context_continuity"
}
```

**Why this is bad (intentionally):**
- ❌ sessionMemory is minimal (18 chars, no specifics)
- ❌ No reference to prior actions or results
- ❌ toolContext is generic (17 chars, no workflow context)
- ❌ Goals are weak and don't show decomposition

---

### Complete Example Pair: Verification Before Action

**Scenario:** Replace operation on important config

#### Positive Example (label: true)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Update the API endpoint in my config to use the new production server"
    },
    {
      "role": "assistant",
      "content": "tool_call: contentManager_readContent\narguments: {\"context\": {\"sessionId\": \"session_1732302000000_x9y3z7a2b\", \"workspaceId\": \"ws_1732302000000_c5d1e6f4g\", \"sessionDescription\": \"Reading config before updating API endpoint\", \"sessionMemory\": \"User wants to update API endpoint to production server. Multiple config files exist (dev.config, prod.config, settings.yaml). Need to read current config to identify exact API endpoint line before replacing to avoid breaking configuration.\", \"toolContext\": \"Reading config file content to locate exact API endpoint string before replacement. Preview ensures we replace correct line and don't inadvertently change other endpoints. Safety step before modify operation.\", \"primaryGoal\": \"Update API endpoint to production server\", \"subgoal\": \"Read config to identify current endpoint location\"}, \"filePath\": \"config/settings.yaml\", \"includeLineNumbers\": true}"
    }
  ],
  "label": true,
  "behavior": "verification_before_action"
}
```

**Why this is good:**
- ✅ Reads before replacing (verification pattern)
- ✅ sessionMemory explains why reading first (152 chars)
- ✅ toolContext shows safety reasoning
- ✅ Uses `includeLineNumbers: true` to enable precise replacement

#### Negative Example (label: false)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Update the API endpoint in my config to use the new production server"
    },
    {
      "role": "assistant",
      "content": "tool_call: contentManager_replaceContent\narguments: {\"context\": {\"sessionId\": \"session_1732302000000_x9y3z7a2b\", \"workspaceId\": \"ws_1732302000000_c5d1e6f4g\", \"sessionDescription\": \"Updating API config\", \"sessionMemory\": \"Replacing endpoint.\", \"toolContext\": \"Changing API URL.\", \"primaryGoal\": \"Update endpoint\", \"subgoal\": \"Replace URL\"}, \"filePath\": \"config/settings.yaml\", \"oldContent\": \"api_endpoint: https://dev.api.com\", \"newContent\": \"api_endpoint: https://prod.api.com\"}"
    }
  ],
  "label": false,
  "behavior": "verification_before_action"
}
```

**Why this is bad (intentionally):**
- ❌ Direct replace without reading first
- ❌ Assumes old endpoint URL without verification
- ❌ Could fail if actual string doesn't match exactly
- ❌ Weak context (sessionMemory 20 chars)

---

## Appendix: Tool Reference

### Tools by Behavior

**Intellectual Humility:**
- `vaultLibrarian_searchContent` - Verify file existence
- `vaultLibrarian_searchDirectory` - Explore structure
- `agentManager_executePrompt` - Escalate to smarter model
- `memoryManager_searchMemory` - Check prior context

**Context Continuity:**
- All tools (focus on context fields quality)

**Verification Before Action:**
- `vaultLibrarian_searchContent` + `vaultManager_deleteNote`
- `contentManager_readContent` + `contentManager_replaceContent`
- `vaultManager_listDirectory` + `vaultManager_deleteFolder`
- `memoryManager_listSessions` + batch session operations

**Strategic Tool Selection:**
- `vaultLibrarian_batch` (instead of multiple searchContent)
- `contentManager_batchContent` (instead of multiple reads/writes)
- `agentManager_batchExecutePrompt` (instead of multiple prompts)

**Error Recovery:**
- Any failed operation → alternative tool
- searchContent with adjusted parameters
- Direct access → Search fallback

**Workspace Awareness:**
- `memoryManager_loadWorkspace` before workspace operations
- `memoryManager_createWorkspace` for new structured work
- Tool calls reference workspace structure

---

## Quick Reference Card

### Positive Example Checklist

- [ ] sessionMemory: 80-150 chars, specific details
- [ ] toolContext: 60-120 chars, explains WHY
- [ ] toolContext is STRING (not object)
- [ ] All 7 context fields present
- [ ] sessionMemory never empty
- [ ] Goals: clear primaryGoal → subgoal decomposition
- [ ] Tool choice demonstrates positive pattern
- [ ] Parameters are realistic and appropriate
- [ ] Single-turn format (no Result in assistant content)

### Negative Example Checklist

- [ ] SAME user request as positive
- [ ] sessionMemory: 0-50 chars OR generic placeholder
- [ ] toolContext: <40 chars, just restates action
- [ ] toolContext is STRING (not object)
- [ ] All 7 context fields present (but weak quality)
- [ ] Tool choice demonstrates anti-pattern
- [ ] Parameters show mistakes (hardcoded, assumed)
- [ ] Still valid structure (realistic mistake, not garbage)
- [ ] Single-turn format (no Result in assistant content)

### Label Assignment

```
label: true  → Demonstrates desired behavior
label: false → Demonstrates anti-pattern (realistic mistake)
```

### Interleaving for KTO

After generating all pairs:
1. Shuffle positive examples
2. Shuffle negative examples
3. Interleave: pos[0], neg[0], pos[1], neg[1], ...
4. Result: Perfect True/False/True/False pattern

---

**Version History:**
- 1.0 (2025-11-22): Initial rubric guide with 6 core behaviors
