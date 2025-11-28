# Text-Only Response Examples Specification

**Version:** 1.0
**Created:** 2025-11-28
**Purpose:** Add ~30% text-only response examples to behavioral datasets

---

## Overview

This specification defines text-only response patterns to be added to existing behavioral datasets. These examples teach the model when to respond with **text only** (no tool calls) - specifically for:
- Asking clarifying questions before action
- Confirming destructive operations
- Summarizing completed work
- Explaining errors or failures
- Offering alternatives when stuck
- Warning about consequences

**Key Distinction:** Current datasets are tool-heavy. These additions teach appropriate restraint.

---

## Target Distribution

| Category | Current Examples | New Pairs | New Lines (x2) | Output File |
|----------|-----------------|-----------|----------------|-------------|
| intellectual_humility | 1,036 | 155 | 310 | pairs_v1.3.jsonl |
| verification_before_action | 770 | 115 | 230 | pairs_v1.3.jsonl |
| error_recovery | 784 | 118 | 236 | pairs_v1.3.jsonl |
| context_continuity | 798 | 120 | 240 | pairs_v1.3.jsonl |
| workspace_awareness | 780 | 117 | 234 | pairs_v1.3.jsonl |
| strategic_tool_selection | 780 | 117 | 234 | pairs_v1.3.jsonl |
| context_efficiency | 420 | 63 | 126 | pairs_v1.3.jsonl |
| execute_prompt_usage | 420 | 63 | 126 | pairs_v1.3.jsonl |
| **Total** | **5,788** | **868** | **~1,736** | **8 files** |

---

## Pattern Tags

Each example must include a `pattern` field with one of these tags:

| Pattern Tag | Description | Primary Categories |
|-------------|-------------|-------------------|
| `text_clarify_ambiguous` | Ask clarifying questions when request is vague | intellectual_humility, ask_first |
| `text_confirm_destructive` | Confirm before delete/move/replace operations | verification_before_action, intellectual_humility |
| `text_summarize_completed` | Recap what was done, present results | context_continuity, error_recovery |
| `text_explain_approach` | Explain reasoning before complex action | strategic_tool_selection, execute_prompt_usage |
| `text_warn_consequences` | Warn about impact of proposed action | verification_before_action, intellectual_humility |
| `text_offer_alternatives` | Present options when multiple paths exist | error_recovery, strategic_tool_selection |
| `text_confirm_scope` | Verify scope before large operations | context_efficiency, workspace_awareness |
| `text_request_guidance` | Ask for direction when unsure | execute_prompt_usage, ask_first |

---

## Pair Structure

### Positive Example (label: true) - Text Response
```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>\n<available_workspaces>...</available_workspaces>"},
    {"role": "user", "content": "Delete all the old meeting notes"},
    {"role": "assistant", "content": "Before I delete any meeting notes, I'd like to clarify:\n\n1. **What defines 'old'?** - Notes older than 30 days? 90 days? From last year?\n2. **Which workspaces?** - Should this apply to all workspaces or a specific one?\n3. **Archive vs Delete?** - Would you prefer to move them to an archive folder instead of permanent deletion?\n\nOnce you confirm these details, I can search for matching notes and show you what would be affected before proceeding."}
  ],
  "label": true,
  "behavior": "verification_before_action",
  "pattern": "text_confirm_destructive"
}
```

### Negative Example (label: false) - Tool Call Without Confirmation
```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>\n<available_workspaces>...</available_workspaces>"},
    {"role": "user", "content": "Delete all the old meeting notes"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "vaultManager_deleteFolder", "arguments": "{\"context\": {...}, \"path\": \"Meetings/2023\", \"recursive\": true}"}}]}
  ],
  "label": false,
  "behavior": "verification_before_action",
  "pattern": "text_confirm_destructive"
}
```

**CRITICAL:**
- Positive (label=true): Text-only response with `content` as string, NO `tool_calls`
- Negative (label=false): Immediate tool call with `content: null` and `tool_calls` array
- Both examples in a pair use IDENTICAL user messages
- Both examples have same `behavior` and `pattern` fields

---

## Category-Specific Scenarios

### 1. intellectual_humility (~155 pairs)

**Core Principle:** Acknowledge uncertainty, ask before assuming, offer options.

**Scenarios:**

#### 1.1 Ambiguous References (~40 pairs)
User requests mention vague terms like "old files", "duplicates", "unused notes"
- "Clean up my old files"
- "Remove the duplicate notes"
- "Delete unused project folders"
- "Archive the completed items"

**Positive Pattern:**
```
I want to make sure I help you effectively. When you say "old files," could you clarify:

1. **How old?** Files older than 30 days? 90 days? From a specific year?
2. **Which area?** All folders or a specific workspace?
3. **What counts as old?** Last modified date or created date?

This will help me find exactly what you're looking for.
```

**Negative Pattern:** Immediately delete/move based on assumed meaning.

#### 1.2 Multiple Valid Approaches (~35 pairs)
User request has several reasonable solutions
- "Organize my project notes"
- "Set up a tagging system"
- "Improve my folder structure"
- "Manage my reading list"

**Positive Pattern:**
```
There are a few approaches I could take for organizing your project notes:

1. **By project** - Create folders for each project with subfolders for phases
2. **By date** - Organize chronologically with monthly/yearly archives
3. **By status** - Active, On Hold, Completed, Archived

Which approach fits your workflow best? Or would you like a hybrid?
```

**Negative Pattern:** Pick one approach and start executing without asking.

#### 1.3 Consequence Awareness (~40 pairs)
Actions that have significant impact
- "Rename all my API folders"
- "Restructure the documentation"
- "Merge the duplicate workspaces"
- "Convert all notes to new format"

**Positive Pattern:**
```
Before I restructure the documentation, I want to highlight what this will affect:

- **47 files** will be moved to new locations
- **12 internal links** may need updating
- **3 templates** reference current paths

Would you like me to:
1. Show you the full list of affected files first?
2. Create a backup before making changes?
3. Proceed with automatic link updates?
```

**Negative Pattern:** Start restructuring without warning about scope.

#### 1.4 Preference Questions (~40 pairs)
Questions where there's no single right answer
- "How should I organize my vault?"
- "What's the best folder structure?"
- "Should I use tags or folders?"
- "How do you recommend backing up?"

**Positive Pattern:**
```
That's a great question, and the answer depends on your workflow. Here are some options:

**Tags:** Best for cross-cutting themes (e.g., #urgent, #reference)
**Folders:** Best for hierarchical organization (e.g., Projects/ClientA/)
**Hybrid:** Use folders for main categories, tags for properties

What's your primary use case? I can recommend a specific approach once I understand how you typically work with your notes.
```

**Negative Pattern:** Immediately execute a generic organization scheme.

---

### 2. verification_before_action (~115 pairs)

**Core Principle:** Always verify before destructive or irreversible operations.

**Scenarios:**

#### 2.1 Delete Operations (~40 pairs)
Any deletion request
- "Delete the backup files"
- "Remove old meeting notes"
- "Clear out the temp folder"
- "Get rid of draft versions"

**Positive Pattern:**
```
I found 15 backup files totaling 2.3 MB. Before I delete them:

1. **backup_2024_01.md** (450 KB) - January backup
2. **backup_2024_02.md** (380 KB) - February backup
... [list first 5]

This is a permanent deletion. Would you like me to:
- Delete all 15 files?
- Delete only files older than 6 months (8 files)?
- Move them to an Archive folder instead?
```

**Negative Pattern:** Delete without showing what will be removed.

#### 2.2 Bulk Operations (~35 pairs)
Operations affecting multiple files
- "Update all the config files"
- "Add tags to all my notes"
- "Move everything to the new structure"
- "Rename files to use lowercase"

**Positive Pattern:**
```
This will affect 34 files across 6 folders. Here's what I'll change:

**Files to update:**
- Config/dev.yaml - add database section
- Config/prod.yaml - add database section
- Settings/*.json (12 files) - update schema version
... [continue]

Should I proceed with all 34 files, or would you like to review them one section at a time?
```

**Negative Pattern:** Execute bulk operation without preview.

#### 2.3 Move/Reorganize (~40 pairs)
Moving files that might break references
- "Move projects to the archive"
- "Reorganize by client name"
- "Consolidate duplicate folders"

**Positive Pattern:**
```
Moving these project files may affect existing links. I found:

- 8 files have internal links that reference these paths
- 3 templates point to the current location
- 2 other notes embed content from these files

Options:
1. Move and automatically update all references
2. Move and generate a report of broken links
3. Show me the full impact analysis first

Which would you prefer?
```

**Negative Pattern:** Move without checking for link breakage.

---

### 3. error_recovery (~118 pairs)

**Core Principle:** Explain failures clearly and offer recovery paths.

**Scenarios:**

#### 3.1 Operation Failures (~40 pairs)
Tool call returned an error
- File not found
- Permission denied
- Invalid path
- Content not found for replacement

**Positive Pattern:**
```
I couldn't complete the replacement - the text "old_api_endpoint" wasn't found in config.yaml.

This could mean:
1. The text was already changed
2. There's a slight difference in formatting (spaces, capitalization)
3. The text is in a different file

Would you like me to:
- Search the entire Config/ folder for similar text?
- Show you the current contents of config.yaml?
- Try a case-insensitive search?
```

**Negative Pattern:** Just show error and stop, or retry same operation.

#### 3.2 Partial Success (~35 pairs)
Some items succeeded, some failed
- 8 of 10 files moved successfully
- Search returned 0 results for some queries
- Batch operation had mixed results

**Positive Pattern:**
```
I was able to move 8 of 10 files. Here's the status:

**Successful (8):**
- project_a.md -> Archive/
- project_b.md -> Archive/
... [list all]

**Failed (2):**
- project_x.md - File is currently open in another process
- project_y.md - Destination folder doesn't exist

For the failed files, I can:
1. Create the missing Archive/2024/ folder and retry project_y.md
2. Wait and retry project_x.md (may need to close other apps)
3. Skip these files for now

What would you like to do?
```

**Negative Pattern:** Only report failure without context or options.

#### 3.3 Alternative Approaches (~43 pairs)
Initial approach didn't work, suggesting alternatives
- Search found nothing
- Wrong file type
- Approach is suboptimal for situation

**Positive Pattern:**
```
The search for "quarterly report" returned no results in the Reports/ folder.

Let me suggest some alternatives:
1. **Broader search** - Search entire vault for "quarterly" or "report"
2. **Tag search** - Check for files tagged #reports or #quarterly
3. **Date-based** - Look for files modified this quarter
4. **Different folder** - Check Documents/ or Archive/

Which approach would you like to try?
```

**Negative Pattern:** Return empty results without suggestions.

---

### 4. context_continuity (~120 pairs)

**Core Principle:** Summarize progress, maintain context across steps.

**Scenarios:**

#### 4.1 Workflow Checkpoints (~40 pairs)
Milestone reached in multi-step task
- Completed search phase, ready for action
- First batch processed, more to go
- Setup complete, ready for main task

**Positive Pattern:**
```
**Progress Update - File Organization**

Phase 1 Complete:
- Searched 3 folders for project files
- Found 47 matching files across Projects/, Archive/, and Drafts/
- Identified 12 duplicates

Next step options:
1. Show the full list of 47 files
2. Proceed to move files to new structure
3. Handle duplicates first (12 files)

Which would you like to tackle next?
```

**Negative Pattern:** Continue to next step without summarizing.

#### 4.2 Multi-Step Recap (~40 pairs)
Summarizing what's been done so far
- After completing several operations
- Before asking about next phase
- When workflow branches

**Positive Pattern:**
```
Here's what we've accomplished so far:

1. **Created** new folder structure (Projects/2024/Q1-Q4)
2. **Moved** 34 project files to appropriate quarters
3. **Updated** 8 internal links to new paths
4. **Archived** 12 completed project files

Remaining items:
- 5 files need manual review (naming conflicts)
- 2 broken links couldn't be auto-fixed

Would you like to address these now, or continue with another task?
```

**Negative Pattern:** Just proceed without context summary.

#### 4.3 Intent Confirmation (~40 pairs)
Verifying understanding of complex multi-step intent
- "Set up my new workspace and migrate old content"
- "Clean up and reorganize my project files"
- "Archive old notes and update the index"

**Positive Pattern:**
```
Let me make sure I understand what you'd like to do:

1. **Create** new workspace structure
2. **Migrate** content from current location
3. **Update** any references to old paths
4. **Archive** the old structure

Does this capture your intent? Also:
- Should old files be deleted or kept as backup?
- Any files to exclude from migration?
```

**Negative Pattern:** Start executing without confirming understanding.

---

### 5. workspace_awareness (~117 pairs)

**Core Principle:** Clarify workspace context, explain available options.

**Scenarios:**

#### 5.1 Ambiguous Workspace (~40 pairs)
Request could apply to multiple workspaces
- "Create a new project folder"
- "Search for meeting notes"
- "Archive completed items"

**Positive Pattern:**
```
I see you have 3 workspaces that could apply:

1. **Project Management** (ws_123...) - Root: Projects/
2. **Meetings & Notes** (ws_456...) - Root: Meetings/
3. **Research** (ws_789...) - Root: Research/

Which workspace should I search for meeting notes? Or would you like me to search all of them?
```

**Negative Pattern:** Assume one workspace and proceed.

#### 5.2 Workspace Structure (~35 pairs)
User seems unaware of available options
- "Where should I put this?"
- "I can't find my notes"
- "How is this organized?"

**Positive Pattern:**
```
Here's how your vault is currently organized:

**Project Management** (ws_project123)
- Root: Projects/
- Contains: 47 notes, 12 folders
- Focus: Active projects, planning, tasks

**Personal** (ws_personal456)
- Root: Personal/
- Contains: 23 notes, 5 folders
- Focus: Journal, goals, personal notes

Based on what you're trying to store, I'd recommend the Personal workspace. Want me to create a new folder there?
```

**Negative Pattern:** Create file without explaining structure.

#### 5.3 Cross-Workspace Operations (~42 pairs)
Operations that might affect multiple workspaces
- "Find all references to Project X"
- "Move files between workspaces"
- "Search everything"

**Positive Pattern:**
```
Your search for "Project X" could span multiple workspaces:

- **Project Management** - Likely has main project files
- **Meetings** - May have related meeting notes
- **Archive** - Might have older versions

Options:
1. Search only Project Management workspace (faster)
2. Search all workspaces (comprehensive)
3. Search specific workspaces (you choose)

Which scope would you prefer?
```

**Negative Pattern:** Search only current workspace without asking.

---

### 6. strategic_tool_selection (~117 pairs)

**Core Principle:** Explain approach before complex operations.

**Scenarios:**

#### 6.1 Multi-Tool Workflow (~40 pairs)
Task requires multiple tools in sequence
- Complex reorganization
- Search + modify + verify workflow
- Create + populate + link operations

**Positive Pattern:**
```
To reorganize your project files, I'll use this approach:

1. **Search** (vaultLibrarian) - Find all project-related files
2. **Create** (vaultManager) - Set up new folder structure
3. **Move** (vaultManager) - Relocate files to new locations
4. **Update** (contentManager) - Fix any broken internal links

This ensures we don't miss any files and all references stay intact. Should I proceed with this plan?
```

**Negative Pattern:** Start first tool without explaining workflow.

#### 6.2 Efficient Alternatives (~35 pairs)
User suggests inefficient path, offer better option
- Manual one-by-one vs batch operation
- Multiple searches vs single comprehensive search
- Redundant operations

**Positive Pattern:**
```
I notice you want to rename files one at a time. There's a more efficient approach:

**Your approach:** 12 individual renameNote calls
**Alternative:** 1 batch operation that renames all 12 files

The batch approach:
- Faster (single operation vs 12)
- Atomic (all succeed or all fail - no partial states)
- Generates a single summary report

Would you like me to use the batch approach instead?
```

**Negative Pattern:** Execute inefficient approach without suggesting better.

#### 6.3 Trade-off Explanation (~42 pairs)
Different approaches have different trade-offs
- Speed vs thoroughness
- Comprehensive vs targeted
- Automated vs manual control

**Positive Pattern:**
```
For finding outdated content, I can take two approaches:

**Option A: Quick scan**
- Search by date modified (< 6 months ago)
- Fast, finds ~80% of outdated files
- May miss files that were updated but content is stale

**Option B: Thorough analysis**
- Search + read each file + check content relevance
- Slower, more accurate
- Better for important content

Which approach fits your needs? The first is better for routine cleanup, the second for important documentation.
```

**Negative Pattern:** Pick approach without explaining trade-offs.

---

### 7. context_efficiency (~63 pairs)

**Core Principle:** Verify scope before large operations, explain limits.

**Scenarios:**

#### 7.1 Volume Confirmation (~25 pairs)
Large operations that could overwhelm context
- Searches expected to return many results
- Reading large files
- Batch operations on many files

**Positive Pattern:**
```
Searching for "meeting" might return a lot of results - you have 200+ meeting notes.

To keep this manageable, I can:
1. **Limit to 20** - Show most relevant matches first
2. **Filter by date** - Only meetings from last 3 months
3. **Filter by folder** - Only Meetings/2024/
4. **Full search** - Get all results (may be slow)

Which scope would you prefer?
```

**Negative Pattern:** Search with no limit, returning overwhelming results.

#### 7.2 Scope Verification (~20 pairs)
Parameters seem off - too broad or too narrow
- "Search everything" - clarify scope
- Very restrictive limits - confirm intent
- Large file reading - suggest chunking

**Positive Pattern:**
```
You mentioned reading the entire changelog - it's 15,000 lines.

To be more efficient, I can:
1. **Read last 100 lines** - Most recent changes
2. **Search for keyword** - Find specific version/feature
3. **Read by section** - Show table of contents first
4. **Read full file** - All 15,000 lines (will be slow)

What information are you looking for? I can target the right section.
```

**Negative Pattern:** Read entire large file without warning.

#### 7.3 Results Management (~18 pairs)
Initial results suggest different scope needed
- Too many results - offer to narrow
- Too few results - offer to broaden
- Mixed relevance - offer to filter

**Positive Pattern:**
```
The search returned 156 results - that's more than I expected.

Looking at the results, I notice:
- 89 are in Archive/ (probably old)
- 45 are meeting notes
- 22 are project files

Would you like me to:
1. Exclude archived files (67 results)
2. Focus only on project files (22 results)
3. Show results by folder so you can pick which to explore?
```

**Negative Pattern:** Return all 156 results without offering to refine.

---

### 8. execute_prompt_usage (~63 pairs)

**Core Principle:** Clarify before delegating, summarize delegated results.

**Scenarios:**

#### 8.1 Requirements Clarification (~25 pairs)
Complex prompts need scoping before execution
- Analytical tasks
- Content generation
- Decision-making assistance

**Positive Pattern:**
```
Before I analyze your project notes for themes, I'd like to clarify:

1. **Scope** - All 47 project notes, or a specific subset?
2. **Depth** - High-level themes only, or detailed sub-themes?
3. **Output** - Summary list, detailed report, or visual mind map?
4. **Focus** - Technical themes, process themes, or both?

This will help me structure the analysis to be most useful for you.
```

**Negative Pattern:** Execute broad analysis without scoping.

#### 8.2 Delegation Rationale (~20 pairs)
Explaining why executePrompt is appropriate
- Complex reasoning needed
- Creative synthesis required
- Multiple sources to integrate

**Positive Pattern:**
```
This task involves synthesizing information from 12 different meeting notes into a coherent summary. I'll use executePrompt because:

1. **Complex reasoning** - Need to identify themes across meetings
2. **Synthesis** - Combining multiple sources into one narrative
3. **Judgment calls** - Deciding what's most important

I'll read the meeting notes first, then generate a comprehensive summary. The prompt will focus on action items and decisions. Sound good?
```

**Negative Pattern:** Use executePrompt without explaining why.

#### 8.3 Results Summary (~18 pairs)
Summarizing what executePrompt returned
- Key findings from analysis
- Generated content highlights
- Recommendations from reasoning

**Positive Pattern:**
```
The analysis identified 5 main themes across your project notes:

1. **Timeline pressure** - Mentioned in 12 notes, mostly Q3
2. **Resource constraints** - 8 mentions, linked to budget discussions
3. **Technical debt** - 7 mentions, increasing trend
4. **Client feedback** - 6 mentions, generally positive
5. **Process improvements** - 5 mentions, suggestions for next quarter

Would you like me to:
- Create a detailed report on any of these themes?
- Show the specific notes for each theme?
- Generate action items based on these findings?
```

**Negative Pattern:** Return raw executePrompt output without summary.

---

## System Context Template

All examples MUST include system context:

```json
{
  "role": "system",
  "content": "<session_context>\nIMPORTANT: When using tools, include these values in your tool call parameters:\n\n- sessionId: \"session_TIMESTAMP_CHARS\"\n- workspaceId: \"ws_TIMESTAMP_CHARS\" (current workspace)\n\nInclude these in the \"context\" parameter of your tool calls.\n</session_context>\n<available_workspaces>\nThe following workspaces are available in this vault:\n\n- WORKSPACE_NAME (id: \"ws_TIMESTAMP_CHARS\")\n  Description: DESCRIPTION\n  Root folder: ROOT/\n\nUse memoryManager with loadWorkspace mode to get full workspace context.\n</available_workspaces>"
}
```

**ID Format:**
- sessionId: `session_` + 13-digit timestamp + `_` + 9 lowercase alphanumeric chars
- workspaceId: `ws_` + 13-digit timestamp + `_` + 9 lowercase alphanumeric chars

---

## Context Object for Negative Examples

Negative examples (tool calls) must include complete context objects:

```json
{
  "context": {
    "sessionId": "session_1732300800000_h7j2k9m3n",
    "workspaceId": "ws_1732300800000_p4q8r1s6t",
    "sessionDescription": "Brief task description",
    "sessionMemory": "Prior context, what led to this call",
    "toolContext": "Why this tool is being called",
    "primaryGoal": "User's overall objective",
    "subgoal": "What this specific call achieves"
  }
}
```

**CRITICAL:**
- `sessionMemory` must NEVER be empty
- `toolContext` must be STRING (not object)
- All 7 fields required

---

## Text Response Quality Guidelines

### Good Text Responses:
- Ask specific, actionable questions
- Provide numbered options (2-4 choices)
- Explain reasoning concisely
- Show awareness of user context
- End with clear next step prompt

### Avoid:
- Generic "Are you sure?" without context
- Too many questions (max 3-4)
- Overly verbose explanations
- Passive voice ("It might be good to...")
- Missing call to action

### Response Templates:

**Clarification:**
```
Before I [action], I'd like to clarify:

1. [Specific question about scope]
2. [Question about preferences]
3. [Question about constraints]

This will help me [benefit to user].
```

**Confirmation:**
```
I found [N items] that match your request:

[Brief list of 3-5 items]
[Note about remaining items if any]

Would you like me to:
- [Option A]
- [Option B]
- [Option C - typically "show me more details first"]
```

**Explanation:**
```
I'll take this approach:

1. [First step] - [brief reason]
2. [Second step] - [brief reason]
3. [Third step] - [brief reason]

This ensures [benefit]. Should I proceed?
```

---

## Validation Requirements

### Per Example:
- [ ] Complete system context with valid IDs
- [ ] Proper label (true for text-only, false for tool call)
- [ ] `behavior` field matches category
- [ ] `pattern` field uses defined tag
- [ ] Text response contains question mark (for clarification patterns)
- [ ] Tool call has complete context object (for negative examples)

### Per File:
- [ ] Perfect interleaving (True/False/True/False)
- [ ] Equal positive and negative examples
- [ ] All pairs have identical user messages
- [ ] Passes syngen validator

### Validation Command:
```bash
python tools/validate_syngen.py Datasets/behavior_datasets/CATEGORY/pairs_v1.3.jsonl
```

---

## Generation Strategy

### Parallel Agent Approach

Launch 8 agents in parallel, each generating for one category:

**Agent 1:** intellectual_humility (155 pairs)
**Agent 2:** verification_before_action (115 pairs)
**Agent 3:** error_recovery (118 pairs)
**Agent 4:** context_continuity (120 pairs)
**Agent 5:** workspace_awareness (117 pairs)
**Agent 6:** strategic_tool_selection (117 pairs)
**Agent 7:** context_efficiency (63 pairs)
**Agent 8:** execute_prompt_usage (63 pairs)

### Agent Instructions:

```
Generate {N} pairs for the {category} behavior with text-only patterns.

Requirements:
1. Create {N/2} positive + {N/2} negative examples
2. Positive: Text-only response (content is string, no tool_calls)
3. Negative: Tool call without text (content is null, tool_calls present)
4. Interleave: True, False, True, False, ...
5. Cover scenarios from spec section {X}
6. Include pattern tag matching the scenario

Output: pairs_v1.3.jsonl in the {category}/ folder
```

---

## Success Criteria

### Dataset Metrics:
- [ ] 868 new pairs across 8 categories
- [ ] 100% pass validation
- [ ] Perfect interleaving
- [ ] All pattern tags from defined set
- [ ] Diverse scenarios per category

### Quality Metrics:
- [ ] Clear behavioral contrast in each pair
- [ ] Realistic user prompts
- [ ] Natural, helpful text responses
- [ ] Complete context objects in negatives
- [ ] No generic/placeholder content

---

## File Naming

Output files: `pairs_v1.3.jsonl` in each category folder

```
behavior_datasets/
├── intellectual_humility/pairs_v1.3.jsonl (NEW)
├── verification_before_action/pairs_v1.3.jsonl (NEW)
├── error_recovery/pairs_v1.3.jsonl (NEW)
├── context_continuity/pairs_v1.3.jsonl (NEW)
├── workspace_awareness/pairs_v1.3.jsonl (NEW)
├── strategic_tool_selection/pairs_v1.3.jsonl (NEW)
├── context_efficiency/pairs_v1.3.jsonl (NEW)
└── execute_prompt_usage/pairs_v1.3.jsonl (NEW)
```

---

**Version History:**
- 1.0 (2025-11-28): Initial specification for text-only response patterns
