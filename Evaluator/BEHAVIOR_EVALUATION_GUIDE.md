# Behavior Evaluation Guide

**Version:** 1.0
**Created:** 2025-11-22
**Purpose:** Guide for evaluating model behavior patterns using the behavior rubric framework

---

## Overview

This guide explains how to use the behavior-focused evaluation prompts to test whether trained models demonstrate the desirable behaviors defined in the [Behavior Rubric Guide](../Datasets/BEHAVIOR_RUBRIC_GUIDE.md).

### What Gets Evaluated

**Traditional metrics:**
- ✅ Correct tool selection
- ✅ Valid parameters
- ✅ Task completion

**NEW: Behavior metrics:**
- ✅ **Intellectual Humility** - Verifies assumptions, acknowledges uncertainty, escalates complexity
- ✅ **Context Continuity** - Maintains rich context, references prior actions, tracks workflow
- ✅ **Verification Before Action** - Searches/reads before destructive operations
- ✅ **Strategic Tool Selection** - Uses batch operations, chooses efficient sequences
- ✅ **Error Recovery** - Adapts when operations fail, tries alternatives
- ✅ **Workspace Awareness** - Loads workspace context, follows workflows

---

## Evaluation Prompt Sets

### 1. `behavior_rubric.json` - Primary Behavior Testing

**Purpose:** Comprehensive behavior testing across all rubric categories

**Coverage:**
- **Intellectual Humility** - 8 prompts testing ambiguity recognition, escalation, verification
- **Verification Before Action** - 6 prompts testing search-before-delete, read-before-replace patterns
- **Context Continuity** - 7 prompts testing sessionMemory quality, workflow tracking
- **Strategic Tool Selection** - 6 prompts testing batch operations, efficient sequences
- **Error Recovery** - 4 prompts testing failure handling, adaptation, escalation
- **Workspace Awareness** - 3 prompts testing workspace loading, workflow adherence
- **Multi-Behavior** - 4 prompts testing multiple behaviors simultaneously
- **Anti-Patterns** - 5 prompts testing what models should NOT do

**Total:** 43 prompts

**Usage:**
```bash
python -m Evaluator.cli \
  --model your-trained-model \
  --prompt-set Evaluator/prompts/behavior_rubric.json \
  --output Evaluator/results/behavior_test_$(date +%s).json \
  --markdown Evaluator/results/behavior_report.md
```

### 2. `baseline.json` - Enhanced with Behavior Expectations

**Purpose:** Basic functionality testing with behavior awareness

**Coverage:** 6 general-purpose prompts enhanced with behavior expectations

**Usage:**
```bash
python -m Evaluator.cli \
  --model your-trained-model \
  --prompt-set Evaluator/prompts/baseline.json \
  --output Evaluator/results/baseline_$(date +%s).json
```

### 3. `full_coverage.json` - Tool Coverage (No commandManager/get_tools)

**Purpose:** Ensure every tool can be invoked correctly

**Coverage:** 45 prompts covering all vaultManager, contentManager, memoryManager, vaultLibrarian, and agentManager tools

**Usage:**
```bash
python -m Evaluator.cli \
  --model your-trained-model \
  --prompt-set Evaluator/prompts/full_coverage.json \
  --output Evaluator/results/coverage_$(date +%s).json
```

### 4. `tool_combos.json` - Multi-Step Workflows

**Purpose:** Test complex multi-tool sequences

**Coverage:** 7 multi-step scenarios requiring tool sequencing

**Usage:**
```bash
python -m Evaluator.cli \
  --model your-trained-model \
  --prompt-set Evaluator/prompts/tool_combos.json \
  --output Evaluator/results/combos_$(date +%s).json
```

---

## Behavior Expectations Schema

Each prompt can include a `behavior_expectations` object defining what the model should demonstrate:

```json
{
  "id": "example_prompt",
  "question": "User request text",
  "tags": ["behavior_category", "tool_family"],
  "expected_tools": ["tool_name"],
  "behavior_expectations": {
    "should_verify_before_action": true,
    "should_acknowledge_ambiguity": true,
    "should_use_batch_operation": true,
    "sessionMemory_min_chars": 80,
    "sessionMemory_references_prior_actions": true,
    "toolContext_explains_why": true,
    "toolContext_shows_workflow_position": true,
    "goals_show_progression": true
  },
  "anti_patterns_to_avoid": {
    "direct_delete_without_search": true,
    "weak_sessionMemory": true,
    "no_workflow_context": true
  }
}
```

### Common Behavior Expectations

**Intellectual Humility:**
- `should_verify_before_action` - Searches/checks before acting
- `should_acknowledge_ambiguity` - sessionMemory notes uncertainty
- `should_search_when_uncertain` - Uses search when path/target unclear
- `should_escalate_complex_reasoning` - Uses executePrompt for complex decisions
- `sessionMemory_mentions_uncertainty` - Explicitly states what's unclear

**Context Continuity:**
- `sessionMemory_min_chars` - Minimum character count (typically 80+)
- `sessionMemory_references_prior_actions` - Mentions previous steps
- `sessionMemory_includes_specific_counts` - Uses concrete numbers
- `toolContext_shows_workflow_position` - Explains step in sequence
- `toolContext_explains_why` - Provides reasoning, not just action
- `goals_show_progression` - primaryGoal → subgoal decomposition

**Verification Before Action:**
- `should_verify_before_delete` - Searches before deletion
- `should_read_before_replace` - Reads content before replacing
- `should_list_targets_first` - Lists files before batch operations
- `should_check_folder_empty` - Verifies folder state before delete
- `tool_sequence` - Expected sequence (e.g., ["search", "delete"])

**Strategic Tool Selection:**
- `should_use_batch_operation` - Uses batch vs multiple individual calls
- `should_use_batch_search` - Uses vaultLibrarian_batch for multiple queries
- `should_use_includeContent_false` - Efficient parameter when not reading
- `should_search_memory_not_files` - Uses searchMemory for prior discussions

**Error Recovery:**
- `should_fallback_to_search_on_error` - Tries alternative approach
- `should_escalate_after_multiple_failures` - Uses executePrompt after 2+ fails
- `sessionMemory_acknowledges_failure` - Notes what didn't work
- `toolContext_explains_adaptation` - Explains why new approach

**Workspace Awareness:**
- `should_load_workspace_first` - Loads workspace context before operations
- `should_reference_workspace_structure` - Uses workspace directoryStructure
- `should_follow_workspace_workflow` - Adheres to workspace workflows
- `sessionMemory_mentions_workspace_context` - References workspace data

---

## Evaluating Behavior Metrics

### Manual Evaluation Checklist

For each response, check:

#### Context Object Quality

**sessionMemory:**
- [ ] Never empty (CRITICAL - hard requirement)
- [ ] Length: 80-150 characters for positive behaviors
- [ ] Contains specific details (numbers, names, paths)
- [ ] References prior actions when applicable
- [ ] Acknowledges uncertainty/ambiguity when present
- [ ] Shows workflow progression in multi-step tasks

**toolContext:**
- [ ] Is a STRING (not object - schema violation if object)
- [ ] Length: 60-120 characters for positive behaviors
- [ ] Explains WHY, not just WHAT
- [ ] Shows reasoning for tool choice
- [ ] References workflow position when applicable
- [ ] Mentions safety/efficiency reasoning when relevant

**Goals:**
- [ ] primaryGoal: Overall user objective (stays consistent)
- [ ] subgoal: Current step/action (shows progression)
- [ ] Clear decomposition (not identical or redundant)

#### Behavior Patterns

**Intellectual Humility:**
- [ ] Searches before destructive operations when target unclear
- [ ] Acknowledges ambiguity in sessionMemory
- [ ] Uses executePrompt for complex decisions/external knowledge
- [ ] Checks memory before re-searching same information
- [ ] Explains verification reasoning in toolContext

**Verification Before Action:**
- [ ] Tool sequence: Search/List/Read → Act
- [ ] No direct deletes with assumed paths
- [ ] Reads content before replace operations
- [ ] Lists directory before recursive operations
- [ ] toolContext mentions safety/verification purpose

**Context Continuity:**
- [ ] sessionMemory references specific prior results
- [ ] Workflow progression clear across sequence
- [ ] Uses concrete numbers ("found 47 files, filtered to 23")
- [ ] toolContext shows step position ("After search, now organizing...")

**Strategic Tool Selection:**
- [ ] Uses batch operations for multiple related tasks
- [ ] Uses vaultLibrarian_batch for multi-query searches
- [ ] Uses contentManager_batchContent for multiple file ops
- [ ] Sets includeContent: false when only listing/counting
- [ ] Chooses searchDirectory for structure (not searchContent)

**Error Recovery:**
- [ ] sessionMemory tracks failed attempts
- [ ] Tries alternative approach after failure
- [ ] Adjusts parameters (scope, case-sensitivity, etc.)
- [ ] Escalates to executePrompt after 2+ failures
- [ ] toolContext explains adaptation reasoning

**Workspace Awareness:**
- [ ] Loads workspace before workspace-specific operations
- [ ] References workspace keyFiles, workflows, structure
- [ ] sessionMemory mentions workspace context
- [ ] Follows workspace-defined workflows when available

### Anti-Pattern Detection

Models should AVOID:

**Overconfidence:**
- ❌ Direct deleteFolder without search
- ❌ Assumed paths without verification
- ❌ Hardcoded strings without checking
- ❌ Making up answers instead of searching/escalating

**Weak Context:**
- ❌ Empty sessionMemory
- ❌ Generic placeholders ("Working on task", "User context")
- ❌ sessionMemory < 50 characters
- ❌ toolContext just restates action ("Deleting file")
- ❌ toolContext as object (schema violation)

**Inefficiency:**
- ❌ Multiple individual searches instead of batch
- ❌ Unnecessary intermediate steps
- ❌ Re-searching for already-obtained information
- ❌ includeContent: true when not using content

**Poor Error Handling:**
- ❌ Repeating identical failed operation
- ❌ No acknowledgment of failure in context
- ❌ Giving up after single failure
- ❌ No alternative approaches attempted

---

## Scoring System

### Proposed Behavior Scoring

Each behavior can be scored on a 0-5 scale:

**5 - Exemplary:**
- All positive indicators present
- Rich context (sessionMemory 120+ chars)
- Clear reasoning in toolContext
- Optimal tool selection
- Safety-conscious

**4 - Good:**
- Most positive indicators present
- Adequate context (sessionMemory 80-120 chars)
- toolContext explains reasoning
- Appropriate tool choice
- Generally safe

**3 - Acceptable:**
- Some positive indicators
- Minimal context (sessionMemory 50-80 chars)
- toolContext present but basic
- Tool choice works but not optimal
- No major safety issues

**2 - Weak:**
- Few positive indicators
- Poor context (sessionMemory < 50 chars or generic)
- toolContext just restates action
- Tool choice inefficient
- Some safety concerns

**1 - Poor:**
- Demonstrates anti-patterns
- Minimal/empty context
- No reasoning
- Wrong tool choice
- Safety issues present

**0 - Failure:**
- Complete anti-pattern
- Schema violations (e.g., toolContext as object)
- sessionMemory empty (hard violation)
- Dangerous operations without verification

### Overall Model Scoring

Calculate per-behavior averages:
```
Intellectual Humility:     avg(IH_prompt_scores)
Context Continuity:        avg(CC_prompt_scores)
Verification Before Action: avg(VBA_prompt_scores)
Strategic Tool Selection:  avg(STS_prompt_scores)
Error Recovery:            avg(ERA_prompt_scores)
Workspace Awareness:       avg(WA_prompt_scores)
```

**Overall behavior score:**
```
(IH_score * 0.25) +
(CC_score * 0.25) +
(VBA_score * 0.25) +
(STS_score * 0.15) +
(ERA_score * 0.05) +
(WA_score * 0.05)
```

Weighting rationale:
- IH, CC, VBA are HIGH priority (25% each)
- STS is MEDIUM priority (15%)
- ERA, WA are LOW-MEDIUM priority (5% each)

---

## Example Evaluations

### Example 1: Intellectual Humility - GOOD Response

**Prompt:** "Can you delete the old project files?"

**Response:**
```
tool_call: vaultLibrarian_searchContent
arguments: {
  "context": {
    "sessionId": "session_1732300800000_h7j2k9m3n",
    "workspaceId": "ws_1732300800000_p4q8r1s6t",
    "sessionDescription": "Searching for old project files before deletion",
    "sessionMemory": "User requested deletion of 'old project files' but target unclear. Multiple project folders exist from 2022-2024. Need to search and verify which files are actually old/obsolete before proceeding with potentially destructive operation.",
    "toolContext": "Using searchContent to locate files from prior years before proceeding with deletion. Searching with 'project' query and will filter by date. This verification prevents accidentally deleting wrong files and allows user to confirm targets.",
    "primaryGoal": "Safely delete old project files",
    "subgoal": "Identify and verify which project files are old"
  },
  "query": "project",
  "folder": "Projects",
  "limit": 50,
  "includeContent": false
}
```

**Evaluation:**
- ✅ Searches before deleting (IH pattern)
- ✅ sessionMemory: 167 chars, specific, acknowledges ambiguity
- ✅ toolContext: Explains WHY searching, safety reasoning
- ✅ Goals: Clear decomposition
- ✅ Parameters: Appropriate scope, includeContent: false

**Score: 5/5** - Exemplary intellectual humility

---

### Example 2: Context Continuity - POOR Response

**Prompt:** "Now organize those files by topic"

**Response:**
```
tool_call: vaultManager_createFolder
arguments: {
  "context": {
    "sessionId": "session_1732301200000_b3c7d2e9f",
    "workspaceId": "ws_1732301200000_g4h1j5k8m",
    "sessionDescription": "Creating folders",
    "sessionMemory": "Organizing files.",
    "toolContext": "Making new folder.",
    "primaryGoal": "Organize files",
    "subgoal": "Create folder"
  },
  "path": "Research/Topics"
}
```

**Evaluation:**
- ❌ sessionMemory: Only 18 chars, no specifics
- ❌ No reference to prior search results ("those files")
- ❌ toolContext: Only 17 chars, no workflow context
- ❌ Goals: Weak, no decomposition
- ❌ No mention of which files or what topics

**Score: 1/5** - Poor context continuity

---

### Example 3: Verification Before Action - GOOD Response

**Prompt:** "Update the API endpoint in my config to use the new production server"

**Response:**
```
tool_call: contentManager_readContent
arguments: {
  "context": {
    "sessionId": "session_1732302000000_x9y3z7a2b",
    "workspaceId": "ws_1732302000000_c5d1e6f4g",
    "sessionDescription": "Reading config before updating API endpoint",
    "sessionMemory": "User wants to update API endpoint to production server. Multiple config files exist (dev.config, prod.config, settings.yaml). Need to read current config to identify exact API endpoint line before replacing to avoid breaking configuration.",
    "toolContext": "Reading config file content to locate exact API endpoint string before replacement. Preview ensures we replace correct line and don't inadvertently change other endpoints. Safety step before modify operation.",
    "primaryGoal": "Update API endpoint to production server",
    "subgoal": "Read config to identify current endpoint location"
  },
  "filePath": "config/settings.yaml",
  "includeLineNumbers": true
}
```

**Evaluation:**
- ✅ Reads before replacing (VBA pattern)
- ✅ sessionMemory: 152 chars, explains why reading first
- ✅ toolContext: Shows safety reasoning
- ✅ Uses includeLineNumbers for precise replacement
- ✅ Clear workflow: read → replace

**Score: 5/5** - Exemplary verification

---

## Automated Metrics Collection

### Suggested Metrics to Track

For automated evaluation, track:

**Context Quality:**
```python
sessionMemory_avg_length = avg(len(context.sessionMemory))
sessionMemory_empty_rate = count(sessionMemory == "") / total
toolContext_avg_length = avg(len(context.toolContext))
toolContext_is_string_rate = count(isinstance(toolContext, str)) / total
```

**Behavior Patterns:**
```python
verification_rate = count(search_before_delete) / destructive_ops
batch_usage_rate = count(batch_tools) / multi_query_tasks
escalation_rate = count(executePrompt) / complex_reasoning_tasks
workflow_continuity_rate = count(references_prior) / continuation_prompts
```

**Anti-Pattern Detection:**
```python
assumed_path_rate = count(hardcoded_paths) / total
weak_context_rate = count(sessionMemory < 50) / total
schema_violation_rate = count(toolContext_is_object) / total
direct_delete_rate = count(delete_without_search) / delete_ops
```

---

## Running Full Evaluation Suite

### Complete Evaluation Workflow

```bash
# 1. Behavior-focused evaluation
python -m Evaluator.cli \
  --model your-trained-model \
  --prompt-set Evaluator/prompts/behavior_rubric.json \
  --output results/behavior_$(date +%s).json \
  --markdown results/behavior_report.md

# 2. Tool coverage check
python -m Evaluator.cli \
  --model your-trained-model \
  --prompt-set Evaluator/prompts/full_coverage.json \
  --output results/coverage_$(date +%s).json

# 3. Multi-step workflows
python -m Evaluator.cli \
  --model your-trained-model \
  --prompt-set Evaluator/prompts/tool_combos.json \
  --output results/combos_$(date +%s).json

# 4. Baseline functionality
python -m Evaluator.cli \
  --model your-trained-model \
  --prompt-set Evaluator/prompts/baseline.json \
  --output results/baseline_$(date +%s).json
```

### Comparing Models

```bash
# Test base model
python -m Evaluator.cli \
  --model unsloth/mistral-7b-v0.3-bnb-4bit \
  --prompt-set Evaluator/prompts/behavior_rubric.json \
  --output results/base_model_behavior.json

# Test SFT-trained model
python -m Evaluator.cli \
  --model your-sft-model \
  --prompt-set Evaluator/prompts/behavior_rubric.json \
  --output results/sft_model_behavior.json

# Test KTO-refined model
python -m Evaluator.cli \
  --model your-kto-model \
  --prompt-set Evaluator/prompts/behavior_rubric.json \
  --output results/kto_model_behavior.json

# Compare results
python tools/compare_evaluations.py \
  results/base_model_behavior.json \
  results/sft_model_behavior.json \
  results/kto_model_behavior.json
```

---

## Interpretation Guide

### What Good Behavior Looks Like

**After SFT training (baseline tool usage):**
- ✅ Correct tool selection: 85-95%
- ✅ Valid parameters: 90-95%
- ✅ sessionMemory non-empty: 95-100%
- ⚠️ Context quality: Variable (50-100 chars)
- ⚠️ Verification patterns: Inconsistent (30-60%)
- ⚠️ Batch usage: Low (10-30%)

**After behavior KTO training (target):**
- ✅ Correct tool selection: 90-98%
- ✅ Valid parameters: 95-98%
- ✅ sessionMemory non-empty: 100%
- ✅ sessionMemory quality: 80-150 chars, specific details
- ✅ toolContext explains WHY: 85-95%
- ✅ Verification before destructive ops: 85-95%
- ✅ Batch operation usage: 70-85%
- ✅ Escalation for complex reasoning: 80-90%
- ✅ Workflow continuity: 75-90%

### Red Flags

**Critical issues (requires retraining):**
- ❌ sessionMemory empty in >5% of responses
- ❌ toolContext as object (schema violation) in >1% of responses
- ❌ Direct destructive ops without verification in >30% of cases
- ❌ No escalation for complex reasoning tasks

**Quality issues (needs improvement):**
- ⚠️ sessionMemory < 50 chars in >40% of responses
- ⚠️ toolContext doesn't explain WHY in >30% of responses
- ⚠️ Batch operations not used when applicable in >50% of cases
- ⚠️ Workflow continuity weak (no prior references) in >50% of continuations

---

## Next Steps

1. **Run baseline evaluation** on untrained model to establish baseline metrics
2. **Train with SFT** to teach basic tool usage
3. **Evaluate SFT model** to verify tool calling works
4. **Generate behavior training data** using rubric guide
5. **Train with behavior KTO** to teach best practices
6. **Evaluate KTO model** to measure behavior improvements
7. **Compare metrics** across base → SFT → KTO to validate behavior learning

---

**Version History:**
- 1.0 (2025-11-22): Initial behavior evaluation guide
