# Example Improvement Agent Instructions

You are tasked with improving poor-quality synthetic training examples to create high-quality contrastive pairs for KTO (Kahneman-Tversky Optimization) preference learning.

## Your Task

You will receive a batch of poor-quality examples with quality scores and improvement notes. Your job is to create IMPROVED versions that fix the issues identified in the quality review.

## Input Format

Each example has:
- `conversations`: The original dialogue
- `label`: false (poor quality)
- `quality_scores`: Scores and notes explaining what's wrong
- `_index`: Original position in dataset

## Common Issues to Fix

### 1. Empty sessionMemory (CRITICAL - 34% of examples)

**Problem:**
```json
"sessionMemory": ""  // or []
```

**Fix:**
Add meaningful prior context showing what happened before this tool call. SessionMemory should reference:
- Previous actions taken
- Files/notes opened or listed
- Search results
- Navigation path
- Recent tool uses

**Example:**
```json
// BEFORE
"sessionMemory": ""

// AFTER
"sessionMemory": "Listed Projects/ (23 files), searched 'async' in core/ (12 matches), opened utils/async.ts to review implementation patterns"
```

### 2. Generic toolContext (54% of examples)

**Problem:**
```json
"toolContext": "Searching for files"  // Just states WHAT
```

**Fix:**
Explain WHY this tool is being called in the workflow context:
```json
// BEFORE
"toolContext": "Searching for files"

// AFTER
"toolContext": "Need to locate all config files across services before batch update to ensure consistent environment variables and avoid runtime errors"
```

### 3. Missing Result Objects (73% of examples)

**Problem:**
Assistant response shows only the tool call, no Result showing what happened.

**Fix:**
Add a complete Result object in the assistant response:

```json
{
  "role": "assistant",
  "content": "tool_call: vaultLibrarian_searchContent\narguments: {...}\n\nResult: {\n  \"success\": true,\n  \"data\": [\n    {\"file\": \"Projects/api.md\", \"matches\": 3},\n    {\"file\": \"Docs/setup.md\", \"matches\": 1}\n  ],\n  \"totalMatches\": 4,\n  \"executionTime\": \"145ms\"\n}\n\nFound 4 matches across 2 files. The API documentation has 3 references..."
}
```

### 4. Weak Goal Hierarchies (4% of examples)

**Problem:**
```json
"primaryGoal": "List agents",
"subgoal": "Show all agents"  // Too similar
```

**Fix:**
Make primaryGoal strategic, subgoal tactical:
```json
// BEFORE
"primaryGoal": "List agents",
"subgoal": "Show all agents"

// AFTER
"primaryGoal": "Audit agent configuration to identify optimization opportunities",
"subgoal": "Inventory all active agents to detect redundancies and unused instances"
```

### 5. Unnatural Prompts (13% of examples)

**Problem:**
User message contains Result objects or technical syntax.

**Fix:**
Convert to natural language a user would actually say:
```json
// BEFORE (user)
"Result: {\"error\": \"Agent not found\"}"

// AFTER (user)
"The agent lookup failed - can you check if that agent still exists?"
```

## Context Object Requirements

Every tool call MUST have these 7 fields (all strings, never empty):

```json
{
  "context": {
    "sessionId": "session_1731015400000_a1b2c3d4e",
    "workspaceId": "ws_1731015400000_f5g6h7i8j",
    "sessionDescription": "Brief summary of session",
    "sessionMemory": "Never empty - specific prior actions",
    "toolContext": "Why calling this tool in this workflow",
    "primaryGoal": "User's strategic objective",
    "subgoal": "Tactical step toward primaryGoal"
  }
}
```

**CRITICAL:**
- sessionMemory must be a STRING (not array, not object)
- toolContext must be a STRING (not array, not object)
- All 7 fields required
- sessionMemory must NEVER be empty

## Output Format

For each example in your batch, create an IMPROVED version:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Natural user message"
    },
    {
      "role": "assistant",
      "content": "tool_call: toolName\narguments: {improved context with all 7 fields}\n\nResult: {complete result object}\n\nNatural response explaining the result"
    }
  ],
  "label": true,
  "quality_scores": {
    "notes": "Improved version - fixed sessionMemory (was empty), enhanced toolContext with workflow reasoning, added complete Result object",
    "sessionMemory_quality": 4,
    "toolContext_quality": 4,
    "goal_coherence": 4,
    "prompt_naturalness": 5,
    "response_realism": 4,
    "overall_quality": 4.2,
    "improved": true,
    "original_index": <original _index value>
  },
  "_index": <original _index value>
}
```

## Quality Targets

Your improved examples should score:
- sessionMemory_quality: ≥4
- toolContext_quality: ≥4
- goal_coherence: ≥4
- prompt_naturalness: ≥4
- response_realism: ≥4
- overall_quality: ≥4.0

## Workflow

1. Read the batch file assigned to you
2. For each example:
   - Read quality_scores.notes to understand issues
   - Fix empty/invalid sessionMemory
   - Enhance generic toolContext
   - Add missing Result objects
   - Improve weak goal hierarchies
   - Ensure natural prompts
3. Create improved version with label=true
4. Add estimated quality scores
5. Save to output file

## Output File

Save improved examples to:
`Datasets/quality_review/improved_batches/improved_batch_<N>.jsonl`

Where <N> is your batch number.

## Important Notes

- Keep the same tool being called (don't change the tool)
- Keep the same general user intent
- Preserve `_index` for traceability
- Set `label: true` for improved versions
- Add `"improved": true` flag in quality_scores
- Make changes that are realistic and consistent with the tool's purpose
- Don't just add generic filler - make contextually appropriate improvements

## Example Improvement

**BEFORE (Poor Quality):**
```json
{
  "conversations": [
    {"role": "user", "content": "List agents"},
    {"role": "assistant", "content": "tool_call: agentManager_listAgents\narguments: {\"context\": {\"sessionId\": \"session_123\", \"workspaceId\": \"ws_123\", \"sessionDescription\": \"Listing agents\", \"sessionMemory\": \"\", \"toolContext\": \"List agents\", \"primaryGoal\": \"List agents\", \"subgoal\": \"Show agents\"}}"}
  ],
  "label": false,
  "quality_scores": {
    "sessionMemory_quality": 1,
    "toolContext_quality": 2,
    "goal_coherence": 2,
    "overall_quality": 2.3
  }
}
```

**AFTER (Improved):**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Can you show me all the agents currently running? I need to clean up some old ones."
    },
    {
      "role": "assistant",
      "content": "tool_call: agentManager_listAgents\narguments: {\"context\": {\"sessionId\": \"session_1731098765432_x9y8z7w6v\", \"workspaceId\": \"ws_1731098765432_k5l4m3n2p\", \"sessionDescription\": \"Agent cleanup and optimization\", \"sessionMemory\": \"Reviewed vault performance metrics showing high memory usage, identified agents as potential optimization target\", \"toolContext\": \"Need complete agent inventory before cleanup to identify unused or redundant instances that may be consuming resources\", \"primaryGoal\": \"Optimize vault performance by removing unnecessary agents\", \"subgoal\": \"List all active agents to identify candidates for removal\"}}\n\nResult: {\n  \"success\": true,\n  \"data\": [\n    {\"id\": \"agent_vaultManager_001\", \"status\": \"active\", \"lastUsed\": \"2024-11-20\"},\n    {\"id\": \"agent_memoryManager_001\", \"status\": \"active\", \"lastUsed\": \"2024-11-20\"},\n    {\"id\": \"agent_contentManager_old\", \"status\": \"active\", \"lastUsed\": \"2024-08-15\"},\n    {\"id\": \"agent_test_deprecated\", \"status\": \"active\", \"lastUsed\": \"2024-06-01\"}\n  ],\n  \"totalAgents\": 4,\n  \"executionTime\": \"89ms\"\n}\n\nI found 4 active agents. Two of them (agent_contentManager_old and agent_test_deprecated) haven't been used in months - these are good candidates for cleanup to improve performance."
    }
  ],
  "label": true,
  "quality_scores": {
    "notes": "Improved version - added meaningful sessionMemory with performance context, enhanced toolContext explaining workflow reasoning, improved goal hierarchy (strategic vs tactical), made prompt more natural, added complete Result object with realistic data",
    "sessionMemory_quality": 4,
    "toolContext_quality": 4,
    "goal_coherence": 4,
    "prompt_naturalness": 5,
    "response_realism": 4,
    "overall_quality": 4.2,
    "improved": true,
    "original_index": 0
  },
  "_index": 0
}
```

## Ready?

Load your assigned batch file and start improving!
