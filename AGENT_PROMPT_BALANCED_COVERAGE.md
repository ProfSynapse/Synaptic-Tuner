# Agent Task: Add Balanced Tool Coverage to KTO Dataset

## Objective

Add 216 new examples (108 good + 108 bad pairs) to the synthetic dataset to achieve balanced tool coverage across all 47 available tools. This will bring all underrepresented tools to at least 15 good examples each.

## Critical Requirements

1. **Perfect 1:1 Label Ratio:** Every good example (label: true) MUST be paired with exactly one bad example (label: false)
2. **Perfect Interleaving:** Examples MUST alternate throughout the entire file (true, false, true, false...)
3. **Zero Errors in Good Examples:** All true-labeled examples must validate perfectly against tool schemas
4. **Intentional Errors in Bad Examples:** All false-labeled examples should have deliberate errors (wrong tool names, missing params, invalid values)
5. **Schema Compliance:** Use only tools and parameters defined in `tools/tool_schemas.json`

## Files to Read

### 1. Current Dataset
**Path:** `syngen_toolset_v1.0.0_claude.jsonl`
- **Purpose:** Understand existing example patterns and structures
- **What to look for:**
  - ChatML message format (system/user/assistant roles)
  - Tool call format: `tool_call: toolName` followed by `arguments:` with JSON
  - Context object structure (sessionId, workspaceId, sessionDescription, sessionMemory, toolContext, primaryGoal, subgoal)
  - Realistic use case patterns
  - Common workflows and conversational flow

### 2. Tool Schemas
**Path:** `tools/tool_schemas.json`
- **Purpose:** Understand exact parameter requirements for each tool
- **What to look for:**
  - Required vs optional parameters for each tool
  - Parameter data types (string, boolean, object, array)
  - Parameter descriptions and constraints
  - Context object requirements (all tools require this)

### 3. Coverage Analysis Output
**Command:** `python3 analyze_tool_coverage.py syngen_toolset_v1.0.0_claude.jsonl`
- **Purpose:** Verify current coverage before starting
- **What to verify:**
  - Confirm 45/47 tools have examples
  - Verify the exact tools needing more examples
  - Check current counts for each priority tool

## Tools Requiring New Examples

### Priority 1: Never-Used Tools (20 examples = 10 good + 10 bad)

| Tool | Current | Target | Need |
|------|---------|--------|------|
| agentManager_batchExecutePrompt | 0 | 10 | +10 good, +10 bad |
| memoryManager_createWorkspace | 0 | 10 | +10 good, +10 bad |

### Priority 2: Severely Underrepresented (86 examples = 43 good + 43 bad)

| Tool | Current | Target | Need Good | Need Bad |
|------|---------|--------|-----------|----------|
| commandManager_executeCommand | 1 | 10 | +9 | +9 |
| commandManager_listCommands | 2 | 10 | +8 | +8 |
| memoryManager_listWorkspaces | 1 | 10 | +9 | +9 |
| memoryManager_updateState | 1 | 10 | +9 | +9 |
| agentManager_listModels | 2 | 10 | +8 | +8 |

### Priority 3: Underrepresented (110 examples = 55 good + 55 bad)

| Tool | Current | Target | Need Good | Need Bad |
|------|---------|--------|-----------|----------|
| agentManager_executePrompt | 9 | 15 | +6 | +6 |
| agentManager_deleteAgent | 5 | 15 | +10 | +10 |
| agentManager_toggleAgent | 5 | 15 | +10 | +10 |
| agentManager_updateAgent | 8 | 15 | +7 | +7 |
| agentManager_getAgent | 10 | 15 | +5 | +5 |
| contentManager_prependContent | 8 | 15 | +7 | +7 |
| vaultManager_moveFolder | 9 | 15 | +6 | +6 |
| memoryManager_updateWorkspace | 11 | 15 | +4 | +4 |

**TOTAL: 216 examples (108 good + 108 bad)**

## Example Format (ChatML with Tool Calls)

### Good Example Structure (label: true)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant integrated with Obsidian vault management."
    },
    {
      "role": "user",
      "content": "Create a new workspace for the Q1 2024 project"
    },
    {
      "role": "assistant",
      "content": "I'll create a new workspace for your Q1 2024 project.\n\ntool_call: memoryManager_createWorkspace\narguments: {\"context\": {\"sessionId\": \"session_1730000000001_abc123\", \"workspaceId\": \"ws_1730000000001_xyz789\", \"sessionDescription\": \"Creating new project workspace\", \"sessionMemory\": \"User starting Q1 2024 planning\", \"toolContext\": \"Setting up dedicated workspace\", \"primaryGoal\": \"Organize Q1 projects\", \"subgoal\": \"Create workspace\"}, \"name\": \"Q1-2024-Projects\", \"description\": \"Workspace for Q1 2024 project planning and tracking\"}\n\nResult: {\"success\": true, \"workspaceId\": \"ws_1730000000002_q1proj\", \"name\": \"Q1-2024-Projects\", \"created\": \"2024-01-15T10:30:00Z\"}\n\nI've successfully created the 'Q1-2024-Projects' workspace. You can now start organizing your project files here."
    }
  ],
  "label": true
}
```

### Bad Example Structure (label: false)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant integrated with Obsidian vault management."
    },
    {
      "role": "user",
      "content": "Create a workspace for archived projects"
    },
    {
      "role": "assistant",
      "content": "I'll create an archive workspace for you.\n\ntool_call: memoryManager_createWorkspace\narguments: {\"context\": {\"sessionId\": \"session_1730000000002_def456\", \"workspaceId\": \"ws_1730000000002_abc999\"}, \"workspaceName\": \"Archive\", \"type\": \"archive\"}\n\nResult: {\"error\": \"Missing required parameter 'name'\", \"success\": false}"
    }
  ],
  "label": false
}
```

**Note:** Bad example has intentional errors:
- Missing `sessionDescription`, `sessionMemory`, `toolContext`, `primaryGoal`, `subgoal` in context
- Uses `workspaceName` instead of correct parameter `name`
- Uses unexpected parameter `type` not in schema

## Process Steps

### Step 1: Preparation
1. Read `syngen_toolset_v1.0.0_claude.jsonl` to understand format and patterns
2. Read `tools/tool_schemas.json` to understand exact requirements for priority tools
3. Run coverage analysis to confirm starting state
4. Study existing examples of similar tools to maintain consistency

### Step 2: Example Generation Strategy

For each priority tool, create diverse, realistic examples:

#### agentManager_batchExecutePrompt Use Cases:
- Batch image generation with different prompts
- Testing multiple agent responses
- Running series of related prompts
- Parallel execution for comparison
- Batch processing workflows

#### memoryManager_createWorkspace Use Cases:
- New project workspace
- Client-specific workspace
- Temporary experimental workspace
- Topic-based organization workspace
- Archive or historical workspace

#### commandManager_executeCommand Use Cases:
- Running custom Obsidian commands
- Executing vault maintenance commands
- Triggering plugin actions
- Automation workflows

#### commandManager_listCommands Use Cases:
- Discovering available commands
- Checking for specific command existence
- Listing commands for user selection

#### memoryManager_listWorkspaces Use Cases:
- Viewing all available workspaces
- Finding specific workspace
- Workspace selection workflows

#### memoryManager_updateState Use Cases:
- Modifying state metadata
- Updating state description
- Adding information to existing state

#### agentManager Tools Use Cases:
- executePrompt: Single prompt execution
- deleteAgent: Removing unused agents
- toggleAgent: Enabling/disabling agents
- updateAgent: Modifying agent settings
- getAgent: Retrieving agent details
- listModels: Checking available AI models

#### contentManager_prependContent Use Cases:
- Adding headers to existing files
- Inserting timestamps at file start
- Prepending metadata or tags

#### vaultManager_moveFolder Use Cases:
- Reorganizing vault structure
- Moving folders to archive
- Restructuring projects

#### memoryManager_updateWorkspace Use Cases:
- Changing workspace description
- Updating workspace metadata
- Renaming workspace

### Step 3: Create Good Examples (108 total)

For each tool:
1. **Use realistic context:**
   - Generate unique sessionId: `session_[timestamp]_[random]`
   - Generate unique workspaceId: `ws_[timestamp]_[random]`
   - Write descriptive sessionDescription
   - Include relevant sessionMemory
   - Specify clear toolContext, primaryGoal, subgoal

2. **Use correct parameters:**
   - Check schema for required parameters
   - Use exact parameter names from schema
   - Include all required parameters
   - Use appropriate optional parameters

3. **Vary the scenarios:**
   - Different user goals
   - Different vault structures
   - Different workflow stages
   - Different content types

4. **Create realistic results:**
   - Success responses with appropriate data
   - Realistic IDs and timestamps
   - Meaningful confirmation messages

### Step 4: Create Bad Examples (108 total)

For each good example, create a paired bad example with intentional errors:

**Error Types to Use (vary across examples):**
1. **Missing required parameters:**
   - Omit `name`, `id`, or other required fields
   - Leave out required context fields

2. **Wrong parameter names:**
   - Use `path` instead of `filePath`
   - Use `workspaceName` instead of `name`
   - Use `agentId` instead of `id`

3. **Unexpected parameters:**
   - Add `snapshot`, `includeContext`, `tags`, etc. that aren't in schema
   - Use parameters from wrong tools

4. **Invalid tool names:**
   - Use `memoryManager_getWorkspace` instead of `memoryManager_loadWorkspace`
   - Use `agentManager_runPrompt` instead of `agentManager_executePrompt`

5. **Invalid values:**
   - Wrong data types (string instead of boolean)
   - Empty required strings
   - Invalid format for IDs

6. **Incomplete context:**
   - Missing sessionDescription, sessionMemory, etc.
   - Partial context objects

### Step 5: Insert with Perfect Interleaving

**CRITICAL:** You CANNOT simply append examples. You MUST insert them maintaining alternating pattern.

**Insertion Strategy:**
1. Read current dataset line by line
2. Track current label pattern
3. Find appropriate insertion points where pattern is maintained
4. Insert good/bad pairs at multiple points throughout file
5. Ensure final file still alternates perfectly: true, false, true, false...

**Example Insertion Logic:**
```
Current file: [true, false, true, false, ...]
If inserting at line 100 (which is currently false):
- Insert new true example at line 100
- Insert new false example at line 101
- This maintains: [..., true, NEW_TRUE, NEW_FALSE, false, ...]
Wait, that breaks pattern!

CORRECT approach:
- Find location where current = true
- Insert new true AFTER it
- Insert new false AFTER the new true
- Pattern: [..., true, NEW_TRUE, NEW_FALSE, ...]
```

**Recommended approach:**
- Distribute insertions evenly throughout file
- Insert ~20-30 pairs at a time
- Validate interleaving after each batch
- Don't insert all at the end

### Step 6: Validation

After inserting new examples:

1. **Run validator:**
   ```bash
   python3 tools/validate_syngen.py syngen_toolset_v1.0.0_claude.jsonl
   ```

2. **Verify results:**
   - TRUE-labeled examples with errors: 0 (MUST be zero)
   - Total examples: 3194 (was 2978, added 216)
   - True-labeled: 1607 (was 1489, added 108)
   - False-labeled: 1587 (was 1489, added 108)
   - Label ratio: 1.00:1

3. **Run coverage analysis:**
   ```bash
   python3 analyze_tool_coverage.py syngen_toolset_v1.0.0_claude.jsonl
   ```

4. **Verify coverage targets met:**
   - agentManager_batchExecutePrompt: ≥10 good examples
   - memoryManager_createWorkspace: ≥10 good examples
   - All Priority 2 tools: ≥10 good examples
   - All Priority 3 tools: ≥15 good examples

5. **If validation fails:**
   - Review error messages
   - Fix schema violations in good examples
   - Ensure bad examples have intentional, appropriate errors
   - Re-run validation

### Step 7: Quality Check

Manually review a sample of new examples:
- Do they look realistic?
- Do conversations flow naturally?
- Are tool calls appropriate for the user request?
- Are results believable?
- Do bad examples have clear, intentional errors?

## Common Pitfalls to Avoid

1. ❌ **Breaking interleaving:** Don't append all examples at end
2. ❌ **Schema violations in good examples:** Always validate parameter names
3. ❌ **Inconsistent context:** Use proper sessionId/workspaceId format
4. ❌ **Unrealistic scenarios:** Make sure use cases make sense
5. ❌ **Copy-paste errors:** Vary the content, don't just duplicate
6. ❌ **Wrong error types in bad examples:** Errors should be parameter/schema issues, not fictional tool failures
7. ❌ **Missing context fields:** All tools require full context object
8. ❌ **Using removed tools:** Don't use vaultManager_updateContent, vaultLibrarian_createNote, etc.

## Success Criteria

✅ **Validation passes:** 0 errors in true-labeled examples
✅ **Perfect ratio:** Exactly 1.00:1 true:false labels (1607:1587)
✅ **Perfect interleaving:** Entire file alternates true/false with no consecutive duplicates
✅ **Coverage achieved:** All priority tools reach target counts
✅ **Quality examples:** Realistic, diverse, well-formatted examples
✅ **Intentional errors:** Bad examples have appropriate schema violations

## Deliverables

1. Updated `syngen_toolset_v1.0.0_claude.jsonl` with 216 new examples
2. Validation report showing 0 errors
3. Coverage report showing improved distribution
4. Summary of new examples added per tool

## Notes

- **Take your time:** Quality is more important than speed
- **Validate frequently:** Run validator after each batch of insertions
- **Study existing examples:** Match the style and format of the current dataset
- **Maintain diversity:** Use different scenarios, goals, and workflows
- **Follow schema exactly:** Any deviation in good examples will cause validation failure
- **Use Edit tool only:** Do NOT use bash commands to modify the dataset

## Example Workflow

1. Read dataset and schemas
2. Create 10 good examples for agentManager_batchExecutePrompt
3. Create 10 matching bad examples
4. Insert these 20 examples maintaining interleaving
5. Validate
6. Move to next tool
7. Repeat until all 216 examples added
8. Final validation and coverage check

Good luck! Focus on creating high-quality, realistic examples that will help the LLM learn proper tool usage patterns.
