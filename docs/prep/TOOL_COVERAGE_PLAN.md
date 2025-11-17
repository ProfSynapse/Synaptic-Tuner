# Tool Coverage Improvement Plan

**Generated:** 2025-11-13
**Dataset:** syngen_toolset_v1.0.0_claude.jsonl
**Current Status:** 45/47 tools have examples (95.7% coverage)

## Current Coverage Summary

- **Total tools available:** 47
- **Tools with examples:** 45 (95.7%)
- **Tools never used:** 2 (4.3%)
- **Total good examples:** 1489
- **Total tool calls in good examples:** 3087

## Tools Never Used (Priority: HIGH)

### 1. agentManager_batchExecutePrompt
- **Manager:** agentManager
- **Current usage:** 0 examples
- **Target:** Add 5-10 examples
- **Use cases:**
  - Running multiple prompts across different agents
  - Batch testing multiple image generation prompts
  - Executing series of related prompts sequentially
  - Parallel prompt execution for comparison

### 2. memoryManager_createWorkspace
- **Manager:** memoryManager
- **Current usage:** 0 examples
- **Target:** Add 5-10 examples
- **Use cases:**
  - Creating new project workspace
  - Setting up workspace for new client
  - Initializing workspace with specific structure
  - Creating temporary workspace for experimentation

## Coverage Distribution Analysis

### Well-Covered Tools (>100 calls in good examples)
1. **contentManager_readContent** - 114 calls ✓
2. **memoryManager_loadWorkspace** - 233 calls ✓ (most used!)
3. **vaultLibrarian_searchContent** - 162 calls ✓
4. **memoryManager_listSessions** - 131 calls ✓

### Moderately-Covered Tools (20-100 calls)
- Most contentManager tools (createContent: 67, replaceContent: 46, appendContent: 26)
- memoryManager session/state tools (createState: 83, loadSession: 72)
- vaultLibrarian tools (batch: 23, searchMemory: 27)
- vaultManager tools (createFolder: 49, moveNote: 48, duplicateNote: 28)

### Lightly-Covered Tools (1-20 calls)
- **agentManager tools** - Most have <30 calls
  - generateImage: 26 ✓
  - createAgent: 20 ✓
  - executePrompt: 9 (could use more)
  - listAgents: 16 ✓
  - Others: <15 calls each

- **commandManager tools** - Very low usage
  - executeCommand: 1 call only ⚠️
  - listCommands: 2 calls only ⚠️

- **vaultManager specialized tools**
  - editFolder: 11 ✓
  - moveFolder: 9 (could use more)
  - openNote: 17 ✓ (but 107 bad examples - very error-prone?)

- **memoryManager workspace tools**
  - updateWorkspace: 11 (could use more)
  - listWorkspaces: 1 only ⚠️
  - updateState: 1 only ⚠️

## Recommended Priorities

### PRIORITY 1 (Immediate): Never-Used Tools (2 tools)
**Target:** Add 5-10 good examples each (10-20 total examples)

| Tool | Target Examples | Rationale |
|------|----------------|-----------|
| agentManager_batchExecutePrompt | 5-10 | Unique batch capability, important workflow tool |
| memoryManager_createWorkspace | 5-10 | Foundational workspace operation, should be well-represented |

### PRIORITY 2 (High): Severely Underrepresented (<5 good examples)

**Target:** Bring each tool to at least 10 good examples (30-50 total examples)

| Tool | Current | Target | Gap |
|------|---------|--------|-----|
| commandManager_executeCommand | 1 | 10 | +9 |
| commandManager_listCommands | 2 | 10 | +8 |
| memoryManager_listWorkspaces | 1 | 10 | +9 |
| memoryManager_updateState | 1 | 10 | +9 |
| agentManager_listModels | 2 | 10 | +8 |

**Total needed:** ~43 new examples

### PRIORITY 3 (Medium): Underrepresented (5-15 good examples)

**Target:** Bring each tool to at least 15 good examples (40-80 total examples)

| Tool | Current | Target | Gap |
|------|---------|--------|-----|
| agentManager_executePrompt | 9 | 15 | +6 |
| agentManager_deleteAgent | 5 | 15 | +10 |
| agentManager_toggleAgent | 5 | 15 | +10 |
| agentManager_updateAgent | 8 | 15 | +7 |
| agentManager_getAgent | 10 | 15 | +5 |
| contentManager_prependContent | 8 | 15 | +7 |
| vaultManager_moveFolder | 9 | 15 | +6 |
| memoryManager_updateWorkspace | 11 | 15 | +4 |

**Total needed:** ~55 new examples

### PRIORITY 4 (Low): Already Decent (>15 examples)
These tools have reasonable coverage. Only add if balancing is needed.

## Estimated Impact

### Adding Priority 1 + 2 (20 + 43 = 63 new examples)
- Would bring never-used tools to baseline
- Would eliminate severely underrepresented category
- New dataset size: 1489 + 63 = **1552 good examples**
- Requires 63 matching bad examples for 1:1 ratio
- **Total new examples: 126** (maintaining perfect interleaving)

### Adding Priority 1 + 2 + 3 (63 + 55 = 118 new examples)
- Would bring all tools to ≥15 examples minimum
- Much more balanced coverage across all tools
- New dataset size: 1489 + 118 = **1607 good examples**
- Requires 118 matching bad examples for 1:1 ratio
- **Total new examples: 236** (maintaining perfect interleaving)

## Implementation Strategy

### Phase 1: Fill the Gaps (Priorities 1 & 2)
1. **Create 10 examples for never-used tools** (agentManager_batchExecutePrompt, memoryManager_createWorkspace)
2. **Create 43 examples for severely underrepresented tools** (commandManager, memoryManager workspace tools)
3. **Create matching 53 bad examples** with intentional errors
4. **Insert with perfect interleaving** to maintain 1:1 ratio throughout

**Deliverable:** Add 106 new examples total (53 good, 53 bad)

### Phase 2: Balance the Distribution (Priority 3)
1. **Create 55 good examples** for underrepresented tools
2. **Create 55 matching bad examples**
3. **Insert with perfect interleaving**

**Deliverable:** Add 110 new examples total (55 good, 55 bad)

### Phase 3: Quality Review
1. Validate all new examples against schemas
2. Ensure realistic use cases
3. Verify intentional errors in bad examples are appropriate
4. Check interleaving and ratio

## Recommended Approach

### Option A: Conservative (Priority 1 + 2 only)
- **New examples:** 106 total (53 pairs)
- **Final dataset size:** 3,084 examples
- **Effort:** ~2-3 hours of careful example creation
- **Benefit:** Eliminates all zero-usage and severe gaps

### Option B: Balanced (Priority 1 + 2 + 3)
- **New examples:** 216 total (108 pairs)
- **Final dataset size:** 3,194 examples
- **Effort:** ~4-5 hours of careful example creation
- **Benefit:** Much more uniform distribution across all tools

### Option C: Minimal (Priority 1 only)
- **New examples:** 20 total (10 pairs)
- **Final dataset size:** 2,998 examples
- **Effort:** ~30-45 minutes
- **Benefit:** Quick fix for never-used tools

## Key Considerations

1. **Quality over Quantity:** Better to have fewer high-quality examples than many mediocre ones
2. **Realistic Use Cases:** Examples should reflect actual user workflows
3. **Error Diversity:** Bad examples should show different types of errors (wrong params, missing context, invalid values)
4. **Maintain Interleaving:** Every new example pair must be inserted maintaining perfect alternation
5. **Context Variety:** Use different sessionIds, workspaceIds, goals, and scenarios

## Next Steps

1. **Decide on approach** (Option A, B, or C)
2. **Create example templates** for each priority tool
3. **Generate synthetic examples** following established patterns
4. **Review and validate** new examples
5. **Insert with interleaving** into dataset
6. **Final validation** to ensure 0 errors and perfect ratio

## Questions to Consider

1. What's the target dataset size? (affects which priority level to pursue)
2. Are certain tools more important than others for the finetuning goal?
3. Should we focus on specific managers (e.g., agentManager has low coverage overall)?
4. Do we want roughly equal distribution or can some tools remain more common?
