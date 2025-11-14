# Tool Coverage Gap Analysis: Copilot v1.0.1

## Current Status

**Dataset:** `syngen_toolset_v1.0.1_copilot.jsonl`
- **Total Examples:** 1,146 (573 true, 573 false)
- **Tool Coverage:** 41 of 47 tools (87%)
- **Validation:** ✅ Perfect (0 errors, 0 warnings)
- **Interleaving:** ✅ Perfect 1:1 ratio

## Coverage Gaps

### Priority 1: ZERO Examples (CRITICAL)
**7 tools with 0 examples - need 140 examples (70 good + 70 bad)**

| Tool | Current | Target | Need |
|------|---------|--------|------|
| `agentManager_batchExecutePrompt` | 0 | 10 | +10 good, +10 bad |
| `agentManager_deleteAgent` | 0 | 10 | +10 good, +10 bad |
| `agentManager_getAgent` | 0 | 10 | +10 good, +10 bad |
| `memoryManager_createWorkspace` | 0 | 10 | +10 good, +10 bad |
| `memoryManager_listWorkspaces` | 0 | 10 | +10 good, +10 bad |
| `memoryManager_loadWorkspace` | 0 | 10 | +10 good, +10 bad |
| `memoryManager_updateState` | 0 | 10 | +10 good, +10 bad |

### Priority 2: Underrepresented (1-9 examples)
**18 tools - need ~164 examples**

| Tool | Current | Need |
|------|---------|------|
| `memoryManager_updateWorkspace` | 1 | +9 good, +9 bad |
| `agentManager_generateImage` | 2 | +8 good, +8 bad |
| `agentManager_toggleAgent` | 2 | +8 good, +8 bad |
| `agentManager_listModels` | 3 | +7 good, +7 bad |
| `agentManager_updateAgent` | 3 | +7 good, +7 bad |
| `vaultLibrarian_batch` | 4 | +6 good, +6 bad |
| `agentManager_listAgents` | 5 | +5 good, +5 bad |
| `commandManager_executeCommand` | 5 | +5 good, +5 bad |
| `commandManager_listCommands` | 6 | +4 good, +4 bad |
| `memoryManager_listStates` | 6 | +4 good, +4 bad |
| `memoryManager_loadState` | 6 | +4 good, +4 bad |
| `contentManager_replaceContent` | 7 | +3 good, +3 bad |
| `contentManager_deleteContent` | 7 | +3 good, +3 bad |
| `vaultManager_editFolder` | 7 | +3 good, +3 bad |
| `agentManager_createAgent` | 8 | +2 good, +2 bad |
| `memoryManager_createState` | 8 | +2 good, +2 bad |
| `memoryManager_loadSession` | 9 | +1 good, +1 bad |
| `vaultManager_deleteFolder` | 9 | +1 good, +1 bad |

### Priority 3: Optional Boost (10-14 examples to 15)
**4 tools - need ~24 examples**

| Tool | Current | Could Add |
|------|---------|-----------|
| `agentManager_executePrompt` | 10 | +5 good, +5 bad |
| `vaultManager_deleteNote` | 11 | +4 good, +4 bad |
| `memoryManager_createSession` | 13 | +2 good, +2 bad |
| `contentManager_prependContent` | 14 | +1 good, +1 bad |

## Summary of Needs

| Priority | Tools | Examples Needed |
|----------|-------|-----------------|
| **P1: Critical (0 examples)** | 7 | **140** |
| **P2: Low coverage (1-9)** | 18 | **164** |
| **P3: Boost to 15 (optional)** | 4 | 24 |
| **TOTAL MINIMUM (P1+P2)** | **25 tools** | **304 examples** |
| **With P3 boost** | 29 tools | 328 examples |

## Recommendations

### Option 1: Minimum Viable (P1 only)
- **Add:** 140 examples (70 good + 70 bad pairs)
- **Focus:** 7 tools with zero coverage
- **Result:** 41 → 41 tools covered (100% basic coverage)
- **Final dataset:** 1,286 examples

### Option 2: Balanced Coverage (P1 + P2)
- **Add:** 304 examples (152 good + 152 bad pairs)
- **Focus:** All 25 underrepresented tools
- **Result:** All tools with ≥10 examples
- **Final dataset:** 1,450 examples

### Option 3: Complete (P1 + P2 + P3)
- **Add:** 328 examples (164 good + 164 bad pairs)
- **Focus:** 29 tools, boost all to ≥15 examples
- **Result:** Maximum coverage quality
- **Final dataset:** 1,474 examples

## Comparison with Claude Dataset

**Claude v1.0.0:**
- Total: 2,978 examples
- Coverage: 45/47 tools (96%)
- Missing: `agentManager_batchExecutePrompt`, `memoryManager_createWorkspace`

**Copilot v1.0.1:**
- Total: 1,146 examples  
- Coverage: 41/47 tools (87%)
- Missing: 6 additional tools

**Gap:** Copilot is 61% smaller and covers 4 fewer tools.

## Next Steps

1. **Choose approach:** Option 1, 2, or 3 above
2. **Generate examples:** Use tool schemas to create realistic examples
3. **Maintain quality:**
   - Perfect 1:1 ratio
   - Perfect interleaving  
   - Zero validation errors
   - Realistic, diverse scenarios
4. **Validate frequently:** After each batch of 20-40 examples

## Key Constraints

✅ Must maintain perfect 1:1 label ratio
✅ Must maintain perfect interleaving (true, false, true, false...)
✅ All true-labeled examples must validate perfectly
✅ All false-labeled examples must have intentional schema violations
✅ Use only tools defined in `tools/tool_schemas.json`

---

**Recommendation:** Start with **Option 2 (Balanced Coverage)** to bring all tools to at least 10 examples. This provides solid coverage for training while being achievable (304 examples to add).
