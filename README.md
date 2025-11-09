# Claudesidian-MCP Synthetic Training Data Repository

High-quality synthetic training dataset for fine-tuning local LLMs to reliably use the Claudesidian-MCP tool suite for Obsidian vault operations.

## Repository Structure

```
Synthetic Conversations/
├── README.md                                    # This file
├── finetuning-strategy.md                      # Master strategy document
├── syngen_toolset_v1.0.0_claude.jsonl          # Main Claude-generated dataset (853 examples)
├── syngen_toolset_v1.0.0_chatgpt.jsonl         # ChatGPT-generated dataset
├── syngen_toolset_v1.0.0_copilot.jsonl         # Copilot-generated dataset
├── docs/                                        # Documentation
│   ├── WORKSPACE_README.md                     # Workspace structure overview
│   ├── WORKSPACE_ANALYSIS_REPORT.md            # Detailed workspace schema
│   ├── WORKSPACE_ARCHITECTURE_DIAGRAM.md       # Visual diagrams
│   ├── WORKSPACE_KEY_FILES_REFERENCE.md        # Source code mapping
│   ├── WORKSPACE_DOCUMENTATION_INDEX.md        # Navigation guide
│   ├── SCHEMA_VERIFICATION_REFERENCE.md        # Tool schema reference
│   └── TOOL_SCHEMA_REFERENCE.md                # Tool definitions
└── tools/                                       # Validation & utilities
    ├── validate_syngen.py                      # Validator script
    └── tool_schemas.json                       # Tool schema definitions
```

## Dataset Overview

### Current Stats (v1.0.0)
- **Total Examples:** 853
- **Desirable Examples:** 634 (74.4%)
- **Undesirable Examples:** 215 (25.2%)
- **Ratio:** 2.95:1 (target 3:1)
- **Format:** ChatML (no system message)

### Coverage

#### Batch Set A: Core Tool Categories (144 examples)
- **Batch 52:** vaultManager tools (file/folder operations)
- **Batch 53:** contentManager tools (CRUD operations)
- **Batch 54:** memoryManager tools (sessions/states/workspace)

#### Batch Set B: Advanced & Multi-Step (144 examples)
- **Batch 55:** vaultLibrarian tools (advanced search, batch operations)
- **Batch 56:** agentManager tools (agent lifecycle, image generation)
- **Batch 57:** Multi-step workflows (2-3 tool chaining with context accumulation)

#### Batch Set C: Advanced Scenarios (144 examples)
- **Batch 58:** Tool discovery (get_tools meta-tool usage)
- **Batch 59:** Error recovery (handling failures and retry logic)
- **Batch 60:** Clarification scenarios (handling ambiguous requests)

#### Future Batches: Workspace-Aware Workflows (in progress)
- **Batch 61:** Workspace selection & loading
- **Batch 62:** Workspace-informed actions (read files → perform action)
- **Batch 63:** Workspace state checkpointing

## Quick Start

### Validate Dataset
```bash
python3 tools/validate_syngen.py syngen_toolset_v1.0.0_claude.jsonl
```

### Generate New Batches
Use the task-based agents to generate new synthetic examples:
```bash
# Agents will write to /tmp/batchXX_claude.jsonl
# Then merge into main dataset
```

### Understanding the Data

Each example follows this structure:
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "User request or question"
    },
    {
      "role": "assistant",
      "content": "tool_call: toolName\narguments: {...}\n\nResult: {...}\n\nAssistant's response to user"
    }
  ],
  "label": "desirable" or "undesirable"
}
```

**Key Requirements:**
- ✓ NO system message (starts directly with user role)
- ✓ Context object as FIRST parameter in all tool calls
- ✓ Complete context with all 7 fields:
  - `sessionId`: `session_<13 digits>_<9 chars>`
  - `workspaceId`: `ws_<13 digits>_<9 chars>`
  - `sessionDescription`: Brief summary
  - `sessionMemory`: NEVER empty, 1-2 sentences of prior context
  - `toolContext`: Why this tool is being called
  - `primaryGoal`: User's overall objective
  - `subgoal`: What this specific call achieves
- ✓ Tool results between tool calls and response

## Context Objects

Every tool call must include a context object as the FIRST parameter:

```json
{
  "context": {
    "sessionId": "session_1731015400000_a1b2c3d4e",
    "workspaceId": "ws_1731015400000_f5g6h7i8j",
    "sessionDescription": "What this session is about",
    "sessionMemory": "Prior context from earlier interactions",
    "toolContext": "Why we're calling this specific tool",
    "primaryGoal": "User's main objective",
    "subgoal": "What this tool call achieves"
  },
  "otherParam": "value"
}
```

## Labels Explained

### Desirable Examples (634)
Demonstrate **correct** tool usage:
- Proper tool selection for the task
- Accurate parameter types and values
- Complete context objects
- Realistic use cases
- Multi-step workflows with proper chaining
- Good error handling and clarification

### Undesirable Examples (215)
Demonstrate **common mistakes** for contrastive learning:
- Missing required parameters
- Wrong tool for task
- Empty or missing sessionMemory
- Context objects in wrong position
- Inconsistent sessionIds in multi-step workflows
- Poor error recovery
- Over/under clarification

## Validator Details

The validator checks:
- ✓ JSON validity
- ✓ ChatML format (no system message)
- ✓ Context object presence and format
- ✓ SessionId/workspaceId format correctness
- ✓ All 7 context fields present
- ✓ SessionMemory never empty
- ✓ Context as first parameter
- ✓ Tool schema compliance
- ✓ Desirable vs undesirable labeling

**Special Handling:** Undesirable examples are allowed to have intentional errors in tool parameters (for training), but structural issues (missing context fields) must follow the pattern.

## Tools & Schemas

See `tools/tool_schemas.json` for complete tool definitions including:
- 42+ tool schemas across 5 agents
- Required/optional parameters
- Parameter types and descriptions
- Context schema requirements
- Tool categories and agent mappings

## Documentation

### For Understanding Workspaces
1. Start: `docs/WORKSPACE_README.md`
2. Details: `docs/WORKSPACE_ANALYSIS_REPORT.md`
3. Visuals: `docs/WORKSPACE_ARCHITECTURE_DIAGRAM.md`
4. Code: `docs/WORKSPACE_KEY_FILES_REFERENCE.md`
5. Navigation: `docs/WORKSPACE_DOCUMENTATION_INDEX.md`

### For Understanding Datasets
- `finetuning-strategy.md` - Master strategy document
- `docs/SCHEMA_VERIFICATION_REFERENCE.md` - Tool schema reference
- `docs/TOOL_SCHEMA_REFERENCE.md` - Tool definitions

## Batch Generation Process

1. **Plan** - Determine focus area and example count
2. **Generate** - Use Task agents to create examples
3. **Validate** - Run validator on generated batch
4. **Merge** - Add validated examples to main dataset
5. **Check Stats** - Verify ratio and count targets

## Contributing New Batches

When generating new batches:
1. Follow the ChatML format strictly (no system message)
2. Always include complete context objects
3. Never leave sessionMemory empty
4. Use proper ID formats (13 digits_9 chars)
5. Ensure no duplicates from existing dataset
6. Maintain ~3:1 desirable:undesirable ratio
7. Test with validator before merging

## Training Usage

This dataset is optimized for:
- **Unsloth universal format** - Simple prompt/completion pairs
- **KTO training** - Paired desirable/undesirable examples
- **Tool calling focus** - No general conversation, only tool use
- **Multi-turn completeness** - Full execution cycles with results
- **Context accumulation** - Learning to track state across steps

## License & Attribution

Generated using Claude (Anthropic) for the Claudesidian-MCP project.

---

**Last Updated:** 2025-11-09
**Total Commits:** See `.git` for full history
**Next Target:** 1,000+ examples with workspace-aware workflows
