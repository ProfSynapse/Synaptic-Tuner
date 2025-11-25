# Tools Datasets

This directory contains tool-calling training data organized by agent category.

## Structure

```
tools_datasets/
├── contentManager/       # CRUD operations on note content
│   └── tools_v1.0.jsonl
├── vaultManager/         # File/folder operations
│   └── tools_v1.0.jsonl
├── memoryManager/        # Session/state/workspace management
│   └── tools_v1.0.jsonl
├── vaultLibrarian/       # Advanced search, batch operations
│   └── tools_v1.0.jsonl
├── agentManager/         # Agent lifecycle, prompt execution
│   └── tools_v1.0.jsonl
├── split_metadata.json   # Statistics from split operation
└── README.md
```

## Agent Categories

| Agent | Description | Key Tools |
|-------|-------------|-----------|
| **contentManager** | Create, read, update, delete note content | `createContent`, `readContent`, `appendContent`, `replaceContent` |
| **vaultManager** | File and folder operations | `createFolder`, `moveNote`, `deleteNote`, `listDirectory` |
| **memoryManager** | Session and workspace state management | `createSession`, `loadWorkspace`, `createState`, `listSessions` |
| **vaultLibrarian** | Search and batch operations | `searchContent`, `searchDirectory`, `searchMemory`, `batch` |
| **agentManager** | AI agent management and execution | `executePrompt`, `createAgent`, `updateAgent`, `listAgents` |

## Dataset Format

Each line in the JSONL files is a single example:

```json
{
  "conversations": [
    {"role": "user", "content": "User request..."},
    {"role": "assistant", "content": null, "tool_calls": [
      {
        "id": "abc123def",
        "type": "function",
        "function": {
          "name": "agentName_toolMethod",
          "arguments": "{\"context\": {...}, ...}"
        }
      }
    ]}
  ],
  "label": true
}
```

**Note:** The `label` field is optional. When present, `true` indicates a positive example and `false` indicates a negative example.

## Usage

### Adding New Examples

1. Edit the appropriate agent's `tools_v1.0.jsonl` file
2. Ensure each example follows the format above
3. Run the merge script to regenerate the combined dataset

### Merging Datasets

```bash
cd Datasets/tools
python merge_tools_datasets.py
```

This creates:
- `syngen_tools_sft_merged_v1.0.jsonl` - Combined dataset
- `syngen_tools_sft_merged_v1.0.metadata.json` - Statistics

### Splitting Source Dataset (One-Time)

If you need to re-split from the original source:

```bash
cd Datasets/tools
python split_tools_dataset.py
```

## Version History

| Version | Date | Notes |
|---------|------|-------|
| v1.0 | 2025-11-25 | Initial split from `syngen_tools_sft_11.24.25_cleaned.jsonl` |

## Context Object Requirements

All tool calls must include a context object with these 7 required fields:

```json
{
  "context": {
    "sessionId": "session_1731015400000_a1b2c3d4e",
    "workspaceId": "ws_1731015400000_f5g6h7i8j",
    "sessionDescription": "Brief summary",
    "sessionMemory": "Prior context (never empty)",
    "toolContext": "Why calling this tool",
    "primaryGoal": "User's main objective",
    "subgoal": "What this call achieves"
  }
}
```

**Important:** `sessionMemory` must never be empty.
