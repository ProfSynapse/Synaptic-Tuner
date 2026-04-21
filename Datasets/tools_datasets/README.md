# Tool Datasets

Canonical tool-calling training data, organized by manager and generation mode.

## Current Scope

Active CLI-first canonical datasets currently live under:

```text
Datasets/tools_datasets/
├── non_thinking/
│   ├── contentManager/
│   ├── memoryManager/
│   ├── promptManager/
│   ├── searchManager/
│   └── storageManager/
├── thinking/                  # Deferred for this migration pass
└── reports/cli_schema/
```

## Current Canonical Versions

- `non_thinking/contentManager/tools_v2.3.jsonl`
- `non_thinking/memoryManager/tools_v2.4.jsonl`
- `non_thinking/promptManager/tools_v2.6.jsonl`
- `non_thinking/searchManager/tools_v2.2.jsonl`
- `non_thinking/storageManager/tools_v2.4.jsonl`

## Dataset Format

Each example is a ChatML conversation. Tool-calling examples use an OpenAI-style `tool_calls` array and a single CLI-first `useTools` wrapper.

```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>"},
    {"role": "user", "content": "Move today's note into archive and read it back."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "useTools",
            "arguments": "{\"workspaceId\":\"default\",\"sessionId\":\"session_123\",\"memory\":\"Need to inspect and reorganize notes.\",\"goal\":\"Move a note and then read it back.\",\"constraints\":\"Do not touch unrelated files.\",\"tool\":\"storage move \\\"notes/today.md\\\" \\\"archive/today.md\\\", content read \\\"archive/today.md\\\"\",\"strategy\":\"serial\"}"
          }
        }
      ]
    }
  ],
  "label": true
}
```

## Migration Workflow

Inventory:
```bash
python3 tools/migrations/05_inventory_cli_schema_datasets.py
```

Rewrite non-thinking datasets into the CLI-first wrapper:
```bash
python3 tools/migrations/06_migrate_cli_schema_datasets.py
```

Inspect reports:
```text
Datasets/tools_datasets/reports/cli_schema/
├── inventory.json
├── migration_report.json
├── migration_report.md
├── regen_queue.jsonl
└── manual_review.jsonl
```

## Validation

Structural validation:
```bash
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py Datasets/tools_datasets/non_thinking/contentManager/tools_v2.3.jsonl
```

Generation smoke test:
```bash
python3 -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_quickcheck.jsonl
```

## Notes

- `thinking/` datasets are intentionally deferred in the current migration plan.
- `taskManager` and `canvasManager` have SynthChat scenario coverage for future generation, but they are not part of the refreshed canonical dataset pass yet.
- Old wrapper-era manager names such as `vaultManager`, `vaultLibrarian`, and `agentManager` are no longer canonical for active datasets.
