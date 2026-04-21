# Tools Directory

Utilities for tool-schema auditing, dataset migration, and dataset analysis.

## Active Files

### `audit_tool_schemas.py`
Regenerates the evaluator-facing YAML schema from the canonical CLI-first catalog in [`tool-schemas.json`](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tool-schemas.json).

Usage:
```bash
python3 tools/audit_tool_schemas.py
```

Outputs:
- `Evaluator/config/tool_schema_corrected.yaml`
- comparison output against `Evaluator/config/tool_schema.yaml`
- test-case audit output for invalid params

### `migrations/`
Active migration pipeline for the CLI-first non-thinking tool datasets.

Current active scripts:
- `tools/migrations/05_inventory_cli_schema_datasets.py`
- `tools/migrations/06_migrate_cli_schema_datasets.py`
- `tools/migrations/08_inventory_synthchat_cli_schema_refs.py`
- `tools/migrations/cli_schema_rules.py`
- `tools/migrations/cli_schema_utils.py`

### `analyze_tool_coverage.py`
Analyzes tool usage distribution across a dataset.

Usage:
```bash
python3 tools/analyze_tool_coverage.py Datasets/tools_datasets/non_thinking/contentManager/tools_v2.3.jsonl
```

## Canonical Schema

### `tool-schemas.json`
Current source of truth for the CLI-first tool surface.

The catalog is organized as:
```json
{
  "generatedAt": "...",
  "toolCount": 60,
  "agentCount": 11,
  "agents": [...],
  "tools": [
    {
      "agent": "storageManager",
      "tool": "move",
      "command": "storage move",
      "usage": "storage move <path> <newPath> [--overwrite]",
      "arguments": [...]
    }
  ]
}
```

Evaluator and environment execution now assume a single wrapper tool call:
```json
{
  "name": "useTools",
  "arguments": {
    "workspaceId": "default",
    "sessionId": "session_123",
    "memory": "Need to inspect and reorganize notes.",
    "goal": "Move a note and then read it back.",
    "constraints": "Do not touch unrelated files.",
    "tool": "storage move \"notes/today.md\" \"archive/today.md\", content read \"archive/today.md\"",
    "strategy": "serial"
  }
}
```

## Validation

Use the canonical validator from the synthetic-data-generation skill:
```bash
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py Datasets/my_dataset.jsonl
```

Use the environment-backed SynthChat path for runtime checks:
```bash
python3 -m SynthChat.run generate \
  --env-backend local \
  --env-tool-schema Evaluator/config/tool_schema.yaml \
  --env-exec-config Evaluator/config/environment_execution.yaml \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3
```

## Legacy Helpers

Wrapper-era and pre-CLI migration helpers have been retired from the active path.
Local archive copies live under ignored `archive/` snapshots when needed for reference.
