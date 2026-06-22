# Case Study: Tool-Calling Pipeline

This project’s current tool-calling stack is CLI-first. The model learns to emit a single `useTools` wrapper whose `tool` field contains one or more CLI commands.

---

## What The Model Must Learn

1. Call `useTools`, not direct tool functions.
2. Put `workspaceId`, `sessionId`, `memory`, `goal`, and `tool` at the top level of `function.arguments`.
3. Express concrete operations as CLI commands in the `tool` string.
4. Use `strategy` when multiple commands are present and ordering matters.
5. Ask for clarification instead of guessing on vague or destructive requests.

---

## Canonical Wrapper

```json
{
  "tool_calls": [
    {
      "id": "call_0001",
      "type": "function",
      "function": {
        "name": "useTools",
        "arguments": "{\"workspaceId\":\"default\",\"sessionId\":\"session_1731071640123_d7m2v9c4r\",\"memory\":\"User wants to reorganize notes and inspect the result.\",\"goal\":\"Move a note and read it back.\",\"constraints\":\"Do not touch unrelated files.\",\"tool\":\"storage move \\\"notes/today.md\\\" \\\"archive/today.md\\\", content read \\\"archive/today.md\\\"\",\"strategy\":\"serial\"}"
      }
    }
  ],
  "content": null
}
```

The inner wrapper payload is:

```json
{
  "workspaceId": "default",
  "sessionId": "session_1731071640123_d7m2v9c4r",
  "memory": "User wants to reorganize notes and inspect the result.",
  "goal": "Move a note and read it back.",
  "constraints": "Do not touch unrelated files.",
  "tool": "storage move \"notes/today.md\" \"archive/today.md\", content read \"archive/today.md\"",
  "strategy": "serial"
  }
}
```

The command catalog comes from [`cli-first-tool-schemas.json`](../../../cli-first-tool-schemas.json).

---

## Active Managers

Current migration and generation scope:
- `contentManager`
- `memoryManager`
- `promptManager`
- `searchManager`
- `storageManager`

Future-facing scenario coverage exists for:
- `canvasManager`
- `taskManager`

---

## System Prompt Contract

System prompts still provide runtime context through sections like:
- `<session_context>`
- `<vault_structure>`
- `<available_workspaces>`
- `<available_prompts>`
- `<selected_workspace>`

But the model must now copy `sessionId` and `workspaceId` into the top level of `useTools.arguments`, not into a nested `context` object.

Example instruction:

```xml
<session_context>
IMPORTANT: When using tools, include these values as top-level fields in your useTools arguments payload.
- sessionId: "session_1731071640123_d7m2v9c4r"
- workspaceId: "default" (no specific workspace selected)
</session_context>
```

---

## Dataset Creation Flow

1. Start from `cli-first-tool-schemas.json`.
2. Generate evaluator YAML with:
   ```bash
   python3 tools/audit_tool_schemas.py
   ```
3. Migrate or regenerate canonical datasets under `Datasets/tools_datasets/`.
4. Use SynthChat quickcheck targets before broad generation.
5. Validate with the canonical validator and environment-backed smoke tests.

---

## Migration Notes

Old wrapper-era formats are no longer canonical:
- nested `context` + `calls`
- direct function names like `vaultManager_openNote`
- legacy managers such as `vaultManager`, `vaultLibrarian`, and `agentManager`

The modern equivalents are:
- `storageManager`
- `searchManager`
- `promptManager`

And for content editing:
- `contentManager_read`
- `contentManager_write`
- `contentManager_insert`
- `contentManager_replace`
- `contentManager_setProperty`

---

## Smoke Test

```bash
python3 -m SynthChat.run generate \
  --provider openrouter \
  --model qwen/qwen3.6-plus \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_quickcheck_qwen_cli.jsonl
```

This should produce `useTools` calls with top-level wrapper fields and a CLI `tool` string that expands cleanly in the environment runtime.
