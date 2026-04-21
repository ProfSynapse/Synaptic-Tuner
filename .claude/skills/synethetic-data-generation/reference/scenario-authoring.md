# Scenario Authoring Reference

Scenarios define what SynthChat should generate. For tool-calling work in this repo, scenarios should target the CLI-first `useTools` wrapper.

---

## Current Tool Scenario Shape

```yaml
scenarios:
  storageManager_move:
    type: tool
    agent: storageManager
    tool: move
    system: true
    rubrics:
      system_prompt: [system_prompt_format]
      response: [tool_alignment, factuality]
    prompts:
      system: |
        Generate a realistic system prompt with:
        - <session_context> containing sessionId and workspaceId
        - <vault_structure>
        - <available_workspaces>
        - <available_prompts>
        - <selected_workspace> JSON
      user: |
        Generate a natural request that requires moving a file.
        OUTPUT ONLY THE REQUEST TEXT.
      assistant: |
        Generate a single OpenAI-style tool call:
        - function.name must be "useTools"
        - arguments must place workspaceId, sessionId, memory, goal, tool at the top level
        - tool must be a CLI command string such as:
          storage move "notes/today.md" "archive/today.md"
```

---

## Required Tool Wrapper

Tool scenarios should ultimately generate:

```json
{
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
```

Do not generate:
- direct per-tool function names such as `storageManager_move`
- legacy `context` + `calls` wrapper payloads

---

## Scenario Types

### Tool
Use for tool-calling datasets. Active managers in the current migration scope:
- `contentManager`
- `memoryManager`
- `promptManager`
- `searchManager`
- `storageManager`

### Behavioral
Use for clarification, verification, destructive safety, and similar behaviors.

### Docs-Based
Use when `--docs` is supplying source material.

---

## Environment Validation

Use the `environment` block when the tool call should be executed locally during generation:

```yaml
environment:
  allowed_tools:
    - storageManager_move
    - contentManager_read
  max_steps: 4
  execution:
    strict_schema: true
  assertions:
    - type: path_exists
      path: "archive/today.md"
```

Even though the generated response uses the `useTools` wrapper, `allowed_tools` and assertions should still reference the expanded concrete tool names.

---

## Authoring Rules

- Use `tool_call_format: default` unless there is a strong reason to override it.
- For content edits, prefer the current tools:
  - `contentManager_read`
  - `contentManager_write`
  - `contentManager_insert`
  - `contentManager_replace`
  - `contentManager_setProperty`
- Keep system prompts explicit that `sessionId` and `workspaceId` belong at the top level of the `useTools` payload.
- Use checked-in target manifests for smoke tests.

---

## Dry-Run First

Before broad generation:

```bash
python3 -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_quickcheck.jsonl
```
