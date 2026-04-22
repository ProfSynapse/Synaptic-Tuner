# Scenario Authoring Reference

Scenarios define what SynthChat should generate. Tool-call structure is config-driven; scenarios should reference the active configured format instead of assuming a hardcoded wrapper in code.

---

## Config-Driven Tool Scenario Shape

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
        Generate a single tool response using the configured tool_call_format.
        Follow the active wrapper name, argument shape, and command format from config.
```

The active format is only a scenario/config choice, not a repo-wide runtime truth.
In this repo, the default expectation is
that scenarios are rubric-backed and judge-backed unless there is a deliberate
reason to run a raw smoke test without them.

For new or production-bound scenarios:
- add `rubrics` for at least the stages you care about
- prefer a response-stage rubric for tool correctness
- add `final_judge` and/or in-loop `judge` when the scenario needs an explicit
  quality gate beyond schema/runtime validation

Judge-less scenarios are allowed for narrow plumbing checks, but they should be
treated as temporary smoke-test scaffolding rather than the standard setup.

---

## Format Rule

The tool wrapper, top-level fields, command style, and any context payload shape
must come from config:
- `SynthChat/config/tool_call_formats.yaml`
- scenario YAML
- evaluator/tool schema config

Do not treat the currently active wrapper as permanent. A different user should
be able to define a different format without changing code.

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

`allowed_tools` and assertions should reference the concrete expanded tool names
for the active environment config. Do not assume one wrapper or one command
format in prose unless that scenario explicitly defines it.

---

## Authoring Rules

- Use `tool_call_format: default` unless there is a strong reason to override it.
- Default to rubric-backed scenarios. Do not leave `rubrics`, `judge`, and
  `final_judge` empty by accident.
- For tool scenarios, the baseline should usually include:
  - `rubrics.response` with tool-format/tool-alignment checks
  - `final_judge` when you need a final pass/fail gate on overall quality
  - `judge.in_loop` only when the scenario benefits from iterative correction
- If environment validation is enabled, make sure the response-stage judge or
  improver can see environment/runtime failures through template variables or
  stage payloads. Those failures should inform improvement recommendations.
- For content edits, prefer the current tools:
  - `contentManager_read`
  - `contentManager_write`
  - `contentManager_insert`
  - `contentManager_replace`
  - `contentManager_setProperty`
- Keep format-specific instructions inside scenario/config text, not runtime code.
- Use checked-in target manifests for smoke tests.
- If you intentionally omit rubrics/judges for a smoke test, note that choice in
  the scenario comments or plan so it is clearly an exception.

---

## Dry-Run First

Before broad generation:

```bash
python3 -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_quickcheck.jsonl
```
