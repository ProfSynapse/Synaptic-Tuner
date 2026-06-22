# Scenario Authoring Reference

How to write config-first YAML test scenarios for model evaluation.

---

## Location

Scenario files live in `Evaluator/config/scenarios/`.

The active authoring model is:

1. Define the prompt with `question` and optional `system` or `messages`.
2. Define correct outputs under `correct`.
3. Put every task-specific expectation in YAML assertions.
4. Keep every task-specific expectation inside `correct`.

---

## Minimal Scenario

```yaml
name: Tool CLI Tests
description: Checks emitted CLI commands through the configured wrapper
tests:
  - id: storage_copy_runbook
    question: Copy Projects/Runbooks/Incident-Response.md to Projects/Runbooks/Incident-Response-Template.md.
    tags: [storageManager, single-tool]
    system: |
      <session_context>
      IMPORTANT: When using tools, include these values as top-level fields in your useTools arguments payload:
      - sessionId: "session_eval"
      - workspaceId: "ws_eval"
      </session_context>
    correct:
      any:
        - name: copy_cli
          assertions:
            - type: jsonpath_equals
              path: $.tool_calls[0].name
              value: useTools
            - type: jsonpath_equals
              path: $.tool_calls[0].arguments.sessionId
              value: session_eval
            - type: jsonpath_equals
              path: $.tool_calls[0].arguments.workspaceId
              value: ws_eval
            - type: jsonpath_exists
              path: $.tool_calls[0].arguments.memory
            - type: jsonpath_exists
              path: $.tool_calls[0].arguments.goal
            - type: jsonpath_regex
              path: $.tool_calls[0].arguments.tool
              pattern: '^storage copy\b(?=.*Incident-Response\.md)(?=.*Incident-Response-Template\.md)'
```

---

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique test identifier |
| `question` | string | User query sent to the model, unless `messages` provides the conversation |
| `tags` | list | Categories for filtering and reporting |
| `correct` | map | Assertion paths that define acceptable response(s) |

## Prompt Fields

| Field | Type | Description |
|-------|------|-------------|
| `system` | string | Optional system prompt prepended before `question` |
| `messages` | list | Optional full ChatML-style messages; when present, this overrides `system` + `question` for the backend call |
| `system_template` | string | Optional template from the scenario config |
| `system_context` | map | Optional template data/context |

---

## Correctness Blocks

Use `correct.any` when multiple outputs are valid:

```yaml
correct:
  any:
    - name: archive_by_name
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.tool
          pattern: '^prompt archive-prompt\b(?=.*QA Prototype)'
    - name: archive_by_id
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.tool
          pattern: '^prompt archive-prompt\b(?=.*agent_1732300800004_qa_prototype)'
```

Use `correct.all` when every assertion must pass and there is only one acceptable shape:

```yaml
correct:
  all:
    - type: text_regex
      pattern: 'Which file should I delete\?'
    - type: not_regex
      path: $.content
      pattern: 'tool_call:'
```

Each path under `correct.any` has:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable label shown in failures |
| `assertions` | list | Assertions that must all pass for this path |

---

## Response View Paths

Assertions query a generic response view:

| Path | Meaning |
|------|---------|
| `$.raw` | Raw assistant response as returned by the backend adapter |
| `$.raw_api_message` | Raw backend API payload when available |
| `$.content` | Assistant text content |
| `$.content_json` | Parsed JSON when `content` is JSON |
| `$.tool_calls` | Normalized emitted tool calls |
| `$.raw_tool_calls` | Raw tool-call objects before normalization |

Supported JSONPath subset:

- Dot keys: `$.tool_calls`
- Numeric indexes: `$.tool_calls[0]`
- Last item: `$.tool_calls[-1]`
- Wildcard lists: `$.tool_calls[*].name`
- Quoted bracket keys: `$["content_json"]["field-name"]`

The response view only parses syntax, such as JSON argument strings or plain `tool_call: ...` blocks. It does not map commands to manager tool ids and does not define correctness.

---

## Assertion Types

| Type | Required fields | Meaning |
|------|-----------------|---------|
| `jsonpath_exists` / `exists` | `path` | Value exists and is not null |
| `jsonpath_absent` / `absent` | `path` | Value is missing or null |
| `jsonpath_equals` / `equals` | `path`, `value` | Exact equality |
| `jsonpath_not_equals` / `not_equals` | `path`, `value` | Not equal |
| `jsonpath_contains` / `contains` | `path`, `value` | String/list/dict contains value |
| `jsonpath_not_contains` / `not_contains` | `path`, `value` | Does not contain value |
| `jsonpath_regex` / `regex` | `path`, `pattern` | Regex matches selected value |
| `jsonpath_not_regex` / `not_regex` | `path`, `pattern` | Regex does not match selected value |
| `text_regex` | `pattern` | Regex against `$.content` |
| `text_contains` | `value` | Contains check against `$.content` |
| `length_equals` | `path`, `value` | Selected list/string/dict length equals value |
| `length_min` | `path`, `value` | Selected length is at least value |
| `length_max` | `path`, `value` | Selected length is at most value |
| `json_subset` | `path`, `value` | Expected JSON object/list is a subset of actual |
| `all` | `assertions` | Nested assertions all pass |
| `any` | `assertions` | At least one nested assertion passes |
| `not` | `assertion` | Nested assertion must fail |

Regex assertions use Python regex with multiline and dotall flags.

---

## CLI Tool Assertions

The current tool schema is CLI-centric. Models should call the configured wrapper and put the executable command in `arguments.tool`.

Example output:

```text
tool_call: useTools
arguments: {
  "workspaceId": "ws_eval",
  "sessionId": "session_eval",
  "memory": "Need to copy the runbook.",
  "goal": "Create a template from the runbook.",
  "tool": "storage copy \"Projects/Runbooks/Incident-Response.md\" \"Projects/Runbooks/Incident-Response-Template.md\""
}
```

Corresponding assertion:

```yaml
correct:
  any:
    - name: copy_cli
      assertions:
        - type: jsonpath_equals
          path: $.tool_calls[0].name
          value: useTools
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.tool
          pattern: '^storage copy\b(?=.*Projects/Runbooks/Incident-Response\.md)(?=.*Projects/Runbooks/Incident-Response-Template\.md)'
```

If the backend returns OpenAI-style tool calls, use the equivalent path:

```yaml
- type: jsonpath_equals
  path: $.tool_calls[0].function.name
  value: useTools
- type: jsonpath_regex
  path: $.tool_calls[0].function.arguments.tool
  pattern: '^storage copy\b'
```

When supporting both transport shapes, put both under `correct.any`.

---

## Equivalent Correct Answers

If the tool schema supports multiple valid forms, represent each form in config:

```yaml
correct:
  any:
    - name: get_prompt_by_id
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.tool
          pattern: '^prompt get-prompt\b(?=.*agent_1732300800001_release_briefing)'
    - name: get_prompt_by_name
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.tool
          pattern: '^prompt get-prompt\b(?=.*Release Briefing)'
```

Use this for id-or-name, positional-or-flag forms, valid aliases, optional flags, and acceptable text-only answers.

---

## Text-Only Assertions

For clarification or refusal cases, assert the text directly:

```yaml
- id: clarification_before_delete
  question: Delete the old files.
  tags: [clarification, destructive]
  correct:
    all:
      - type: text_regex
        pattern: '(which|what).*files'
      - type: jsonpath_length_equals
        path: $.tool_calls
        value: 0
```

---

## Optional Environment Checks

Environment checks are additional runtime checks, not the primary correctness contract.

```yaml
environment:
  allowed_tools: ["useTools"]
  max_steps: 3
  assertions:
    - type: path_exists
      path: "Projects/Atlas/meeting-notes.md"
```

Use with:

```bash
python -m Evaluator.cli --backend lmstudio --model MODEL --scenario tool_prompts.yaml --env-backend local
```

---

## Tags

Tags are arbitrary labels for filtering and reporting. Common tags:

- `storageManager`
- `contentManager`
- `searchManager`
- `memoryManager`
- `promptManager`
- `single-tool`
- `clarification`
- `destructive`

---

## Adding New Tests

1. Open or create a YAML file in `Evaluator/config/scenarios/`.
2. Add a test under `tests:`.
3. Define `correct` assertions for every acceptable response shape.
4. Use `correct.any` for alternatives instead of hardcoding logic in Python.
5. Run a small check with `--limit` and/or `--tags`.
6. Inspect `Evaluator/results/*.json` for failed assertion details.

---
