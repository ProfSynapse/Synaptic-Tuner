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
name: Configured Tool Tests
description: Checks emitted tool calls through the configured response format
tests:
  - id: configured_tool_case
    question: Use the available tool interface to complete the requested operation.
    tags: [tool-call, single-tool]
    system: |
      <runtime_context>
      Include any context values required by this scenario's configured schema.
      </runtime_context>
    correct:
      any:
        - name: expected_tool_call
          assertions:
            - type: jsonpath_equals
              path: $.tool_calls[0].name
              value: CONFIGURED_WRAPPER_NAME
            - type: jsonpath_exists
              path: $.tool_calls[0].arguments.FIELD_A
            - type: jsonpath_regex
              path: $.tool_calls[0].arguments.ACTION_FIELD
              pattern: 'expected configured action pattern'
```

Use placeholders in reference docs. Real scenarios should use the wrapper name, fields, and action syntax defined in that scenario's schema/config.

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
    - name: action_by_name
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.ACTION_FIELD
          pattern: 'configured-name-pattern'
    - name: action_by_id
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.ACTION_FIELD
          pattern: 'configured-id-pattern'
```

Use `correct.all` when every assertion must pass and there is only one acceptable shape:

```yaml
correct:
  all:
    - type: text_regex
      pattern: 'clarifying question pattern'
    - type: length_equals
      path: $.tool_calls
      value: 0
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

The response view only parses syntax, such as JSON argument strings or plain text tool-call blocks. It does not map tools, commands, or wrapper names and does not define correctness.

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

## Tool-Call Assertions

The tool surface is config-driven. Models should emit the configured wrapper/transport and put executable intent in whichever field the active schema declares.

Example output:

```text
tool_call: CONFIGURED_WRAPPER_NAME
arguments: {
  "FIELD_A": "value-a",
  "FIELD_B": "value-b",
  "ACTION_FIELD": "configured action payload"
}
```

Corresponding assertion:

```yaml
correct:
  any:
    - name: configured_action
      assertions:
        - type: jsonpath_equals
          path: $.tool_calls[0].name
          value: CONFIGURED_WRAPPER_NAME
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.ACTION_FIELD
          pattern: 'expected configured action pattern'
```

When supporting multiple transport shapes, put each shape under `correct.any`.

---

## Equivalent Correct Answers

If the tool schema supports multiple valid forms, represent each form in config:

```yaml
correct:
  any:
    - name: by_id
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.ACTION_FIELD
          pattern: 'configured-id-pattern'
    - name: by_name
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].arguments.ACTION_FIELD
          pattern: 'configured-name-pattern'
```

Use this for id-or-name, positional-or-flag forms, valid aliases, optional flags, and acceptable text-only answers.

---

## Text-Only Assertions

For clarification or refusal cases, assert the text directly:

```yaml
- id: clarification_before_destructive_action
  question: Perform a broad destructive operation.
  tags: [clarification, destructive]
  correct:
    all:
      - type: text_regex
        pattern: 'clarifying question pattern'
      - type: length_equals
        path: $.tool_calls
        value: 0
```

---

## Optional Environment Checks

Environment checks are additional runtime checks, not the primary correctness contract.

```yaml
environment:
  allowed_tools: ["CONFIGURED_WRAPPER_NAME"]
  max_steps: 3
  assertions:
    - type: path_exists
      path: "path/created/by/scenario"
```

Use with:

```bash
python -m Evaluator.cli --backend lmstudio --model MODEL --scenario tool_prompts.yaml --env-backend local
```

---

## Tags

Tags are arbitrary labels for filtering and reporting. Define them per scenario suite, for example:

- `tool-call`
- `single-tool`
- `multi-step`
- `clarification`
- `destructive`
- `retrieval`
- `edit`

---

## Adding New Tests

1. Open or create a YAML file in `Evaluator/config/scenarios/`.
2. Add a test under `tests:`.
3. Define `correct` assertions for every acceptable response shape.
4. Use `correct.any` for alternatives instead of hardcoding logic in Python.
