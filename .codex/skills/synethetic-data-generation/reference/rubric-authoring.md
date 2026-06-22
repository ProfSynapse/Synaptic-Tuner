# Rubric Authoring Reference

**Location:** `SynthChat/rubrics/*.yaml`

Rubrics define how SynthChat judges and improves generated examples. Tool-call
formats are config-driven. Rubrics should teach and validate the active
configured format, not assume one wrapper is globally correct in code.

---

## Available Rubrics

| Rubric | Scope | Threshold | What It Checks |
|--------|-------|-----------|----------------|
| `system_prompt_format` | system_prompt | 0.8 | 5 required XML sections, hierarchy consistency |
| `thinking_quality` | thinking | 0.8 | Goal clarity, memory context, req/plan distinction, confidence |
| `tool_alignment` | response | 0.8 | Tool calls match thinking block's goal/plan |
| `response_quality` | response | 0.8 | Response matches risk assessment |
| `factuality` | response | 0.8 | Details are grounded in prompt/context |
| `content_writing_quality` | response | 0.85 | Frontmatter, structure, substance, Obsidian formatting |
| `confidence_calibration` | thinking | 0.8 | Confidence matches operation risk |
| `context_alignment` | response | 0.8 | Context IDs match system prompt |
| `quality_labels` | response | 0.8 | KTO label assignment |
| `destructive_safety` | response | 0.8 | Proper handling of destructive operations |
| `contentManager_tools` | response | 0.8 | CLI-first content tool call correctness |
| `memoryManager_tools` | response | 0.8 | CLI-first memory tool call correctness |
| `promptManager_tools` | response | 0.8 | CLI-first prompt tool call correctness |
| `searchManager_tools` | response | 0.8 | CLI-first search tool call correctness |
| `storageManager_tools` | response | 0.8 | CLI-first storage tool call correctness |

---

## Format Discipline

For active tool rubrics in this repo:
- validate against the format selected in config/scenario, not a hardcoded runtime assumption
- keep wrapper names, top-level fields, and command examples in rubric text/config only
- do not teach or validate one wrapper as globally correct unless that rubric is intentionally scoped to that configured format
- if a rubric is tied to a specific tool-call format, say so explicitly in the rubric text and validation schema

---

## Rubric Skeleton

```yaml
name: Human-Readable Name
description: What this rubric evaluates
scope: system_prompt | thinking | response | user
pass_threshold: 0.0-1.0

judge_prompt: |
  Explain what is being evaluated.
  Return JSON: {"rubrickey_score": 0.0-1.0}

improver_prompt: |
  Explain how to fix the content.
  If improving a tool response, preserve the CLI-first `useTools` wrapper.

output_schema:
  type: object
  properties:
    rubrickey_score:
      type: number
      min: 0.0
      max: 1.0
  required:
    - rubrickey_score
  additionalProperties: false

output_format:
  type: assistant_message

validations:
  - tools:
      YOUR_WRAPPER_NAME:
        _required: [YOUR_REQUIRED_FIELDS]
    error: "Configured tool wrapper validation failed: {details}"
```

---

## Template Variables

| Variable | Description |
|----------|-------------|
| `{current_content}` | Content being judged/improved |
| `{system_prompt}` | The system message |
| `{user_request}` | The user message |
| `{assistant_response}` | The full assistant response |
| `{feedback}` | Judge feedback |
| `{thinking_content}` | Extracted thinking block |
| `{thinking_block}` | Thinking block for alignment checks |
| `{original_tool_calls}` | Tool calls as JSON |
| `{environment_result_json}` | Structured environment/runtime result for the example |
| `{environment_passed}` | Boolean environment pass/fail when available |

---

## Validation Patterns

### Tool Wrapper Validation

```yaml
validations:
  - tools:
      YOUR_WRAPPER_NAME:
        _required: [FIELD_A, FIELD_B]
        _additionalProperties: false
    error: "Configured wrapper validation failed: {details}"
```

### Cross-Scope ID Validation

```yaml
  - cross_scope:
      from: response
      to: system_prompt
      extract:
        fields: [sessionId, workspaceId]
        pattern: "(?:sessionId|workspaceId).*?[\"']([^\"']+)[\"']"
      validate_in: [session_context]
    error: "Context ID mismatch: '{value}' not found in session_context"
```

---

## Authoring Rules

1. Keep format assumptions in rubric/config text, never as unspoken runtime truths.
2. Keep judge and improver examples aligned with the active configured schema.
3. If environment validation is enabled, include environment/runtime failures in the judge/improver context so recommendations can address real execution errors.
4. If a rubric is intentionally tied to one format, scope and label it clearly.
5. If a rubric is intentionally backward-compat only, archive it instead of leaving it active.
6. Prefer structural validation for wrapper shape and use the judge for command semantics.
7. Re-run a small judged smoke test after rubric changes so you verify the improver path, not just raw generation.
