# Rubric YAML Structure

This document defines the structure for rubric YAML files in the improvement engine.

## Overview

Rubrics are used to:
1. **Validate** - Programmatically check structure (fields, tags, patterns)
2. **Judge** - LLM evaluation of quality
3. **Improve** - Generate better versions when quality is low

## Quick Start

```yaml
name: My Rubric
description: What this rubric checks
scope: thinking  # system_prompt | thinking | response
pass_threshold: 0.8

# Structural validation (runs BEFORE judge)
validations:
  - field: goal
    type: string
    min: 10
    error: "Goal must be at least 10 characters"

# LLM judge evaluation
judge_prompt: |
  Evaluate the content quality...
  Output JSON: {"my_score": 0.0-1.0}

output_schema:
  type: object
  properties:
    my_score:
      type: number
  required: [my_score]
```

## Validations

The `validations` list provides structural validation that runs BEFORE the LLM judge. Failures automatically set score to 0.0.

### Field Validation

Validate fields in parsed JSON content (e.g., thinking blocks).

```yaml
validations:
  # Basic field check
  - field: goal
    type: string
    min: 10
    error: "Goal must be at least {min} characters"

  # Array with minimum items
  - field: requirements
    type: array
    min: 1
    error: "Add at least {min} requirement(s)"

  # Nested field (dot notation)
  - field: assessment.risky
    type: boolean
    error: "assessment.risky must be boolean"

  # Number with range
  - field: confidence
    type: number
    min: 0.0
    max: 1.0
    error: "Confidence must be {min}-{max}"
```

**Keys:**
| Key | Description |
|-----|-------------|
| `field` | Field path (dot notation for nesting) |
| `type` | `string`, `number`, `array`, `object`, `boolean` |
| `min` | Minimum (length for string/array, value for number) |
| `max` | Maximum (length for string/array, value for number) |
| `error` | Message with `{field}`, `{type}`, `{min}`, `{max}`, `{actual}` |

### XML Tag Validation

Validate XML tags exist (checks BOTH opening and closing).

```yaml
validations:
  - match: session_context
    type: xml
    error: "Missing <session_context>...</session_context>"

  - match: vault_structure
    type: xml
    error: "Missing <vault_structure>...</vault_structure>"
```

### JSON Field Validation

Validate JSON fields exist within content or specific tags.

```yaml
validations:
  # Top-level JSON field (looks for "field": pattern)
  - match: context
    type: json
    in_tag: selected_workspace  # Scope to this tag
    error: "Missing 'context' field"

  # Nested JSON field (parses JSON, uses dot notation)
  - match: context.rootFolder
    type: json
    in_tag: selected_workspace
    error: "Missing 'context.rootFolder'"
```

### Regex Validation

```yaml
validations:
  - match: "tool_call:\\s+\\w+"
    type: regex
    error: "Invalid tool call format"
```

### Contains Validation (Default)

Simple text contains check.

```yaml
validations:
  - match: "some required text"
    error: "Missing required text"
```

### Cross-Scope Validation

Extract values from one scope and validate they exist in another.

```yaml
validations:
  - cross_scope:
      from: thinking
      to: system_prompt
      extract:
        fields: [goal, memory, plan]
        pattern: '[\w/-]+\.(?:md|txt|json)'
      skip_if:
        - pattern: '(?:create|new).*{value}'
      validate_in: [vault_structure, selected_workspace]
    error: "HALLUCINATED: '{value}' not found in vault"
```

**Keys:**
| Key | Description |
|-----|-------------|
| `from` | Source scope to extract from |
| `to` | Target scope to validate against |
| `extract.fields` | Fields to extract values from |
| `extract.pattern` | Regex to match values |
| `skip_if` | Skip validation if value matches pattern |
| `validate_in` | Tags in target to search |
| `error` | Message with `{value}` interpolation |

## Validation Type Summary

| Type | Key | Purpose |
|------|-----|---------|
| Field | `field` | Validate parsed dict fields |
| XML | `match` + `type: xml` | Check `<tag>` and `</tag>` exist |
| JSON | `match` + `type: json` | Check JSON field exists |
| Regex | `match` + `type: regex` | Regex pattern match |
| Contains | `match` (no type) | Simple text contains |
| Cross-scope | `cross_scope` | Validate across scopes |

## Complete Examples

### Thinking Block Validation

```yaml
name: Thinking Quality
scope: thinking
pass_threshold: 0.8

validations:
  - field: goal
    type: string
    min: 10
    error: "Goal must be at least {min} characters"

  - field: requirements
    type: array
    min: 1
    error: "Add at least {min} requirement(s)"

  - field: assessment.risky
    type: boolean
    error: "assessment.risky must be boolean"

  - field: confidence
    type: number
    min: 0.0
    max: 1.0
    error: "Confidence must be {min}-{max}"

judge_prompt: |
  Evaluate thinking quality...
  Output: {"thinking_score": 0.0-1.0}

output_schema:
  type: object
  properties:
    thinking_score:
      type: number
  required: [thinking_score]
```

### System Prompt Validation

```yaml
name: System Prompt Format
scope: system_prompt
pass_threshold: 0.8

validations:
  # Required XML tags
  - match: session_context
    type: xml
    error: "Missing <session_context>...</session_context>"

  - match: vault_structure
    type: xml
    error: "Missing <vault_structure>...</vault_structure>"

  - match: selected_workspace
    type: xml
    error: "Missing <selected_workspace>...</selected_workspace>"

  # JSON fields in selected_workspace
  - match: context
    type: json
    in_tag: selected_workspace
    error: "Missing 'context' field"

  - match: context.rootFolder
    type: json
    in_tag: selected_workspace
    error: "Missing 'context.rootFolder'"

judge_prompt: |
  Evaluate system prompt completeness...
  Output: {"format_score": 0.0-1.0}

output_schema:
  type: object
  properties:
    format_score:
      type: number
  required: [format_score]
```

### Cross-Scope Factuality Check

```yaml
name: Factuality
scope: [thinking, response]
pass_threshold: 0.8

validations:
  - cross_scope:
      from: thinking
      to: system_prompt
      extract:
        fields: [goal, memory, requirements, plan]
        pattern: '(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.(?:md|txt)'
      skip_if:
        - pattern: '(?:create|new|generate).*{value}'
      validate_in: [vault_structure, selected_workspace]
    error: "HALLUCINATED PATH: '{value}' not found"

judge_prompt: |
  Check for fabricated information...
  Output: {"factuality_score": 0.0-1.0}

output_schema:
  type: object
  properties:
    factuality_score:
      type: number
  required: [factuality_score]
```

## Other Fields

### Judge Prompt

LLM prompt to evaluate quality. Template variables:
- `{current_content}` - Content being evaluated
- `{system_prompt}` - System prompt (if available)
- `{user_request}` - User message

### Improver Prompt

LLM prompt to improve content. Template variables:
- `{current_content}` - Content to improve
- `{feedback}` - Judge's feedback
- `{format_instructions}` - Format spec content

### Output Schema

Defines what the judge must return:

```yaml
output_schema:
  type: object
  properties:
    my_score:
      type: number
  required: [my_score]
```

## Scope Types

| Scope | Content | Use For |
|-------|---------|---------|
| `system_prompt` | String | Format validation, tags |
| `thinking` | Dict (parsed JSON) | Field validation |
| `response` | String + tool_calls | Tool alignment |

## Testing

```bash
# Run rubric on dataset
python -m improvement_engine.services.rubric_runner \
  --file dataset.jsonl \
  --output improved.jsonl \
  --rubrics thinking_quality,system_prompt_format \
  --backend lmstudio

# List available rubrics
python -m improvement_engine.services.rubric_runner --list
```

## Best Practices

1. **Use validations for structural checks** - Faster than LLM, gives specific errors
2. **Reserve judge for quality evaluation** - Nuanced assessment LLM does better
3. **Write actionable error messages** - Help the improver fix issues
4. **Use cross_scope for factuality** - Catch hallucinated paths/names
5. **Set appropriate thresholds** - 0.8 normal, 0.9 strict, 0.7 lenient
