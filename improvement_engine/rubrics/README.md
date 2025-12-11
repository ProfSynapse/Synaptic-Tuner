# Rubric YAML Structure

This document defines the required structure for rubric YAML files in the improvement engine.

## Overview

Rubrics are used to:
1. **Judge** - Evaluate if an example meets quality standards
2. **Validate** - Programmatically check structure (XML tags, JSON fields, etc.)
3. **Improve** - Generate better versions when quality is low

## Required Fields

```yaml
name: <string>
  # Display name for the rubric (e.g., "System Prompt Format")

description: <string>
  # Brief description of what this rubric validates

scope: <string>
  # Which part of the example to evaluate
  # Options: "system_prompt" | "thinking" | "response"

pass_threshold: <float>
  # Minimum score (0.0-1.0) required to pass
  # Typical values: 0.8 (strict), 0.9 (very strict)

judge_prompt: |
  # Prompt for LLM judge to evaluate quality
  # Should include:
  #   - Evaluation criteria
  #   - Scoring guidance (0.0 = fail, 1.0 = perfect)
  #   - Output format (JSON with score field)

improver_prompt: |
  # Prompt for LLM to improve content (OPTIONAL)
  # Template variables available:
  #   - {current_content}: The content being improved
  #   - {feedback}: Judge's feedback on what's wrong
  #   - {format_instructions}: The format_spec content
  #
  # If omitted, this rubric is judge-only (no improvement)

format_spec: |
  # Detailed format specification (OPTIONAL)
  # Used in improver_prompt as {format_instructions}
  # Should provide:
  #   - Examples of correct format
  #   - Required sections/fields
  #   - Quality criteria

output_schema:
  # JSON Schema for judge output validation
  # Required fields should match what judge_prompt says it will output
  type: object
  properties:
    <score_field_name>:
      type: number
      minimum: 0.0
      maximum: 1.0
  required:
    - <score_field_name>

schema_validation:
  # Programmatic validation rules (OPTIONAL)
  # Runs BEFORE judge to catch structural errors
  # Types: xml, json, regex, yaml, code
  types: [xml, json]

  xml:
    required_tags:
      - "<tag_name>"

  json:
    sections:
      - tag: xml_tag_containing_json
        extract_pattern: '\{[\s\S]*\}'
        required_fields: [field1, field2]
```

## Minimal Example

```yaml
name: Simple Quality Check
description: Basic quality validation
scope: response
pass_threshold: 0.8

judge_prompt: |
  Evaluate if the response is clear and helpful.

  Score 1.0 if excellent, 0.0 if poor.

  Output JSON: {"quality_score": 0.0-1.0}

output_schema:
  type: object
  properties:
    quality_score:
      type: number
      minimum: 0.0
      maximum: 1.0
  required:
    - quality_score
```

## Full Example with Improvement

```yaml
name: System Prompt Format
description: Validates system prompt structure
scope: system_prompt
pass_threshold: 0.9

judge_prompt: |
  Evaluate if system prompt has required sections:
  - <session_context>
  - <vault_structure>
  - <available_workspaces>

  Score:
  - 1.0 = All sections present
  - 0.5 = Some sections missing
  - 0.0 = Empty or wrong format

  Output JSON: {"format_score": 0.0-1.0}

improver_prompt: |
  Fix the system prompt based on feedback.

  **Current:**
  ```
  {current_content}
  ```

  **Feedback:**
  ```
  {feedback}
  ```

  **Format Requirements:**
  {format_instructions}

  Output the improved system prompt.

format_spec: |
  Required sections:
  1. <session_context> with sessionId and workspaceId
  2. <vault_structure> with folders and files
  3. <available_workspaces> with at least 1 workspace

output_schema:
  type: object
  properties:
    format_score:
      type: number
      minimum: 0.0
      maximum: 1.0
  required:
    - format_score

schema_validation:
  types: [xml]
  xml:
    required_tags:
      - "<session_context>"
      - "<vault_structure>"
      - "<available_workspaces>"
```

## Scope Types

### `system_prompt`
- Evaluates the system role message in conversations
- Content: String (system prompt text)
- Use for: Format validation, completeness checks

### `thinking`
- Evaluates the `<thinking>` block in assistant responses
- Content: Dict (JSON structure inside thinking tags)
- Use for: Goal clarity, plan quality, confidence calibration

### `response`
- Evaluates the full assistant response (text + tool calls)
- Content: Dict with "content" (string) and "tool_calls" (array)
- Use for: Tool alignment, destructive safety, factuality

## Schema Validation Types

### XML Validation
```yaml
schema_validation:
  types: [xml]
  xml:
    required_tags:
      - "<tag_name>"
      - "<another_tag>"
```

### JSON Validation
```yaml
schema_validation:
  types: [json]
  json:
    # Standalone JSON
    required_fields: [field1, field2]

    # JSON inside XML tags
    sections:
      - tag: workspace_info
        extract_pattern: '\{[\s\S]*\}'
        required_fields: [context, structure]
```

### Regex Validation
```yaml
schema_validation:
  types: [regex]
  regex:
    patterns:
      - pattern: 'tool_call:\s*\w+'
        description: "Must have tool_call: format"
```

### Multiple Types
```yaml
schema_validation:
  types: [xml, json]  # Both validators run
  xml:
    required_tags: ["<selected_workspace>"]
  json:
    sections:
      - tag: selected_workspace
        required_fields: [context, workspaceStructure]
```

## Judge Output Schema

The `output_schema` defines what JSON the judge must return. Must have:
- `type: object`
- `properties` with score field(s)
- Each score field: `type: number`, `minimum: 0.0`, `maximum: 1.0`
- `required` array with score field name(s)

The engine looks for the first score field to determine pass/fail.

## Template Variables

Available in `improver_prompt`:

- `{current_content}` - The content being improved (string for system_prompt, dict for thinking)
- `{feedback}` - The judge's feedback on what's wrong
- `{format_instructions}` - The `format_spec` content (if defined)

## Best Practices

1. **Judge Prompts**
   - Be specific about scoring criteria
   - Provide examples of good (1.0) vs bad (0.0)
   - Always specify exact JSON output format

2. **Improver Prompts**
   - Include both current content and feedback
   - Reference format_spec for detailed requirements
   - Request raw output (no markdown wrappers)

3. **Schema Validation**
   - Use for structural checks (tags, fields exist)
   - Run before judge to catch obvious errors
   - Provides specific error messages to judge

4. **Pass Thresholds**
   - 0.8 = Normal strictness
   - 0.9 = High strictness (use for critical safety checks)
   - 0.7 = Lenient (use for style preferences)

5. **Scope Selection**
   - `system_prompt` = Format and completeness
   - `thinking` = Reasoning quality, structure
   - `response` = Tool calls, safety, factuality

## Skip Files

The following files are NOT rubrics and are automatically skipped:
- `quality_labels.yaml` - Labeling configuration
- `README.md` - This file

To skip additional files, configure `skip_files` in code.

## File Naming

- Use snake_case: `system_prompt_format.yaml`
- Rubric key = filename stem (without .yaml)
- Rubric name can be any display string

## Testing Rubrics

Test a rubric on examples:

```bash
python -m improvement_engine.services.rubric_runner \
  --file dataset.jsonl \
  --output improved.jsonl \
  --rubrics system_prompt_format,thinking_quality \
  --backend lmstudio \
  --start-line 1 --end-line 10 \
  --max-iterations 3
```

List available rubrics:

```bash
python -m improvement_engine.services.rubric_runner --list
```

## Common Patterns

### Judge-Only Rubric
```yaml
# Omit improver_prompt for evaluation-only rubrics
name: Factuality Check
# ... judge_prompt, output_schema ...
# NO improver_prompt = this rubric only evaluates, doesn't improve
```

### Rubric with Schema Pre-Validation
```yaml
# schema_validation runs BEFORE judge
# Results are shown to judge in prompt
schema_validation:
  types: [xml, json]
  xml:
    required_tags: ["<tag>"]
  json:
    required_fields: [field1]
```

### Multi-Criteria Rubric
```yaml
# Judge can check multiple aspects, output single score
judge_prompt: |
  Check:
  1. Format correctness
  2. Content quality
  3. Completeness

  Average score: {"overall_score": 0.0-1.0}
```

---

**Note:** All rubric files are cached after first load for performance. Restart the runner to reload changes.
