# Rubric Authoring Reference

**Location:** `SynthChat/rubrics/*.yaml`

Rubrics define HOW to judge and improve generated examples. Each targets a specific quality dimension.

---

## Available Rubrics

| Rubric | Scope | Threshold | What It Checks |
|--------|-------|-----------|----------------|
| `system_prompt_format` | system_prompt | 0.8 | 5 required XML sections, hierarchy consistency |
| `thinking_quality` | thinking | 0.8 | Goal clarity, memory context, req/plan distinction, confidence |
| `tool_alignment` | response | 0.8 | Tool calls match thinking block's goal/plan |
| `response_quality` | response | 0.8 | Response matches risk assessment (confirm if risky) |
| `factuality` | response | 0.8 | All details grounded in system prompt or user request |
| `content_writing_quality` | response | 0.85 | Frontmatter, structure, substance, Obsidian formatting |
| `confidence_calibration` | thinking | 0.8 | Confidence matches operation risk level |
| `context_alignment` | response | 0.8 | Context IDs match system prompt |
| `quality_labels` | response | 0.8 | KTO label assignment (true/false) |
| `destructive_safety` | response | 0.8 | Proper handling of destructive operations |
| `contentManager_tools` | response | 0.8 | ContentManager tool call correctness |
| `commandManager_tools` | response | 0.8 | CommandManager tool call correctness |
| `memoryManager_tools` | response | 0.8 | MemoryManager tool call correctness |
| `promptManager_tools` | response | 0.8 | PromptManager tool call correctness |
| `searchManager_tools` | response | 0.8 | SearchManager tool call correctness |
| `storageManager_tools` | response | 0.8 | StorageManager tool call correctness |
| `createState_transform` | response | 0.8 | State transformation correctness |

---

## Scopes

Each rubric targets a scope — a part of the conversation:

| Scope | Extracts From | How |
|-------|--------------|-----|
| `system_prompt` | System message | By role |
| `user` | User message | By role |
| `thinking` | `<thinking>` block in assistant | Regex pattern |
| `tool_calls` | Tool invocations in assistant | Pattern parsing |
| `response` | Assistant text minus thinking/tools | Exclusion |

Scope definitions live in `SynthChat/config/validation.yaml`.

---

## Rubric Schema

```yaml
name: Human-Readable Name
description: What this rubric evaluates
scope: system_prompt | thinking | response | tool_calls | user
pass_threshold: 0.0-1.0          # Score below this = fail

# JUDGE PROMPT — evaluates content, returns score(s)
judge_prompt: |
  # CONTEXT
  What's being evaluated.

  **Content:** {current_content}
  **System Prompt:** {system_prompt}

  # INSTRUCTIONS
  Evaluation criteria with scoring guidelines.

  # FORMAT
  Return JSON: {"rubrickey_score": 0.0-1.0}

# IMPROVER PROMPT — fixes issues found by judge
improver_prompt: |
  **Current:** {current_content}
  **Feedback:** {feedback}

  Generate improved version.
  Output ONLY the improved content.

# OUTPUT SCHEMA — expected judge output fields
output_schema:
  type: object
  properties:
    rubrickey_score:               # Convention: rubricname_score (lowercase, no spaces)
      type: number
      min: 0.0
      max: 1.0
  required:
    - rubrickey_score
  additionalProperties: false

# VALIDATIONS — structural checks run BEFORE judge (no LLM needed)
validations: [...]                # See Validation Types below

# OPTIONAL — tool call validation
validation:
  tool_calls:
    enabled: true
    check_context_ids: true       # Validate sessionId/workspaceId match system prompt

# OPTIONAL — output format hint for improver
output_format:
  type: assistant_message         # Tells engine to parse as full assistant message
```

---

## Available Template Variables

| Variable | Description |
|----------|-------------|
| `{current_content}` | Content being judged/improved (scope-specific) |
| `{system_prompt}` | The system message |
| `{user_request}` | The user message |
| `{assistant_response}` | The full assistant response |
| `{feedback}` | Judge's feedback (improver prompt only) |
| `{thinking_content}` | Extracted thinking block |
| `{thinking_block}` | Thinking block (for alignment checks) |
| `{original_tool_calls}` | Tool calls as JSON |

---

## Validation Types

Structural checks that run before the LLM judge. Catches issues without burning tokens.

### XML Tag Validation

```yaml
validations:
  - match: session_context
    type: xml
    error: "Missing required <session_context> tag"
```

### Scoped JSON Field Validation

```yaml
  - match: workspaceStructure
    type: json
    in_tag: selected_workspace      # Only check inside this tag
    error: "Missing 'workspaceStructure' in selected_workspace"
```

### Nested JSON Field (Dot Notation)

```yaml
  - match: context.rootFolder
    type: json
    in_tag: selected_workspace
    error: "Missing 'context.rootFolder'"
```

### String Field Validation

```yaml
  - field: goal
    type: string
    min: 10
    error: "Goal must be at least {min} characters."
```

### Array Field Validation

```yaml
  - field: requirements
    type: array
    min: 1
    error: "Must have at least {min} requirement(s)."
```

### Number Field Validation

```yaml
  - field: confidence
    type: number
    min: 0.0
    max: 1.0
    error: "Confidence must be {min}-{max}."
```

### Boolean / Object Field Validation

```yaml
  - field: assessment.risky
    type: boolean
    error: "Must be a boolean."

  - field: assessment
    type: object
    error: "Must be an object."
```

### Regex Validation

```yaml
  - match: "^#{1,3}\\s"
    type: regex
    error: "Content should start with a markdown heading"
```

### Cross-Scope Validation

Checks values extracted from one scope exist in another (e.g., paths in thinking exist in system prompt).

```yaml
  - cross_scope:
      from: thinking                # Extract from this scope
      to: system_prompt             # Validate against this scope
      extract:
        fields: [goal, memory]      # Fields to search
        pattern: 'path/regex'       # Pattern to extract values
      skip_if:                      # Optional: skip matches
        - pattern: 'create.*{value}'
      validate_in: [vault_structure, selected_workspace]
    error: "HALLUCINATED PATH: '{value}' not found"
```

### Tool Structure Validation

```yaml
  - tools:
      useTools:
        context:
          workspaceId: string
          sessionId: string
          memory: string
          goal: string
        calls:
          _item_schema:
            agent: string
            tool: string
            params: object
          _subtools:
            contentManager:
              write:
                _required: [path, content]
                path: string
                content: string
    error: "useTools validation failed: {details}"
```

---

## Writing a New Rubric

### Step-by-step

1. **Choose scope** — what part of the conversation to evaluate
2. **Name the score field** — `yourrubricname_score` (lowercase, no spaces, underscores ok)
3. **Write judge prompt** — returns `{"yourrubricname_score": 0.0-1.0}`
4. **Write improver prompt** — uses `{current_content}` and `{feedback}`
5. **Add validations** — structural checks that don't need an LLM
6. **Set pass_threshold** — typically 0.8

### Example: Heading Structure Rubric

```yaml
name: Heading Structure
description: Validates proper heading hierarchy in content
scope: response
pass_threshold: 0.8

judge_prompt: |
  Evaluate heading structure.

  **Content:** {current_content}

  Check for:
  1. Starts with heading
  2. Hierarchy follows (no skipping levels)
  3. At least 3 headings for substantial content

  Return JSON: {"headingstructure_score": 0.0-1.0}

improver_prompt: |
  Fix heading structure.

  **Current:** {current_content}
  **Feedback:** {feedback}

  Add proper heading hierarchy. Output ONLY improved content.

output_schema:
  type: object
  properties:
    headingstructure_score:
      type: number
      min: 0.0
      max: 1.0
  required:
    - headingstructure_score

validations:
  - match: "^#{1,3}\\s"
    type: regex
    error: "Content should start with a markdown heading"
```

### Testing Your Rubric

```bash
# Create a test JSONL with a few examples, then:
python -m SynthChat.run validate -i test.jsonl --rubrics your_rubric_name

# If failures, try improving:
python -m SynthChat.run improve -i test.jsonl --rubrics your_rubric_name --max-iterations 3
```
