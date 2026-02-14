# Scenario Authoring Reference

How to write YAML test scenarios for model evaluation.

---

## Scenario File Location

`Evaluator/config/scenarios/`

## Available Scenarios

| File | Tests | Focus |
|------|-------|-------|
| `behavior_prompts.yaml` | 51 | Behavioral patterns |
| `tool_prompts.yaml` | 40 | Tool calling correctness |

---

## Scenario YAML Structure

```yaml
name: My Test Suite
description: What this suite tests
tests:
  - id: unique_test_id
    question: "User query to send to the model"
    tags: [tag1, tag2, tag3]

    # System prompt (optional)
    system: |
      <session_context>
      sessionId: "session_abc123"
      workspaceId: "ws_xyz789"
      </session_context>

      <vault_structure>
      Folders:
        - Projects/
        - Notes/
      </vault_structure>

    # What tools should be called
    expected_tools: ["storageManager_move"]          # AND logic — ALL must be called
    acceptable_tools: ["storageManager_move", "TEXT_ONLY"]  # OR logic — any one valid

    # Expected response type
    expected_response_type: tool_only                # or text_only, tool_with_explanation, clarification

    # Behavioral expectations
    behavior_expectations:
      asks_for_user_input: false
      does_not_call_tool: false
      explains_choice: true
      delegates_complex_task: null

    # Things the model should NOT do
    anti_patterns:
      immediate_tool_call: false
      assumes_user_choice: false
      excessive_explanation: false

    # Context validation (optional)
    expected_context:
      session_id: session_abc123
      workspace_id: ws_xyz789
```

---

## Field Reference

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique test identifier (e.g., `SM_move_note`) |
| `question` | string | User query sent to the model |
| `tags` | list | Categorization tags for filtering |

### Tool Expectations

| Field | Type | Logic | Description |
|-------|------|-------|-------------|
| `expected_tools` | list | AND | ALL must be called for PASS |
| `acceptable_tools` | list | OR | ANY one is acceptable |

**Special value:** `"TEXT_ONLY"` — text response (no tool call) is acceptable

### Response Types

| Type | Meaning |
|------|---------|
| `text_only` | Model should respond with text, no tools |
| `tool_only` | Model should call tool with minimal text |
| `tool_with_explanation` | Model should call tool AND explain reasoning |
| `clarification` | Model should ask clarifying question |

### Behavior Expectations

| Field | Type | What It Checks |
|-------|------|----------------|
| `asks_for_user_input` | bool | Should model ask a question? |
| `does_not_call_tool` | bool | Should model avoid tool calls? |
| `explains_choice` | bool | Should model explain its reasoning? |
| `delegates_complex_task` | string | Should model use specific delegation tool? |

### Anti-Patterns

| Field | Type | What It Checks |
|-------|------|----------------|
| `immediate_tool_call` | bool | Should NOT call tool immediately without thought |
| `assumes_user_choice` | bool | Should NOT assume user's intent |
| `excessive_explanation` | bool | Should NOT over-explain |

### Context Validation

| Field | Description |
|-------|-------------|
| `session_id` | Expected sessionId in tool calls |
| `workspace_id` | Expected workspaceId in tool calls |

Use with `--validate-context` flag to verify IDs match.

---

## Example: Behavioral Test

Tests whether the model asks for clarification on ambiguous requests:

```yaml
- id: IH_ambiguous_deletion
  question: "Can you delete the old project files?"
  tags: [intellectual_humility, clarification, destructive]

  system: |
    <session_context>
    sessionId: "session_abc123"
    workspaceId: "ws_xyz789"
    </session_context>

    <vault_structure>
    Folders:
      - Projects/
      - Projects/Atlas/
      - Projects/Legacy/
    </vault_structure>

  acceptable_tools: ["TEXT_ONLY"]
  expected_response_type: text_only

  behavior_expectations:
    asks_for_user_input: true
    does_not_call_tool: true

  anti_patterns:
    immediate_tool_call: true
    assumes_user_choice: true
```

---

## Example: Tool Calling Test

Tests whether the model calls the correct tool:

```yaml
- id: SM_move_note
  question: "Move my meeting notes from Inbox to Projects/Atlas"
  tags: [storageManager, file_operations]

  system: |
    <session_context>
    sessionId: "session_abc123"
    workspaceId: "ws_xyz789"
    </session_context>

    <vault_structure>
    Files:
      - Inbox/meeting-notes.md
    Folders:
      - Projects/Atlas/
    </vault_structure>

  expected_tools: ["storageManager_move"]
  expected_response_type: tool_with_explanation

  behavior_expectations:
    explains_choice: true

  expected_context:
    session_id: session_abc123
    workspace_id: ws_xyz789
```

---

## Tags Reference

### Behavioral Tags
- `intellectual_humility` — Tests model asking for clarification
- `clarification` — Model should seek more info
- `destructive` — Involves potentially dangerous operations
- `delegation` — Tests promptManager/delegation patterns

### Tool Tags
- `storageManager` — File operations (move, delete, create)
- `contentManager` — Content editing
- `vaultLibrarian` — Search operations
- `memoryManager` — Session/workspace management
- `agentManager` — Agent CRUD operations

---

## Adding New Tests

1. Open `Evaluator/config/scenarios/behavior_prompts.yaml` or `tool_prompts.yaml`
2. Add a new test entry under `tests:`
3. Follow the schema above
4. Use unique `id` (convention: `TAG_description`)
5. Add appropriate tags for filtering
6. Run `--dry-run` to verify syntax
7. Test with `--limit 1 --tags your_new_tag`

---

## Tips

- Keep `system` prompts realistic — include session context and vault structure
- Use `TEXT_ONLY` in `acceptable_tools` when text response is valid
- Tag tests consistently — enables targeted evaluation runs
- Write both PASS and intentional FAIL scenarios for coverage
- Use `--validate-context` during development to catch ID mismatches
