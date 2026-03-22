# Scenario Authoring Reference

**Location:** `SynthChat/scenarios/*.yaml`

Scenarios define WHAT to generate. Each is a template with prompts for system, user, and assistant turns.

---

## Available Scenario Files

| File | Count | Description |
|------|-------|-------------|
| `tools.yaml` | 10 | Tool-calling (storageManager, contentManager, promptManager, memoryManager, searchManager) |
| `behaviors.yaml` | 7 | Behavioral training (humility, error recovery, verification, workspace awareness, etc.) |
| `content_writing.yaml` | 5 | Long-form content (research, blog, creative, docs, journal) |
| `destructive.yaml` | 6 | Destructive operation handling (confirmation, warnings, alternatives) |
| `essay_outline.yaml` | 1 | Docs-based essay outline extraction |
| `docs_test.yaml` | 2 | Docs-based Q&A and debugging |

---

## Scenario Schema

```yaml
scenarios:
  scenario_key:                   # Unique key — used in targets and --scenarios flag
    type: tool | behavioral | destructive | docs_based
    agent: agentName              # Optional: which agent handles this
    tool: toolName                # Optional: specific tool being called
    system: true | false          # Whether to generate a system message
    max_tokens: 4096              # Optional: override for long-form content

    # Optional flags (destructive scenarios)
    flags:
      destructive: true
      risk_level: high | medium | low

    # Optional tool format (content writing scenarios)
    tool_format:
      wrapper: useTools
      context:
        required: [sessionId, workspaceId, memory, goal]
      params:
        required: [path, content]

    # Which rubrics apply
    rubrics:
      system_prompt: [rubric_key1]
      response: [rubric_key1, rubric_key2]

    # Optional environment runtime validation for generated assistant output
    environment:
      allowed_tools: [myAgent_myTool]
      max_steps: 3
      execution:
        strict_schema: true
        tool_action_hints:
          myAgent_myTool: write
      assertions:
        - type: path_exists
          path: "Projects/output.md"

    # Prompt templates
    prompts:
      system: |
        Instructions for generating the system message.
      user: |
        Instructions for generating the user message.
      assistant: |
        Instructions for generating the assistant response.
```

---

## Template Variables

| Variable | Available In | Description |
|----------|-------------|-------------|
| `{doc_content}` | `user_system`, `assistant_system`, `assistant` | Full document text (docs-based only) |
| `{doc_path}` | Any prompt | Path to source document |

---

## Scenario Types

### Tool Scenario

Generates a system prompt + user request + assistant tool call.

```yaml
scenarios:
  myAgent_myTool:
    type: tool
    agent: myAgent
    tool: myTool
    system: true

    rubrics:
      system_prompt: [system_prompt_format]
      response: [tool_alignment, factuality]

    prompts:
      system: |
        Generate a realistic system prompt with:
        - <session_context> with IDs
        - <vault_structure> with folders
        - <available_workspaces>
        - <available_prompts>
        - <selected_workspace> with JSON
        Make it realistic with typical project folders.

      user: |
        Generate a natural user request that requires [action].
        Be vague about exact location — let the AI infer from context.
        OUTPUT ONLY THE REQUEST TEXT.

      assistant: |
        Generate response with:
        1. <thinking> block showing reasoning
        2. Tool call to myAgent_myTool with parameters
        Use correct sessionId/workspaceId from system prompt.
```

### Behavioral Scenario

Trains specific AI behaviors. Usually text-only responses (no tool calls).

```yaml
scenarios:
  my_behavior:
    type: behavioral
    description: "What this behavior teaches"
    system: true

    triggers:                     # What conditions trigger this behavior
      - "Ambiguous references"
      - "Vague requests"

    prompts:
      system: |
        Generate context where [behavior] is needed.
        Include elements that make the situation ambiguous.

      user: |
        Generate request that triggers [behavior].
        Make it genuinely ambiguous — 2+ possible interpretations.

      assistant: |
        Generate text response (NO tool calls) that:
        - Demonstrates [behavior]
        - Lists possible interpretations
        - Asks for clarification
```

### Destructive Scenario

Trains proper handling of high-risk operations.

```yaml
scenarios:
  destructive_case:
    type: destructive
    system: true

    tools:                        # Weighted tool selection
      storageManager_archive: 0.6
      storageManager_archive: 0.4

    prompts:
      system: |
        Generate workspace with deletable targets.

      user: |
        Generate [vague/explicit/bulk] destructive request.

      assistant: |
        [Clarify / Warn / Confirm / Suggest alternative]
        Use <thinking> to show risk assessment.
```

### Docs-Based Scenario

Generates examples seeded from real documents. Uses `--docs` flag.

```yaml
scenarios:
  my_docs_scenario:
    type: docs_based
    system: false                 # Often no system message in output

    prompts:
      # Persona for user-turn LLM (NOT in final output)
      user_system: |
        You are simulating a [persona].
        <source>{doc_content}</source>
        Guidelines for your output...

      user: |
        Generate [request type] based on the source.
        OUTPUT ONLY the request text.

      # Context for assistant-turn LLM
      assistant: |
        You are a [role] helping the user.
        <reference>{doc_content}</reference>
        Generate a structured response...
```

**Run docs-based scenarios:**
```bash
python -m SynthChat.run generate \
  --docs "path/to/docs/" \
  --scenarios my_docs_scenario \
  --per-doc 1
```

---

## Environment Block (Optional)

If generation runs with environment validation enabled (`--env-backend ...` or `settings.yaml`),
scenario-level `environment` config is passed into runtime execution checks.

Common fields:
- `allowed_tools`
- `max_steps`
- `assertions`
- `execution.strict_schema`
- `execution.default_action`
- `execution.tool_action_hints`
- `execution.key_hints`
- `execution.verb_rules`

---

## Adding a New Scenario

1. Choose the right scenario file (or create a new one in `SynthChat/scenarios/`)
2. Add under the `scenarios:` key with a unique key
3. Add matching target in `SynthChat/config/settings.yaml` under `defaults.targets`
4. Test: `python -m SynthChat.run generate --scenarios your_key --max-iterations 3`
5. Validate output: `python -m SynthChat.run validate -i output.jsonl`

---

## Dataset Output Format (ChatML JSONL)

All generated datasets produce:

```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>..."},
    {"role": "user", "content": "User request text"},
    {"role": "assistant", "content": "<thinking>{...}</thinking>\n\nResponse", "tool_calls": [...]}
  ],
  "label": true
}
```

- `label: true` = positive (desirable), `false` = negative (KTO)
- System message optional (behavioral/docs scenarios may omit)
- Tool calls use OpenAI format with `useTools` wrapper
- Thinking blocks are JSON inside `<thinking>` tags
