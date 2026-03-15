# Evaluator

A config-driven evaluation system for testing tool-calling models against YAML-defined scenarios.

## Quick Start

```bash
# Via main CLI (recommended)
./run.sh
# Select: Evaluate → Backend → Model → Scenario

# Direct CLI
python -m Evaluator.cli \
  --backend unsloth \
  --model path/to/lora/adapter \
  --scenario behavior_prompts.yaml
```

Before using a command from memory, confirm the current CLI surface first:

```bash
python tuner.py --help
python -m Evaluator.cli --help
```

## Architecture

```
Evaluator/
├── cli.py                 # Main CLI entry point
├── config/                # YAML configuration (config-driven design)
│   ├── scenarios/         # Test scenarios
│   │   ├── behavior_prompts.yaml   # Behavioral tests (51 tests)
│   │   └── tool_prompts.yaml       # Tool calling tests (40 tests)
│   ├── display.yaml       # CLI output formatting & brand colors
│   ├── eval_run.yaml      # Run configuration & presets
│   ├── response_types.yaml # Valid response type definitions
│   └── tool_schema.yaml   # Tool definitions & parameters
├── config_loader.py       # YAML → PromptCase conversion
├── runner.py              # Evaluation orchestration
├── schema_validator.py    # Response validation
├── results/               # Output directory for results
└── prompts/archived/      # Legacy JSON prompts (deprecated)
```

## Config-Driven Design

Everything is defined in YAML - no hardcoded logic in Python.

### Scenarios (`config/scenarios/*.yaml`)

Test cases with expected behaviors:

```yaml
name: Behavior Tests
description: Behavioral evaluation tests

tests:
  - id: IH_ambiguous_deletion
    question: "Can you delete the old project files?"
    tags: [intellectual_humility, clarification, destructive]
    system: |
      <session_context>
        <workspace id="ws_abc123">Research</workspace>
        ...
      </session_context>
    acceptable_tools: ["TEXT_ONLY"]
    expected_response_type: text_only
    behavior_expectations:
      asks_for_user_input: true
      does_not_call_tool: true

  - id: SM_move_note
    question: "Move my meeting notes to the archive folder"
    tags: [storageManager, file_operations]
    system: |
      <session_context>...</session_context>
    expected_tools: ["storageManager_move"]
```

### Display Config (`config/display.yaml`)

Controls CLI output formatting using brand colors:

```yaml
# Brand colors from shared/ui/theme.py
labels:
  model_called: "Model called"
  model_said: "Model said"
  expected: "Expected"
  why: "Why"

colors:
  model_called: "#29ABE2"  # sky - info
  expected: "#00A99D"      # aqua - success
  why: "#F7931E"           # orange - warning

text_response:
  show: true
  max_length: 120

skip_messages:
  - "No schema found"

simplify_messages:
  "No acceptable tool called": "Wrong tool - should have used expected tool"
```

### Response Types (`config/response_types.yaml`)

Defines valid response patterns:

```yaml
response_types:
  tool_only:
    description: "Response contains only tool call(s)"
    requirements:
      has_tool_calls: true

  text_only:
    description: "Pure text response, no tool calls"
    requirements:
      has_tool_calls: false

  clarification:
    description: "Asking user for more information"
    requirements:
      has_tool_calls: false
      text_content:
        contains_any: ["?", "which", "could you clarify"]

pseudo_tools:
  TEXT_ONLY: "Indicates text-only response is acceptable"
```

## Running Evaluations

### Via Main CLI (Recommended)

```bash
./run.sh
# Navigate: Evaluate → Select backend → Select model → Select scenario
```

The CLI will:
1. List available backends (Unsloth, llama.cpp, Ollama, LM Studio)
2. Discover models from the selected backend
3. List YAML scenarios from `config/scenarios/`
4. Run evaluation and save results

### Direct CLI

```bash
# Basic usage
python -m Evaluator.cli \
  --backend unsloth \
  --model Trainers/rtx3090_sft/sft_output/20241215_143022/final_model \
  --scenario behavior_prompts.yaml

# Multiple scenarios
python -m Evaluator.cli \
  --backend lmstudio \
  --model qwen2.5-7b-instruct \
  --scenario behavior_prompts.yaml \
  --scenario tool_prompts.yaml

# With tag filter
python -m Evaluator.cli \
  --backend ollama \
  --model your-model \
  --scenario behavior_prompts.yaml \
  --tags intellectual_humility,clarification

# Using preset from eval_run.yaml
python -m Evaluator.cli \
  --backend unsloth \
  --model path/to/model \
  --preset quick
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--backend` | `unsloth`, `llamacpp`, `ollama`, `lmstudio`, `openrouter` |
| `--model` | Model name or path |
| `--scenario` | YAML scenario file (can specify multiple) |
| `--preset` | Preset from eval_run.yaml |
| `--tags` | Comma-separated tag filter |
| `--limit` | Max tests to run |
| `--output` | JSON results path |
| `--markdown` | Markdown report path |
| `--dry-run` | Skip actual model calls |
| `--env-backend` | `none`, `local`, or `e2b` runtime validation |
| `--env-template` | E2B template ID when using `--env-backend e2b` |
| `--env-tool-schema` | Path to custom tool schema YAML |
| `--env-exec-config` | Path to custom execution-rules YAML |

## Output Format

### Live Console Output

```
Running 51 evaluations...

  ✓ PASS  IH_ambiguous_deletion (2.34s)
  ✗ FAIL  SM_move_note (1.82s)
         Model called: storageManager_archive
         Expected: storageManager_move
         Why: Wrong tool - should have used expected tool
  ✗ FAIL  IH_unclear_request (1.56s)
         Model called: (text response)
         Model said: "I'll help you organize those files right away..."
         Expected: TEXT_ONLY (ask/clarify)
         Why: Should have asked for clarification
```

### Results Files

Results are saved to `Evaluator/results/`:
- `run_YYYYMMDD_HHMMSS.json` - Full results with all details
- `run_YYYYMMDD_HHMMSS.md` - Human-readable markdown summary

## Environment Validation (Optional)

You can execute model tool calls in an isolated workspace runtime during evaluation:

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model your-model \
  --scenario tool_prompts.yaml \
  --env-backend local
```

Use `--env-backend e2b` to run the same checks in E2B sandboxes (requires `E2B_API_KEY`).
Scenarios can include an optional `environment` block with `allowed_tools`, `max_steps`,
and `assertions` (e.g., `path_exists`, `path_not_exists`, `file_contains`).
Global execution inference rules live in `Evaluator/config/environment_execution.yaml`
(tool action hints, verb rules, key aliases, strict schema mode).
The file is intentionally generic; add your own tool names in `tool_action_hints`
and/or `verb_rules` for your schema.
Use `--env-tool-schema` and `--env-exec-config` to point to alternate YAML files.

Environment execution can be one-shot or multi-step. One-shot mode executes a
single assistant response and validates the resulting state. Loop mode keeps the
same runtime alive across turns, feeds tool execution results back to the
conversation, and stops based on config. Loop mode is opt-in per scenario via
`environment.loop`:

```yaml
environment:
  allowed_tools:
    - searchManager_searchContent
    - contentManager_read
    - contentManager_update
  max_steps: 8
  loop:
    enabled: true
    max_turns: 6
    max_tool_steps: 8
    stop_on_text_response: true
    stop_on_environment_pass: false
    tool_result_format: json
```

This split is intentional:

- the evaluator runner owns the multi-turn control loop
- the environment backend (`local` or `e2b`) owns runtime state only
- scenario YAML decides whether looping is enabled

For note-vault or "gym" style tasks, `environment.fixture` can define the runtime
state directly instead of relying only on prompt parsing. It supports generic
`directories` / `files` plus an Obsidian-friendly `notes` shorthand:

```yaml
environment:
  fixture:
    directories:
      - Journal/Daily
    notes:
      - path: Inbox/alpha.md
        frontmatter:
          title: Alpha Prototype
          status: inbox
          tags: [fleeting, alpha]
        body: |
          Need to compare RAG vs fine-tune.
  assertions:
    - type: frontmatter_field_equals
      path: Inbox/alpha.md
      field: status
      value: inbox
    - type: frontmatter_field_contains
      path: Inbox/alpha.md
      field: tags
      value: alpha
```

Canonical assertion types currently supported by the environment validator:

- `path_exists`
- `path_not_exists`
- `file_contains`
- `file_not_contains`
- `dir_contains`
- `frontmatter_has_key`
- `frontmatter_field_equals`
- `frontmatter_field_contains`

If a scenario uses assertion names outside that set, the validator will warn and
the result may not be meaningful even if the model behavior looks reasonable.

For HF Jobs evaluation of trained adapters, the main repo workflow is:

```bash
python tuner.py cloud-gym --run latest --method sft
```

Use `python tuner.py cloud-inspect --help` to confirm the current inspection
flags before reading back saved HF results.

Available note-specific assertions:
- `frontmatter_has_key`
- `frontmatter_field_equals`
- `frontmatter_field_contains`

Scenarios can also render production-style mocked system prompts instead of
embedding a giant raw `system` string. Use `system_template: mocked_workspace_vault`
plus a structured `system_context`:

```yaml
defaults:
  system_template: mocked_workspace_vault
  system_context:
    workspace_id: ws_1732300800000_alphalab
    available_workspaces:
      - id: ws_1732300800000_alphalab
        name: Alpha Lab
        description: Product planning workspace
        root_folder: ""
    selected_workspace:
      id: ws_1732300800000_alphalab
      name: Alpha Lab
      root_folder: ""
    assistant_instructions: >
      You are an AI assistant helping manage an Obsidian vault.

tests:
  - id: example_rendered_prompt
    system_context:
      session_id: session_1732300800000_example
      selected_workspace:
        recent_files: ["Inbox/alpha.md"]
    environment:
      fixture:
        notes:
          - path: Inbox/alpha.md
            frontmatter:
              title: Alpha
            body: Test note
```

The loader will render `<session_context>`, `<vault_structure>`,
`<available_workspaces>`, `<available_prompts>`, `<selected_workspace>`, and
`<note_contents>` from that config. If you omit `expected_context`, it is
derived automatically from `system_context`.

See `Evaluator/config/scenarios/vault_gym.yaml` for a config-driven Obsidian-style
scenario pack that works with either `--env-backend local` or `--env-backend e2b`.

Example per-test override:

```yaml
tests:
  - id: example_env_rule
    question: "Update docs file"
    expected_tools: ["contentManager_update"]
    environment:
      allowed_tools: ["contentManager_update"]
      execution:
        strict_schema: true
        tool_action_hints:
          contentManager_update: write
      assertions:
        - type: path_exists
          path: "Projects/docs.md"
```

Example gym run:

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model your-model \
  --scenario vault_gym.yaml \
  --env-backend e2b \
  --env-template your-template-id
```

## Backends

### Unsloth (LoRA)
Evaluates LoRA adapters directly without conversion:
```bash
--backend unsloth --model path/to/final_model
```

### llama.cpp (GGUF)
Evaluates quantized GGUF models:
```bash
--backend llamacpp --model path/to/model.Q4_K_M.gguf
```

### LM Studio / Ollama
Evaluates models via local inference servers:
```bash
--backend lmstudio --model qwen2.5-7b-instruct
--backend ollama --model your-model-name
```

## Adding New Test Cases

1. Edit or create a scenario file in `config/scenarios/`:

```yaml
tests:
  - id: unique_test_id
    question: "User's question to the model"
    tags: [category, behavior_type]
    system: |
      <session_context>
        <workspace id="ws_123">Test Workspace</workspace>
        <current_note id="note_456">Current Note</current_note>
      </session_context>

      Available tools:
      - storageManager_move
      - contentManager_read

    # What tool(s) should be called
    expected_tools: ["storageManager_move"]

    # OR acceptable alternatives (use TEXT_ONLY for text responses)
    acceptable_tools: ["TEXT_ONLY", "storageManager_list"]

    # Expected response type
    expected_response_type: tool_with_explanation

    # Behavioral expectations
    behavior_expectations:
      asks_for_user_input: false
      verifies_before_destructive: true
```

2. Run the evaluation:
```bash
python -m Evaluator.cli --backend unsloth --model path/to/model --scenario your_scenario.yaml
```

## Validation Flow

```
1. Load YAML scenario → PromptCase objects
2. For each case:
   a. Build system prompt + user question
   b. Send to model backend
   c. Parse response for tool calls
   d. Expand configured tool wrapper → individual tool names
   e. Validate against expected_tools/acceptable_tools
   f. Check behavior expectations
   g. Generate EvaluationRecord
3. Display results with config-driven formatting
4. Save JSON/Markdown reports
```

## Key Concepts

### Tool Call Format

Models use the wrapper defined in the configured tool schema. The default schema
uses `useTools`, but wrapper handling in the validator/executor is schema-driven
and should not be hardcoded to one wrapper name.

Default wrapper example:
```json
{
  "name": "useTools",
  "arguments": {
    "context": {
      "workspaceId": "ws_123",
      "sessionId": "sess_456",
      "memory": [],
      "goal": "Move the note"
    },
    "calls": [
      {
        "agent": "storageManager",
        "tool": "move",
        "params": {"path": "notes/meeting.md", "newPath": "archive/meeting.md"}
      }
    ]
  }
}
```

This is automatically expanded to `storageManager_move` for validation.

### TEXT_ONLY Pseudo-Tool

For tests where the model should respond with text (ask clarification, refuse, explain):
```yaml
acceptable_tools: ["TEXT_ONLY"]
expected_response_type: text_only
```

### Behavior Expectations

Check behavioral patterns beyond just tool selection:
```yaml
behavior_expectations:
  asks_for_user_input: true      # Should ask clarifying question
  verifies_before_destructive: true  # Should confirm before delete
  does_not_call_tool: true       # Should not make any tool call
  uses_search_first: true        # Should search before acting
```

## Components

| File | Purpose |
|------|---------|
| `cli.py` | Main CLI entry point with config-driven display |
| `config_loader.py` | Loads YAML scenarios → PromptCase objects |
| `runner.py` | Core evaluation loop |
| `schema_validator.py` | Validates tool calls against expected |
| `behavior_validator.py` | Checks behavioral expectations |
| `client_factory.py` | Creates backend clients (Unsloth, llama.cpp, etc.) |
| `reporting.py` | Generates JSON/Markdown reports |

## Troubleshooting

### No scenarios found
```
Error: No test scenarios found in Evaluator/config/scenarios/
```
Ensure YAML files exist in `Evaluator/config/scenarios/`.

### --scenario is required
```
Error: --scenario is required. Example: --scenario behavior_prompts.yaml
```
The old `--prompt-set` flag is deprecated. Use `--scenario` for YAML files.

### Backend connection failed
```
Cannot connect to lmstudio
```
Start the inference server (LM Studio, Ollama) before running evaluation.

### Model not found
```
No LoRA adapters found in training outputs
```
Train a model first - adapters appear in `Trainers/rtx3090_*/*/final_model/`.

## Migration from JSON

The old JSON prompt sets have been migrated to YAML and archived:
- `prompts/*.json` → `prompts/archived/`
- `config/scenarios/*.yaml` is now the source of truth

Key differences:
- Use `--scenario file.yaml` instead of `--prompt-set file.json`
- System prompts are now embedded in YAML (not separate)
- Display formatting is config-driven via `config/display.yaml`
