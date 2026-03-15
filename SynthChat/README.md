# SynthChat

Synthetic dataset generation and improvement for training data.

## Quick Start

```bash
# From repo root
cd /path/to/Synthetic-Conversations

# Generate examples (uses settings.yaml defaults)
PYTHONPATH=. python3 -m SynthChat.run generate \
  --targets-file scratch/fixtures/synthchat/test_content_targets.json \
  --output scratch/fixtures/synthchat/output.jsonl

# Improve existing dataset
PYTHONPATH=. python3 -m SynthChat.run improve \
  --input Datasets/existing.jsonl \
  --rubrics system_prompt_format,tool_alignment

# Validate without improving
PYTHONPATH=. python3 -m SynthChat.run validate \
  --input Datasets/existing.jsonl \
  --rubrics system_prompt_format
```

Before using a command from memory, confirm the live CLI surface first:

```bash
python tuner.py --help
python -m SynthChat.run --help
python -m SynthChat.run generate --help
```

## Configuration

### Two-Layer Config System

1. **settings.yaml** - Default configuration (providers, models, iterations)
2. **CLI args** - Override specific settings per run

### settings.yaml Location

```
SynthChat/config/settings.yaml
```

### Key Settings

```yaml
llm:
  # Generation LLM (creates raw examples)
  generation:
    provider: openrouter          # lmstudio | ollama | openrouter | unsloth
    model: google/gemini-2.0-flash-001
    temperature: 0.7
    max_tokens: 4096

  # Improvement LLM (judges and improves - needs strong model)
  improvement:
    provider: openrouter
    model: openai/gpt-4o
    temperature: 0.1              # Low for deterministic judging

improvement:
  max_iterations: 10              # Max improvement loops per example
  default_rubrics:                # Applied when no --rubrics specified
    - system_prompt_format
    - tool_alignment

generation:
  stage_validation: true          # Validate each stage before proceeding
```

### Provider Options

| Provider | Use Case | Config |
|----------|----------|--------|
| `openrouter` | Cloud models (recommended) | Requires `OPENROUTER_API_KEY` in `.env` |
| `lmstudio` | Local models | Set `host` and `port` |
| `ollama` | Local models | Set `host` (default: localhost:11434) |
| `unsloth` | Fine-tuned LoRA | Set `model` to adapter path |

## Commands

### generate

Create new synthetic examples from scenario definitions.

```bash
PYTHONPATH=. python3 -m SynthChat.run generate [options]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--config-dir` | Config directory | `SynthChat/config` |
| `--scenarios-dir` | Scenarios directory | `SynthChat/scenarios` |
| `--rubrics-dir` | Rubrics directory | `SynthChat/rubrics` |
| `--output, -o` | Output JSONL file | Auto-generated with timestamp |
| `--targets-file` | JSON with scenario counts | Uses `defaults.targets` from settings |
| `--scenarios` | Filter to specific scenarios | All in targets |
| `--max-iterations` | Override max improvement loops | From settings.yaml |
| `--provider` | LLM provider (lmstudio, ollama, openrouter) | From settings.yaml |
| `--model` | Model name | From settings.yaml |
| `--docs` | Seed docs for generation | None |
| `--per-doc` | Examples per doc | 1 |
| `--env-backend` | Environment execution backend (`none`, `local`, `e2b`) | settings.yaml / disabled |
| `--env-template` | E2B template ID (when backend is `e2b`) | None |
| `--env-tool-schema` | Path to custom tool schema YAML | settings.yaml / default |
| `--env-exec-config` | Path to custom execution-rules YAML | settings.yaml / default |

Environment execution behavior (tool-action hints, inference rules) is config-driven via:
`Evaluator/config/environment_execution.yaml`
and can be replaced per run with `--env-tool-schema` / `--env-exec-config`.

For environment-backed tool-use generation, scenarios can opt into structured
generation instead of relying on unconstrained freeform JSON:

- `environment_generation.schema: canonical_environment`
- `assistant_generation.schema: use_tools_response`

Use that path when you want executable tool data, not just tool-shaped text.

Important pattern:
- The tool wrapper name is schema-driven, not hardcoded.
- Core generation/execution reads the wrapper from the configured tool schema
  (`--env-tool-schema` or the default evaluator schema).
- Keep wrapper choice in config/YAML. Do not add code branches for one wrapper
  name or one scenario pack.

**Example: Generate 5 content writing examples**

```bash
# Create targets file
echo '{
  "contentManager_write_research": 1,
  "contentManager_write_blog": 1,
  "contentManager_write_creative": 1,
  "contentManager_write_documentation": 1,
  "contentManager_write_journal": 1
}' > targets.json

# Generate
PYTHONPATH=. python3 -m SynthChat.run generate \
  --targets-file targets.json \
  --output docs/content_examples.jsonl \
  --max-iterations 3
```

### improve

Improve existing dataset using rubrics.

```bash
PYTHONPATH=. python3 -m SynthChat.run improve [options]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Input JSONL file (required) | - |
| `--output, -o` | Output JSONL file | Input with timestamp |
| `--rubrics` | Comma-separated rubric names | From settings.yaml |
| `--start-line` | Start line (1-indexed) | 1 |
| `--end-line` | End line (inclusive) | Last line |
| `--max-iterations` | Max improvement loops | From settings.yaml |
| `--provider` | LLM provider (lmstudio, ollama, openrouter) | From settings.yaml |
| `--model` | Model name | From settings.yaml |

**Example: Improve lines 1-10 with specific rubrics**

```bash
PYTHONPATH=. python3 -m SynthChat.run improve \
  --input Datasets/raw.jsonl \
  --output Datasets/improved.jsonl \
  --rubrics system_prompt_format,content_writing_quality \
  --start-line 1 \
  --end-line 10 \
  --max-iterations 5
```

### validate

Check dataset against rubrics without modifying.

```bash
PYTHONPATH=. python3 -m SynthChat.run validate [options]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Input JSONL file (required) | - |
| `--rubrics` | Comma-separated rubric names | From settings.yaml |
| `--provider` | LLM provider (lmstudio, ollama, openrouter) | From settings.yaml |
| `--model` | Model name | From settings.yaml |

**Example: Validate all examples**

```bash
PYTHONPATH=. python3 -m SynthChat.run validate \
  --input Datasets/my_dataset.jsonl \
  --rubrics system_prompt_format,tool_alignment
```

## Directory Structure

```
SynthChat/
├── config/
│   ├── settings.yaml       # Main configuration
│   └── validation.yaml     # Validation settings
├── scenarios/              # Scenario definitions (YAML)
│   ├── content_writing.yaml
│   └── tool_scenarios.yaml
├── rubrics/                # Quality rubrics (YAML)
│   ├── README.md           # Rubric format documentation
│   ├── system_prompt_format.yaml
│   ├── tool_alignment.yaml
│   └── content_writing_quality.yaml
├── interactions/           # Logged judge/improver interactions (for KTO)
├── run.py                  # CLI entry point
├── generator.py            # Generation pipeline
└── engine.py               # Improvement engine
```

## Scenarios

Scenarios define what to generate. Each scenario specifies:
- Type (tool, behavior, chat)
- Prompts for each stage (system, user, thinking, assistant)
- Rubrics to validate each stage

SynthChat also supports an optional structured environment-generation stage for
tool-use scenarios. This lets a scenario generate a filesystem-like environment
first, then render a production-style mocked system prompt from that environment
before generating the user and assistant turns.

Those environment-backed scenarios can target either:

- one-shot tool responses
- multi-step evaluator episodes, where the same runtime persists across turns
  and tool outputs are fed back to the model

The loop itself is owned by the evaluator's `environment.loop` config, not by
SynthChat-specific code or one hardcoded tool wrapper.

Environment sourcing is explicit via `environment_mode`:

- `provided`: use the hand-authored `environment` block only
- `generated`: generate the environment from `environment_generation`
- `hybrid`: start from the hand-authored `environment` block and merge generated
  values over it

Example shape:

```yaml
scenarios:
  envfs_update_config_note:
    type: tool
    tool: contentManager_update
    environment_mode: generated
    system_template: mocked_workspace_vault
    system_context:
      available_workspaces:
        - id: ws_generated_ops
          name: Operations Workspace
          description: Operational notes
          root_folder: ""
    environment_generation:
      schema: canonical_environment
      prompt: |
        Output JSON with:
        - environment.fixture (directories/files/notes)
        - environment.assertions
        - system_context.session_id / workspace_id / selected_workspace
    assistant_generation:
      schema: use_tools_response
    prompts:
      user: |
        Generate the user request.
      assistant: |
        Generate the assistant tool response as JSON.
```

For hybrid scenarios:

```yaml
scenarios:
  envfs_hybrid_case:
    type: tool
    tool: contentManager_update
    environment_mode: hybrid
    environment:
      fixture:
        directories: ["Ops"]
    environment_generation:
      prompt: |
        Output JSON with environment.fixture.files and assertions.
```

Generated examples store the environment spec under
`metadata.generated_environment`. If environment validation is enabled, the
runtime execution trace still appears under `metadata.environment`.

Generated examples also store filterable labels under `metadata.labels`:

- `metadata.labels.flat`: simple tags such as `scenario:...`,
  `environment_passed:true`, `kto_candidate:negative`,
  `issue:missing_expected_tool`
- `metadata.labels.filter`: structured fields for slicing datasets later, such
  as `scenario_key`, `tool_name`, `stage_failures`, `issue_labels`,
  `executed_tools`, and `kto_candidate_label`

This is intended for downstream filtering, not just immediate training labels.
For example, you can keep all environment-backed generations, then later choose
to train KTO only on:

- strong positives: `kto_candidate_label == true`
- behaviorally meaningful negatives: `kto_candidate_label == false`
- excluding noisy cases: `schema_error` only, or other harness-only failures

See `SynthChat/scenarios/content_writing.yaml` for examples.
See `SynthChat/scenarios/tool_environments.yaml` for environment-backed examples.

### Canonical Assertion Types

When using structured environment generation, keep assertions to the canonical
types currently supported by the environment validator:

- `path_exists`
- `path_not_exists`
- `file_contains`
- `file_not_contains`
- `dir_contains`
- `frontmatter_has_key`
- `frontmatter_field_equals`
- `frontmatter_field_contains`

If the generator produces assertion names like `exists`, `yaml_front_matter`,
or `file_unchanged`, the validator will treat them as unknown.

### HF Pilot Flow

The current remote-first pattern for a small SynthChat pilot is:

```bash
python tuner.py cloud-run --job-config Trainers/cloud/jobs/synthchat_vault_kto_pilot.yaml
```

That job:

- runs SynthChat remotely on HF Jobs
- writes dataset output under `/workspace/outputs/synthchat/`
- syncs the dataset JSONL to the configured HF Bucket prefix
- copies any `interactions_*.jsonl` files it finds into the same synced output

Use a small pilot first. A 10-example batch is the right smoke test before
scaling to larger counts.

## Rubrics

Rubrics define quality criteria. Each rubric has:
- Validations (schema checks, regex, code validators)
- Judge prompt (LLM-based quality assessment)
- Pass threshold (0.0-1.0)

See `SynthChat/rubrics/README.md` for format documentation.

## Environment Variables

Create `.env` in repo root:

```bash
# Required for OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Optional for LM Studio (if using local)
LMSTUDIO_HOST=localhost
LMSTUDIO_PORT=1234

# Optional for E2B-backed environment execution
E2B_API_KEY=e2b_...
```

## Output Format

Generated JSONL with metadata header:

```jsonl
{"_meta": {"synthchat_version": "1.0.0", "generated_at": "...", "stats": {...}}}
{"conversations": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": null, "tool_calls": [...]}], "metadata": {...}}
```

The first line may be a metadata/header record. Skip it when counting examples
or sampling training rows.

## Troubleshooting

### ModuleNotFoundError: No module named 'shared'

Run with `PYTHONPATH=.`:
```bash
PYTHONPATH=. python3 -m SynthChat.run generate ...
```

### Provider not working

Check settings.yaml has correct provider config and `.env` has API keys.

### Validation failing

1. Check rubric exists: `ls SynthChat/rubrics/`
2. Validate rubric format: See `SynthChat/rubrics/README.md`
3. Run validate mode to see specific errors
