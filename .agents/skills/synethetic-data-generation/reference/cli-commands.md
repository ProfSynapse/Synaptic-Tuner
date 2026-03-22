# CLI Commands Reference

## Entry Points

```bash
python -m SynthChat.run [generate|improve|validate] [options]   # Direct
python -m Evaluator.cli [options]                                # Evaluator
./run.sh                                                         # Interactive menu
python tuner.py                                                  # Main CLI
```

---

## `generate` — Create New Dataset

Generates examples from scenario templates, running each through judge/improve loop.

```bash
python -m SynthChat.run generate [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config-dir PATH` | Config directory | `SynthChat/config` |
| `--scenarios-dir PATH` | Scenarios directory | `SynthChat/scenarios` |
| `--rubrics-dir PATH` | Rubrics directory | `SynthChat/rubrics` |
| `--output, -o PATH` | Output JSONL file | Auto-generated with timestamp |
| `--targets-file PATH` | JSON file with `{scenario_key: count}` | Uses `defaults.targets` from settings.yaml |
| `--scenarios KEY [KEY...]` | Generate only these scenarios (filters targets) | All targets |
| `--max-iterations N` | Max judge/improve loops per example | From settings.yaml |
| `--provider PROVIDER` | LLM provider override (`openrouter`, `lmstudio`, `ollama`, `unsloth`) | From settings.yaml |
| `--model MODEL` | Model name override | From settings.yaml |
| `--workers, -w N` | Parallel worker threads | `1` |
| `--docs PATH` | Doc file or folder for docs-based generation | None |
| `--per-doc N` | Examples to generate per doc | `1` |
| `--env-backend` | Runtime validation backend (`none`, `local`, `e2b`) | From settings / `none` |
| `--env-template` | E2B template ID (for `--env-backend e2b`) | None |
| `--env-timeout` | Runtime timeout in seconds | From settings / `120` |
| `--env-api-key` | E2B API key override | `E2B_API_KEY` env var |
| `--env-tool-schema` | Path to custom tool schema YAML | From settings / default |
| `--env-exec-config` | Path to custom execution-rules YAML | From settings / default |

**Examples:**

```bash
# Full dataset with defaults
python -m SynthChat.run generate

# Specific scenarios, 4 workers
python -m SynthChat.run generate --scenarios storageManager_createFolder storageManager_move --workers 4

# OpenRouter with specific model
python -m SynthChat.run generate --provider openrouter --model google/gemini-2.0-flash-001

# Docs-based generation
python -m SynthChat.run generate --docs "essays/" --scenarios essay_outline --per-doc 1

# Custom targets file, limited iterations
python -m SynthChat.run generate --targets-file my_targets.json --max-iterations 3 -o test_run.jsonl

# Enable local environment validation
python -m SynthChat.run generate --env-backend local

# Bring your own tool schema and execution rules
python -m SynthChat.run generate \
  --env-backend local \
  --env-tool-schema ./my_config/tool_schema.yaml \
  --env-exec-config ./my_config/environment_execution.yaml
```

---

## `improve` — Improve Existing Dataset

Runs existing JSONL examples through judge/improve loop against selected rubrics.

```bash
python -m SynthChat.run improve [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input, -i PATH` | **Required.** Input JSONL file | — |
| `--output, -o PATH` | Output JSONL file | Auto-timestamped |
| `--config-dir PATH` | Config directory | `SynthChat/config` |
| `--rubrics-dir PATH` | Rubrics directory | `SynthChat/rubrics` |
| `--rubrics NAMES` | Comma-separated rubric names | From settings.yaml `improvement.default_rubrics` |
| `--start-line N` | Start line (1-indexed) | `1` |
| `--end-line N` | End line (inclusive) | Last line |
| `--max-iterations N` | Max improvement loops | From settings.yaml |
| `--provider PROVIDER` | LLM provider override | From settings.yaml |
| `--model MODEL` | Model name override | From settings.yaml |
| `--workers, -w N` | Parallel workers | `1` |

**Examples:**

```bash
# Default rubrics
python -m SynthChat.run improve -i raw.jsonl

# Specific rubrics, line range
python -m SynthChat.run improve -i data.jsonl -o data_v2.jsonl \
  --rubrics system_prompt_format,thinking_quality \
  --start-line 1 --end-line 50

# Powerful model for judging
python -m SynthChat.run improve -i data.jsonl \
  --provider openrouter --model openai/gpt-oss-120b --max-iterations 5
```

---

## `validate` — Check Quality (No Changes)

Judges each example and reports pass/fail without modifying data.

```bash
python -m SynthChat.run validate [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input, -i PATH` | **Required.** Input JSONL file | — |
| `--config-dir PATH` | Config directory | `SynthChat/config` |
| `--rubrics-dir PATH` | Rubrics directory | `SynthChat/rubrics` |
| `--rubrics NAMES` | Comma-separated rubric names | From settings.yaml |
| `--provider PROVIDER` | LLM provider override | From settings.yaml |
| `--model MODEL` | Model name override | From settings.yaml |

**Output:** Pass rate, failing lines, failures grouped by rubric. Suggests `improve` command to fix.

```bash
python -m SynthChat.run validate -i data.jsonl --rubrics system_prompt_format,factuality
```

---

## Evaluator CLI

Evaluates a fine-tuned model against test scenarios.

```bash
python -m Evaluator.cli [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--model NAME` | **Required.** Model name | — |
| `--backend` | `ollama`, `lmstudio`, `llamacpp`, `openrouter`, `mlc`, `unsloth` | `lmstudio` |
| `--config-dir PATH` | Evaluator config dir | `Evaluator/config` |
| `--scenario PATH` | Scenario file(s) | All in config |
| `--preset NAME` | Preset from eval_run.yaml | — |
| `--tags TAG,TAG` | Filter tests by tag | — |
| `--limit N` | Max prompts | All |
| `--temperature FLOAT` | Generation temp | `0.2` |
| `--top-p FLOAT` | Top-p | `0.9` |
| `--max-tokens INT` | Max tokens | `1024` |
| `--seed INT` | Seed | — |
| `--host HOST` | Backend host | `localhost` |
| `--port PORT` | Backend port | `1234` |
| `--output PATH` | JSON results | — |
| `--markdown PATH` | Markdown summary | — |
| `--dry-run` | Skip backend calls | `false` |
| `--validate-context` | Validate IDs from system prompt | `false` |
| `--no-browser` | Don't auto-open | `false` |
| `--env-backend` | Runtime validation backend (`none`, `local`, `e2b`) | `none` |
| `--env-template` | E2B template ID | — |
| `--env-timeout` | Runtime timeout in seconds | `120` |
| `--env-api-key` | E2B API key override | env var |
| `--env-tool-schema` | Custom tool schema YAML path | default config |
| `--env-exec-config` | Custom execution-rules YAML path | default config |

**Examples:**

```bash
# Ollama evaluation
python -m Evaluator.cli --model my-model --backend ollama --port 11434

# Filter by tags, limit count
python -m Evaluator.cli --model my-model --backend lmstudio --tags single-tool --limit 20

# Dry run
python -m Evaluator.cli --model test --dry-run --validate-context

# Runtime validation with custom toolset
python -m Evaluator.cli --model my-model --backend lmstudio \
  --scenario tool_prompts.yaml \
  --env-backend local \
  --env-tool-schema ./my_config/tool_schema.yaml \
  --env-exec-config ./my_config/environment_execution.yaml
```

---

## Structural Validator

Quick JSONL structure check (no LLM needed):

```bash
python tools/validate_syngen.py Datasets/your_dataset.jsonl
```

Checks: valid JSON, conversation structure, tool schemas, parameter validation.
