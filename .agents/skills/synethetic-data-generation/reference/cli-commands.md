# CLI Commands Reference

## Entry Points

```bash
python -m SynthChat.run [generate|improve|validate|sanitize] [options]   # Direct
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
| `--privacy-profile NAME` | Privacy preprocess profile for raw docs before generation | None |
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

# OpenRouter with Groq-routed GPT-OSS
python -m SynthChat.run generate --provider openrouter --model openai/gpt-oss-120b

# Docs-based generation
python -m SynthChat.run generate --docs "essays/" --scenarios essay_outline --per-doc 1

# Docs-based generation with privacy preprocessing
python -m SynthChat.run generate --docs tests/fixtures/privacy/raw_seed_docs \
  --targets-file SynthChat/config/targets_privacy_docs_smoke.json \
  --privacy-profile realistic_pseudonyms

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
| `--lines SPEC` | Arbitrary 1-indexed line selectors (`3,7,10-15`) | None |
| `--line-file PATH` | File with line selectors (one per line or comma-separated, `#` comments allowed) | None |
| `--max-iterations N` | Max improvement loops | From settings.yaml |
| `--provider PROVIDER` | LLM provider override | From settings.yaml |
| `--model MODEL` | Model name override | From settings.yaml |
| `--workers, -w N` | Parallel workers | `1` |
| `--privacy-profile NAME` | Privacy preprocess profile for input JSONL before improvement | None |

**Examples:**

```bash
# Default rubrics
python -m SynthChat.run improve -i raw.jsonl

# Specific rubrics, line range
python -m SynthChat.run improve -i data.jsonl -o data_v2.jsonl \
  --rubrics system_prompt_format,thinking_quality \
  --start-line 1 --end-line 50

# Arbitrary scattered rows
python -m SynthChat.run improve -i data.jsonl -o regen_slice.jsonl \
  --rubrics promptManager_tools \
  --lines 7,12,20-25 \
  --workers 8

# Checked-in line manifest
python -m SynthChat.run improve -i data.jsonl -o regen_slice.jsonl \
  --rubrics contentManager_tools \
  --line-file Datasets/tools_datasets/reports/cli_schema/regen_lines.txt \
  --workers 12

# Powerful model for judging
python -m SynthChat.run improve -i data.jsonl \
  --provider openrouter --model openai/gpt-oss-120b --max-iterations 5

# Sanitize dataset content before improvement sends it to the model
python -m SynthChat.run improve -i data.jsonl \
  --privacy-profile realistic_pseudonyms \
  --max-iterations 1
```

Notes:
- `--lines` and `--line-file` select arbitrary rows from the input file; they do not need to be contiguous.
- When a subset is selected, the `.improve_report.json` still records the original input `line_number` values, which is what you want for deterministic merge/replace workflows.

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
| `--privacy-profile NAME` | Privacy preprocess profile for input JSONL before validation | None |

**Output:** Pass rate, failing lines, failures grouped by rubric. Suggests `improve` command to fix.

```bash
python -m SynthChat.run validate -i data.jsonl --rubrics system_prompt_format,factuality
```

```bash
python -m SynthChat.run validate -i tests/fixtures/privacy/raw_seed_dataset.jsonl \
  --privacy-profile realistic_pseudonyms
```

---

## `sanitize` — Apply Privacy Preprocessing

Sanitizes docs or JSONL datasets without running the SynthChat judge/improve loop.

```bash
python -m SynthChat.run sanitize [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input, -i PATH` | **Required.** Input file or directory | — |
| `--output, -o PATH` | Output file or directory | Auto-derived |
| `--config-dir PATH` | Config directory | `SynthChat/config` |
| `--privacy-profile NAME` | Privacy preprocess profile to apply | From settings / required in practice |

**Examples:**

```bash
# Mask-only docs sanitize
python -m SynthChat.run sanitize \
  -i tests/fixtures/privacy/raw_seed_docs \
  -o tmp/privacy_mask_only_docs \
  --privacy-profile mask_only

# Realistic pseudonymization for JSONL
python -m SynthChat.run sanitize \
  -i tests/fixtures/privacy/raw_seed_dataset.jsonl \
  -o tmp/privacy_pseudonyms_dataset.jsonl \
  --privacy-profile realistic_pseudonyms
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
python3 scripts/validate_syngen.py Datasets/your_dataset.jsonl
```

Checks: valid JSON, conversation structure, tool schemas, parameter validation.
