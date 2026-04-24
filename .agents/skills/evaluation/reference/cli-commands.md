# Evaluator CLI Commands Reference

Full CLI flag reference for the model evaluation system.

---

## Main Command

```bash
python -m Evaluator.cli [options]
```

---

## Common Flags

### Backend & Model

| Flag | Description | Example |
|------|-------------|---------|
| `--backend` | Backend type | `vllm`, `lmstudio`, `ollama`, `llamacpp`, `unsloth`, `openrouter`, `mlc` |
| `--model` | Model name or path | `finetuned`, `qwen2.5-7b-instruct`, `path/to/lora` |
| `--host` | Backend host override | `127.0.0.1` |
| `--port` | Backend port override | `8011` |

### Generation

| Flag | Description | Example |
|------|-------------|---------|
| `--temperature` | Sampling temperature | `0` |
| `--top-p` | Nucleus sampling | `0.9` |
| `--max-tokens` | Max generated tokens | `768` |
| `--seed` | Optional seed | `42` |

### Scenario Selection

| Flag | Description | Example |
|------|-------------|---------|
| `--scenario` | YAML scenario file; can repeat | `tool_prompts.yaml` |
| `--preset` | Preset from `eval_run.yaml` | `quick`, `full`, `strict` |
| `--tags` | Comma-separated tag filter | `storageManager,single-tool` |
| `--limit` | Max tests to run | `10` |

### Output

| Flag | Description | Example |
|------|-------------|---------|
| `--output` | JSON results path | `Evaluator/results/run.json` |
| `--markdown` | Markdown report path | `Evaluator/results/run.md` |
| `--no-dashboard` | Disable live dashboard | Useful for scripted runs |
| `--progress-jsonl` | Write streaming JSONL progress | Cloud/local replay workflows |

### Validation

| Flag | Description |
|------|-------------|
| `--dry-run` | Load config and skip model calls |
| `--validate-context` | Validate configured session/workspace context structurally |

### Environment Runtime Validation

| Flag | Description | Example |
|------|-------------|---------|
| `--env-backend` | Runtime backend | `none`, `local`, `e2b` |
| `--env-template` | E2B template ID | `tmpl_abc123` |
| `--env-timeout` | Runtime command timeout seconds | `120` |
| `--env-api-key` | E2B API key override | `e2b_...` |
| `--env-tool-schema` | Tool schema YAML | `path/to/tool_schema.yaml` |
| `--env-exec-config` | Execution rules YAML | `path/to/environment_execution.yaml` |

### LLM-as-Judge

| Flag | Description |
|------|-------------|
| `--judge` | Enable judge validation |
| `--judge-mode` | `and`, `or`, or `judge_only` |
| `--judge-provider` | Judge provider, e.g. `openrouter`, `lmstudio`, `ollama` |
| `--judge-model` | Judge model |
| `--judge-rubrics` | Comma-separated rubric names |
| `--judge-rubrics-dir` | Rubric directory |
| `--no-judge-log` | Disable judge interaction logging |

### HuggingFace Integration

| Flag | Description |
|------|-------------|
| `--lineage` | Save evaluation lineage JSON |
| `--upload-to-hf` | Upload results to HuggingFace repo |
| `--update-model-card` | Update README with eval results; requires `--upload-to-hf` |

---

## Examples

### Local vLLM Tool Eval

```bash
python -m Evaluator.cli \
  --backend vllm \
  --model finetuned \
  --scenario tool_prompts.yaml \
  --host 127.0.0.1 \
  --port 8011 \
  --temperature 0 \
  --max-tokens 768 \
  --no-dashboard \
  --output Evaluator/results/local_vllm_tool_eval.json \
  --markdown Evaluator/results/local_vllm_tool_eval.md
```

### Quick Scenario Smoke

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model MODEL \
  --scenario tool_prompts.yaml \
  --limit 3
```

### Filter By Tags

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model MODEL \
  --scenario tool_prompts.yaml \
  --tags storageManager
```

### Eval With Judge

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model MODEL \
  --scenario tool_prompts.yaml \
  --judge \
  --judge-rubrics tool_call_quality \
  --judge-mode and
```

### Eval With Runtime Environment

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model MODEL \
  --scenario tool_prompts.yaml \
  --env-backend local
```

### Custom Tool Schema + Runtime Rules

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model MODEL \
  --scenario tool_prompts.yaml \
  --env-backend local \
  --env-tool-schema ./my_config/tool_schema.yaml \
  --env-exec-config ./my_config/environment_execution.yaml
```

### Compare Two Models

```bash
python -m Evaluator.cli --backend vllm --model base \
  --scenario tool_prompts.yaml \
  --output Evaluator/results/base.json

python -m Evaluator.cli --backend vllm --model finetuned \
  --scenario tool_prompts.yaml \
  --output Evaluator/results/finetuned.json
```

---

## Via Interactive Menu

```bash
./run.sh
# Select: Evaluate
# Choose backend
# Choose model
# Choose scenario(s)
```

The interactive menu walks through common options. Use the CLI directly for exact reproducible runs.
