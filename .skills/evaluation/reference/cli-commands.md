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
| `--no-load-in-4bit` | `unsloth` only: load at full/half precision instead of the default 4-bit | Full-precision reference runs |

### Model Artifact

Recorded in `metadata.model_artifact` of the results JSON and in the lineage.

| Flag | Description | Example |
|------|-------------|---------|
| `--quantization` | Quant label of the evaluated artifact; default is detected from the model path/name | `Q4_K_M` |
| `--artifact-manifest` | `gguf_manifest.json`; the entry matching the model file name and the calibration block are recorded | `gguf/gguf_manifest.json` |

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

python -m Evaluator.compare \
  --reference Evaluator/results/base.json \
  --candidate finetuned=Evaluator/results/finetuned.json
```

---

## Compare Command

```bash
python -m Evaluator.compare --reference REF.json --candidate [LABEL=]CAND.json [--candidate ...] [options]
```

Compares results files written by `Evaluator.cli`: deltas, per-tag deltas,
per-case flips, request errors, latency ratio and an exact McNemar p-value.
Without a label, a candidate is labelled by its recorded quantization, a quant
name in the model/file name, or the file stem.

| Flag | Description |
|------|-------------|
| `--reference` | Reference results JSON (e.g. full-precision model) |
| `--candidate` | Candidate results JSON, optionally `LABEL=PATH`; repeatable |
| `--max-pass-rate-drop PP` | Gate: max overall pass-rate drop in percentage points |
| `--max-correctness-drop PP` | Gate: max correctness pass-rate drop |
| `--max-tag-drop PP` | Gate: max per-tag pass-rate drop |
| `--min-tag-cases N` | Only gate tags with at least N shared cases (default 1) |
| `--max-regressions N` | Gate: max cases flipping pass to fail |
| `--alpha A` | Drops only breach when McNemar p < A |
| `--allow-mismatch` | Do not fail candidates whose case ids or settings differ |
| `--output` | Comparison JSON path |
| `--markdown` | Comparison Markdown path |

No thresholds means report only (exit 0). Exit `1` when a gate breaches, `2` on
input errors. See `results-metrics.md` "Quantization Regression Check" for the
full-precision vs GGUF recipe.

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
