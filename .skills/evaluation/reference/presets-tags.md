# Presets & Tags Reference

Evaluation presets and tag-based filtering.

---

## Presets

Presets are defined in `Evaluator/config/eval_run.yaml`.

| Preset | Description | Use Case |
|--------|-------------|----------|
| `quick` | Small smoke subset | Fast iteration |
| `full` | Configured full suite | Pre-release or checkpoint comparison |
| `strict` | Higher pass threshold | Release gate |

Presets can lag active scenario migrations. When precision matters, pass `--scenario` explicitly.

### Examples

```bash
# Quick smoke test
python -m Evaluator.cli --backend lmstudio --model MODEL --preset quick

# Full configured suite
python -m Evaluator.cli --backend lmstudio --model MODEL --preset full

# Explicit tool scenario
python -m Evaluator.cli --backend vllm --model finetuned --scenario tool_prompts.yaml --host 127.0.0.1 --port 8011
```

---

## Tags

Tags are arbitrary YAML labels used for filtering and reporting. They are not validators.

Common tags:

| Tag | What It Usually Groups |
|-----|------------------------|
| `storageManager` | Storage CLI commands |
| `contentManager` | Content read/write/edit commands |
| `searchManager` | Search commands |
| `memoryManager` | Memory/workspace commands |
| `promptManager` | Prompt CRUD/execution commands |
| `single-tool` | One wrapper call expected |
| `clarification` | Text clarification cases |
| `destructive` | Delete/archive/overwrite-sensitive cases |

### Tag Filtering

```bash
# Single tag
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario tool_prompts.yaml \
  --tags storageManager

# Multiple tags, comma-separated
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario tool_prompts.yaml \
  --tags promptManager,single-tool
```

---

## Custom Run Configuration

```bash
# Run only 5 tests
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario tool_prompts.yaml --limit 5

# Run a capability slice with explicit outputs
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario tool_prompts.yaml \
  --tags storageManager \
  --output Evaluator/results/storage_tests.json \
  --markdown Evaluator/results/storage_tests.md

# Dry run to verify config loads
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario tool_prompts.yaml --dry-run
```

---

## Recommended Strategy

| Situation | Command Pattern |
|-----------|-----------------|
| Iterating on YAML | `--scenario FILE --limit 1 --tags TAG` |
| Checking a local vLLM model | `--backend vllm --host 127.0.0.1 --port 8011 --scenario tool_prompts.yaml` |
| Comparing checkpoints | Same `--scenario`, different `--model`, separate `--output` files |
| Investigating failures | Open result JSON and inspect `record.correctness.paths` |
| Release gate | Use a fixed preset or explicit scenario list and compare `correctness_pass_rate` |
