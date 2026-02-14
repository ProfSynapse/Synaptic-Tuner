# Evaluator CLI Commands Reference

Full CLI flag reference for the model evaluation system.

---

## Main Command

```bash
python -m Evaluator.cli [options]
```

---

## All Flags

### Backend & Model
| Flag | Description | Example |
|------|-------------|---------|
| `--backend` | Backend type | `unsloth`, `llamacpp`, `ollama`, `lmstudio`, `openrouter`, `mlc` |
| `--model` | Model name or path | `qwen2.5-7b-instruct` or `path/to/lora` |

### Scenario Selection
| Flag | Description | Example |
|------|-------------|---------|
| `--scenario` | YAML scenario file(s) | `behavior_prompts.yaml` (can repeat) |
| `--preset` | Preset from eval_run.yaml | `quick`, `full`, `behavior_only`, `strict` |
| `--tags` | Tag filter (comma-separated) | `intellectual_humility,clarification` |
| `--limit` | Max tests to run | `10` |

### Output
| Flag | Description | Example |
|------|-------------|---------|
| `--output` | JSON results path | `Evaluator/results/run.json` |
| `--markdown` | Markdown report path | `Evaluator/results/run.md` |

### HuggingFace Integration
| Flag | Description |
|------|-------------|
| `--lineage` | Save evaluation lineage JSON |
| `--upload-to-hf` | Upload results to HuggingFace repo |
| `--update-model-card` | Update README with eval results (requires `--upload-to-hf`) |

### Validation
| Flag | Description |
|------|-------------|
| `--validate-context` | Validate sessionId/workspaceId match prompt |
| `--dry-run` | Skip actual model calls (test config) |

---

## Examples

**Full evaluation with both scenario files:**
```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model qwen2.5-7b-instruct \
  --scenario behavior_prompts.yaml \
  --scenario tool_prompts.yaml \
  --output Evaluator/results/full_eval.json \
  --markdown Evaluator/results/full_eval.md
```

**Quick smoke test:**
```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model qwen2.5-7b-instruct \
  --preset quick
```

**Evaluate LoRA with Unsloth backend:**
```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model ./Trainers/rtx3090_sft/sft_output_rtx3090/20250114/final_model \
  --scenario behavior_prompts.yaml
```

**Filter by tags:**
```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model qwen2.5-7b-instruct \
  --scenario behavior_prompts.yaml \
  --tags intellectual_humility,clarification
```

**Evaluate and upload to HuggingFace:**
```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model path/to/model \
  --scenario behavior_prompts.yaml \
  --scenario tool_prompts.yaml \
  --lineage eval_lineage.json \
  --upload-to-hf username/model-name \
  --update-model-card
```

**Limit number of tests:**
```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model MODEL \
  --scenario behavior_prompts.yaml \
  --limit 10
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

The interactive menu walks through all options with arrow-key selection.
