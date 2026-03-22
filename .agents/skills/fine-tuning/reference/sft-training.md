# SFT Training Reference

Supervised Fine-Tuning — teaches the model tool-calling format and behaviors from positive examples.

---

## CLI Flags

```bash
python train_sft.py [options]
```

### Model Selection
| Flag | Description | Default |
|------|-------------|---------|
| `--model-size {3b,7b,13b,20b}` | Use preset configuration | — |
| `--config PATH` | Load custom Python config file | — |

### Complexity Tiers
| Flag | Description |
|------|-------------|
| `--tier {quick,standard,thorough}` | Use preset complexity tier (overrides individual training params) |

| Tier | LoRA Rank | LR | Epochs | Steps | Time | Use Case |
|------|-----------|------|--------|-------|------|----------|
| `quick` | r=8, alpha=16 | 5e-4 | 1 | 200 max | ~5 min | Rapid prototyping, idea validation |
| `standard` | r=64, alpha=128 | 2e-4 | 1 | — | ~30-60 min | Production training |
| `thorough` | r=128, alpha=256 | 1e-4 | 3 | — | ~2-4 hrs | Maximum quality, publication |

Tier configs: `Trainers/sft/configs/tiers/{quick,standard,thorough}.yaml`

Explicit CLI flags (e.g., `--learning-rate`) override tier defaults.

### Training Parameters
| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size N` | Per-device batch size | 2 |
| `--gradient-accumulation N` | Gradient accumulation steps | 2 |
| `--learning-rate FLOAT` | Learning rate | 2e-4 |
| `--num-epochs N` | Number of epochs | 3 |
| `--max-steps N` | Max steps (overrides epochs) | — |
| `--max-seq-length N` | Max sequence length | 2048 |

### Dataset
| Flag | Description | Default |
|------|-------------|---------|
| `--dataset-name STR` | HuggingFace dataset name | config value |
| `--dataset-file STR` | Specific file in HF dataset | config value |
| `--local-file PATH` | Local JSONL file (overrides HF) | — |
| `--split-dataset` | Create train/validation split | false |

### Experiment Tracking
| Flag | Description | Default |
|------|-------------|---------|
| `--wandb` | Enable W&B logging | false |
| `--wandb-project STR` | W&B project name | — |
| `--wandb-run-name STR` | W&B run name | — |

### Utility
| Flag | Description |
|------|-------------|
| `--dry-run` | Setup only, don't train |
| `--resume-from-checkpoint PATH` | Resume from checkpoint |
| `--hf-token STR` | HuggingFace token (gated models) |
| `--no-dashboard` | Disable live dashboard, use table output |
| `--quiet` | Suppress verbose library logs |

---

## Key SFT Settings

### What Makes SFT Different
- **No reference model** — simpler than KTO/GRPO
- **Higher learning rate** (2e-4 vs KTO's 1e-6) — more aggressive learning
- **Multiple epochs** (3 vs KTO's 1) — learn patterns thoroughly
- **Positive examples only** — no True/False labels needed
- **Packing support** — 2.5-5x faster training

### Packing (Recommended)
When `packing: true` in config:
1. Multiple examples packed into single sequences
2. **2.5-5x faster** training due to better GPU utilization
3. Dataset auto-preprocessed with chat template

### Completion-Only Loss
When `completion_only_loss: true` (default):
- Loss computed only on assistant response tokens
- User prompt tokens ignored during training
- Prevents model from learning to generate user messages

---

## Training Workflow

1. **Setup environment**: `bash setup.sh` (creates conda env, installs deps)
2. **Prepare dataset**: JSONL with `conversations` field, positive examples only
3. **Test setup**: `python train_sft.py --model-size 7b --tier quick --dry-run`
4. **Quick iteration**: `python train_sft.py --model-size 7b --tier quick` (~5 min)
5. **Production run**: `python train_sft.py --model-size 7b --tier standard`
6. **Monitor**: Watch live dashboard or `tail -f logs/training_latest.jsonl`
7. **Upload**: `python3 .skills/upload-deployment/scripts/upload_model.py ./final_model user/repo --save-method merged_16bit`

---

## Typical SFT Performance (7B on RTX 3090)

| Metric | Value |
|--------|-------|
| VRAM usage | ~7-9 GB |
| Speed (with packing) | ~500-800 tokens/sec |
| Time per epoch (~2700 examples) | ~15 min |
| Total (3 epochs) | ~45 min |

---

## Config File

**Location:** `Trainers/rtx3090_sft/configs/config.yaml`

Key sections:
- `model` — model name, seq length, quantization
- `lora` — rank, alpha, dropout, target modules
- `training` — batch size, LR, epochs, packing, etc.
- `dataset` — source, filtering, split
- `evolutionary` — experimental gradient evolution (disabled by default)

See `reference/training-config.md` for full config documentation.
