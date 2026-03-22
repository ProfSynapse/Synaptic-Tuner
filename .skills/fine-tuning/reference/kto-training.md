# KTO Training Reference

Kahneman-Tversky Optimization — refines model behavior using preference pairs (desirable vs undesirable examples).

---

## CLI Flags

```bash
python train_kto.py [options]
```

### Model Selection
| Flag | Description |
|------|-------------|
| `--model-size {3b,7b,13b,20b}` | Use preset configuration |
| `--qwen-3b`, `--llama-3b` | 3B model shortcuts |
| `--mistral-7b`, `--llama-8b`, `--qwen-7b` | 7-8B model shortcuts |
| `--magistral`, `--deepseek-7b` | Specialized 7B models |
| `--qwen-vl-8b`, `--qwen-thinking-8b` | Vision/reasoning models |
| `--llama-13b`, `--gemma-12b`, `--deepseek-14b` | 13-14B models |
| `--gpt-20b`, `--mistral-24b` | 20-24B models |
| `--model-name MODEL` | Manual model override |

### Complexity Tiers
| Flag | Description |
|------|-------------|
| `--tier {quick,standard,thorough}` | Use preset complexity tier (overrides individual training params) |

| Tier | LoRA Rank | LR | Epochs | Steps | Time | Use Case |
|------|-----------|------|--------|-------|------|----------|
| `quick` | r=8, alpha=16 | 5e-4 | 1 | 200 max | ~5 min | Rapid prototyping |
| `standard` | r=64, alpha=128 | 1e-6 | 1 | — | ~30-60 min | Production training |
| `thorough` | r=128, alpha=256 | 1e-4 | 3 | — | ~2-4 hrs | Maximum quality |

Tier configs: `Trainers/kto/configs/tiers/{quick,standard,thorough}.yaml`

Explicit CLI flags override tier defaults.

### KTO-Specific
| Flag | Description | Default |
|------|-------------|---------|
| `--beta FLOAT` | KL divergence penalty | 0.1 |
| `--two-stage-lr` | Enable two-stage LR schedule | false |
| `--lr-reduction-step N` | Step to reduce LR | 50 |
| `--lr-reduction-factor FLOAT` | LR reduction multiplier | 0.5 |

### Training Parameters
| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size N` | Per-device batch size | 2 |
| `--gradient-accumulation N` | Gradient accumulation | 4 |
| `--learning-rate FLOAT` | Learning rate | 1e-6 |
| `--num-epochs N` | Epochs | 1 |
| `--max-steps N` | Max steps (overrides epochs) | — |
| `--adaptive-memory` | Auto-adjust batch for VRAM | false |
| `--target-vram-util FLOAT` | VRAM utilization target | 0.80 |

### Dataset
| Flag | Description |
|------|-------------|
| `--local-file PATH` | Local JSONL file |
| `--dataset-name STR` | HuggingFace dataset |
| `--dataset-file STR` | File within HF dataset |
| `--split-dataset` | Create train/val split |

### Utility
| Flag | Description |
|------|-------------|
| `--dry-run` | Setup only |
| `--resume-from-checkpoint PATH` | Resume training |
| `--wandb` | Enable W&B |
| `--debug` | Detailed debug logging |

---

## CRITICAL: Dataset Interleaving

**KTO datasets MUST be interleaved True/False.** TRL's KTOTrainer crashes on homogeneous batches.

**The data loader handles this automatically:**
1. Splits into True and False groups
2. Balances counts (truncates majority)
3. Shuffles each group
4. Interleaves: [True, False, True, False, ...]
5. Uses SequentialSampler to preserve order

**You don't need to pre-interleave** — just provide a dataset with both `label: true` and `label: false` examples.

---

## KTO-Specific Parameters

### Beta (KL Penalty)
Controls how far the model can diverge from its reference:
- **Lower (0.01-0.05)** — More divergence allowed, faster learning, less stable
- **Default (0.1)** — Balanced
- **Higher (0.5-1.0)** — Stays closer to reference, more conservative

### Desirable/Undesirable Weights
In `config.yaml`:
```yaml
training:
  desirable_weight: 1.0      # Weight for True examples
  undesirable_weight: 1.0    # Weight for False examples
```
Keep at 1.0/1.0 for balanced datasets.

### KTO-S (SIGN Correction)
```yaml
training:
  use_kto_s: false    # Enable KTO-S variant
```
- **false** (default) — Standard KTO
- **true** — KTO-S with SIGN correction, more stable for base models
- Recommended when training from non-fine-tuned base models

### Two-Stage Learning Rate
```yaml
training:
  use_two_stage_lr: false
  lr_reduction_step: 50
  lr_reduction_factor: 0.5
```
Reduces LR midway through training to prevent instability.

---

## Key Metrics to Monitor

| Metric | Healthy Range | Warning |
|--------|--------------|---------|
| `rewards/margins` | Increasing 0→1.0-2.0 | Flat or decreasing = not learning |
| `rewards/chosen` | Increasing | — |
| `rewards/rejected` | Decreasing or stable | Increasing = model getting worse |
| `kl` | < 0.1 early, stable | Spikes = instability |
| `loss` | Decreasing | NaN/Inf = critical error |

---

## Training Workflow

1. **Prepare dataset**: JSONL with `conversations` + `label` (true/false)
2. **Verify labels**: Must have both True and False examples
3. **Test setup**: `python train_kto.py --model-size 7b --dry-run`
4. **Train**: `python train_kto.py --model-size 7b --local-file ../../Datasets/kto_data.jsonl`
5. **Monitor margins**: Should steadily increase
6. **Upload**: `python src/upload_to_hf.py ./final_model user/repo`

---

## SFT vs KTO Comparison

| Aspect | SFT | KTO |
|--------|-----|-----|
| Purpose | Teach format | Refine quality |
| Learning rate | 2e-4 | 1e-6 (100x lower) |
| Epochs | 3 | 1 |
| Dataset | Positive only | Interleaved True/False |
| Reference model | None | Frozen copy (implicit) |
| Key metric | Loss | Margins |
| Order | First | After SFT |
