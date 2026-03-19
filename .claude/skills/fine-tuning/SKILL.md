---
name: fine-tuning
description: Complete reference for the fine-tuning pipeline (SFT, KTO, GRPO). Covers training CLI flags, YAML configuration, model presets, dataset requirements, LoRA settings, training monitoring, and the full train workflow. Use when training models, configuring training runs, choosing hyperparameters, or troubleshooting training issues. This skill is about USING the training system via CLI and YAML — never modifying source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Fine-Tuning Pipeline

Train language models with SFT (supervised), KTO (preference), and GRPO (reward optimization) on NVIDIA RTX 3090 or supported cloud providers.

## Quick Reference

| Task | Command |
|------|---------|
| Interactive menu | `./run.sh` → Train |
| SFT training | `cd Trainers/rtx3090_sft && python train_sft.py --model-size 7b` |
| KTO training | `cd Trainers/rtx3090_kto && python train_kto.py --model-size 7b` |
| GRPO training | `cd Trainers/rtx3090_grpo && python train_grpo.py` |
| Dry run (test setup) | `python train_sft.py --model-size 7b --dry-run` |
| Resume checkpoint | `python train_sft.py --resume-from-checkpoint PATH` |
| Monitor training | `tail -f logs/training_latest.jsonl` |
| Environment setup | `cd Trainers/rtx3090_sft && bash setup.sh` |

## Training Methods at a Glance

| Method | Purpose | LR | Epochs | Dataset | When to Use |
|--------|---------|-------|--------|---------|-------------|
| **SFT** | Teach format/behavior | 2e-4 | 3 | Positive examples only | First — teaches WHAT to do |
| **KTO** | Refine with preferences | 1e-6 | 1 | Interleaved True/False | Second — teaches WHICH is better |
| **GRPO** | Optimize for rewards | 5e-6 | 1 | Prompts + ground truth | Third — optimizes for specific metrics |

**Recommended pipeline:** SFT → KTO → GRPO (each builds on the previous)

## Key Directories

- `Trainers/rtx3090_sft/` — SFT trainer (configs, scripts, src)
- `Trainers/rtx3090_kto/` — KTO trainer (configs, scripts, src)
- `Trainers/rtx3090_grpo/` — GRPO trainer (configs, rewards, src)
- `Trainers/shared/` — Shared UI components
- `Datasets/` — Training datasets (JSONL)

## Progressive Reference

Load the specific reference you need:

| Reference | When to Load | Path |
|-----------|-------------|------|
| **SFT Training** | Running SFT, configuring SFT params | `reference/sft-training.md` |
| **KTO Training** | Running KTO, dataset interleaving, preference tuning | `reference/kto-training.md` |
| **GRPO Training** | Running GRPO, reward config, GSPO variant | `reference/grpo-training.md` |
| **Model Presets** | Choosing models, VRAM planning, LoRA settings | `reference/model-presets.md` |
| **Dataset Formats** | Preparing datasets, format requirements per method | `reference/dataset-formats.md` |
| **Training Config** | YAML config deep-dive, all settings explained | `reference/training-config.md` |
| **Cloud Training** | Provider-native persistence, exact-commit rules, cloud smoke tests | `reference/cloud-training.md` |
| **Troubleshooting** | OOM errors, training instability, platform issues | `reference/troubleshooting.md` |

## Common Patterns

**Quick SFT test run:**
```bash
cd Trainers/rtx3090_sft
python train_sft.py --model-size 3b --max-steps 50 --dry-run
```

**KTO with local dataset:**
```bash
cd Trainers/rtx3090_kto
python train_kto.py --model-size 7b --local-file ../../Datasets/my_kto_data.jsonl
```

**GRPO continuing from SFT checkpoint:**
```bash
# Edit configs/config.yaml to set model.lora_path to SFT checkpoint
cd Trainers/rtx3090_grpo
python train_grpo.py
```

**Enable W&B logging:**
```bash
python train_sft.py --model-size 7b --wandb --wandb-project my-project
```

**Cloud smoke test:**
```bash
python tuner.py cloud
# Choose provider + method after confirming the working tree is clean and pushed
```

## Environment Variables

```bash
HF_TOKEN=hf_...                       # HuggingFace (gated models + uploads)
WANDB_API_KEY=...                     # Weights & Biases (optional)
MODAL_TOKEN_ID=...                    # Modal cloud auth (optional)
MODAL_TOKEN_SECRET=...                # Modal cloud auth (optional)
RUNPOD_API_KEY=...                    # RunPod cloud auth (optional)
```

## Output Structure

Local trainers produce timestamped run directories:
```
{method}_output_rtx3090/YYYYMMDD_HHMMSS/
├── checkpoints/           # Training checkpoints (last 3 kept)
├── logs/                  # JSONL metrics + symlink to latest
├── final_model/           # Final LoRA adapters
└── training_lineage.json  # Complete training provenance
```

Cloud runs use the canonical provider-native layout:
```
runs/{provider}/{method}/{timestamp}-{shortsha}/
├── checkpoints/
├── logs/
├── final_model/
├── training_lineage.json
└── manifest.json
```

Provider-native storage defaults:
- `hf_jobs` → Hugging Face Bucket
- `modal` → Modal Volume
- `runpod` → RunPod Network Volume

Optional final-model publishing to Hugging Face Hub is off by default and only uploads `final_model`.

## Tips

- Always `--dry-run` first to verify setup without training
- Use `--model-size 3b` for fast iteration, `7b` for production
- SFT with `packing: true` is 2.5-5x faster
- KTO datasets MUST be interleaved True/False (auto-handled by data loader)
- GRPO rewards are YAML-driven — edit `configs/rewards/` not Python
- Monitor `training_latest.jsonl` for real-time metrics
- Keep VRAM headroom — reduce `--batch-size` if OOM
- `training_lineage.json` tracks full provenance for reproducibility
- Cloud runs require a clean tracked worktree and a pushed commit; remote jobs clone the exact branch and commit you launched
