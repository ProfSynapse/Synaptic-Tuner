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
| SFT training | `python tuner.py train --method sft` |
| KTO training | `python tuner.py train --method kto` |
| GRPO training | `python tuner.py train --method grpo` |
| Dry run (test setup) | `python train_sft.py --model-size 7b --dry-run` |
| Resume checkpoint | `python train_sft.py --resume-from-checkpoint PATH` |
| Monitor training | `tail -f logs/training_latest.jsonl` |
| Environment setup | `cd Trainers/rtx3090_sft && bash setup.sh` |
| HF custom job | `python tuner.py cloud-run --job-config Trainers/cloud/jobs/<job>.yaml` |
| HF gym against trained model | `python tuner.py cloud-gym --run latest --method sft` |

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
- `Trainers/cloud/jobs/` — Reusable HF Jobs configs for custom runs like SynthChat pilots
- `SynthChat/scenarios/` — Synthetic data scenario packs, including environment-backed pilots

## CLI Discipline

- Do not guess command names or flags from memory.
- Before giving command guidance, check the current parser definitions or run the real command with `--help`.
- In this repo, the source of truth is the actual CLI surface in `tuner/cli/parser.py`, `tuner/cli/router.py`, and tool-specific `--help` output.
- This matters for cloud workflows in particular because commands like `cloud-run`, `cloud-gym`, and `cloud-inspect` are evolving.

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

**Custom HF job from checked-in config:**
```bash
python tuner.py cloud-run --job-config Trainers/cloud/jobs/synthchat_vault_kto_pilot.yaml
```

**Cloud evaluation against latest HF run (vLLM, no GGUF):**
```bash
python tuner.py cloud-eval --run latest --preset full
```

**Cloud training + eval in one flow:**
```bash
python tuner.py cloud-pipeline --method sft --preset full
```

**Environment-backed gym against latest trained adapter on HF:**
```bash
python tuner.py cloud-gym --run latest --method sft
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
├── capacity_features.json
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

HF Jobs-specific cloud behavior:
- launch from a clean tracked worktree and a pushed commit only; the remote container checks out that exact SHA
- keep the main training environment compatible with Unsloth/Transformers; any Buckets-only `huggingface_hub` upgrade must stay isolated from the trainer runtime
- pass `HF_TOKEN` into the cloud job explicitly; do not assume HF Jobs injects it into the container environment
- treat blank `HF_TOKEN` / `HF_API_KEY` values as unset, otherwise bucket sync can fail with `Authorization: Bearer `
- if a bucket name is bare, let the launcher resolve it to the canonical namespaced bucket ID before training starts
- bucket creation / identity checks should happen once up front; repeated `whoami` / `create_bucket` calls during log sync can hit HF rate limits
- for post-training cloud evaluation, prefer `python tuner.py cloud-eval --run latest --preset full`; it launches an HF Job, downloads the bucketed LoRA, runs direct Unsloth inference remotely, and syncs results back to the same bucket under `.../evaluations/vllm/...`
- for the common train-then-evaluate flow, prefer `python tuner.py cloud-pipeline --method sft --preset full`; it runs HF Jobs training first, then launches cloud eval against that exact run without making you reselect it
- to inspect a finished HF cloud evaluation run from the bucket, use `python tuner.py cloud-inspect --run latest --eval-run latest --method sft`
- avoid trying to force vLLM into the Unsloth HF Jobs image for this path; direct Unsloth inference is the stable default unless you intentionally move evaluation to a dedicated vLLM runtime
- if preset-based eval fails with missing scenario files, check `Evaluator/config/eval_run.yaml` before debugging the loader; stale preset filenames are an easy failure mode
- cloud eval results are saved to the same HF bucket. Inspect them in this order: `evaluation_results.json` for summary + records, `evaluation_results.md` for human-readable report, `evaluation_lineage.json` for provenance, and `logs/eval_progress.jsonl` for live/replayed progress
- completed SFT/KTO runs now also save a flat `capacity_features.json` artifact with model params, hardware, throughput, peak memory, headroom, and outcome fields designed for tabular modeling or capacity prediction
- when reading `evaluation_results.json`, focus on how failures happened, not just how many there were
- separate evaluator or parser noise from actual model behavior failures; warnings produced by a lower-level validator should not automatically be treated as model regressions
- HF custom jobs can be used for non-training workloads like SynthChat generation; keep those jobs config-driven in `Trainers/cloud/jobs/*.yaml` and launch them with `python tuner.py cloud-run --job-config ...`
- for environment-backed SynthChat pilots, prefer a small remote smoke test first, such as 10 total examples, before scaling counts
- for environment-backed tool generation, prefer structured output for both generated environments and assistant tool responses so the artifact is executable instead of “tool-shaped prose”
- keep tool wrapper choice config-driven via the canonical tool schema; do not assume `useTools` is a hardcoded invariant in new prompts, validators, or generators
- for environment-backed gym work, keep multi-step agent loops config-driven via `environment.loop`; the evaluator runner owns the loop, while `local` and `e2b` are interchangeable runtime backends underneath it
- if the goal is "can it navigate the space and recover", prefer `environment.loop.mode: agentic`; keep final environment state as the hard success criterion and use `scoring.paths` for preferred vs acceptable workflows
- the reusable inspection method is:
  compare expected behavior to actual behavior
  inspect parsed tool/action records before raw text when both are present
  group failures by mechanism, such as wrong action selected, acted instead of clarifying, malformed structured output, missing required fields, or behavior-expectation mismatch
- keep examples project-agnostic; use the saved evaluation record schema to understand the failure shape instead of hardcoding one model, one toolset, or one prompt format

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
- For HF Jobs bucket troubleshooting, load `reference/cloud-training.md`
