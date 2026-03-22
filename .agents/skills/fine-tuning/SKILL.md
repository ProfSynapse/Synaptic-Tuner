---
name: fine-tuning
description: Complete reference for the fine-tuning pipeline (SFT, KTO, GRPO), cloud HF Jobs workflows, autonomous experiment search, checkpoint evaluation, and LoRA surgery. Covers training CLI flags, YAML configuration, model presets, dataset requirements, LoRA settings, training monitoring, hyperparameter search, and post-training optimization. Use when training models, configuring training runs, choosing hyperparameters, running cloud experiments, inspecting HF jobs, or troubleshooting training issues. This skill is about USING the training system via CLI and YAML — never modifying source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Fine-Tuning Pipeline

Train language models with SFT, KTO, and GRPO locally or on supported cloud providers. This skill also covers the Karpathy-style experiment loop, checkpoint evaluation, LoRA surgery, and the current HF Jobs operational path.

## Quick Reference

| Task | Command |
|------|---------|
| Interactive menu | `./run.sh` → Train |
| SFT training | `cd Trainers/rtx3090_sft && python train_sft.py --model-size 7b` |
| KTO training | `cd Trainers/rtx3090_kto && python train_kto.py --model-size 7b` |
| GRPO training | `cd Trainers/rtx3090_grpo && python train_grpo.py` |
| Experiment loop | `python tuner.py experiment-loop --experiment-config configs/flywheel/experiment_loop.yaml` |
| LoRA surgery | `python tuner.py surgery --surgery-config configs/lora_surgery.yaml` |
| HF custom job | `python tuner.py cloud-run --job-config Trainers/cloud/jobs/<job>.yaml` |
| Canonical HF train+eval | `python tuner.py cloud-pipeline --method sft --preset full` |
| Full experiment bundle | `python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/<spec>.yaml --yes` |
| Blind hardware plan | `python tuner.py plan-hardware --experiment-spec Trainers/cloud/experiments/<spec>.yaml` |
| Analyze finished experiment | `python tuner.py analyze-experiment --experiment-id latest` |
| Live HF job list | `python tuner.py cloud-jobs list` |
| Live HF job logs | `python tuner.py cloud-jobs logs --job professorsynapse/<job-id> --tail 200` |
| Cloud eval against a run | `python tuner.py cloud-eval --run latest --preset full` |
| HF gym against trained model | `python tuner.py cloud-gym --run latest --method sft` |
| Battle-of-models catalog | `python3 scripts/battle_of_models.py list` |
| ML training | `python tuner.py ml train --config Trainers/ml/configs/templates/regression.yaml` |

## Training Methods at a Glance

| Method | Purpose | LR | Epochs | Dataset | When to Use |
|--------|---------|----|--------|---------|-------------|
| **SFT** | Teach format and behavior | 2e-4 | 3 | Positive examples only | First stage |
| **KTO** | Refine with preferences | 1e-6 | 1 | Interleaved True/False | Second stage |
| **GRPO** | Optimize against rewards | 5e-6 | 1 | Prompts + ground truth | Final online stage |

**Recommended pipeline:** SFT → KTO → GRPO

## Complexity Tiers

Use `--tier` on the local SFT and KTO trainers when you want a preset instead of hand-tuning LoRA rank and LR.

| Tier | LoRA Rank | LR | Time | Use Case |
|------|-----------|----|------|----------|
| `quick` | r=8 | 5e-4 | ~5 min | Prototyping and smoke runs |
| `standard` | r=64 | 2e-4 | ~30-60 min | Normal training |
| `thorough` | r=128 | 1e-4 | ~2-4 hrs | Quality-focused final runs |

## Key Directories

- `Trainers/rtx3090_sft/` — SFT trainer
- `Trainers/rtx3090_kto/` — KTO trainer
- `Trainers/rtx3090_grpo/` — GRPO trainer
- `Trainers/cloud/jobs/` — checked-in HF Jobs configs
- `Datasets/` — JSONL training datasets
- `SynthChat/scenarios/` — synthetic data and environment-backed scenarios
- `shared/flywheel/` — autonomous experiment-loop logic and LightGBM surrogate search
- `Trainers/ml/` — traditional ML / LightGBM training for tabular experiment analysis
- `shared/experiment_tracking/` — unified run and experiment tracking

## CLI Discipline

- Do not guess command names or flags from memory.
- Before giving command guidance, check `tuner/cli/parser.py`, `tuner/cli/router.py`, or the real `--help` output.
- Prefer repo CLIs and checked-in scripts over ad hoc Python snippets.
- For local trainer iteration, use the checked-in `train_sft.py`, `train_kto.py`, and `train_grpo.py` entrypoints.
- For canonical HF experiments, prefer `python tuner.py cloud-pipeline ...` over `cloud-run`.
- For full train → eval → exact loss → analysis → recommendation runs, prefer `python tuner.py run-experiment ...`.
- For blind stage hardware selection before launch, use `python tuner.py plan-hardware ...`.
- For live HF status and traceback inspection, use `python tuner.py cloud-jobs ...`.
- For finished experiment bundles and next-run suggestions, use `python tuner.py analyze-experiment ...`.
- For hyperparameter search, use `python tuner.py experiment-loop ...`; this is the built-in LLM + LightGBM surrogate path.
- For tabular post-hoc models, use `python tuner.py ml ...` and the configs under `Trainers/ml/configs/templates/`.

## Progressive Reference

Load the specific reference you need:

| Reference | When to Load | Path |
|-----------|-------------|------|
| **SFT Training** | Running SFT, configuring SFT params | `reference/sft-training.md` |
| **KTO Training** | Running KTO, dataset interleaving, preference tuning | `reference/kto-training.md` |
| **GRPO Training** | Running GRPO, reward config, GSPO variant | `reference/grpo-training.md` |
| **Model Presets** | Choosing models, VRAM planning, LoRA settings | `reference/model-presets.md` |
| **Dataset Formats** | Preparing datasets, format requirements per method | `reference/dataset-formats.md` |
| **Training Config** | YAML config deep-dive | `reference/training-config.md` |
| **Cloud Training** | Provider-native persistence, exact-commit rules, cloud smoke tests | `reference/cloud-training.md` |
| **Cloud Experiments** | Canonical train→eval launches with `--train-*` overrides | `reference/cloud-experiment-launching.md` |
| **Checkpoint Evaluation** | Best-checkpoint selection via eval | `reference/checkpoint-evaluation.md` |
| **Experiment Loop** | Autonomous hyperparameter search (LLM + LightGBM) | `reference/experiment-loop.md` |
| **LoRA Surgery** | Eval-guided post-training weight optimization | `reference/lora-surgery.md` |
| **Troubleshooting** | OOM errors, instability, platform issues | `reference/troubleshooting.md` |
| **Env Alignment Protocol** | Canonical SynthChat → SFT → merge/publish → KTO → env-GRPO flow | `protocols/environment-backed-alignment-pipeline.md` |

## Common Patterns

**Quick SFT test run:**
```bash
cd Trainers/rtx3090_sft
python train_sft.py --model-size 3b --tier quick --dry-run
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

**Autonomous hyperparameter search:**
```bash
python tuner.py experiment-loop \
  --experiment-config configs/flywheel/experiment_loop.yaml \
  --max-experiments 10
```

**Post-training LoRA surgery:**
```bash
python tuner.py surgery --surgery-config configs/lora_surgery.yaml
```

**Inspect live HF jobs:**
```bash
python tuner.py cloud-jobs list --limit 20
python tuner.py cloud-jobs show --job professorsynapse/<job-id>
python tuner.py cloud-jobs logs --job professorsynapse/<job-id> --tail 200
```

**Canonical one-off HF experiment with direct overrides:**
```bash
python tuner.py cloud-pipeline --method sft --preset full \
  --train-model-name Qwen/Qwen3.5-2B \
  --train-image-profile next \
  --train-gpu a10g-small \
  --train-batch-size 8 \
  --train-gradient-accumulation 4 \
  --train-no-load-in-4bit \
  --train-lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --yes
```

**Battle-of-models helper:**
```bash
python3 scripts/battle_of_models.py list
python3 scripts/battle_of_models.py commands --smoke
python3 scripts/battle_of_models.py launch --smoke qwen35-2b --image-profile next
```

**Cloud training + eval in one flow:**
```bash
python tuner.py cloud-pipeline --method sft --preset full
```

**Full experiment with train → eval → exact loss → analysis:**
```bash
python tuner.py run-experiment \
  --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_full.yaml \
  --yes
```

**Blind hardware planning before launch:**
```bash
python tuner.py plan-hardware \
  --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_full.yaml \
  --optimize-for balanced
```

**Run experiment with auto hardware selection:**
```bash
python tuner.py run-experiment \
  --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_full.yaml \
  --auto-hardware \
  --optimize-for cost \
  --yes
```

**Prepare a fixed-tier GPU benchmark matrix for the same model:**
```bash
python tuner.py plan-hardware --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_benchmark_l4x1.yaml --optimize-for cost
python tuner.py plan-hardware --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_benchmark_a10g_small.yaml --optimize-for cost
python tuner.py plan-hardware --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_benchmark_a100_large.yaml --optimize-for cost
```
Use this when you want three comparable full-pipeline runs of the same model across increasing GPU tiers. Keep the GPU pinned in the spec, leave training batch/grad accumulation unset, and let `run-experiment --auto-hardware` fill the batch shape for that tier.

**Resume or slice the experiment pipeline:**
```bash
python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_full.yaml --from-stage evaluation --yes
python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_full.yaml --only-stage loss --yes
python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_full.yaml --skip-stage analysis --skip-stage recommendation --yes
```

**Inspect the finished experiment bundle and next-run candidates:**
```bash
python tuner.py analyze-experiment --experiment-id latest
python tuner.py analyze-experiment --experiment-id exp_20260321_221651 --json
```

**Environment-backed gym against latest trained adapter on HF:**
```bash
python tuner.py cloud-gym --run latest --method sft
```

**Tabular LightGBM experiment:**
```bash
python tuner.py ml train --config Trainers/ml/configs/templates/regression.yaml
```

## Environment Variables

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
RUNPOD_API_KEY=...
```

## Output Structure

Local trainers produce timestamped run directories:
```
{method}_output_rtx3090/YYYYMMDD_HHMMSS/
├── checkpoints/
├── logs/
├── final_model/
└── training_lineage.json
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

## HF Jobs Notes

- Launch from a clean tracked worktree and a pushed commit only; the remote container checks out that exact SHA.
- Keep the main training interpreter compatible with Unsloth and Transformers; any Buckets-only `huggingface_hub` upgrade must stay isolated from the trainer runtime.
- Pass `HF_TOKEN` into the cloud job explicitly; do not assume HF Jobs injects it automatically.
- Treat blank `HF_TOKEN` / `HF_API_KEY` values as unset, otherwise bucket sync can fail with `Authorization: Bearer `.
- For post-training cloud evaluation, prefer `python tuner.py cloud-eval --run latest --preset full`.
- `run-experiment` now supports stage controls: `--only-stage`, `--from-stage`, and repeated `--skip-stage`.
- `run-experiment --auto-hardware` uses a blind planner: model size, method, seq length, quantization, and live HF flavor pricing. It does not require prior telemetry.
- `plan-hardware` is the inspection surface for that same planner.
- Finished experiments now write `.tracking/experiments/<id>/analysis/` with:
  `experiment_summary.json`, `run_matrix.csv`, `feature_dataset.{jsonl,csv}`, `next_run_candidates.json`, `draft_next_spec.yaml`.
- For the common train-then-evaluate flow, prefer `python tuner.py cloud-pipeline --method sft --preset full`.
- `cloud-pipeline` is currently a two-job orchestration on HF Jobs, not a single remote composite job.
- `run-experiment` is the higher-level experiment loop: training stays provider-native, eval can use `vllm`, and exact dataset loss runs afterward with `transformers`.
- Use `evaluation.runtime: vllm` in experiment specs when you want the fast eval server path. The exact loss stage still uses a post-eval `transformers` forward pass.
- Checkpoint-vs-checkpoint comparison is not automatic in smoke runs; you only get that if the trainer emitted multiple checkpoints and you intentionally run checkpoint evaluation / experiment-loop workflows.
- For SFT model-comparison experiments, use `cloud-pipeline` with `--train-*` overrides so the experiment lands in canonical HF training storage instead of `runs/hf_jobs/custom/...`.
- When testing newer upstream Unsloth runtimes, switch images with `--train-image-profile next` instead of upgrading packages in the old stable image.
- To inspect a finished HF cloud evaluation run from the bucket, use `python tuner.py cloud-inspect --run latest --eval-run latest --method sft`.
- Completed SFT/KTO runs save a flat `capacity_features.json` artifact designed for tabular modeling or capacity prediction.

## Tips

- Always `--dry-run` first to verify setup without training.
- Use `--tier quick` for fast prototyping and `--tier thorough` for final quality runs.
- Use `--model-size 3b` for fast iteration and `7b` for production-style runs.
- SFT with `packing: true` is much faster.
- KTO datasets must be interleaved True/False.
- GRPO rewards are YAML-driven; edit `configs/rewards/`, not Python.
- `fitness.yaml` uses `FitnessEvaluator` for structural validation.
- After training, use surgery only after you have a stable baseline and evaluation scenario.
- `training_lineage.json` tracks full provenance for reproducibility.
