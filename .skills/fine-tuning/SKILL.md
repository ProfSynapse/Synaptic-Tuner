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
| GRPO training | `cd Trainers/grpo && python train_grpo.py` |
| Pivot-profile GRPO dataset | `cd Trainers/grpo && python train_grpo.py --config configs/pivot_config.yaml --pivot-profile-only` |
| GRPO with pivot filtering | `cd Trainers/grpo && python train_grpo.py --config configs/pivot_config.yaml` |
| Env-backed GRPO | `cd Trainers/grpo && python train_env_grpo.py --config ./configs/env_config.yaml --dry-run` |
| Experiment loop | `python tuner.py experiment-loop --experiment-config configs/flywheel/experiment_loop.yaml` |
| LoRA surgery | `python tuner.py surgery --surgery-config configs/lora_surgery.yaml` |
| HF custom job | `python tuner.py cloud-run --job-config Trainers/cloud/jobs/<job>.yaml` |
| Canonical HF train+eval | `python tuner.py cloud-pipeline --method sft --preset full` |
| Full experiment bundle | `python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/<spec>.yaml --yes` |
| Evolutionary SFT smoke test | `python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/<evolutionary-spec>.yaml --yes` |
| Staggered experiment batch | `python3 scripts/launch_experiment_batch.py Trainers/cloud/experiments/<spec1>.yaml Trainers/cloud/experiments/<spec2>.yaml --yes` |
| Blind hardware plan | `python tuner.py plan-hardware --experiment-spec Trainers/cloud/experiments/<spec>.yaml` |
| Analyze finished experiment | `python tuner.py analyze-experiment --experiment-id latest` |
| Analyze/prune dataset from loss | `python3 scripts/prune_dataset_from_loss.py --dataset-path ... --experiment-id ... --analyze-only` |
| Analyze bucket-backed run | `python tuner.py bucket analyze --path runs/hf_jobs/sft/<run-prefix>/` |
| Read bucket artifact | `python tuner.py bucket read --path runs/.../logs/training_latest.jsonl --jsonl-latest --pretty` |
| List bucket prefix | `python tuner.py bucket list --path runs/hf_jobs/sft/<run-prefix>/ --limit 20` |
| Pull bucket prefix locally | `python tuner.py bucket pull --path runs/hf_jobs/sft/<run-prefix>/ --dest .` |
| Push local artifact to bucket | `python tuner.py bucket push --path local/results.json --dest runs/manual_uploads/` |
| Live HF job list | `python tuner.py cloud-jobs list` |
| Live HF job logs | `python tuner.py cloud-jobs logs --job professorsynapse/<job-id> --tail 200` |
| Cloud eval against a run | `python tuner.py cloud-eval --run latest --preset full` |
| HF gym against trained model | `python tuner.py cloud-gym --run latest --method sft` |
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
- `Trainers/grpo/` — GRPO and env-GRPO trainer
- `Trainers/cloud/jobs/` — checked-in HF Jobs configs
- `Datasets/` — JSONL training datasets
- `SynthChat/scenarios/` — synthetic data and environment-backed scenarios
- `shared/flywheel/` — autonomous experiment-loop logic and LightGBM surrogate search
- `Trainers/ml/` — traditional ML / LightGBM training for tabular experiment analysis
- `shared/experiment_tracking/` — unified run and experiment tracking

## CLI Discipline

- Never cancel a job, delete bucket artifacts, remove files, or relaunch a cost-incurring cloud run unless the user has explicitly approved that exact action in the current conversation.
- Treat cancel/delete/relaunch as irreversible or materially destructive operator actions. Do not infer permission from surrounding context or from a user's broader goal.
- Do not guess command names or flags from memory.
- Before giving command guidance, check `tuner/cli/parser.py`, `tuner/cli/router.py`, or the real `--help` output.
- Prefer repo CLIs and checked-in scripts over ad hoc Python snippets.
- After benchmark runs complete, treat the checked-in benchmark ledger as part of the workflow:
  - [model_hardware_benchmark_ledger.md](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/docs/benchmarks/model_hardware_benchmark_ledger.md)
  - [model_hardware_benchmark_ledger.csv](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/docs/benchmarks/model_hardware_benchmark_ledger.csv)
- Treat stage lineage as the source of truth for automation:
  - training: `training_lineage.json`
  - evaluation: `evaluation_lineage.json`
  - exact loss: `loss_lineage.json`
- Treat `loss_summary.json` as a supporting artifact, not the canonical final loss metadata file.
- The ledger should accumulate real model-size / hardware / timing / cost data so future hardware planning can optimize against observed evidence instead of memory.
- For local trainer iteration, use the checked-in `train_sft.py`, `train_kto.py`, and `train_grpo.py` entrypoints.
- For canonical HF experiments, prefer `python tuner.py cloud-pipeline ...` over `cloud-run`.
- For full train → eval → exact loss → analysis → recommendation runs, prefer `python tuner.py run-experiment ...`.
- Evolutionary SFT is experimental but now first-class in the cloud experiment path. Prefer a checked-in experiment spec or `cloud-pipeline --train-evolutionary-*` overrides over editing trainer YAMLs by hand.
- Use evolutionary training for technical validation or targeted experiments first. It adds significant per-step compute overhead, so smoke tests should usually cap `max_steps` before you commit to a long run.
- For multi-spec benchmark launches, prefer `python3 scripts/launch_experiment_batch.py ...` over back-to-back manual submissions. It defaults to a 5-second stagger.
- For blind stage hardware selection before launch, use `python tuner.py plan-hardware ...`.
- For live HF status and traceback inspection, use `python tuner.py cloud-jobs ...`.
- For finished experiment bundles and next-run suggestions, use `python tuner.py analyze-experiment ...`.
- For loss-driven dataset cleanup, start with `python3 scripts/prune_dataset_from_loss.py ... --analyze-only`; only apply a pruning rule after checking which families are actually enriched in the high-loss slice.
- Prefer the generic pruning strategies first (`loss_threshold`, `top_percent`). Use repo-specific presets only when the analysis output shows that exact family is genuinely overrepresented.
- For in-flight cloud-run health checks, inspect the bucket-backed artifacts first (`training_latest.jsonl`, `stage_summary.json`, `training_lineage.json`, eval/loss partials). Use raw HF logs only as a fallback when the bucket prefix has not started writing yet.
- For quick bucket spot checks, use `python tuner.py bucket read ...` or `python tuner.py bucket list ...` instead of manual `hf buckets cp` commands.
- For local inspection or offline diffing, use `python tuner.py bucket pull ...` to sync a bucket-relative path into the current workspace while preserving its relative path.
- For one-off uploads back into the HF artifact bucket, use `python tuner.py bucket push ...` instead of ad hoc `sync_bucket` snippets.
- For `a100-large` or larger tiers, bias toward aggressive packing. Do not lower batch just because the adapter recipe changed. Start from the highest known-good packed shape for the same model family and only back off after a real OOM or clear instability signal.
- Treat large unused VRAM on `a100-large` as a mistake, not a comfort margin. If `training_lineage.json` shows tens of GB of reserved headroom, the run is underpacked and the next iteration should push batch size harder even if that risks OOM.
- For vLLM eval on multi-GPU hardware, prequantized BitsAndBytes base models (for example `*-bnb-4bit`) cannot use tensor parallelism. Do not assume `x4` means vLLM will shard generation across all GPUs; in this path, eval may need to fall back to single-GPU while exact loss still fans out across all visible GPUs afterward.
- For experiment specs in this repo, set `evaluation.runtime: vllm` and `evaluation.image_profile: fast_vllm` by default. Do not silently switch an experiment to `unsloth` eval just because the model family is new. If vLLM compatibility is uncertain, verify current official support and ask the user before deviating from the vLLM path.
- When the user says "latest run" or "latest experiment", distinguish `latest attempt` from `latest completed baseline`. Check `.tracking/experiments/*/experiment.json` sorted by `created_at` first, then state explicitly which interpretation you are using before copying hyperparameters forward.
- When choosing an A100 packed shape, prefer the nearest latest attempt that actually exercised the hardware over an older completed baseline that clearly underpacked the card.
- If `run-experiment` refuses to launch because the tracked worktree is dirty, prefer creating a clean temporary git worktree and launching from there over asking the user to stash or cleaning their checkout.
- If a cloud run fails before bucket artifacts appear, treat it as a bootstrap/runtime problem first. Inspect `cloud-jobs logs` before changing training hyperparameters.
- For newly released architectures or day-zero model launches, verify official Docker Hub tags for `unsloth/unsloth` and `vllm/vllm-openai` before trusting the repo's pinned image profiles. As of 2026-04-02, Docker Hub shows `unsloth/unsloth:latest` updated 1 day ago and `vllm/vllm-openai:latest` / `v0.17.1` updated about 17 hours ago.
- If a named training image profile is broken, prefer an explicit `training.cloud_image` override to a currently verified official image tag over changing unrelated parts of the experiment such as evaluation backend.
- If a run needs newer package versions but the right base image is otherwise close, use stage-local experiment-spec `pip_packages` pins under `training:`, `evaluation:`, or `loss:` instead of a one-off helper script or a repo-global image pin.
- For hyperparameter search, use `python tuner.py experiment-loop ...`; this is the built-in LLM + LightGBM surrogate path.
- For tabular post-hoc models, use `python tuner.py ml ...` and the configs under `Trainers/ml/configs/templates/`.

## Research Note Lifecycle

Use [`.skills/research-reporting/SKILL.md`](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/.skills/research-reporting/SKILL.md) alongside this skill whenever the user wants experiment tracking, research summaries, or reusable post-run analysis.

- When a training run or experiment is launched, create the research note immediately rather than waiting for all stages to finish.
- The initial note should capture config intent from `experiment.json` and `spec_path`, with `stage_statuses` set from the current known state and metrics left `null` or empty where evidence does not exist yet.
- After training finishes, update the same note with training provenance from `training_lineage.json` or `stage_details.training`.
- After evaluation finishes, update the same note with evaluation metrics and observed failure patterns.
- After loss finishes, update the same note with loss metrics, high-loss hashes, and any dataset-review signals.
- After analysis/recommendation finishes, update the same note with hypotheses, selected candidate rank, and recommended next action.
- Do not fork separate notes per stage unless the user explicitly asks for per-stage notes. Default is one note per experiment, updated over time.
- Treat the note as a living artifact for the run: launch state, in-flight updates, and final postmortem should all land in the same document.
- If a stage has not run yet or failed, preserve the section and leave missing values explicit instead of inventing placeholders.

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
| **LoRA Techniques** | LoRA variants, init methods, config recipes | `reference/lora-techniques.md` |
| **Evolutionary Config** | Experimental gradient-selection config schema and defaults | `reference/training-config.md` |
| **LoRA Surgery** | Eval-guided post-training weight optimization | `reference/lora-surgery.md` |
| **Troubleshooting** | OOM errors, instability, platform issues | `reference/troubleshooting.md` |
| **Env Alignment Protocol** | Canonical SynthChat → SFT → merge/publish → KTO → env-GRPO flow | `protocols/environment-backed-alignment-pipeline.md` |

## LoRA Technique Configs

Reference config templates for LoRA variants in `.skills/fine-tuning/configs/`:

| Template | Technique | When to Use |
|----------|-----------|-------------|
| `regret_free.yaml` | High-rank + rsLoRA + all-linear | Full-FT quality from LoRA |
| `dora.yaml` | DoRA (weight-decomposed) | Drop-in quality boost |
| `qlora_dora.yaml` | QLoRA + DoRA | Single GPU, VRAM-limited |
| `pissa.yaml` | PiSSA (SVD init) | Fast convergence |
| `eva.yaml` | EVA (activation SVD init) | Small datasets |
| `olora.yaml` | OLoRA (QR init) | Fast convergence, simpler than EVA |
| `loftq.yaml` | LoftQ (quant-aware init) | Minimize 4-bit quality loss |
| `grpo_minimal.yaml` | Minimal rank for RL | GRPO/reasoning tasks |

See `reference/lora-techniques.md` for full details, integration status, and compatibility notes.

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
cd Trainers/grpo
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
Gotcha:
- `cloud-jobs show` is useful, but for real progress checks prefer the bucket-backed artifacts once they exist. Early raw logs are often just bootstrap noise; the bucket tells you when training/eval/loss has actually started producing useful state.

**Read the latest training record from a bucket-backed run:**
```bash
python tuner.py bucket read \
  --path runs/hf_jobs/sft/<run-prefix>/logs/training_latest.jsonl \
  --jsonl-latest \
  --pretty
```

**Tail the structured progress log or summary artifact directly from the bucket:**
```bash
python tuner.py bucket analyze \
  --path runs/hf_jobs/sft/<run-prefix>/

python tuner.py bucket read \
  --path runs/hf_jobs/sft/<run-prefix>/logs/training_latest.jsonl \
  --tail 5

python tuner.py bucket read \
  --path runs/hf_jobs/sft/<run-prefix>/logs/stage_summary.json \
  --pretty

python tuner.py bucket list \
  --path runs/hf_jobs/sft/<run-prefix>/ \
  --limit 20

python tuner.py bucket pull \
  --path runs/hf_jobs/sft/<run-prefix>/analysis/loss/ \
  --dest .

python tuner.py bucket push \
  --path local/analysis/notes.json \
  --dest runs/manual_uploads/
```

Gotcha:
- Use `bucket analyze` first when a run has finished. It summarizes `training_lineage.json`, the newest `evaluation_lineage.json`, and `loss_lineage.json` in one pass.
- If the same training run has multiple eval reruns or alternate loss prefixes, pass `--eval-path` and/or `--loss-path` explicitly so the summary cannot attach to the wrong post-training artifacts.
- If schema pass rate is strong but behavior pass rate is still weak, do not assume you need more tool-call syntax SFT. That pattern usually means the next bottleneck is text-only restraint, clarification, or verify-before-action behavior.

**Canonical one-off HF experiment with direct overrides:**
```bash
python tuner.py cloud-pipeline --method sft --preset full \
  --train-model-name Qwen/Qwen3.5-2B \
  --train-image-profile next \
  --train-gpu a10g-small \
  --train-batch-size 8 \
  --train-gradient-accumulation 4 \
  --train-lora-r 128 \
  --train-lora-alpha 256 \
  --train-use-rslora \
  --train-use-dora \
  --train-no-load-in-4bit \
  --train-lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --yes

python tuner.py cloud-pipeline --method sft \
  --train-model-name Qwen/Qwen3-4B \
  --train-gpu a100-large \
  --train-batch-size 24 \
  --train-gradient-accumulation 2 \
  --train-learning-rate 1e-4 \
  --train-max-steps 60 \
  --train-evolutionary-enabled \
  --train-evolutionary-candidates 4 \
  --train-evolutionary-eval-batch-size 2 \
  --train-evolutionary-validation-config configs/fitness/tool_calling.yaml \
  --train-evolutionary-strategy gradient_noise \
  --train-evolutionary-noise-scale 0.03 \
  --train-evolutionary-max-grad-norm 1.0 \
  --train-evolutionary-selection-method best \
  --train-evolutionary-min-improvement 0.01 \
  --train-evolutionary-eval-frequency 5 \
  --train-evolutionary-warmup-steps 200 \
  --yes
```
For SFT experiments, the cloud and experiment surfaces now accept:
- `--train-save-steps` — Override checkpoint save frequency (steps)
- `--train-save-total-limit` — Override max checkpoints kept
- `--train-lora-r`
- `--train-lora-alpha`
- `--train-lora-dropout`
- `--train-use-dora`
- `--train-use-rslora`
- `--train-init-lora-weights`
- `--train-lora-target-modules`

Gotcha:
- `init_lora_weights=loftq` is the only research-style init currently wired for the stable Unsloth SFT path. EVA, PiSSA, and OLoRA still need a separate PEFT-first path.
- `lora_target_modules: all-linear` is forwarded now, but the legacy Unsloth path is still the riskier runtime. Use an explicit module list for the stable baseline.

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
Gotcha:
- `--auto-hardware` can change `batch_size` and `gradient_accumulation` per hardware tier. When comparing speed/cost, always record the resolved training shape alongside the GPU flavor. Otherwise you can misread a faster run as “better hardware” when part of the gain came from a larger planner-selected batch.
- `run-experiment` analysis now appends or updates the checked-in benchmark ledger automatically when analysis runs. If you are comparing ad hoc runs outside experiment orchestration, add or backfill the ledger deliberately.
- When launching several experiment specs in one sweep, do not submit them back-to-back by hand. Use `python3 scripts/launch_experiment_batch.py ... --stagger-seconds 5`. That avoids same-second job submission races and keeps artifact prefixes easier to track.

**Launch a staggered experiment-spec benchmark batch:**
```bash
python3 scripts/launch_experiment_batch.py \
  Trainers/cloud/experiments/qwen3_4b_full_cycle_benchmark_l40sx1_pruned.yaml \
  Trainers/cloud/experiments/qwen3_4b_full_cycle_benchmark_a100_large_pruned.yaml \
  --auto-hardware \
  --optimize-for cost \
  --yes
```

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

**Analyze a dataset against finished per-example loss before pruning:**
```bash
python3 scripts/prune_dataset_from_loss.py \
  --dataset-path Datasets/synthchat/my_dataset.jsonl \
  --experiment-id exp_20260322_103122 \
  --analyze-only
```

**Apply a generic pruning rule:**
```bash
python3 scripts/prune_dataset_from_loss.py \
  --dataset-path Datasets/synthchat/my_dataset.jsonl \
  --experiment-id exp_20260322_103122 \
  --strategy loss_threshold \
  --min-loss 2.0
```

**Apply a capped generic prune budget instead of a hard loss threshold:**
```bash
python3 scripts/prune_dataset_from_loss.py \
  --dataset-path Datasets/synthchat/my_dataset.jsonl \
  --experiment-id exp_20260322_103122 \
  --strategy top_percent \
  --top-percent 0.02
```

**Apply the current repo-specific result-echo preset and publish in one step:**
```bash
python3 scripts/prune_dataset_from_loss.py \
  --dataset-path Datasets/synthchat/my_dataset.jsonl \
  --experiment-id exp_20260322_103122 \
  --strategy result_echo_linesdelta_recommendations \
  --publish-repo professorsynapse/claudesidian-synthetic-dataset
```
Use this only when the analysis report shows that `text_only` result-echo recap rows are genuinely overrepresented in the high-loss slice. Do not assume that family is always the right prune target.

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
- When using `--auto-hardware`, treat the resolved `batch_size` / `gradient_accumulation` as part of the experiment definition. Compare wall-clock and cost only after checking those resolved values.
- After training finishes, read `training_lineage.json` before declaring the hardware/batch shape “good enough.” If peak reserved VRAM is still far below device capacity, the run is underpacked and the next iteration should push batch utilization harder.
- On `a100-large`, the default posture should be: push it until it breaks, then back off one notch. A large A100 headroom cushion is usually wasted iteration speed.
- Finished experiments now write `.tracking/experiments/<id>/analysis/` with:
  `experiment_summary.json`, `run_matrix.csv`, `feature_dataset.{jsonl,csv}`, `next_run_candidates.json`, `draft_next_spec.yaml`.
- For the common train-then-evaluate flow, prefer `python tuner.py cloud-pipeline --method sft --preset full`.
- `cloud-pipeline` is currently a two-job orchestration on HF Jobs, not a single remote composite job.
- `run-experiment` is the higher-level experiment loop: training stays provider-native, then evaluation and exact dataset loss run as separate sibling post-training jobs by default.
- Use `evaluation.runtime: vllm` in experiment specs when you want the fast eval server path. The exact loss stage still uses a post-eval `transformers` forward pass.
- If you explicitly want the older embedded path for a smoke run, set `post_training.mode: same_job` in the experiment spec. Default is `parallel`.
- For `cloud-eval --with-loss` or `post_training.mode: same_job`, stage-local overlay packages such as `peft` must be on the evaluator process `PYTHONPATH`, not only the bucket-sync helper path. If exact loss says a package is "required" even though the command installed it into `/tmp/hf-eval-site`, inspect the runtime `PYTHONPATH` export first.
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
