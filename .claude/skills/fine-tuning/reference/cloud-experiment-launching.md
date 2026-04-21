# Cloud Experiment Launching

Use this reference when the goal is a real HF experiment, not a one-off custom job.

## Canonical Rule

For SFT model comparisons on HF Jobs:
- use `python tuner.py cloud-pipeline --method sft --preset full`
- pass model and training changes via `--train-*` CLI overrides
- prefer `--train-image-profile stable|next` over in-job package upgrades when the experiment needs a different Unsloth runtime
- for experiment evaluation in this repo, keep the eval runtime on `vllm` unless the user explicitly approves a fallback
- avoid `cloud-run` unless the job is genuinely custom

This ensures:
- training lands under `runs/hf_jobs/sft/{timestamp}-{sha}/`
- cloud evaluation attaches to that exact run automatically
- downstream discovery (`cloud-eval`, `cloud-inspect`, run comparison) works

## Useful Overrides

- `--train-model-name`
- `--train-image-profile`
- `--train-cloud-image`
- `--train-gpu`
- `--train-timeout-hours`
- `--train-batch-size`
- `--train-gradient-accumulation`
- `--train-learning-rate`
- `--train-num-epochs`
- `--train-max-steps`
- `--train-max-seq-length`
- `--train-lora-r`
- `--train-lora-alpha`
- `--train-lora-dropout`
- `--train-use-dora`
- `--train-use-rslora`
- `--train-init-lora-weights`
- `--train-no-load-in-4bit`
- `--train-lora-target-modules`

## Example

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
```

## Experiment Spec Surface

The same SFT LoRA controls are also available in `run-experiment` YAML under `training:`:

```yaml
training:
  model_name: Qwen/Qwen3-4B
  lora_r: 128
  lora_alpha: 256
  lora_dropout: 0.05
  use_dora: true
  use_rslora: true
  init_lora_weights: loftq
  lora_target_modules: all-linear
```

Use `"all-linear"` only when you intend to exercise the newer Unsloth model path. On the legacy path, keep an explicit module list for the stable baseline.

For day-zero model launches or runtime compatibility fixes, the experiment spec can also pin stage-local pip packages without changing the global cloud image profile:

```yaml
training:
  model_name: google/gemma-4-E4B-it
  cloud_image: unsloth/unsloth:latest
  pip_packages:
    - unsloth==2026.4.2
    - unsloth-zoo==2026.4.2
    - transformers==5.3.0

evaluation:
  runtime: vllm
  image_profile: fast_vllm
  pip_packages:
    - vllm==0.12.0
```

These entries are passed straight to `python -m pip install --upgrade ...` inside the stage job. Keep them scoped to the specific run that needs them instead of mutating the repo-wide default image pin.

For experiment evaluation, prefer an explicit `vllm` stanza:

```yaml
evaluation:
  enabled: true
  preset: full
  runtime: vllm
  image_profile: fast_vllm
```

Do not silently swap this to `unsloth` because a model family is new. If vLLM support is uncertain, verify current official support and image freshness first, then ask the user before deviating from the vLLM path.

Post-training orchestration is also configurable in the experiment spec:

```yaml
post_training:
  mode: parallel   # parallel | same_job
```

- `parallel` is the default and launches evaluation and exact loss as separate sibling jobs after training
- `same_job` keeps the older embedded eval+loss path for smoke or fallback usage

## A100 Packing Rule

For `a100-large` experiments, the default should be aggressive packing, not conservative validation:

- do not shrink batch just because the LoRA variant changed
- use the best known packed batch shape from the nearest comparable run as the starting point
- read `training_lineage.json` after the run and treat large reserved-VRAM headroom as underpacking
- if the card still has tens of GB free, push batch harder next time
- for this repo, a first-pass OOM on A100 is often a better signal than a slow run with half the card unused

## Promotion Path

1. Run SFT comparisons through `cloud-pipeline`
2. Inspect evaluation outputs and choose the winner
3. Merge/publish the winning SFT artifact
4. Run KTO from the merged/published model
5. Run env-GRPO as the final online stage

## Image Discipline

- `stable` keeps the currently pinned Unsloth image used by existing HF Jobs runs
- `next` is the opt-in path for newer official Unsloth Docker images when smoke-testing newer architectures
- smoke-test `next` before trusting it for Qwen3.5 or Ministral; current upstream docs can get ahead of the exact image contents
- prefer image-profile switches to ad hoc `pip install --upgrade transformers` in the training container
- if a named image profile fails during bootstrap before model load, inspect HF job logs first and treat it as an image/runtime problem, not a training-config problem
- if the profile itself is broken, prefer an explicit `training.cloud_image` override to a currently verified official tag rather than changing evaluation backend
- if the image is close but still too old for a just-released model, prefer stage-local `pip_packages` pins in the experiment spec over inventing a one-off helper script or silently changing unrelated stages
- for newly released architectures, verify the latest official `unsloth/unsloth` and `vllm/vllm-openai` tags on Docker Hub before relying on stale local pins; as of 2026-04-02, `unsloth/unsloth:latest` was updated 1 day ago and `vllm/vllm-openai:latest` / `v0.17.1` about 17 hours ago

## Run Selection Guardrails

- When the user asks to mirror the "latest" experiment, check whether they mean the latest attempt or the latest completed baseline.
- For A100 packing, a newer failed smoke run can still be the right source for batch shape if it exercised the hardware more realistically than an older underpacked completed run.

## Warm Iteration Strategy

HF Jobs are still the right default for recorded train/eval artifacts, but they are not a warm compute pool. If repeated eval/debug retries are dominated by:
- repulling the same model artifacts
- reinstalling the same runtime packages
- reproing the same vLLM startup issue

then switch the iteration loop to a Docker GPU Space with persistent storage instead of repeatedly submitting new HF Jobs.

The checked-in reusable CLI for this path is:

```bash
python3 Trainers/cloud/scripts/manage_space.py deploy \
  --space-id <user>/<space> \
  --template vllm_warm \
  --base-image ghcr.io/<org>/<image>:<tag> \
  --hardware a10g-small \
  --sleep-time 3600 \
  --var BASE_MODEL=<model>
```

Use repeated `--var KEY=VALUE` and `--secret KEY=VALUE` flags to keep the Space generic instead of encoding model-specific behavior into the scaffold itself.

For the full workflow and operational guidance, load `reference/hf-spaces-warm-iteration.md`.
