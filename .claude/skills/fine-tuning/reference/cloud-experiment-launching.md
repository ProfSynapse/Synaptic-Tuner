# Cloud Experiment Launching

Use this reference when the goal is a real HF experiment, not a one-off custom job.

## Canonical Rule

For SFT model comparisons on HF Jobs:
- use `python tuner.py cloud-pipeline --method sft --preset full`
- pass model and training changes via `--train-*` CLI overrides
- prefer `--train-image-profile stable|next` over in-job package upgrades when the experiment needs a different Unsloth runtime
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

Post-training orchestration is also configurable in the experiment spec:

```yaml
post_training:
  mode: parallel   # parallel | same_job
```

- `parallel` is the default and launches evaluation and exact loss as separate sibling jobs after training
- `same_job` keeps the older embedded eval+loss path for smoke or fallback usage

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
