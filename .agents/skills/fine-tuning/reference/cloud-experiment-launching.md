# Cloud Experiment Launching

Use this reference when the goal is a real HF experiment, not a one-off custom job.

## Canonical Rule

For SFT model comparisons on HF Jobs:
- use `python tuner.py cloud-pipeline --method sft --preset full`
- pass model and training changes via `--train-*` CLI overrides
- avoid `cloud-run` unless the job is genuinely custom

This ensures:
- training lands under `runs/hf_jobs/sft/{timestamp}-{sha}/`
- cloud evaluation attaches to that exact run automatically
- downstream discovery (`cloud-eval`, `cloud-inspect`, run comparison) works

## Useful Overrides

- `--train-model-name`
- `--train-gpu`
- `--train-timeout-hours`
- `--train-batch-size`
- `--train-gradient-accumulation`
- `--train-learning-rate`
- `--train-num-epochs`
- `--train-max-steps`
- `--train-max-seq-length`
- `--train-no-load-in-4bit`
- `--train-lora-target-modules`

## Example

```bash
python tuner.py cloud-pipeline --method sft --preset full \
  --train-model-name Qwen/Qwen3.5-2B \
  --train-gpu a10g-small \
  --train-batch-size 8 \
  --train-gradient-accumulation 4 \
  --train-no-load-in-4bit \
  --train-lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --yes
```

## Promotion Path

1. Run SFT comparisons through `cloud-pipeline`
2. Inspect evaluation outputs and choose the winner
3. Merge/publish the winning SFT artifact
4. Run KTO from the merged/published model
5. Run env-GRPO as the final online stage
