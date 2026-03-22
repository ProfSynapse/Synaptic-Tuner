# Checkpoint Evaluation Reference

Best-checkpoint selection via eval — finds the actual best checkpoint, not just the one with lowest training loss.

---

## Why Checkpoint Evaluation?

Training loss and eval quality don't always correlate. A checkpoint with lower loss may have overfit. CheckpointEvaluator runs actual eval scenarios to find the checkpoint that performs best on real tasks.

---

## How It Works

1. **Discover** — Scan the run directory for all checkpoints
2. **Pre-filter** — Read training log to get per-checkpoint loss, keep top N by lowest loss
3. **Evaluate** — Run full evaluation on surviving checkpoints
4. **Rank** — Sort by eval score (descending)
5. **Report** — Return `CheckpointReport` with rankings and best checkpoint

The pre-filter step is a cost optimization: full evaluation is expensive, so we only evaluate the most promising checkpoints.

---

## Data Structures

```python
CheckpointInfo(path, step, training_loss)
CheckpointResult(path, step, training_loss, eval_score, rank)
CheckpointReport(checkpoints_evaluated, best, final_model_rank, results)
```

---

## Usage

CheckpointEvaluator is used internally by:
- **Experiment Loop** — Scores each experiment's checkpoints
- **LoRA Surgery** — Evaluates before/after surgery modifications

It depends on `EvalBackend` (protocol in `shared/eval_backend.py`) which supports:
- **Local evaluation** — On-device inference
- **Cloud evaluation** — Remote inference via cloud provider

---

## Configuration

CheckpointEvaluator is configured programmatically (not via standalone YAML), but the eval backend and scenarios it uses are configured by the calling system:

- Experiment loop: `configs/flywheel/experiment_loop.yaml` → `eval_backend` field
- LoRA surgery: `configs/lora_surgery.yaml` → `eval_backend` field

---

## Tips

- Pre-filtering is aggressive by design — it avoids running expensive evals on clearly bad checkpoints
- The `final_model_rank` field tells you if the last checkpoint was actually the best
- If `final_model_rank > 1`, an earlier checkpoint outperformed the final model — common with overfitting
- CheckpointEvaluator works with any EvalBackend implementation (local or cloud)
