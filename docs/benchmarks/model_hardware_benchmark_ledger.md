# Model Hardware Benchmark Ledger

This is the checked-in benchmark ledger for model-size and hardware decisions.

Use it to answer:
- which hardware is the current default for a given model size
- what the typical train / eval / loss wall clock looks like
- what the typical cost looks like
- where throughput gains stop paying for themselves

The source of truth is the CSV beside this file:
- [model_hardware_benchmark_ledger.csv](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/docs/benchmarks/model_hardware_benchmark_ledger.csv)

## How To Use It

After any meaningful benchmark round:
1. Read bucket artifacts first.
2. Pull `training_lineage.json`, `evaluation_results.json`, and `loss_summary.json` when present.
3. Use live HF hardware pricing when available, not stale guesses.
4. Add or update one row per comparable run configuration.
5. Record caveats in `notes` instead of hiding them.

## Required Fields

Each row should capture:
- model and rough size bucket
- dataset variant
- method and epochs
- train / eval / loss hardware
- resolved train batch shape
- stage times
- stage costs
- eval quality summary
- loss status
- notable caveats

## Interpretation Rules

- Treat `loss_status != completed` as an incomplete cost/speed comparison.
- If `eval_flavor` or `loss_flavor` differs from `train_flavor`, the row is a pipeline comparison, not a pure training benchmark.
- Always compare quality and cost together. Faster is not automatically better.
- When planner-resolved batch shapes differ, that is part of the benchmark result, not noise to ignore.

## Current Working Read

From the first `Qwen3-4B` pruned SFT speed round:
- `a100-large` is the current default training tier for this 4B setup.
- `l40sx1` is viable but ran too close to OOM to be the preferred default.
- `a100x4` is not yet a valid post-training winner because the exact-loss step failed and eval generation had to fall back for the `bnb-4bit` base model.
