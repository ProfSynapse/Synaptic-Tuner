# Artifact Map

Use this reference to decide which repo artifact supports which part of the research note.

## Primary Artifacts

| Artifact | Primary Use | Notes |
|----------|-------------|-------|
| `.tracking/experiments/<id>/experiment.json` | Experiment identity, dataset, provider, run ids, derived output paths | Good first file for orientation |
| Experiment spec at `spec_path` | Actual stage config values | Best source for training/eval/loss knobs like GPU, LR, batch size, LoRA params, and max sequence length |
| `analysis/experiment_summary.json` | Top-level experiment snapshot | Canonical source for status, method, objective, eval summary, artifact roots, stage lineage paths |
| `analysis/next_run_candidates.json` | Ranked recommendations and loss snapshot | Canonical source for `selected_candidate_rank`, confidence, signals, and next-run proposal |
| `analysis/hypothesis_context.json` | Rich supporting evidence | Use for tag-level failures, supporting examples, and interpretation support |
| `analysis/run_matrix.csv` | Stage-by-stage completion/status | Useful for quick validation of what ran successfully |

## Optional Artifacts

| Artifact | When to Use | Notes |
|----------|-------------|-------|
| `analysis/failure_slices/eval_failures.jsonl` | You need representative evaluation failures | Summarize patterns, do not paste long raw examples |
| `analysis/failure_slices/high_loss_examples.jsonl` | You need representative hard examples | Pair with high-loss hashes from `next_run_candidates.json` |
| `analysis/experiment_summary.md` | Human-readable skim only | Prefer the JSON for structured reporting |

## Provenance Artifacts

| Artifact | Primary Use |
|----------|-------------|
| Experiment spec at `spec_path` | Config snapshot for dataset, training, evaluation, loss, and feature stages |
| `training_lineage.json` | Training provenance, commit, runtime metadata |
| `evaluation_lineage.json` | Eval provenance and output lineage |
| `loss_lineage.json` | Loss provenance and output lineage |

## Section-to-Artifact Mapping

| Note Section | Best Source |
|--------------|-------------|
| Frontmatter identity fields | `experiment.json`, `experiment_summary.json` |
| Config snapshot | experiment spec at `spec_path`, then lineage files if needed |
| Metrics block | `experiment_summary.json`, `next_run_candidates.json`, `run_matrix.csv` |
| Failure Analysis | `hypothesis_context.json`, failure slices |
| Hypotheses | `next_run_candidates.json` first, `hypothesis_context.json` second |
| Next Run | `next_run_candidates.json` |
| Sources | Every artifact actually read while drafting |

## Missing Data Rules

- Missing `next_run_candidates.json`: leave recommendation fields empty and do not invent a ranked next step.
- Missing loss outputs: keep loss metrics null and explain that the loss stage is unavailable or failed.
- Missing failure slices: summarize only aggregate patterns from summary/context artifacts.
- Missing `spec_path` or missing spec file: populate `config_snapshot` from `experiment.json` and lineage tags only, and mark missing knobs as `null` instead of guessing.
- Multiple reruns: prefer the paths referenced by the current experiment bundle instead of manually picking nearby artifacts.
