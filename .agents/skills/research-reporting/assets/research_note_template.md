---
note_type: experiment-research-note
generated_at: YYYY-MM-DDTHH:MM:SSZ
experiment_id: exp_YYYYMMDD_HHMMSS
baseline_experiment_id: null
comparison_experiment_ids: []
title: concise-experiment-title
slug: concise-experiment-title
status: completed
provider: hf_jobs
method: sft
objective: short_objective
objective_met: null
primary_bottleneck: null
model_name: org/model
dataset_source: org/dataset/file.jsonl
spec_path: /abs/path/to/experiment_spec.yaml
source_commit: null
training_run_id: null
evaluation_run_id: null
loss_run_id: null
stage_statuses:
  training: completed
  evaluation: completed
  loss: completed
artifacts:
  experiment_summary_json: .tracking/experiments/exp_YYYYMMDD_HHMMSS/analysis/experiment_summary.json
  next_run_candidates_json: .tracking/experiments/exp_YYYYMMDD_HHMMSS/analysis/next_run_candidates.json
  hypothesis_context_json: .tracking/experiments/exp_YYYYMMDD_HHMMSS/analysis/hypothesis_context.json
  run_matrix_csv: .tracking/experiments/exp_YYYYMMDD_HHMMSS/analysis/run_matrix.csv
config_snapshot:
  dataset:
    source: org/dataset
    file: file.jsonl
    hash: null
  training:
    gpu: null
    image_profile: null
    batch_size: null
    gradient_accumulation: null
    learning_rate: null
    num_epochs: null
    max_steps: null
    max_seq_length: null
    load_in_4bit: null
    lora_r: null
    lora_alpha: null
    lora_dropout: null
    use_dora: null
    use_rslora: null
    init_lora_weights: null
    lora_target_modules: []
  evaluation:
    enabled: null
    preset: null
    runtime: null
    gpu: null
    image_profile: null
    max_seq_length: null
  loss:
    enabled: null
    gpu: null
    max_seq_length: null
    completion_only: null
  features: {}
runtime_snapshot:
  hardware:
    training: null
    evaluation: null
    loss: null
  timings:
    created_at: null
    training_started_at: null
    training_finished_at: null
    evaluation_started_at: null
    evaluation_finished_at: null
    loss_started_at: null
    loss_finished_at: null
  cost:
    currency: USD
    total: null
    training: null
    evaluation: null
    loss: null
metrics:
  primary:
    name: null
    value: null
    higher_is_better: null
    source: null
  summary: {}
  groups:
    evaluation: {}
    loss: {}
    training: {}
dominant_failure_tags: []
high_loss_hashes: []
selected_candidate_rank: null
selected_candidate_confidence: null
recommended_next_action: null
open_questions: []
sources: []
---

# Summary

One paragraph on the overall result. State whether the run met its objective, what the main bottleneck is, and what should happen next.

# Run Context

- Experiment: `exp_YYYYMMDD_HHMMSS`
- Objective: `short_objective`
- Model / method: `org/model` / `sft`
- Dataset: `org/dataset/file.jsonl`
- Stage status: training=`completed|failed|missing`, evaluation=`completed|failed|missing`, loss=`completed|failed|missing`

# Observed Results

- Primary metric: `name=value` if the run defines one
- Important metric groups: evaluation, loss, training, or custom groups present in artifacts
- Notable tags or slices: name the strongest supported patterns only
- Config context: mention the small number of knobs most relevant to interpretation

# Failure Analysis

State the strongest observed failure patterns. Prefer tags, failure slices, or high-loss clusters over vague language.

- Pattern 1:
- Pattern 2:
- Pattern 3:

# Hypotheses

Separate observation from interpretation. Each hypothesis should cite the supporting signal.

1. Hypothesis:
   Evidence:
   Confidence:
2. Hypothesis:
   Evidence:
   Confidence:

# Next Run

- Selected candidate rank:
- Recommended action:
- Why this is the next action:
- What would falsify it:

# Sources

- `.tracking/experiments/.../experiment.json`
- `.tracking/experiments/.../analysis/experiment_summary.json`
- `.tracking/experiments/.../analysis/next_run_candidates.json`
