# Decision-Model Training (`decision` method)

Jev-style "System One" models: a state plus typed questions in, calibrated
option probabilities out, one forward pass, no generation. The method lives in
`Trainers/decision/`; `Trainers/decision/README.md` is the full description.

## When to use

- Routing, triage and policy checks that pick from a known option set
  ("which team", "needs human?", "severity 1–4"), where you need a probability
  you can threshold, not text you must parse.
- Use SFT instead when the output is free text or a tool call.

## Question types

| Type | Options rendered | Answer |
|------|------------------|--------|
| `noul` | `false`, `true` (with optional criteria text) | P(true) |
| `choice` | the named options from `criteria` | best option + distribution |
| `score` | ordered levels, lowest first | expected level + distribution |

## Configuration (all YAML)

- `Trainers/decision/configs/model_registry.yaml` is the torso SSOT (hf_id, pinned
  revision, loader `causal_lm` | `text_tower` + `causal_lm_class`, dtype,
  max_length, LoRA targets). Select with `model.registry_name`; add a model by
  adding one block. Registered: `qwen35-{0.8b,2b,4b,9b}-base`, `qwen3-{1.7b,4b}-base`.
- Run configs: `config.yaml` (pointer), `letter_logits.yaml`, and smoke copies
  `smoke_pointer.yaml` / `smoke_letter_logits.yaml` (only sizes differ; tests
  enforce identical model/prompt/lora/augmentation/loss sections).
- `prompt.marker_style` and `prompt.headers` (one line per kind) define the
  rendered prompt and are saved into `final_model/decision_config.json`, so serving
  renders exactly what training saw.
- Per-run registry overrides live under `model:` (`revision`, `max_length`,
  `torch_dtype`, `attn_implementation`) and `lora.target_modules`; `null` keeps the
  registry value.
- Recipes only point `--config` at a config file (plus artifact paths). Do not add
  `--max-steps` etc. to a recipe; make a config instead.

## Readouts

| `model.readout` | Config | Parameters | Notes |
|-----------------|--------|-----------|-------|
| `pointer` | `configs/config.yaml` | ~1M head + LoRA | Strands Decider v19 design. Unbounded option count. Default KL to frozen readout 0.3. |
| `letter_logits` | `configs/letter_logits.yaml` | LoRA only | OpenJev design. Reads the LM's marker-token logits at `<answer>`. Rows with more options than single-token markers (letters: 26, numbers: 9) are dropped. Brier 0.1. |

Keep everything else identical when comparing readouts. The two configs differ
only in readout, marker style and loss weights.

## Workflow

```bash
# Corpus (Docker; the strands-decider builder imports torch)
python tuner.py local-run --job-config Trainers/recipes/decision_strands_corpus_build.yaml --yes

# Validate config + data without loading a model
python Trainers/decision/train_decision.py --config Trainers/decision/configs/config.yaml --dry-run

# Smoke: train -> calibrate -> evaluate -> lineage
python tuner.py local-run --job-config Trainers/recipes/decision_qwen35_2b_pointer_smoke.yaml --yes
python tuner.py local-run --job-config Trainers/recipes/decision_qwen35_2b_letter_logits_smoke.yaml --yes

# Full runs (whole corpus, one epoch)
python tuner.py local-run --job-config Trainers/recipes/decision_qwen35_2b_pointer.yaml --yes
python tuner.py local-run --job-config Trainers/recipes/decision_qwen35_2b_letter_logits.yaml --yes

# Stages on an existing checkpoint
python Trainers/decision/train_decision.py --stage calibrate --checkpoint <run>/final_model
python Trainers/decision/train_decision.py --stage evaluate  --checkpoint <run>/final_model
```

CLI overrides (for ad hoc use; recipes use configs): `--model <registry_name>`, `--readout`, `--train-file`
(repeatable), `--eval-file`, `--max-steps`, `--max-train-examples`,
`--max-eval-examples`, `--output-root`, `--run-timestamp`. All other knobs are
YAML. Unknown YAML keys are rejected.

## Data rules

- `decision-row/v1` rows (`kind, state, instructions, options, label, task,
  weight, instruction_variants`) or Jev request rows with an `answers` map.
- `label` indexes the canonical option order. The collator shuffles options
  per draw, so never pre-shuffle to augment.
- Training drops rows longer than `model.max_length`; it never truncates them.
  At evaluation, an over-long state is cut from the front.
- Calibration rows must not be the reported eval rows. With empty
  `data.calibration_files`, `calibration.holdout_fraction` of the eval rows
  (stratified by task) are carved off for calibration.
- `--max-*-examples` take a seeded sample, not a head slice, because corpus
  files are grouped by task.

## Reading results

`evaluation/decision_eval.json` and `training_lineage.json["decision"]`:

- `calibrated` / `uncalibrated`: accuracy, NLL, Brier, ECE. Temperature
  scaling should lower NLL and ECE without changing accuracy.
- `by_kind`, `by_task`: look for weak held-out tasks. Strands' four held-out
  tasks (emotion, massive_intent, sarcasm, hate_severity) have label sets never
  seen in training, which makes them the generalisation test.
- `order_consistency.answer_change_rate`: the share of rows whose answer flips
  when options are reordered. OpenJev reports 2.3% after tuning, against 18.5%
  for the untuned base model. A high rate means the readout learned slot
  positions instead of option content.
- The run registry's primary metric is held-out `eval_accuracy` when the
  evaluate stage ran.

## Confidence analysis

The question: is the confidence calibrated, and does an internal probe know
more than the readout says? The model always chooses (no abstention). This
pass is read-only over a trained `final_model`:

```bash
python tuner.py local-run --job-config Trainers/recipes/decision_confidence_analysis_pointer.yaml --yes
```

It writes `confidence_report.json`:

- **Arms:** R0 raw, R1 CAL-refit temperature, P-dial correctness probe
  (MechInterp layer sweep), P-gate real-vs-ablated probe, and S stacked.
- **Hypotheses:** `h2_*` paired AUROC diffs. H2 asks whether the probe beats
  the readout.
- **Ablated twins:** `humility_ablated` (max-p vs chance on evidence-removed twins).
- **Sets:** `conformal` (LAC sets per kind: coverage and size, real vs ablated).

Thresholds, splits, layers and alphas live in
`configs/experiments/confidence_analysis_*.yaml`. Rows that carry a
prior-knowledge label in `meta.knowledge` (`known` / `unknown` / `ambiguous`,
from an external labeling protocol) also get a `knowledge` block in the report.

Optional outputs (see `Trainers/decision/README.md` for the full field list):

- **`export.per_row`** (default `true`): `test_rows.jsonl` records gain
  `row_index`, `probs_r0` / `probs_r1` (canonical option order), `dial_score`,
  `stack_score`, `ku_probe_score` (`null` if that probe was not fit) and
  `direction_scores`. Every record carries `split` (`test`, `cal` or `fit`).
- **`export.cal_rows`** (default `false`): also writes `cal_rows.jsonl`, the
  same records for CAL rows (`row_index` within CAL). R1 uses the same CAL-fit
  temperatures, and probe, stacker and direction scores are computed as for
  TEST. Fit abstention thresholds on CAL and apply them to TEST.
- **`export.fit_rows`** (default `false`): also writes `fit_rows.jsonl` for FIT
  rows; probe scores there are in-sample.
- **`export.states`** (default `false`): writes `test_states.npz` with the TEST
  `<answer>` states, keyed `L{i}` (float16) plus `row_index`. `export.layers`
  picks a subset of the captured layers to keep the file small.
- **`directions`** (default `[]`): a list of `{name, path, layer?}`. Each `path`
  is a `mechinterp-direction/v1` JSON from `MechInterp.probe.fit.freeze_direction`.
  Every TEST row is scored with the direction's own logistic decision value,
  `raw_norm * (h @ vector) + intercept`, the scale its `sigma` describes. Each
  direction adds a `directions.<name>` report block:
  - AUROC against correctness, and against `meta.knowledge` when present
  - a paired-bootstrap difference against R1
  - class-conditioned mean and std of the score

  A missing file, wrong schema, uncaptured layer or hidden-size mismatch is a
  hard error. Under `local-run`, add the JSON to the recipe's `setup.copy`.

A layer index is a position in the decoder's `output_hidden_states` tuple:
`0` is the embeddings and `i` is the output of decoder block `i`. This is the
same convention as `capture.layers` and `MechInterp.extraction`.

## Gotchas

- Qwen3.5 needs transformers 5.x. The recipes overlay `transformers==5.17.0` and
  `peft==0.21.0` on the Unsloth image. These pins are candidates from the
  strands-decider README and have not been captured from a passing smoke here.
  Do not reuse the `qwen35-sft-v1` profile: it admits only Qwen3.5-4B for SFT.
- `flash-linear-attention` is Linux-only. Without it, the Gated DeltaNet layers
  fall back to a PyTorch path that is about 2.3× slower.
- LoRA targets must include Qwen3.5's linear-attention projections
  (`in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`). Otherwise LoRA
  reaches only 6 of 24 layers.
- `pointer` uses `head_learning_rate` (1e-3) for its randomly initialised head.
  `letter_logits` has no head and ignores it.
- `train --method decision` on HF Jobs is not wired, because the command builder
  emits SFT flags. Use `cloud-run` with explicit steps, or local Docker.
