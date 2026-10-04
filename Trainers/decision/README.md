# Decision models (`decision` method)

Train a "System One" decision model in the style of TypeSafe's Jev and its open
reproductions (AWS Strands Decider 2B, Open-Jev, OpenJev). A decision model
never generates text. It reads a **state** and a set of named, typed questions,
and returns calibrated probabilities in one forward pass:

| Type | Answer | Example |
|------|--------|---------|
| `noul` | P(true) | "Does this need human approval?" |
| `choice` | one of N named options + distribution | "Which team handles this?" |
| `score` | expected level on an ordered scale + distribution | "How severe, low to critical?" |

## How it works

A causal-LM torso from `configs/model_registry.yaml` (default `qwen35-2b-base`) gets a LoRA adapter and one
of two readouts:

- **`pointer`** (Strands Decider v19): a ~1M-parameter head scores option *k*
  as an attention score between the hidden state at `<answer>` and the hidden
  state at the last token of option *k*'s line. No per-slot weights, so the
  option count is unbounded.
- **`letter_logits`** (OpenJev): no new parameters. Option *k*'s logit is the
  LM's own next-token logit for its marker (`A`, `B`, ... or `1`..`9`) at
  `<answer>`. Rows with more options than single-token markers are dropped.

Both share the prompt (`decision_core/prompting.py`), the collator
(`decision_core/collate.py`), the loss (`decision_core/losses.py`) and the
calibration and evaluation path.

- **Option shuffling.** Options are re-permuted on every draw and the label is
  remapped to follow. Score levels are never shuffled, only reversed as a whole.
- **Loss.** Soft cross-entropy (with ordinal smoothing on score rows), plus an
  optional `brier_weight` × Brier term, plus an optional `kl_frozen_weight` ×
  KL(frozen torso's marker readout ‖ student).
- **Calibration.** After training, one temperature per question kind is fitted
  on held-out rows by NLL and saved in `final_model/decision_config.json`.
- **Evaluation.** Accuracy, NLL, Brier and ECE, reported overall, per kind and
  per task. It also reports how often the answer changes when options are
  reordered.

## Configuration

Everything is YAML; the Python carries no model- or prompt-specific behaviour.

| File | What it holds |
|------|---------------|
| `configs/model_registry.yaml` | Torso SSOT: `hf_id`, pinned `revision`, `loader` (`causal_lm` or `text_tower` + `causal_lm_class`), dtype, `max_length`, LoRA targets, notes. Adding a model is one block. |
| `configs/config.yaml` | Full pointer run: `model.registry_name`, readout, `prompt` (marker style + per-kind headers), LoRA, data, augmentation, loss, training, calibration, evaluation. |
| `configs/letter_logits.yaml` | Same run with the letter-logit readout. |
| `configs/smoke_pointer.yaml`, `configs/smoke_letter_logits.yaml` | Smoke-sized copies (sample sizes, steps, cadence); model/prompt/loss sections must match the full configs (enforced by tests). |
| `configs/corpus_strands_v5.yaml` | Corpus build definition. |

Switching torso is `model.registry_name: qwen3-4b-base` (or `--model qwen3-4b-base`).
Registry values (revision, max_length, dtype, attention, LoRA targets) can be
overridden per run in `model:` / `lora.target_modules`; `null` keeps the registry
value. The resolved model config -- including the prompt headers -- is written to
`final_model/decision_config.json`, so a checkpoint reloads and serves with exactly
the prompt it was trained on, without the registry or run config.

## Data

Rows are `decision-row/v1`, field-compatible with the strands-decider corpus:

```json
{"kind": "choice", "state": "My card was charged twice.", "instructions": "Which team?",
 "options": [["billing", "payments"], ["technical", "bugs"]], "label": 0,
 "task": "support_routing", "weight": 1.0, "instruction_variants": []}
```

Jev request rows with gold answers also load. Each question becomes one row:

```json
{"state": "...", "questions": {"team": {"type": "choice", "instructions": "Which team?",
  "criteria": {"billing": "...", "technical": "..."}}}, "answers": {"team": "billing"}}
```

The public corpus (21 training tasks plus 7 held-out tasks, built by the pinned
strands-decider builder) is defined in `configs/corpus_strands_v5.yaml`. The
build writes `Datasets/decision/strands_v5/` (git-ignored) and records each
file's hash against the published reproduction hashes.

## Run

```bash
# 1. Build the corpus (Docker; the builder imports torch)
python tuner.py local-run --job-config Trainers/recipes/decision_strands_corpus_build.yaml --yes

# 2. Check config + data, render a sample prompt (no model load)
python Trainers/decision/train_decision.py --config Trainers/decision/configs/config.yaml --dry-run

# 3. Smoke train -> calibrate -> evaluate (Docker)
python tuner.py local-run --job-config Trainers/recipes/decision_qwen35_2b_pointer_smoke.yaml --yes
python tuner.py local-run --job-config Trainers/recipes/decision_qwen35_2b_letter_logits_smoke.yaml --yes

# 4. Full runs (whole corpus, one epoch)
python tuner.py local-run --job-config Trainers/recipes/decision_qwen35_2b_pointer.yaml --yes
python tuner.py local-run --job-config Trainers/recipes/decision_qwen35_2b_letter_logits.yaml --yes

# Re-calibrate or re-evaluate an existing checkpoint
python Trainers/decision/train_decision.py --stage evaluate --checkpoint <run>/final_model
```

A run writes `<output_root>/<timestamp>/` containing:

- `checkpoints/`
- `logs/`
- `final_model/`: the adapter, `readout_head.safetensors` (pointer only) and `decision_config.json`
- `evaluation/decision_eval.json`
- `run_config.json`
- `training_lineage.json`

## Confidence analysis

`analyze_confidence.py` is a read-only pass over a trained `final_model`. It asks
whether the model's confidence is calibrated, and whether an internal probe knows more
than the readout says. It runs on held-out rows plus **state-ablated twins**
(the evidence removed, so the question becomes unknowable).

- **Confidence arms:**
  - raw (R0)
  - calibrated (R1, temperatures refitted on its own CAL split)
  - a correctness probe at `<answer>` (P-dial; `MechInterp.probe.fit` layer sweep)
  - a real-vs-ablated gate probe (P-gate)
  - a stack of R1 and P-dial (S)
- **Metrics:**
  - ECE / NLL / Brier, AUROC with a bootstrap floor, AURC
  - confident-wrong and underconfident-right rates
  - the confidence-over-chance gap on ablated twins
  - split-conformal LAC sets per kind and ordinal intervals

The model never abstains. If rows carry a prior-knowledge label in
`meta.knowledge` (`known` / `unknown` / `ambiguous`), supplied by an external
labeling protocol, the report adds a `knowledge` block: per-group accuracy and
confidence, and whether the readout, the P-dial probe and a known-vs-unknown
probe separate the groups.

```bash
python tuner.py local-run --job-config Trainers/recipes/decision_confidence_analysis_pointer.yaml --yes
python tuner.py local-run --job-config Trainers/recipes/decision_confidence_analysis_letter_logits.yaml --yes
```

Config: `configs/experiments/confidence_analysis_*.yaml`. The checkpoint can be
`latest:<run root>`.

Outputs, under `<output_root>/<timestamp>/`:

- `confidence_report.json`: every arm and metric.
- `analysis_config.json`: the resolved config.
- `test_rows.jsonl`: one record per TEST row, keyed by `row_index`. With
  `export.per_row: true` (the default) each record also carries:
  - `probs_r0` / `probs_r1`: raw and calibrated option probabilities, in
    canonical option order
  - `dial_score`, `stack_score`: the P-dial probe's and the stacker's decision
    values (logits; `p_dial` and `p_stack` are their sigmoids)
  - `ku_probe_score`: the known-vs-unknown probe's decision value, or `null`
    when that probe was not fit
  - `direction_scores`: `{name: score}` for each external direction
- `test_states.npz` (only with `export.states: true`): the `<answer>` hidden
  states of TEST rows, one float16 `(n_test, hidden_size)` array `L{i}` per
  hidden-state index, plus `row_index`. `export.layers` limits it to a subset of
  the captured layers.

**Layer indices.** Everywhere in this config (`capture.layers`,
`export.layers`, `directions[].layer`) a layer is an index into the decoder's
`output_hidden_states` tuple: `0` is the embedding output and `i >= 1` is the
output of decoder block `i` (`layers[i - 1]`). In HF Llama/Qwen-style decoders
the last index usually carries the final norm. `MechInterp.extraction` captures
the same tuple, so a direction frozen at index `i` there is scored at index `i`
here. The states come from the decision model with its LoRA adapter applied.

**External directions.** `directions` lists frozen directions to score TEST
rows along:

```yaml
directions:
  - name: my_direction
    path: path/to/direction.json   # repo-relative or absolute
    layer: 18                      # optional; overrides the JSON's layer
```

Each `path` is a `mechinterp-direction/v1` JSON written by
`MechInterp.probe.fit.freeze_direction`. The score is the direction's own
logistic decision value, `raw_norm * (h @ vector) + intercept` for a normalized
vector, which is exactly `h @ coef + intercept`. It is the score that the JSON's
`sigma` and `calibration` class statistics describe, so per-row values can be
compared with them. It ranks rows exactly like the `mu`-centred projection on
the unit vector, so the AUROCs are the same either way.

Each direction adds a `directions.<name>` block:

- `layer`, `source_layer` (from the JSON), `score_rule`, `n`
- `auroc_correct`: AUROC against the model's own correctness on all TEST rows
- `auroc_correct_minus_r1`: paired-bootstrap AUROC difference against the
  calibrated readout (R1)
- `by_correctness`: `n` / `mean` / `std` of the score on correct and wrong rows
- when rows carry `meta.knowledge`: `auroc_known_vs_unknown`,
  `auroc_known_vs_unknown_minus_r1` and `by_knowledge`

The run fails before any forward pass if a file is missing, has another
schema, or its vector does not match the model's hidden size. It fails after
capture if the direction's layer was not captured. Under `local-run`, add each
direction file to the recipe's `setup.copy`. `--dry-run` also loads and checks
the direction files.

## Status and gaps

- The recipe pins (`transformers==5.17.0`, `peft==0.21.0`) follow the
  strands-decider README and have **not** been captured from a passing smoke
  here. Qualify a runtime profile before treating a run as reproducible.
- The configs train on the v5 short-task corpus only. The published v19
  checkpoint also trains on:
  - multi-step rows (ContractNLI, MuSiQue, BoardgameQA)
  - generated document questions
  - HelpSteer2 adequacy rows
  - teacher/parent replay distributions

  Expect lower scores on long-document and multi-step questions until those
  rows are added.
- Cloud: `cloud-run` with explicit steps works, since it ignores the method.
  The canonical `train --method decision` HF Jobs path is not wired: the command
  builder emits SFT-style flags, and the corpus is git-ignored.
- `resume_from_checkpoint` is not supported. Checkpoints hold only the adapter
  and the head.
- `Trainers/shared` callbacks import `shared.training_capacity`, which needs the
  Unix `resource` module, so the trainer runs on Linux (Docker or WSL), not on
  native Windows.
