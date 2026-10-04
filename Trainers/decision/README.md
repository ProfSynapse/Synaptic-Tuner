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
