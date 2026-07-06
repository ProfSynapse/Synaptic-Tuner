# Mechanistic-Interpretation Cells

The `MechInterp` package gives any researcher a project-agnostic toolkit for
reading and writing a model's internal activations during generation, driven by
declarative recipe YAML through the tuner CLI. Nothing in it is tied to a
particular research question, dataset, model, or grading rule: the vocabulary is
neutral (direction, readout, selection score, intervention, cell) and every
project-specific decision is a plug-in point named in a recipe.

A **cell** is one intervention or readout experiment described entirely by a
recipe. There are four verbs:

| Verb | What it does |
|------|--------------|
| `mechinterp extract` | generate over rows and capture hidden states to safetensors + a manifest |
| `mechinterp probe-fit` | fit a linear readout from extracted activations and freeze a direction JSON |
| `mechinterp steer` | run the six-block declarative intervention cell (smoke-gated) |
| `mechinterp score-gates` | evaluate a declarative `gates.yaml` against a per-row output JSONL |

Example recipes ship under `MechInterp/configs/templates/`. List them with
`python tuner.py mechinterp list-configs`.

## The six-block cell model

A `steer` cell recipe has six blocks. The first five are in the cell YAML; gates
are scored separately so a run and its adjudication stay independent.

1. **surface**: where rows come from (`rows_path`, a JSONL), the generation
   contract (`max_new_tokens`, greedy vs sampled, `seed`), and an optional
   `expected_config_sha` that pins reproducibility: if set, the run aborts unless
   the recipe still hashes to it.
2. **readouts**: the frozen direction files the cell reads or writes along. Each
   is a self-describing JSON produced by `probe-fit` (layer, vector, mean offset,
   `sigma` scale, class-projection calibration).
3. **law**: the intervention law and its shared parameters:
   - `additive` push: `h' = h + strength * d` shifts the residual stream along a
     unit direction.
   - `erase_write` setpoint: `h' = h - (h . d) d + (strength * sigma) d` removes
     the current projection onto `d` and writes a commanded coordinate, so the
     post-write projection equals `strength * sigma` exactly while the orthogonal
     complement is untouched.
   - `position` selects which token columns are edited: `anchor` (the last prompt
     token), `anchor_onward` (from a column to the end), `final` (each row's true
     last non-pad token, correct under left and right padding), or `answer_window`
     (only the generated tokens, from `window_start` onward). `answer_window` is a
     genuinely narrowed window rather than an alias of `anchor_onward`: the caller
     sets `window_start` to the first generated-token index so the prompt is
     excluded, and the engine refuses an `answer_window` with no `window_start`
     instead of silently steering the whole sequence. To also exclude a leading
     thinking span, advance `window_start` past it; the engine cannot find a
     thinking boundary itself because that marker is tokenizer-specific.
   - `generation_mode` selects how the edit propagates during `generate()`:
     `anchor` (edit only the prefill anchor; the KV cache carries it forward) or
     `gen_stream` (edit every decode step).
4. **arms**: named strength overrides that select which rows are active:
   - fixed `strength` (a baseline uses `0.0`);
   - `score_field` + `threshold` (activate rows whose selection score passes);
   - `flag_field` (activate rows whose named boolean is true);
   - `permuted_control_of` + `control_seed` (a seeded, count-matched random draw
     that probes the same dose on a different population).
5. **execution**: lane-agnostic run controls: `output_path` for the per-row
   JSONL, `resume` (skip rows already present), and an optional `grader`.
6. **smoke**: readback tolerances. Before the full arms run, a small smoke pass
   applies the intervention to `n_rows` and reads back the realized projection.
   `steer` refuses the full arms until a smoke passes for this exact config sha;
   `--force-full-run` overrides.

## Plug-in points (kept out of the toolkit)

The tuner ships no notion of what a prompt looks like, what "correct" means, or
which rows matter. Each is a callable or a file named in the recipe:

- **render function** (`--render-fn module.path:callable`): maps a row dict to a
  prompt string. You apply any chat template here.
- **content-end resolver** (`content_end_fn` in an extract recipe): maps
  `(full_ids, prompt_len, tokenizer)` to the index of the last content token.
- **grader** (`grader: module.path:callable` in a steer recipe): maps a per-row
  output dict to a grade dict, merged back into the row so gates can read it.
  `MechInterp.grading.interface:example_grader` is a trivial default (positive
  when the generated text is non-empty); replace it with your own.
- **row pool** (`rows_path`): any JSONL your project produces, one object per row
  with a `row_key` (or `id`/`key`).

## Typical workflow

```bash
# 1. Extract hidden states over a labeled row pool.
python tuner.py mechinterp extract \
  --mi-config my_project/extract.yaml \
  --model <base-model> \
  --render-fn my_project.prompts:render \
  --i-know-this-runs-on-gpu

# 2. Fit a linear readout and freeze a direction (CPU; sweeps layers by OOF AUROC).
python tuner.py mechinterp probe-fit --mi-config my_project/probe_fit.yaml

# 3. Run the intervention cell (smoke first, then the full arms).
python tuner.py mechinterp steer \
  --mi-config my_project/steer_cell.yaml \
  --model <base-model> \
  --render-fn my_project.prompts:render \
  --i-know-this-runs-on-gpu

# 4. Adjudicate with declarative gates.
python tuner.py mechinterp score-gates \
  --gates-config my_project/gates.yaml \
  --rows-path MechInterp/runs/example/rows.jsonl
```

`extract` and `steer` load a model and use a GPU, so they refuse to run without
`--i-know-this-runs-on-gpu`. `probe-fit` and `score-gates` are CPU-only.

## Gates

A `gates.yaml` declares named gates over the per-row output, grouped by an `arm`
field. Each gate names a primitive and the row fields it reads; `overall_pass` is
true only if every gate passes. The primitives are all seeded, so a verdict is
reproducible from its recorded seed:

- `count_flips`: rows whose outcome moved from one boolean state to another
  (for example, a monitored predicate that held before the intervention and no
  longer holds after it).
- `kill_diff_vs_control`: the positive-count difference between a primary arm
  and a count-matched control, with a seeded row-bootstrap confidence interval.
- `permutation_p`: a one-sided permutation p-value for a count-matched positive
  count against a pool (add-one smoothed, never exactly zero).
- `auroc_floor`: a tie-safe AUROC point estimate with a Hanley-McNeil analytic
  standard error and a seeded bootstrap lower confidence bound.

See `MechInterp/configs/templates/gates.yaml` for a worked example.

## Direction JSON schema

`probe-fit` writes a `mechinterp-direction/v1` record: `layer`, `hidden_dim`,
`vector` (unit-norm when `normalized`), `mu` (mean offset), `sigma` (the setpoint
scale, the readout score standard deviation), a `calibration` block with the
class-conditioned projection statistics, the fit `recipe`, and free-form
`provenance`. This is the object `steer` loads to intervene and that any external
reader can consume.

## Design notes

- The intervention math is unit-tested on stub tensors, so the hook semantics are
  verifiable without downloading a model. The `final` position policy resolves
  each row's last non-pad token from the attention mask and handles left and
  right padding identically. The hidden tensor is cloned before any in-place
  write so no autograd view alias is mutated.
- Extraction runs a clean forward pass over prompt + completion (KV cache off)
  and slices hidden states at the requested position families across the
  requested layer range, one safetensors file per row and family.
- Probe fitting reduces each activation matrix with a randomized PCA fit
  label-agnostically, then fits a saga logistic classifier; out-of-fold scoring
  fits PCA and classifier on train folds only, so the reported AUROC never sees
  test-fold information. The full-data direction folds the classifier weight back
  through the PCA basis into raw activation space.

## Launch checklist

Two operational gotchas have each killed a paid remote run more than once. Check
both before launching a cell, especially on a fresh or retried remote container.

1. **Negative-leading values need the equals form.** A grid value that can start
   with a minus sign (a dose ladder like `-2,-1,0,1,2`, or a single negative gain)
   passed as its own argv entry is read by argparse as a flag, and the run dies
   with `expected one argument`. Two defenses, in order of preference:
   - Put dose ladders and negative gains in the recipe, not on the command line.
     Arm strengths (including negatives) live in the `arms` block of the steer
     recipe, so a normal launch never puts a bare negative on argv. This is the
     structural fix and the reason the verbs take a recipe rather than a grid flag.
   - If you must pass a negative on the command line to any tool, use the equals
     form: `--flag=-2,-1,0,1,2`, never `--flag -2,-1,0,1,2`.
2. **Clone and download steps must be idempotent.** A retried remote function
   lands on a warm container where the workspace already exists, so an
   unconditional `git clone` fails. Guard every clone or download with an
   existence check first, then `fetch` and `checkout` the pinned commit. The same
   applies to model and dataset pulls: check for the artifact before fetching.
   Resume-from-output in the steer cell already follows this pattern for its own
   per-row output; extend it to any wrapper that clones the repo or pulls weights.

The steer cell's own smoke-first gate and `expected_config_sha` guard cover a
third failure mode (running an edited or unverified cell), so a full launch is
gated on a recorded smoke pass for the exact config.
