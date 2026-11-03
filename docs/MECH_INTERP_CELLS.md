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
| `mechinterp weights-inspect` | (experimental) read-only report on a LOCAL safetensors checkpoint, with bounded numeric projections |
| `mechinterp weights-patch` | (experimental) validate or create-only export a declarative base-weight patch |

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
`--i-know-this-runs-on-gpu`. `probe-fit`, `score-gates`, `weights-inspect` and
`weights-patch` are CPU-only (`weights-*` need torch only to validate/apply a patch
or to compute projections, never a GPU).

`mechinterp run` stage commands are compiled as `python -m tuner mechinterp ...`
(the installed-wheel form; there is no `tuner.py` in a wheel). Stages run in the
project workdir, and a source checkout's engine root is put on `PYTHONPATH` so the
module resolves; a host project must not contain its own top-level `tuner/`
package. `dose-calibrate` honors `--model-revision` (stage or pipeline
`model_revision`) exactly like `steer` and `extract`.

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

## Experimental: weight patches, module-site capture, weight inspection

These are explicitly experimental MechInterp CLI capabilities, not part of the
stable public API. They are generic: every tensor name, shape, value, index and
restriction comes from a spec/recipe, never from engine code.

### `mechinterp-weight-patch/v1`

```yaml
schema_version: mechinterp-weight-patch/v1
base:
  model_id: org/name                 # informational
  revision: <40-hex commit id>       # required; the run's --model-revision must equal it
  checkpoint: path/to/local_dir_or_file.safetensors   # needed by weights-patch only
  index_sha256: <sha256>             # of model.safetensors.index.json (or the single file)
  files_sha256: {shard.safetensors: <sha256>}         # optional, whole-file pins
  allow_symlinked_files: false       # true for HF cache snapshot dirs (links to regular files only)
  runtime_prefix_rules: []           # [{checkpoint_prefix, runtime_prefix}] if loaded names differ
  tensors:                           # EVERY targeted tensor, pinned to exact stored bytes
    <name>: {dtype: bfloat16, shape: [out, in], sha256: <sha256 of raw dtype bytes>}
budget:                              # all required except the last two
  allowed_tensors: ["<fnmatch pattern>"]
  max_modified_tensors: 4
  max_modified_entries: 100000       # entries whose STORED value actually changes
  max_relative_frobenius: 0.05       # per tensor, ||W'-W||_F / ||W||_F on the realized delta
  max_payload_bytes: 268435456
  max_index_count: 1000000
payload: {path: patch.safetensors, sha256: <sha256>}   # all numeric arrays live here
operations:
  - {id: a, tensor: <name>, op: add_rank_one, scale: 1.0, cols: [3, 9],
     arrays: {u: a.u, v: a.v}}                # W[R][:, C] += scale * outer(u, v)
  - {id: b, tensor: <name>, op: project, axis: output, normalize: false,
     arrays: {direction: b.d}}                # W[:, C] -= scale * d (d^T W[:, C])  (axis: input -> rows)
  - {id: c, tensor: <name>, op: replace_rows, rows: [2, 5], arrays: {values: c.v}}
  - {id: d, tensor: <name>, op: replace_cols, arrays: {values: d.v, cols: d.idx}}
export: {output_dir: path/new_checkpoint_dir, sidecar_files: ["*.json"]}   # weights-patch --export
```

Rules enforced before anything is applied or written: schema (unknown fields,
NaN/Inf, bool-as-number, duplicate YAML keys all refuse); every operation tensor
pinned and matched by `allowed_tensors`; indices distinct and in range (inline, or
integer payload arrays via `arrays.rows`/`arrays.cols`, never both); payload sha256
and exact array inventory (extra arrays refuse); finite payloads; base tensor
dtype/shape/byte hash; finite budgets on the realized result. `project` needs a
unit direction unless `normalize: true`. A zero-valued (or sub-ulp) patch changes no
byte: only entries whose value differs numerically are rewritten, so it is a true
no-op (`noop: true` in the receipt) and unchanged entries (even `-0.0`) keep their
exact bytes. bfloat16 is hashed and written as bf16 bytes; arithmetic is float32
(float64 for float64 tensors) then cast.

**Source binding.** Checkpoint files must be local regular files (no symlinks
unless `allow_symlinked_files`), index shard names plain, every shard header
strictly valid, and the index and headers must agree exactly. A bare revision
string cannot be authenticated from file bytes, so binding has two legs: exact
content pins (`index_sha256`/`files_sha256`, plus the per-tensor byte hashes),
and local revision evidence when present (an HF `snapshots/<sha>` directory name or
`.cache/huggingface/download/*.metadata`). Contradicting evidence refuses; with no
evidence the content pins are required. `model.config` names are never used.

**CLI**

```bash
python tuner.py mechinterp weights-inspect --weights-config inspect.yaml [--json]
python tuner.py mechinterp weights-patch   --weights-config patch.yaml --validate [--json]
python tuner.py mechinterp weights-patch   --weights-config patch.yaml --export   [--json]
```

Exactly one of `--validate` / `--export`. `--validate` writes nothing. `--export`
stages a sibling directory, copies every shard and overwrites only the modified
tensors' byte ranges (headers, metadata and untouched tensors stay byte-identical,
verified by re-hashing), copies the index and `sidecar_files`, writes
`mechinterp-weight-export.manifest.json` (spec/payload/base-file digests, per-tensor
before/after hashes, realized deltas, output file hashes, `patch_digest`), then
renames into place. An existing destination is refused, never overwritten.
Refusals exit 1 with `WEIGHTS_<CODE>` (`SCHEMA PATH INDEX BINDING BASE_HASH_MISMATCH
PAYLOAD NONFINITE BUDGET OUTPUT_EXISTS APPLY`).

### `weight_patch` in `steer` / `dose-calibrate` / `extract`

Steer and dose generation records retain `generated_token_ids` (through the first
EOS, inclusive). Use these for exact token parity; decoded-text equality alone
does not establish token equality. Pool metadata cannot override this field.

```yaml
surface:                       # extract: top-level `weight_patch:`
  weight_patch: {path: patches/edit.yaml, sha256: <sha256 of the spec file>}
```

Applied in memory to the freshly loaded BASE model (before any adapter, generation
or capture); no checkpoint is exported per candidate. Before the model loads the
spec bytes and payload are verified and `--model-revision` must equal
`base.revision`. After load, the targeted tensors of the loaded module are checked
against the pinned byte hashes (a wrong base refuses whatever its config says),
a resolved `config._commit_hash` must agree, and the patch is committed
all-or-nothing (rolled back and re-verified on any failure). Internal implementation
(not a public API): `MechInterp.weights.runtime.prepare_patch`, `apply_patch(...)`
(returns an `AppliedPatch`; `.revert()` or `with patched(model, prepared)`).

Steer/dose lineage: an absent patch leaves behavior and `compute_config_sha` unchanged. With a
patch, the pinned sha is part of the config sha; per-row records carry
`weight_patch_digest`; the smoke state records it (a smoke taken without or under
another patch is not reusable); manifests carry the receipt; and
`<output>.weight_patch.json` binds the output rows to the patch (the digest covers
spec, payload, revision and exact before/after tensor bytes). A resume over rows from
another or no patch is refused. Exit codes: `6` stale lineage, `7` patch refused
(before any row is written).

Limits: a Hub-id load is bound by the revision it was loaded at plus the targeted
tensor hashes (only targeted tensors are verified, not the whole model); the
default loader is bfloat16 so patches over non-bf16 checkpoints refuse by hash;
quantized params are refused; an adapter is attached after the patch.

### `capture_sites`, `use_cache`, `forward_only` (extract)

```yaml
capture_sites:
  - {layer: 20, module: mlp.down_proj, tensor: input}    # or output
use_cache: false      # teacher-forced capture forward; true for hybrid/KV-sharing models
forward_only: false   # true: no generation; families must be [anchor]; content_end_fn unused
resume: true          # append/fsync per completed row; verify identity and files before skipping
```

`layer` is the decoder BLOCK index from `get_decoder_layer` (the `law.layer`
convention), not the `hidden_states` index used by residual `layers` (index k there
is block k-1's output). `module` is a dotted path inside the block; `input` is the
first positional argument. Only the selected family positions are indexed inside the
hook (no sequence cache); each is written float32 to
`<row>__<family>.sites.safetensors` (a separate file, so probe-fit never sees it)
as `L<layer>.<module>.<tensor>` with file metadata, and per-row/manifest metadata
records site, layer, convention, shape, source dtype and positions. Hooks exist only
for the capture forward and are removed on any error. Residual tensor values are
unchanged when these options are absent (`use_cache` defaults to false). For an
anchor-only read use `forward_only: true` (or `max_new_tokens: 1` with a
`content_end_fn`).

Extraction output directories must be empty unless `resume: true` opens their
matching `capture.runlog.jsonl`. The resume fingerprint binds rendered prompts,
row keys, capture settings, the patch receipt, file-backed render/content-end
callables, and model sources (pinned Hub revision or complete local file hashes).
Hub adapters cannot resume without a local, hashable snapshot. Every row's tensor
files are fsynced and hashed before its journal append is fsynced. Resume verifies
all committed files and skips those rows; interrupted uncommitted rows are retried.
Use one writer per directory. `manifest.json` is published atomically on completion.
Probe fitting reads only that manifest's captured rows and verifies their hashes;
it refuses incomplete captures and ignores stray files. New manifests are
`mechinterp-capture/v2`; unpatched historical manifests can still be read.

Simple portable row keys keep their filenames. Other keys use `row-<sha256>`.
Missing keys, duplicates, case-insensitive filename collisions, and encoded-name
collisions are refused before generation. Per-row `capture_files` and
`site_captures` map original keys to the actual filenames and hashes.

### `weights-inspect` projections (bounded numeric reads)

```yaml
schema_version: mechinterp-weights-inspect/v1
checkpoint: path/to/local_checkpoint
tensors: ["<pattern>"]              # list matching tensors (capped by max_listed_tensors)
hash_tensors: true                  # adds sha256 + a paste-ready base.tensors snippet
max_output_values: 16384            # total numbers all projections may emit (<= 1_000_000)
max_tensor_bytes: 2147483648        # largest tensor a projection may read
projections:
  - id: col_dots
    tensor: <2-D float tensor>
    axis: cols                      # cols -> W[:, j]; rows -> W[i, :]
    indices: [12, 40]               # omit for every line (still capped)
    other_indices: [0, 1, 2]        # optional: restrict positions along each line
    kind: dot                       # dot | cosine | slice
    include_norms: true             # dot/cosine: also |line|
    vector:                         # exactly one source
      safetensors: {path: v.safetensors, key: axis, sha256: <sha256>}
      # direction_json: {path: direction.json, sha256: <sha256>}   # mechinterp-direction/v1
      # inline: [0.1, 0.2]                                         # <= 65536 numbers
```

Reads one tensor at a time, converts only the selected lines to float64, and
refuses (`BUDGET`) before reading when the emitted value count would exceed
`max_output_values`. `dot`/`cosine` emit one number per selected line (`cosine` is
`null` for a zero line); `slice` emits raw `[lines, positions]` values. Each result
carries the tensor sha256, shape/dtype and vector provenance. No model is loaded.

Readback measures before and after projections in the same float32 direction
with float64 accumulation, including when the model writes bf16 tensors. Using
a bf16-rounded direction for only the after projection fabricates off-target
movement on unchanged rows. The inactive-row no-op regression requires exact
zero movement; the smoke tolerance and intervention arithmetic are unchanged.
