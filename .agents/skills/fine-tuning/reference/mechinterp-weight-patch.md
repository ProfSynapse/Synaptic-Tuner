# MechInterp weight patches, module-site capture and weight inspection (experimental)

Experimental, CLI-only MechInterp capabilities (not part of any stable public API).
Everything is config-driven; engine code names no model, tensor, direction or
dataset. The full contract and examples live in `docs/MECH_INTERP_CELLS.md`
("Experimental: weight patches, module-site capture, weight inspection"); this page
carries the working rules.

## Commands

```bash
python tuner.py mechinterp weights-inspect --weights-config inspect.yaml --json
python tuner.py mechinterp weights-patch   --weights-config patch.yaml --validate --json
python tuner.py mechinterp weights-patch   --weights-config patch.yaml --export   --json
```

Use `mechinterp run --dry-run` for planning; `steer --dry-run` does not exist.
Pipeline stages are compiled as `python -m tuner mechinterp ...`; `dose-calibrate`
forwards `--model-revision`.

## Rules

- Specs are `mechinterp-weight-patch/v1` / `mechinterp-weights-inspect/v1` YAML or
  JSON; pin every targeted tensor's dtype, shape and raw-byte sha256, the exact
  40-hex `base.revision`, and `index_sha256`/`files_sha256`. Author the pins from
  `weights-inspect` with `hash_tensors: true`.
- Budgets are required and finite (`allowed_tensors`, `max_modified_tensors`,
  `max_modified_entries`, per-tensor `max_relative_frobenius`). The patch is
  all-or-nothing; a refusal changes nothing.
- Always `--validate` before `--export`. Export is create-only into a new directory
  and verifies untouched bytes by re-hashing; it never overwrites. Large outputs
  (including to external drives) are chosen by `export.output_dir`, never by code.
- Evaluate candidates with `surface.weight_patch: {path, sha256}` (in-memory, no
  export per candidate). The run must pass `--model-revision` equal to
  `base.revision`; use a fresh output path when the patch changes (stale lineage
  exits 6; a refused patch exits 7 before any row).
- `extract` `capture_sites` use decoder BLOCK indices (`law.layer` convention), not
  `hidden_states` indices. Set `use_cache: true` for hybrid/KV-sharing models, and
  `forward_only: true` with `families: [anchor]` for anchor-only reads.
- Use `resume: true` for long extraction. `capture.runlog.jsonl` commits each row
  after its tensor files are fsynced and hashed. Resume verifies source/settings/
  patch identity and every completed file, then skips completed rows. Without
  resume the output directory must be empty. Probe fitting requires a completed
  manifest and verifies its captured files. Use the manifest's filenames: unsafe
  keys are hash-encoded for portable storage, and collisions are refused.
- Inspect projections are bounded numeric reads (dot/cosine/slice over selected
  rows or columns); keep `max_output_values` small and do the interpretation in
  project-side analysis, not in engine code.
- A bare revision string is not authenticated by file bytes: the byte pins are the
  binding. Hub-id loads verify only the targeted tensors.

## Test

```bash
# dedicated CPU-only venv (never the main Unsloth/GPU env): torch (CPU wheel),
# safetensors, pydantic, pyyaml, pytest, transformers, pandas, scikit-learn,
# jsonschema, python-dotenv, packaging
python -m pytest tests/mech_interp
```
