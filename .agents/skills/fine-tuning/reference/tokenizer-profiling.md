# Offline Tokenizer Profiling

Use the profiler before choosing a sequence length or completion reserve. It is a read-only, config-first capability: it reads JSONL plus an already materialized local tokenizer snapshot and never downloads a model, tokenizer, or dataset.

```bash
python .skills/fine-tuning/scripts/profile_tokenizer_lengths.py \
  create \
  --config .skills/fine-tuning/configs/qwen35_4b_token_profile.yaml \
  --output-prefix private/token-profile/qwen35_4b
```

The output prefix produces one immutable `<prefix>.token-profile/` directory containing `manifest.json`, `profile.json`, and `profile.csv`. The profile contains model/tokenizer revisions, the runtime-lock and runtime-schema hashes, the runtime's exact Transformers version, semantic config hash, tokenizer load-closure inventory and hash, input JSONL hash, aggregate component distributions, total-token distribution, and configured-budget over-limit/slack distributions. The manifest binds the profile files by byte count and SHA-256. No artifact contains dataset text, rendered chat prompts, or host-local paths.

The directory is prepared privately and committed as one native no-replace rename. Existing destinations are always collisions, including byte-identical replays; they are never overwritten, completed, or repaired in place. On Windows this uses NTFS rename semantics. Linux requires `renameat2(RENAME_NOREPLACE)`; unsupported platforms fail closed. Dotted prefixes are preserved: `run.alpha` publishes to `run.alpha.token-profile/`, never `run.token-profile/`.

## Config contract

For exact SFT prompt/completion profiling, opt in under `input`:

```yaml
mode: messages
prompt_render: prompt_completion
chat_template_kwargs:
  enable_thinking: false
```

This invokes the reviewed `shared.sft_preprocessing.materialize_sft_example`
on the actual row, including declared v2 format and split validation. The final
message must be an assistant target and a prompt must precede it. Exact mode
requires the native `messages_field: messages`; alternate field projection is
rejected so profiling cannot replace a record's native conversation. Messages must
contain exactly `role` and nonempty string `content`. Prefix messages use the
generation scaffold with the configured template arguments; the target is
encoded raw and receives the tokenizer-derived EOS exactly once, as in training.
Per-role counts remain content diagnostics, excluding scaffold and EOS. The
complete sequence is counted without fitting it to the configured budget:
over-budget rows are reported, while the hard `max_tokens_per_record` bound
rejects the run. Missing EOS, invalid records, and materializer failures are
sanitized errors, never fallback full-chat counts.

`add_generation_prompt` must remain false/absent in this mode because the shared
materializer owns the generation boundary. Template kwargs are accepted only
with this explicit render setting: at most 32 identifier keys, 4096 UTF-8 JSON
bytes, nesting depth 4, 64 list elements, and finite JSON values. Tokenization,
truncation, template replacement and other control arguments are reserved.
Model-specific settings belong in configuration. Absent render/kwargs keys
preserve legacy config normalization, evidence shape and identities.

Exact-mode `config_provenance.rendering` binds the render setting, kwargs digest, fixed
implementation name, and reviewed helper SHA-256 into the immutable profile
identity. Creation stable-reads and executes only bytes matching the profiler's
inspectable `SFT_HELPER_SHA256`; verification checks the closed provenance shape
and reviewed digest. This is explicitly reviewed host-source provenance, **not**
attestation by the packaged Modal runtime lock. Neither the helper nor profiler
is added to that frozen inventory. Source hash refreshes require source review;
they must not silently follow arbitrary host edits. Profiling remains tokenizer
only, offline, CPU-only, and never loads model weights. Template kwargs values
never appear in artifacts; only their canonical semantic digest is published.
The generic Modal lock's Transformers admission is separate from named training
image profiles. Exact rendering does not establish tokenizer runtime equivalence
with a training image using a different Transformers stack; matching authenticated
runtime admission or reviewed equivalence is needed before describing an actual
model profile as training-stack qualification.

`model.ref`, `model.model_revision`, and `model.tokenizer_revision` are required. Each revision must be a lowercase 40-hex immutable commit. `model.local_tokenizer_path` must reference a materialized snapshot. The profiler proves the requested tokenizer commit through either the snapshot directory name or `tokenizer_config.json`'s `_commit_hash`; if `name_or_path` is present it must match `model.ref`. Missing or ambiguous proof fails closed with `LOCAL_TOKENIZER_IDENTITY_AMBIGUOUS`.

The legacy `runtime` branch declares `provider: modal`, the packaged runtime lock path, and its
exact SHA-256. The profiler validates that lock against the runtime-lock schema
and requires the installed `transformers.__version__` to equal the lock's
committed Transformers version. Missing, stale, or mismatched evidence fails
closed before tokenizer loading.

Alternatively, select a named training runtime profile with this closed shape:

```yaml
runtime:
  kind: named_profile
  name: reviewed-profile-name
  profiles_dir: Trainers/runtime_profiles
  expected_profile_sha256: <bare lowercase 64-hex digest>
  expected_inventory_sha256: <bare lowercase 64-hex digest>
```

The profiler reuses `tuner.runtime_profiles.load_runtime_profile` and its SFT
model/revision compatibility resolver. Both caller-pinned hashes must match the
authenticated profile and complete distribution inventory; API `sha256:` values
are compared explicitly with the config's bare digests. The inventory is then
stable-read and its exact bytes must match the already authenticated digest
before deriving the tokenizers version. Installed CPython patch version,
Transformers version and tokenizers version must match the inventory before
loading any tokenizer. This host-validator reuse establishes tokenizer-only
version correspondence. It does not attest execution inside the image, loaded
module origins, the complete local ML stack, or Modal runtime-lock membership.

Named evidence binds profile name, both hashes, immutable image, admitted model,
revision and SFT method, Python and tokenizer package versions, and that narrow
claim. `config_provenance.runtime_binding_sha256` hashes the normalized runtime
pins/name and admitted model without local paths. Verification recomputes this
binding, cross-checks the admitted model against top-level model evidence, and
checks the closed evidence shape. The complete runtime evidence also participates
in the semantic config and profile identity. `profiles_dir` is excluded from
published artifacts and semantic identity; private values are never copied from
inventory runtime facts. The legacy runtime branch's identities remain unchanged.

`input.jsonl_path` is required. Select exactly one input mode:

- `text`: `text_field` (default `text`) is a string per row.
- `components`: `components` maps output component names to string fields; total is their sum.
- `messages`: `messages_field` (default `messages`) is an array of `{role, content}` objects. Roles are restricted to the fixed buckets `system`, `developer`, `user`, `assistant`, and `tool`; every bucket appears in evidence for every run, with zero counts for rows where a role is absent. Arbitrary source roles fail closed and never become output keys. With `use_chat_template: true` (the default), total tokens are calculated with the loaded tokenizer's actual `apply_chat_template`; per-role counts are content-only diagnostics and are not expected to sum to the rendered total. Set `add_generation_prompt` only when it mirrors the training format.

Configuration validation is strict and pure: unknown keys and invalid cross-mode fields are rejected, booleans are never accepted as integer budgets, and defaults are applied to a new normalized value without mutating caller data. `semantic_config_sha256` hashes that exact normalized semantic representation. It includes model/tokenizer revisions, input-shape semantics, budgets, and the validated runtime commitment, but excludes the host-local tokenizer and JSONL paths. Moving identical inputs and snapshots therefore does not change configuration identity.

`budgets.max_sequence_tokens` is optional; `completion_reserve_tokens` defaults to zero. When a sequence limit is set, a row is over limit exactly when `total + completion_reserve > max_sequence_tokens`. The result records `evaluated_record_count`, `over_limit_count`, and deterministic integer `over_limit_fraction_ppm`, alongside over-limit magnitude and slack distributions. With no maximum budget, the count, fraction, magnitude, and slack fields are `null` (and the CSV omits budget series). All distributions expose `n`, `min`, `p50`, `p90`, `p95`, `p99`, and `max`; percentile selection is deterministic nearest-rank.

Stable machine-readable errors are written as a small JSON object on stdout with exit code 2. Common codes include `INVALID_IMMUTABLE_REVISION`, `INVALID_TRANSFORMERS_VERSION`, `TRANSFORMERS_VERSION_UNAVAILABLE`, `TRANSFORMERS_VERSION_MISMATCH`, `LOCAL_TOKENIZER_SNAPSHOT_MISSING`, `LOCAL_TOKENIZER_IDENTITY_AMBIGUOUS`, `LOCAL_TOKENIZER_LOAD_FAILED`, `CHAT_TEMPLATE_UNAVAILABLE`, `INVALID_JSONL_ROW`, `OUTPUT_COLLISION`, and `ATOMIC_NOREPLACE_UNAVAILABLE`.

The supplied Qwen example pins model and tokenizer to commit
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, selects the actual `qwen35-sft-v1`
profile and inventory hashes, and mirrors the existing 32K training recipe's
native messages, prompt/completion render, `enable_thinking: false`, 32768 sequence
budget and zero extra completion reserve. Its authenticated inventory requires
CPython 3.12.3, Transformers 5.17.0 and tokenizers 0.23.2. Local tokenizer and JSONL
paths remain deliberate placeholders: profiling does not fetch missing snapshots.
The example's version requirements do not authorize dependency installation or
qualify GPU capacity or model writing quality. Chat-template profiling also
requires Jinja2 and MarkupSafe from the selected profile's authenticated inventory
(the actual Qwen profile pins Jinja2 3.1.6 and MarkupSafe 3.0.3). Installing core
Transformers dependencies does not necessarily install Jinja2. Prepare these
template dependencies in the reviewed tokenizer-only CPU environment; Torch and
model weights are unnecessary. The profiler's named-runtime checks remain limited
to CPython, Transformers and tokenizers version correspondence, without claiming
complete image or local dependency attestation.
