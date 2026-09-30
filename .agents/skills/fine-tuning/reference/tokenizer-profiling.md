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

`model.ref`, `model.model_revision`, and `model.tokenizer_revision` are required. Each revision must be a lowercase 40-hex immutable commit. `model.local_tokenizer_path` must reference a materialized snapshot. The profiler proves the requested tokenizer commit through either the snapshot directory name or `tokenizer_config.json`'s `_commit_hash`; if `name_or_path` is present it must match `model.ref`. Missing or ambiguous proof fails closed with `LOCAL_TOKENIZER_IDENTITY_AMBIGUOUS`.

`runtime` must declare `provider: modal`, the packaged runtime lock path, and its
exact SHA-256. The profiler validates that lock against the runtime-lock schema
and requires the installed `transformers.__version__` to equal the lock's
committed Transformers version. Missing, stale, or mismatched evidence fails
closed before tokenizer loading.

`input.jsonl_path` is required. Select exactly one input mode:

- `text`: `text_field` (default `text`) is a string per row.
- `components`: `components` maps output component names to string fields; total is their sum.
- `messages`: `messages_field` (default `messages`) is an array of `{role, content}` objects. Roles are restricted to the fixed buckets `system`, `developer`, `user`, `assistant`, and `tool`; every bucket appears in evidence for every run, with zero counts for rows where a role is absent. Arbitrary source roles fail closed and never become output keys. With `use_chat_template: true` (the default), total tokens are calculated with the loaded tokenizer's actual `apply_chat_template`; per-role counts are content-only diagnostics and are not expected to sum to the rendered total. Set `add_generation_prompt` only when it mirrors the training format.

Configuration validation is strict and pure: unknown keys and invalid cross-mode fields are rejected, booleans are never accepted as integer budgets, and defaults are applied to a new normalized value without mutating caller data. `semantic_config_sha256` hashes that exact normalized semantic representation. It includes model/tokenizer revisions, input-shape semantics, budgets, and the validated runtime commitment, but excludes the host-local tokenizer and JSONL paths. Moving identical inputs and snapshots therefore does not change configuration identity.

`budgets.max_sequence_tokens` is optional; `completion_reserve_tokens` defaults to zero. When a sequence limit is set, a row is over limit exactly when `total + completion_reserve > max_sequence_tokens`. The result records `evaluated_record_count`, `over_limit_count`, and deterministic integer `over_limit_fraction_ppm`, alongside over-limit magnitude and slack distributions. With no maximum budget, the count, fraction, magnitude, and slack fields are `null` (and the CSV omits budget series). All distributions expose `n`, `min`, `p50`, `p90`, `p95`, `p99`, and `max`; percentile selection is deterministic nearest-rank.

Stable machine-readable errors are written as a small JSON object on stdout with exit code 2. Common codes include `INVALID_IMMUTABLE_REVISION`, `INVALID_TRANSFORMERS_VERSION`, `TRANSFORMERS_VERSION_UNAVAILABLE`, `TRANSFORMERS_VERSION_MISMATCH`, `LOCAL_TOKENIZER_SNAPSHOT_MISSING`, `LOCAL_TOKENIZER_IDENTITY_AMBIGUOUS`, `LOCAL_TOKENIZER_LOAD_FAILED`, `CHAT_TEMPLATE_UNAVAILABLE`, `INVALID_JSONL_ROW`, `OUTPUT_COLLISION`, and `ATOMIC_NOREPLACE_UNAVAILABLE`.

The supplied Qwen example is pinned to the official resolved commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` for both model and tokenizer. Its paths are deliberate placeholders: profiling does not fetch missing snapshots.
