# Batch inference: `batch-generate` and `batch-capture`

Two generic, engine-selectable batch verbs with **crash-safe incremental
persistence and resume**. They take prompts / sequences in and emit completions
/ per-layer hidden states out — nothing project-specific (no pools, grading, or
outcome taxonomies). A consuming project renders prompts, then calls these
verbs, then grades / scores the outputs itself.

The whole point is durability under preemption: a killed or preempted job never
loses completed work and resumes exactly where it left off.

---

## `batch-generate` — prompts in, completions out

```bash
python tuner.py batch-generate \
  --prompts prompts.jsonl \
  --model <hf-id-or-path> \
  --out-dir runs/gen \
  --engine hf-batched \        # or: vllm
  --batch-size 32 \            # auto-halves on CUDA OOM, down to 1
  --max-new-tokens 48 \
  --seed 0                     # greedy by default
# sampled decode:
#   --do-sample --temperature 0.7 --top-p 0.9
# stop strings (repeatable):
#   --stop-string $'\n\n' --stop-string END
```

**Input** (`--prompts`): JSONL rows `{"id": str, "prompt": str}`. Any extra
fields are passed through untouched to the output rows.

**Output** (`<out-dir>/completions.jsonl`): one row per prompt, appended
incrementally per batch:

```json
{"id": "...", "completion_text": "...", "completion_token_ids": [...],
 "prompt_token_len": 12, "finish_reason": "eos|length|stop", "...passthrough": ...}
```

hf-batched engine specifics: length-sorted micro-batching, **left padding** for
generation with correct attention masks, `use_cache=True`, bf16 on supported
GPUs / fp32 on CPU. The pad token falls back to EOS (an existing special token)
without adding a new special id.

---

## `batch-capture` — sequences in, hidden states out

```bash
python tuner.py batch-capture \
  --rows rows.jsonl \
  --model <hf-id-or-path> \
  --out-dir runs/cap \
  --engine hf-batched \        # vllm capture: see "Engine selection" below
  --layers all \               # or a comma list: --layers 20,22
  --persist-dtype float32 \    # or bfloat16
  --batch-size 16
```

**Input** (`--rows`): JSONL rows `{"id": str, "text": str}` OR
`{"id": str, "token_ids": [int]}`, each with a `"positions"` object mapping a
name to an absolute token index into the tokenized sequence, or the literal
`"last"` (final real token). Extra fields pass through.

```json
{"id": "s1", "text": "...", "positions": {"end_of_prompt": 41, "last": "last"}}
```

**Output**:

- Per-row safetensors at `<out-dir>/tensors/<sanitized-id>.safetensors`, keyed
  `"<position>__L<layer>"` (one 1-D vector per position × layer).
- An appended index `<out-dir>/capture.jsonl`:
  `{"id", "file", "n_layers", "hidden_dim", "positions", "...passthrough"}`.

hf-batched engine specifics: batched forward, **right padding** + attention
mask, `output_hidden_states=True`. `layers` indexes the hidden-states tuple
(index 0 = embeddings, 1..N = block outputs), so `all` yields `num_layers + 1`
vectors per position.

---

## Persistence & resume contract (both verbs)

- **Flush after every batch.** JSONL rows are appended as complete
  `line + "\n"` writes, then `flush()` + `fsync()`; tensor files are written to
  a temp name and atomically `os.replace`-d into place. A crash therefore leaves
  every row either fully persisted or absent — never a torn JSON line or a
  half-written tensor.
- **`checkpoint.json`** in the out-dir tracks the set of done ids plus a
  `config_hash` of the invocation (model, engine, decode params, seed, layers,
  persist dtype). It is reconciled against the actual JSONL index at load time,
  so a row that landed in the index but not the checkpoint (crash between the
  two writes) still counts as done.
- **`--resume`** re-invokes against the same `--out-dir`: it loads the
  checkpoint, **verifies the config hash matches** (refuses with a clear error
  across a different config — a changed model/seed/decode param), skips all done
  ids **by id**, and continues. A completed-then-resumed run produces the
  identical artifact set as an uninterrupted run.
- Without `--resume`, pointing at an out-dir that already holds a run is refused
  (no silent overwrite / mixing).

## Sync hook (partial-artifact push)

`--sync-every N` + `--sync-cmd '<shell>'`: after every `N` newly persisted rows
(and once at the end) the shell command runs with two env vars:

- `TUNER_SYNC_DIR` — the out-dir to push.
- `TUNER_SYNC_REASON` — `periodic` or `final`.

Sync failures **warn and continue** — a network blip or expired credential
never kills a compute job that is making progress. This is the generic hook a
cloud wrapper uses to ship partial artifacts to durable storage.

## Engine selection

| engine | generate | capture |
|--------|----------|---------|
| `hf-batched` (default) | length-sorted, left-pad, batched `generate` | batched `output_hidden_states` forward |
| `vllm` | continuous batching (greedy/sampled, seed) | not wired in this generic engine yet — use `hf-batched` |

`vllm` is an optional, soft dependency: it is never imported at load time and is
not a tuner requirement. Selecting `--engine vllm` without vllm installed exits
with a clear "vllm not installed" message. vLLM gained native per-layer
hidden-state extraction in v0.18.0 (prefill-only, connector-based), but its API
is version-specific; until a vLLM is pinned and verified, `batch-capture
--engine vllm` raises a clear error pointing at `hf-batched` (whose capture
forward is already fast — generation, not capture, was the bottleneck).

---

## Worked example: a preemption-safe cloud job

Generate with periodic sync so preemption never loses data:

```bash
python tuner.py batch-generate \
  --prompts prompts.jsonl \
  --model Qwen/Qwen3-4B \
  --out-dir /scratch/gen \
  --engine hf-batched \
  --batch-size 48 \
  --max-new-tokens 48 \
  --seed 0 \
  --sync-every 200 \
  --sync-cmd 'aws s3 sync "$TUNER_SYNC_DIR" s3://my-bucket/run42/ --quiet'
```

If the job is preempted and restarted, first restore the partial artifacts into
the same out-dir (e.g. `aws s3 sync s3://my-bucket/run42/ /scratch/gen`), then
resume — done ids are skipped, only the remainder runs:

```bash
python tuner.py batch-generate \
  --prompts prompts.jsonl \
  --model Qwen/Qwen3-4B \
  --out-dir /scratch/gen \
  --engine hf-batched --batch-size 48 --max-new-tokens 48 --seed 0 \
  --sync-every 200 \
  --sync-cmd 'aws s3 sync "$TUNER_SYNC_DIR" s3://my-bucket/run42/ --quiet' \
  --resume
```

Capture pass over answered sequences (same durability + sync + resume):

```bash
python tuner.py batch-capture \
  --rows answered_rows.jsonl \
  --model Qwen/Qwen3-4B \
  --out-dir /scratch/cap \
  --layers all \
  --persist-dtype float32 \
  --batch-size 16 \
  --sync-every 200 \
  --sync-cmd 'aws s3 sync "$TUNER_SYNC_DIR" s3://my-bucket/run42-cap/ --quiet'
```

> Local lane on a slow disk (e.g. a 9P-mounted drive): point `--out-dir` at fast
> local scratch (ext4 `~/` or `/tmp`) and use `--sync-cmd` to move artifacts to
> the slow drive once, rather than writing tensors there per row.
