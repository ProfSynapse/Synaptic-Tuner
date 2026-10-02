# Dataset Formats Reference

Dataset requirements for the current CLI-first tool-calling stack.

---

## SFT Dataset Formats

### Verified raw text

Use this shape when a declarative dataset-prep recipe projects one verified
normalized source item directly into one training row. The public command is:

```bash
python tuner.py prepare-dataset --config <config.json> --json
```

The resulting content-addressed dataset contains a manifest plus JSONL rows with
exactly these fields:

```jsonl
{"schema_version":"syntunia-sft-row/v1","format":"raw_text","row_id":"<stable-id>","source_item_id":"<stable-source-id>","split":"train","text":"<projected text>"}
```

Key rules:

- Authority requires both `schema_version: syntunia-sft-row/v1` and
  `format: raw_text`; an arbitrary `text` column is not sufficient.
- Every row declares exactly `train` or `validation`. The trainer consumes those
  splits directly and never creates a random split for this format.
- The payload field is `text`; `raw_text` names the format.
- Raw-text rows are tokenized directly, terminated with the tokenizer-derived
  EOS token, and trained with full-sequence labels. They are not chat-rendered.
- Do not mix raw-text rows with conversations or prompt/completion rows.
- The public manifest and CLI response contain stable identities and counts, not
  corpus prose, host paths, or the raw split seed.

### Authoritative prompt/completion messages

Use this shape when a dataset builder has already chosen the exact context prompt,
target prose, grouping, and split:

```jsonl
{"schema_version":"syntunia-sft-row/v2","format":"messages","row_id":"row-<stable-id>","target_item_id":"item-<stable-id>","context_item_ids":["item-<stable-id>"],"group_id":"<group>","split":"train","messages":[{"role":"user","content":"<context bundle and request>"},{"role":"assistant","content":"<target prose>"}]}
```

Key rules:

- Authority requires the exact v2 schema and `format: messages`; legacy message
  rows retain legacy behavior.
- Every row has exactly one user turn followed by one assistant turn and a
  preassigned `train` or `validation` split.
- Train only the assistant completion. Do not include source metadata in the
  target unless it is intentionally part of the prose.
- Authoritative rows fail instead of truncating. Choose `max_seq_length` from an
  exact tokenizer profile that admits every intended row.
- Context document order remains dataset-builder policy; the trainer does not
  impose an outline-first or other corpus-specific convention.

### Conversational SFT

Positive examples only. Tool-calling examples should use OpenAI-style `tool_calls`.

```jsonl
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>"},
    {"role": "user", "content": "Archive today's note and then read it back."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "useTools",
            "arguments": "{\"workspaceId\":\"default\",\"sessionId\":\"session_123\",\"memory\":\"Need to inspect and reorganize notes.\",\"goal\":\"Move a note and then read it back.\",\"constraints\":\"Do not touch unrelated files.\",\"tool\":\"storage move \\\"notes/today.md\\\" \\\"archive/today.md\\\", content read \\\"archive/today.md\\\"\",\"strategy\":\"serial\"}"
          }
        }
      ]
    }
  ]
}
```

Key rules:
- Assistant tool-calling turns should use `content: null` with `tool_calls`
- The wrapped function name is always `useTools`
- `function.arguments` must use the CLI-first top-level wrapper fields
- The actual tool operations live in the `tool` command string

---

## KTO Dataset Format

Interleaved desirable and undesirable examples.

```jsonl
{"conversations":[{"role":"user","content":"..."},{"role":"assistant","content":"good response"}],"label":true}
{"conversations":[{"role":"user","content":"..."},{"role":"assistant","content":"bad response"}],"label":false}
```

Key rules:
- `label: true` = desirable
- `label: false` = undesirable
- Keep paired positive/negative coverage where practical

---

## GRPO Dataset Format

Prompts plus a ground-truth tool wrapper for reward scoring.

```jsonl
{
  "prompt": [
    {"role": "system", "content": "<session_context>...</session_context>"},
    {"role": "user", "content": "Move the note and read it back."}
  ],
  "ground_truth_tool": "useTools",
  "ground_truth_args_json": "{\"workspaceId\":\"default\",\"sessionId\":\"session_123\",\"memory\":\"Need to inspect and reorganize notes.\",\"goal\":\"Move a note and then read it back.\",\"constraints\":\"Do not touch unrelated files.\",\"tool\":\"storage move \\\"notes/today.md\\\" \\\"archive/today.md\\\", content read \\\"archive/today.md\\\"\",\"strategy\":\"serial\"}"
}
```

---

## Embedding Dataset Format (triplets / pairs)

The `embedding` method (SentenceTransformer bi-encoders) trains on retrieval
triplets or pairs, NOT conversations. One JSONL record per line, read by
`Trainers/embedding/src/data_loader.py`.

```jsonl
{"query": "How do I reset my password?", "positive": "Open Settings → Security → Reset Password.", "negatives": ["Our refund policy allows returns within 30 days.", "Dark mode is under Appearance."]}
{"query": "What payment methods are accepted?", "positive": "We accept Visa, Mastercard, PayPal, and Apple Pay."}
```

Key rules:
- Anchor aliases: `query` / `anchor` / `question`. Positive: `positive` / `pos`.
  Negatives: `negatives` (list) / `negative` / `neg` (scalar or list).
- A `negatives` list explodes into one `(anchor, positive, negative)` row per
  negative (standard hard-negative shape).
- Do NOT mix pair and triplet records in one file — if any record has negatives,
  pair rows are dropped.
- The registry spec's `query_prompt` / `passage_prompt` are applied
  automatically; E5-style models require them, so reference models by
  `registry_name`.

Retrieval **evaluation** data is separate (corpus / queries / qrels JSONL). Full
details + the canonical `Datasets/embedding/examples/` fixtures are in the
`embedding-training` skill (`reference/triplet-data.md`).

---

## Example Tool-Calling Recipe

The wrapper below is one declarative dataset recipe, not a parser or trainer
runtime truth. Keep wrapper names and required fields in scenario/config YAML so
other datasets can use different schemas without runtime code changes.

The canonical tool-call format is:

```json
{
  "tool_calls": [
    {
      "id": "call_0001",
      "type": "function",
      "function": {
        "name": "useTools",
        "arguments": "{\"workspaceId\":\"default\",\"sessionId\":\"session_123\",\"memory\":\"Need to inspect and reorganize notes.\",\"goal\":\"Move a note and then read it back.\",\"constraints\":\"Do not touch unrelated files.\",\"tool\":\"storage move \\\"notes/today.md\\\" \\\"archive/today.md\\\", content read \\\"archive/today.md\\\"\",\"strategy\":\"serial\"}"
      }
    }
  ],
  "content": null
}
```

The inner wrapper payload is:

```json
{
  "workspaceId": "default",
  "sessionId": "session_123",
  "memory": "Need to inspect and reorganize notes.",
  "goal": "Move a note and then read it back.",
  "constraints": "Do not touch unrelated files.",
  "tool": "storage move \"notes/today.md\" \"archive/today.md\", content read \"archive/today.md\"",
  "strategy": "serial"
  }
}
```

Required top-level fields:
- `workspaceId`
- `sessionId`
- `memory`
- `goal`
- `tool`

Optional top-level fields:
- `constraints`
- `strategy`

The `tool` value is a CLI command string. Multiple operations are comma-separated.

---

## Canonical Dataset Locations

```text
Datasets/tools_datasets/non_thinking/
├── contentManager/
├── memoryManager/
├── promptManager/
├── searchManager/
└── storageManager/
```

Current canonical versions:
- `contentManager/tools_v2.3.jsonl`
- `memoryManager/tools_v2.4.jsonl`
- `promptManager/tools_v2.6.jsonl`
- `searchManager/tools_v2.2.jsonl`
- `storageManager/tools_v2.4.jsonl`

---

## Validation

```bash
python3 -m shared.validation.dataset_validator Datasets/my_dataset.jsonl
```

Before SFT, check the loss mask the trainer will actually build for this
dataset and tokenizer. `doctor sft-mask` runs the trainer's dataset contract
check and a row sample through its own preprocessing (`materialize_sft_row`)
with only the tokenizer loaded, so it needs no GPU or model weights. It works on
legacy conversation rows and on `prepare-dataset` outputs (raw text and
authoritative messages):

```bash
# Reuse the trainer config's model, dataset, max_seq_length, chat_template_kwargs and prompt_render
python tuner.py doctor sft-mask --sft-config Trainers/sft/configs/config.yaml

# Explicit flags override the trainer config, as train_sft.py flags do
python tuner.py doctor sft-mask --dataset-path Datasets/my_dataset.jsonl \
  --model <tokenizer-id-or-path> --max-seq-length 4096 \
  --chat-template-kwargs '{"enable_thinking": false}' --preview-rows 0,12 --json
```

Hard failures (exit 1): a dataset the trainer's contract would reject,
preprocessing errors (including authoritative rows that do not fit), more rows
the trainer would drop than `training.max_dropped_row_fraction` allows (rows
with zero trained tokens, and masks that stop before the end of the prompt
render), assistant-only rows that fall back to full-sequence loss, trained spans
that do not end with the template's end-of-turn token (`eos_token` for raw
text), and doubled BOS. Warnings: individual dropped rows, truncation rate above
the configured threshold (p50/p95/max lengths are reported) and end-of-turn
tokens lost to truncation. Info: multi-turn rows whose earlier assistant turns
are untrained. Doctor defaults live in `Trainers/sft/configs/mask_doctor.yaml`;
the end-of-turn derivation and drop policy are shared with the trainer. Fix the
data, `max_seq_length`, `chat_template_kwargs` or `prompt_render` until it
passes; do not launch SFT on a failing report. Training runs log the same
dropped-row counts after dataset preparation.

**SFT label change (preprocessing contract version 2).** Assistant-only labels
now stop at the final turn's end-of-turn token (the template's trailing newline
after it is masked); `prompt_render: prompt_completion` closes completions with
the template's end-of-turn token instead of `eos_token_id` (identical when they
coincide) and honours `completion_only_loss: false`; and rows with no supervised tokens left after
truncation, or whose assistant-only mask stopped before the end of the prompt
render, are dropped with a logged count. The run fails when dropped rows exceed
`training.max_dropped_row_fraction` (default `0.01`). Losses and checkpoints
from runs before this change are not directly comparable with later runs;
`training_lineage.json` records `dataset.preprocessing.contract_version: 2`.

Use the migration pipeline for corpus refreshes instead of ad hoc rewriting:

```bash
python3 tools/migrations/05_inventory_cli_schema_datasets.py
python3 tools/migrations/06_migrate_cli_schema_datasets.py
```
