# Dataset Formats Reference

Dataset requirements for the current CLI-first tool-calling stack.

---

## SFT Dataset Format

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

## Current Tool Wrapper

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
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py Datasets/my_dataset.jsonl
```

Use the migration pipeline for corpus refreshes instead of ad hoc rewriting:

```bash
python3 tools/migrations/05_inventory_cli_schema_datasets.py
python3 tools/migrations/06_migrate_cli_schema_datasets.py
```
