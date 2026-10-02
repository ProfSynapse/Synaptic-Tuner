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

### Chronological group holdouts for prepared messages

The existing v2 `prepare-dataset` command supports this optional split declaration:

```json
{"kind":"group_sequence_tail","allocations":[{"name":"train","weight":4},{"name":"validation","weight":1}]}
```

This policy has no seed. It orders target items by authenticated
`ItemLineageV2.sequence` within each configured group, using a training prefix and
validation tail. Equal-sequence cohorts stay together. Among boundaries with
both splits nonempty, choose the validation count nearest the weighted allocation;
an equally near boundary favors the larger validation tail. Groups with no such
boundary fail, including singleton groups. Counts can therefore differ from the
requested ratio; inspect the actual per-group counts before launch.

Target revision families and target derivation chains cannot cross splits.
Training contexts cannot contain held-out targets or their revision/derivative
lineage unless an exact support item is explicitly declared under the
retrospective-conditioning policy below. Validation contexts may include
earlier training targets, matching a
next-item prediction workflow. Conflicting lineage fails rather than silently
moving examples. The manifest retains bounded ID-only split lineage so artifact
verification can independently recompute assignments and leakage checks.

Publish through `python tuner.py prepare-dataset --config <config.json> --json`;
preserve the old immutable artifact. This changes split policy, not prompts,
assistant prose, document ordering, or the trainer. The existing seeded
`group_hash_rank` policy remains unchanged for whole-group holdouts. Sequence-tail
validation measures later items within known groups, not unseen-group performance.

### Independent context and target projections

The v2 preparation config optionally accepts a top-level `context_projection`
with the same closed `{ "structure_ref": ..., "name": ... }` shape as
`target_projection`. Both names resolve declared string-field projections in the
verified bundle; each context item must match its selected context structure,
while targets must match the target structure. For example, a declared source
projection may retain document metadata in inputs while a separate body
projection keeps assistant outputs prose-only. Field and projection names are
configuration, not builder conventions. Missing, null or non-string document
fields reject rather than silently falling back.

When omitted, context uses `target_projection` and legacy serialization, row
identities, dataset bytes and manifests remain unchanged. When explicitly
provided, the selected context projection and resolved digest bind row and
dataset identities even if both projections produce identical text. The public
verifier checks that recipe, resolved projection, digest and row bindings agree.
This option does not change source selection, document order, lineage or leakage
safeguards; review actual complete context packages before authorizing training.

### Intentional target-derived conditioning

For explicitly approved conditioned writing, v2 optionally accepts
`context_package.conditioning_policy: {"kind": "target_derived_support/v1"}`.
A distinct-family support document may descend directly or transitively from
**its own exact target**, in the same group. Literal target text, alternate target
revisions, unrelated newer context and newer coancestors remain prohibited.
Only the exact target ancestor is exempted from the revision check; only its
descendants receive the context chronology exemption. Training contexts still
reject held-out targets, revisions and derivatives. Keep lineage truthful: this
is intentional outline-guided writing, not unconditioned continuation or proof
of unseen generalization.

Omission preserves legacy serialization, causal checks and identities. Enabled
policy binds row and dataset identity. Hash-rank artifacts retain bounded full
declared `conditioning_lineage`; sequence-tail reuses `split_lineage`. Public
verification checks the declared graph, closure, revision/group/split relations
and policy binding, not semantic derivation, copied prose or source truth. It
does not reconstruct hash-rank ordering from a private seed's digest. Review
source provenance and complete examples separately; use the existing CLI to
publish a fresh immutable dataset.

### Declared retrospective support documents

For an explicitly approved full-document planning context, v2 also accepts a
closed, opt-in `context_package.conditioning_policy`:

```json
{"kind":"declared_retrospective_support/v1","support_item_ids":["item-<64hex>"]}
```

`support_item_ids` must be sorted, unique, nonempty, at most 256 exact item IDs,
and every ID must resolve in the verified source bundle and actually appear as
a context item. A listed support item cannot itself be a selected target or
share a revision family with any selected target. Only those declared items
receive the retrospective-context exception: their truthful ancestry may
include multiple chapter targets, including later and validation chapters.
Retain all known `derived_from` edges and assign a causal ordering rank no
earlier than every declared ancestor. Do not invent missing source IDs or label
a retrospective document independent to make preparation pass.
The policy kind and exact support-ID list bind row and dataset identities; the
public verifier checks the declared graph, ancestry closure, and allowed
exception against that binding. Verification cannot prove the source's semantic
truth or detect every copied passage.

The exception concerns the selected support item's per-row chronology and
split checks; it does not exempt literal target prose, alternate revisions,
other newer contexts, lineage cycles, source graph causality, or group/revision
integrity. Own-target outlines remain permitted under this policy without
adding each outline to `support_item_ids`. Omitting a conditioning policy keeps
the strict default; `target_derived_support/v1` remains the narrower option.
Review the actual support content and label evaluation with retrospective
planning documents as **conditioned drafting**, not blind prediction of held-out
story events. A structurally verified artifact does not establish that its
source prose is free of copied target passages.

### Configured cleanup of prepared messages and context documents

The v2 preparation config accepts these optional fields inside `context_package`
(this fragment augments the required lineage, packages, prompts and target rules):

```json
{
  "context_transforms": {
    "drop_fenced_block_info_strings": ["widget"],
    "drop_standalone_line_prefixes": ["<iframe", "  ![["]
  },
  "paragraph_gap_policy": {
    "kind": "collapse_blank_lines/v1",
    "group_ids": ["configured-group"]
  }
}
```

`context_transforms` reuses the existing `target_transforms` drop rules, but
applies them separately to each context document before joining. Target and
context selections remain independent. Prefix matching is literal at the start
of a line: configured indentation matters. Selected fenced blocks must close;
this is configured removal, not a general HTML sanitizer. Keep corpus-specific
embed signatures in config and preserve ordinary prose and links.

`paragraph_gap_policy` applies to targets and context documents whose **own**
declared lineage `group_id` is selected, not the enclosing target row's group.
Selectors must be unique declared groups (at most 256); context-only groups are
valid. Outside preserved backtick/tilde fences, whitespace-only lines become
empty and consecutive blank lines collapse to one. Fence recognition permits
up to three leading spaces and requires the matching marker and sufficient
closing length. Nonblank lines, their indentation, scene markers, intentional
single line breaks and preserved fence contents remain unchanged. Retained line
endings, including CRLF, keep their spelling. Fence state resets per document;
normalization never runs over the joined prompt or crosses document boundaries.

Explicit nulls and unknown fields reject. Context drop lists are bounded to 256
selections each. Cleanup rejects documents left with only whitespace. Omitting
both options preserves legacy config serialization, semantic identities and row
bytes; raw-text v1 is unchanged. Enabled options bind into the semantic recipe
and resulting content-addressed dataset identity.

The builder validates group membership against the complete declared item
lineage resolved to the verified source bundle. The public artifact verifier
checks policy structure, counts/digests and semantic identity; legacy manifests
do not expose complete context-only group membership, so verification alone
cannot reconstruct that builder proof.

Rebuild through `python tuner.py prepare-dataset --config <config.json> --json`
to a fresh immutable publication, preserving the source bundle and previous
dataset. Compare row/split counts, lineage and intended target/context changes;
re-profile tokenizer lengths before consuming changed rows. If the CLI reports
`publication_uncertain` / `parent_durability`, retain its exact records and use
the existing `verify_prepared_dataset_v2` public API to verify the exact retained
manifest and JSONL. Integrity verification is **not** a durability acknowledgment.
Do not rerun publication, overwrite the artifact or weaken checks to manufacture
that acknowledgment; preserve the uncertainty explicitly.

### Legacy conversational SFT

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
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py Datasets/my_dataset.jsonl
```

Use the migration pipeline for corpus refreshes instead of ad hoc rewriting:

```bash
python3 tools/migrations/05_inventory_cli_schema_datasets.py
python3 tools/migrations/06_migrate_cli_schema_datasets.py
```
