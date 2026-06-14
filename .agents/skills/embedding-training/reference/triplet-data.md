# Triplet & Retrieval Data Reference

How to author the two kinds of embedding data — training triplets/pairs and
retrieval eval fixtures — so they match the landed loaders. Companion to
`SKILL.md`.

---

## Training Data: Triplets / Pairs

Training data is JSONL, one record per line, read by
`Trainers/embedding/src/data_loader.py`. A record is a `{query, positive,
negatives}` triplet (or a `{query, positive}` pair).

### Field aliases

The loader is forgiving about field names so you can reuse data from other tools:

| Canonical | Accepted aliases |
|-----------|------------------|
| anchor/query | `query` · `anchor` · `question` |
| positive | `positive` · `pos` |
| negatives | `negatives` (list) · `negative` / `neg` (scalar or list) |

A record missing a query or a positive raises `ValueError` (the offending record
is named).

### Triplet example (hard negatives)

```jsonl
{"query": "How do I reset my password?", "positive": "Open Settings → Security → Reset Password and follow the emailed link.", "negatives": ["Our refund policy allows returns within 30 days.", "Dark mode is under Appearance in Settings."]}
```

A `negatives` **list explodes** into one `(anchor, positive, negative)` row per
negative — each row reuses the same anchor/positive. This is the standard ST
hard-negative shape and is why a 20-record file with ~2 negatives each
materializes ~40+ training rows.

### Pair example (no negatives)

```jsonl
{"query": "What payment methods are accepted?", "positive": "We accept Visa, Mastercard, PayPal, and Apple Pay."}
```

Pairs train with in-batch negatives under MNRL.

> **Do not mix pair and triplet records in one file.** If ANY record has
> negatives, the loader switches to triplet shape and DROPS the pair rows (it
> won't feed an empty negative to the loss). Keep a file all-triplet or all-pair.

### Prompt prefixes

The loader prepends the registry spec's `query_prompt` to anchors and
`passage_prompt` to positives/negatives. E5-style models **require** these
prefixes (`prompt_required: true`), so always reference a model by its
`registry_name` (which carries the prompts) rather than a bare HF id.

---

## Retrieval Eval Data: corpus / queries / qrels

Three JSONL files read by the `retrieval` verifier
(`shared/verifiers/builtins/retrieval_verifier.py`). See `retrieval-eval.md` for
how they are scored.

**corpus.jsonl** and **queries.jsonl** — `{"id", "text"}`:

```jsonl
{"id": "doc_password_reset", "text": "To reset your password, open Settings → Security → Reset Password."}
```

**qrels.jsonl** — `{"query_id", "doc_id", "relevance"?}` (defaults to `1`):

```jsonl
{"query_id": "q_password_reset", "doc_id": "doc_password_reset", "relevance": 1}
```

A missing `id`/`text` (corpus/queries) or `query_id`/`doc_id` (qrels) raises a
`ValueError` naming the file and line.

---

## Canonical Committed Fixtures

The checked-in smoke fixtures live at `Datasets/embedding/examples/`:

| File | Shape | Purpose |
|------|-------|---------|
| `triplets_smoke.jsonl` | `{query, positive, negatives}` × 20 | training-side smoke (explodes to ~41 rows) |
| `corpus.jsonl` | `{id, text}` × 20 | retrieval corpus |
| `queries.jsonl` | `{id, text}` × 20 | retrieval queries |
| `qrels.jsonl` | `{query_id, doc_id, relevance}` × 20 | one relevant doc per query |

These are deliberately tiny, CPU-runnable, and **topically diverse** — each
query has a clearly distinct relevant passage — so an UNTRAINED bge-base-en
already clears the smoke thresholds (ndcg@10 ≥ 0.30, recall@10 ≥ 0.40). That is
the point of a smoke fixture: prove the wiring and that the data is retrievable
(not adversarial), not to stress-test ranking. The retrieval positives mirror the
triplet positives, so the same content exercises both the training and eval
paths.

> For a REAL evaluation, replace these with a larger, harder labeled set (more
> corpus distractors, graded relevance, multiple relevant docs per query). The
> threshold floors in the scenario YAML are calibrated for the smoke set; raise
> them for a real corpus.

---

## Validating Your Data

The cheapest validation is to run the smoke recipe `--dry-run` (the loader parses
and materializes the dataset without training) and the retrieval scenario against
the untrained base (the verifier loads corpus/queries/qrels and reports metrics).
Both paths surface a malformed record as a `ValueError` naming the file and line,
so a bad fixture fails loudly before any training or embedding work.

Synthetic triplet GENERATION (scaling beyond hand-written fixtures) is a Phase-2
item — see the `synethetic-data-generation` skill for the forward reference.
