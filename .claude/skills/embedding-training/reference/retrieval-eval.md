# Retrieval Evaluation Reference

How embedding models are scored: corpus-level retrieval metrics via the
`retrieval` verifier, the metric-spec grammar, and the pass/warn/fail ladder.
Companion to `SKILL.md`; for the broader evaluator surface see the `evaluation`
skill.

---

## Corpus-Level, Not Per-Completion

Most evaluator scenarios score one completion at a time (a chat turn → a scalar).
Retrieval is different: there is **no backend `chat()` turn**. The `retrieval`
verifier embeds a whole document corpus and a query set, runs FAISS top-k
nearest-neighbour retrieval, and aggregates IR metrics against qrels. It is
invoked once per retrieval scenario as a **sibling** to the per-completion loop,
through a dedicated corpus-level entry point — not the `verify(VerifierInput)`
path.

The verifier lives at `shared/verifiers/builtins/retrieval_verifier.py` and is
registered under the type key `retrieval`. It is `shared/`-pure: it imports
nothing from `Evaluator/` or `Trainers/`; the model id/path is passed in already
resolved by the Evaluator caller.

---

## Metric-Spec Grammar

The canonical spec string is `"<metric>@<k>"`:

| Metric | Meaning |
|--------|---------|
| `recall@k` | fraction of relevant docs found in the top-k |
| `mrr@k` | mean reciprocal rank of the first relevant doc |
| `ndcg@k` | normalized discounted cumulative gain (graded, uses qrels grades) |
| `map@k` | mean average precision |

`k` is a positive int. Examples: `recall@10`, `mrr@10`, `ndcg@10`, `map@100`.
This grammar is shared by the scenario YAML, the verifier, and the lineage — one
format, no variants. The pure metric functions live in
`shared/ml/retrieval_metrics.py` (a numpy-only sibling to the
classification-oriented `shared/ml/metrics.py`).

---

## Data Shapes (qrels-based)

Three JSONL files, all matching the landed loaders:

**corpus.jsonl / queries.jsonl** — `{"id", "text"}`:

```jsonl
{"id": "doc_password_reset", "text": "To reset your password, open Settings → Security → Reset Password."}
```
```jsonl
{"id": "q_password_reset", "text": "How do I reset my account password?"}
```

**qrels.jsonl** — `{"query_id", "doc_id", "relevance"?}` (relevance defaults to
`1`; grade `> 0` = relevant; higher grades feed graded nDCG):

```jsonl
{"query_id": "q_password_reset", "doc_id": "doc_password_reset", "relevance": 1}
```

A query may have multiple qrels rows (multiple relevant docs, possibly with
different grades).

---

## Scenario YAML (thresholds live here)

Thresholds are config-driven, never in Python. A scenario test declares a
`retrieval_config` block:

```yaml
tests:
  - id: bge_base_retrieval_smoke
    tags: [embedding, retrieval, smoke]
    retrieval_config:
      corpus: Datasets/embedding/examples/corpus.jsonl
      queries: Datasets/embedding/examples/queries.jsonl
      qrels: Datasets/embedding/examples/qrels.jsonl
      metrics: [recall@10, mrr@10, ndcg@10]
      model:
        registry_name: bge-base-en      # resolved to hf_id by the Evaluator caller
      thresholds:
        min:
          ndcg@10: 0.30                  # below min → fail
          recall@10: 0.40
        warn_margin: 0.05                # within [min, min + margin) on any → warn
      primary_metric: ndcg@10            # headline metric (defaults to metrics[0])
```

The canonical committed scenario is
`Evaluator/config/scenarios/embedding_retrieval_smoke.yaml`.

### The pass/warn/fail ladder (R4)

- `passed = all(value >= min[m] for m in min)` — a gated metric below its `min`
  is a hard fail.
- `warned = passed and any(min[m] <= value < min[m] + warn_margin)` — a passing
  result that is uncomfortably close to a floor is flagged.
- Metrics without a `min` entry are reported but never gate.

`EvaluationRecord.status` returns `fail` / `warn` / `pass` from this ladder; the
retrieval branch is evaluated on its own ladder, ahead of the correctness branch.

---

## Evaluating a Trained Adapter

To score a fine-tuned embedder instead of the untrained base, swap the model
reference for a trained run/adapter path:

```yaml
      model:
        path: embedding_output/20260614_120000/final_model
```

Everything else (corpus, queries, qrels, metrics, thresholds) stays the same, so
you compare a trained model against the same labeled set and thresholds you used
for the baseline. Point `corpus`/`queries`/`qrels` at a larger labeled set for a
real evaluation; the committed `Datasets/embedding/examples/` set is a smoke
fixture (tiny + topically diverse → an untrained bge already clears the floors).

---

## Reading the Result

A run yields a `RetrievalValidationResult`:

| Field | Meaning |
|-------|---------|
| `metrics` | `{spec: mean_value}` across the query set |
| `passed` / `warned` | the ladder decision |
| `primary_metric_name` / `primary_metric` | the headline metric + value |
| `detail.gating` | per-metric `{value, min, ok, warn}` diagnostics |
| `detail.num_corpus` / `num_queries` / `top_k` | run shape |

The `primary_metric_name` flows into the run record's headline metric, so a
retrieval run is comparable in the experiment tracker alongside other methods.
