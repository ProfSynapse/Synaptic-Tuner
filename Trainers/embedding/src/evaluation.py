"""
In-training IR evaluator builder (recall@k / MRR / nDCG on a dev split).

Location: Trainers/embedding/src/evaluation.py
Purpose:  Build a sentence-transformers InformationRetrievalEvaluator from a
          corpus/queries/qrels triple so the embedding trainer can report
          recall@k / MRR / nDCG during training (eval_strategy). Pure construction
          + metric-spec parsing — the heavy retrieval/aggregation for the *final*
          eval lives in the WU-B retrieval verifier (shared/), not here.
Used by:  Trainers/embedding/train_embedding.py (optional in-training eval).

Contract: docs/architecture/embedding-reranker-phase1/01_CONTRACTS.md §6.

Metric-spec grammar (the canonical "<metric>@<k>" form, §4.1): the config
`evaluation.metrics` list (e.g. ["recall@10", "mrr@10", "ndcg@10"]) is parsed into
the k-lists InformationRetrievalEvaluator expects (accuracy_at_k / mrr_at_k /
ndcg_at_k / map_at_k). recall@k maps onto ST's precision_recall_at_k.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

# metric name -> the InformationRetrievalEvaluator kwarg whose k-list it feeds.
_METRIC_TO_IR_KWARG = {
    "recall": "precision_recall_at_k",
    "mrr": "mrr_at_k",
    "ndcg": "ndcg_at_k",
    "map": "map_at_k",
    # accuracy@k is ST-native too; accept it for completeness.
    "accuracy": "accuracy_at_k",
}

# Column names emitted by data_loader.load_embedding_dataset — used to derive an
# in-training IR set (queries/corpus/qrels) from the held-out eval split.
_IR_FROM_DATASET_QUERY_COLUMN = "anchor"
_IR_FROM_DATASET_POSITIVE_COLUMN = "positive"
_IR_FROM_DATASET_NEGATIVE_COLUMN = "negative"


def parse_metric_specs(metric_specs: Iterable[str]) -> dict[str, list[int]]:
    """Parse ["recall@10", "mrr@10", ...] into IR-evaluator k-list kwargs.

    Returns a dict like {"precision_recall_at_k": [10], "mrr_at_k": [10], ...}
    with duplicate k-values de-duplicated and sorted ascending.

    Raises:
        ValueError: a spec is malformed, names an unknown metric, or has a
                    non-positive k.
    """
    accum: dict[str, set[int]] = {}
    for raw in metric_specs:
        spec = str(raw).strip().lower()
        if "@" not in spec:
            raise ValueError(f"Malformed metric spec {raw!r}; expected '<metric>@<k>'")
        metric, _, k_str = spec.partition("@")
        metric = metric.strip()
        if metric not in _METRIC_TO_IR_KWARG:
            raise ValueError(
                f"Unknown metric {metric!r} in spec {raw!r}; "
                f"supported: {sorted(_METRIC_TO_IR_KWARG)}"
            )
        try:
            k = int(k_str)
        except ValueError as exc:
            raise ValueError(f"Non-integer k in metric spec {raw!r}") from exc
        if k <= 0:
            raise ValueError(f"k must be positive in metric spec {raw!r}")
        accum.setdefault(_METRIC_TO_IR_KWARG[metric], set()).add(k)

    return {kwarg: sorted(ks) for kwarg, ks in accum.items()}


def build_ir_evaluator(
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
    relevant_docs: Mapping[str, set[str]],
    metric_specs: Sequence[str],
    *,
    name: str = "embedding-dev",
    batch_size: int = 32,
) -> Any:
    """Build an InformationRetrievalEvaluator from queries/corpus/qrels.

    Args:
        queries:       {qid: query_text}.
        corpus:        {doc_id: doc_text}.
        relevant_docs: {qid: {relevant_doc_id, ...}}.
        metric_specs:  canonical "<metric>@<k>" list (parsed for the k-lists).
        name:          evaluator name (prefixes the reported metric keys).
        batch_size:    encode batch size.

    Returns:
        A configured InformationRetrievalEvaluator (callable on a model).
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    k_kwargs = parse_metric_specs(metric_specs)

    return InformationRetrievalEvaluator(
        queries=dict(queries),
        corpus=dict(corpus),
        relevant_docs={qid: set(docs) for qid, docs in relevant_docs.items()},
        name=name,
        batch_size=batch_size,
        show_progress_bar=False,
        **k_kwargs,
    )


def build_ir_evaluator_from_dataset(
    eval_dataset: Any,
    metric_specs: Sequence[str],
    *,
    name: str = "embedding-dev",
    batch_size: int = 32,
) -> Any | None:
    """Derive an InformationRetrievalEvaluator from a held-out triplet/pairs split.

    The embedding trainer's eval split is a datasets.Dataset of (anchor, positive
    [, negative]) rows — not a corpus/queries/qrels triple. To run recall@k / MRR /
    nDCG during training, this folds the split into an IR set: each row's anchor is
    a query whose single relevant doc is its positive; the corpus is every distinct
    positive (and negative, when present) across the split, so non-positive docs act
    as distractors. Identical texts collapse onto one doc id, so a query is scored
    against the shared corpus rather than a private positive.

    Returns the configured evaluator, or None when no usable eval set exists (the
    caller then trains without in-training IR eval rather than crashing).

    Args:
        eval_dataset:  the held-out datasets.Dataset (or None).
        metric_specs:  canonical "<metric>@<k>" list (drives the IR k-lists).
        name:          evaluator name (prefixes the reported metric keys).
        batch_size:    encode batch size.
    """
    if eval_dataset is None or len(eval_dataset) == 0:
        return None

    columns = set(getattr(eval_dataset, "column_names", []) or [])
    if _IR_FROM_DATASET_QUERY_COLUMN not in columns or _IR_FROM_DATASET_POSITIVE_COLUMN not in columns:
        # The split does not carry the expected (anchor, positive) columns — skip
        # in-training IR eval rather than mis-derive an IR set.
        return None

    corpus: dict[str, str] = {}
    doc_id_by_text: dict[str, str] = {}

    def _doc_id(text: str) -> str:
        """Return a stable doc id for a passage text, collapsing duplicates."""
        if text not in doc_id_by_text:
            doc_id = f"d{len(doc_id_by_text)}"
            doc_id_by_text[text] = doc_id
            corpus[doc_id] = text
        return doc_id_by_text[text]

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    has_negative = _IR_FROM_DATASET_NEGATIVE_COLUMN in columns
    for index, row in enumerate(eval_dataset):
        anchor = row.get(_IR_FROM_DATASET_QUERY_COLUMN)
        positive = row.get(_IR_FROM_DATASET_POSITIVE_COLUMN)
        if not anchor or not positive:
            continue
        qid = f"q{index}"
        queries[qid] = anchor
        relevant_docs[qid] = {_doc_id(positive)}
        if has_negative:
            negative = row.get(_IR_FROM_DATASET_NEGATIVE_COLUMN)
            if negative:
                _doc_id(negative)  # register as a distractor in the corpus

    if not queries:
        return None

    return build_ir_evaluator(
        queries,
        corpus,
        relevant_docs,
        metric_specs,
        name=name,
        batch_size=batch_size,
    )
