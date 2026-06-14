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
