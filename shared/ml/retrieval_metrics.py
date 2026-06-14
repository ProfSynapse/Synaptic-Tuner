"""
shared/ml/retrieval_metrics.py

Pure-function retrieval (information-retrieval) metrics: recall@k, MRR, nDCG@k,
and MAP. Operates on ranked retrieved id lists plus relevance ground truth
(a relevant-id set, or an id->grade mapping for graded nDCG). All functions
return a float in [0, 1].

This is a SIBLING to ``shared/ml/metrics.py`` (which is sklearn-backed
classification/regression). Retrieval metrics operate on a fundamentally
different data shape (ranked id lists + qrels), so they live here rather than
extending ``compute_metrics``. The metric functions themselves carry no heavy
dependencies (numpy at most) and import nothing from ``Evaluator/`` or
``Trainers/`` — ``shared/`` purity is preserved.

Used by:
- ``shared/verifiers/builtins/retrieval_verifier.py`` (corpus-level retrieval
  verifier), via :func:`aggregate_retrieval_metrics`.
- Scenario YAML and retrieval lineage reference the canonical ``"<metric>@<k>"``
  spec grammar parsed here (see :func:`parse_metric_spec`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Set

# Canonical metric-spec grammar: "<metric>@<k>" where metric is one of the
# supported names and k is a positive integer. This is the single spec format
# shared by the scenario YAML, the verifier, and the retrieval lineage.
_SUPPORTED_METRICS = ("recall", "mrr", "ndcg", "map")
_METRIC_SPEC_RE = re.compile(r"^\s*(recall|mrr|ndcg|map)\s*@\s*(\d+)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pure metric functions — ranked ids + ground truth -> float in [0, 1]
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant ids found within the top-``k`` retrieved ids.

    Args:
        retrieved: Ranked list of retrieved ids (best first).
        relevant: Set of ground-truth relevant ids.
        k: Cutoff rank (must be a positive int).

    Returns:
        ``|relevant ∩ top_k| / |relevant|`` in ``[0, 1]``. Returns ``0.0`` when
        there are no relevant ids (nothing to recall).

    Raises:
        ValueError: If ``k`` is not a positive integer.
    """
    _require_positive_k(k)
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for rid in top_k if rid in relevant)
    return hits / len(relevant)


def mrr(retrieved: Sequence[str], relevant: Set[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant id (optionally within top-``k``).

    Args:
        retrieved: Ranked list of retrieved ids (best first).
        relevant: Set of ground-truth relevant ids.
        k: Optional cutoff rank. ``None`` considers the full ranked list.

    Returns:
        ``1 / rank`` of the first relevant id (1-indexed), or ``0.0`` if no
        relevant id appears within the considered window.

    Raises:
        ValueError: If ``k`` is provided and is not a positive integer.
    """
    if k is not None:
        _require_positive_k(k)
    if not relevant:
        return 0.0
    window = retrieved[:k] if k is not None else retrieved
    for rank, rid in enumerate(window, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Graded normalized discounted cumulative gain at rank ``k``.

    Uses the standard ``(2^grade - 1) / log2(rank + 1)`` gain formulation,
    normalized by the ideal DCG (the same gains ranked in descending grade
    order). ``relevance`` maps id -> non-negative integer grade (e.g. 0..3);
    binary relevance is the special case where every relevant id has grade 1.

    Args:
        retrieved: Ranked list of retrieved ids (best first).
        relevance: Mapping id -> relevance grade. Ids absent from the mapping
            (or with grade 0) contribute zero gain.
        k: Cutoff rank (must be a positive int).

    Returns:
        ``DCG@k / IDCG@k`` in ``[0, 1]``. Returns ``0.0`` when the ideal DCG is
        zero (no graded-relevant ids), avoiding division by zero.

    Raises:
        ValueError: If ``k`` is not a positive integer.
    """
    _require_positive_k(k)
    dcg = 0.0
    for rank, rid in enumerate(retrieved[:k], start=1):
        grade = relevance.get(rid, 0)
        if grade > 0:
            dcg += (2 ** grade - 1) / math.log2(rank + 1)

    ideal_grades = sorted((g for g in relevance.values() if g > 0), reverse=True)
    idcg = 0.0
    for rank, grade in enumerate(ideal_grades[:k], start=1):
        idcg += (2 ** grade - 1) / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def map_score(retrieved: Sequence[str], relevant: Set[str], k: int | None = None) -> float:
    """Average precision for a single query (the per-query term of MAP).

    Precision is accumulated at each rank where a relevant id is retrieved, then
    averaged over the number of relevant ids (the standard AP normalization).
    With one query this is "AP"; the mean across queries (computed by
    :func:`aggregate_retrieval_metrics`) is "MAP".

    Args:
        retrieved: Ranked list of retrieved ids (best first).
        relevant: Set of ground-truth relevant ids.
        k: Optional cutoff rank. ``None`` considers the full ranked list.

    Returns:
        Average precision in ``[0, 1]``. Returns ``0.0`` when there are no
        relevant ids.

    Raises:
        ValueError: If ``k`` is provided and is not a positive integer.
    """
    if k is not None:
        _require_positive_k(k)
    if not relevant:
        return 0.0
    window = retrieved[:k] if k is not None else retrieved
    hits = 0
    precision_sum = 0.0
    for rank, rid in enumerate(window, start=1):
        if rid in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


# ---------------------------------------------------------------------------
# Per-query container + spec parsing + aggregator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryResult:
    """Per-query retrieval outcome handed to :func:`aggregate_retrieval_metrics`.

    Attributes:
        retrieved: Ranked list of retrieved ids (best first).
        relevant: Set of binary-relevant ids (used by recall/mrr/map).
        relevance: Optional id -> grade mapping for graded nDCG. When omitted,
            it is derived from ``relevant`` as grade-1 binary relevance.
    """

    retrieved: Sequence[str]
    relevant: Set[str] = field(default_factory=set)
    relevance: Mapping[str, int] | None = None

    def graded_relevance(self) -> Mapping[str, int]:
        """Return the graded relevance map, deriving binary grades if absent."""
        if self.relevance is not None:
            return self.relevance
        return {rid: 1 for rid in self.relevant}


def parse_metric_spec(spec: str) -> tuple[str, int]:
    """Parse a canonical ``"<metric>@<k>"`` spec into ``(metric, k)``.

    Args:
        spec: A spec string such as ``"recall@10"`` or ``"ndcg@10"``. The metric
            name is matched case-insensitively and lowered; ``k`` must be a
            positive integer.

    Returns:
        A ``(metric_lowercase, k)`` tuple, e.g. ``("recall", 10)``.

    Raises:
        ValueError: If the spec does not match the grammar or ``k <= 0``.
    """
    match = _METRIC_SPEC_RE.match(spec)
    if match is None:
        raise ValueError(
            f"Invalid metric spec {spec!r}. Expected '<metric>@<k>' where metric "
            f"is one of {_SUPPORTED_METRICS} and k is a positive integer."
        )
    metric = match.group(1).lower()
    k = int(match.group(2))
    if k <= 0:
        raise ValueError(f"Invalid metric spec {spec!r}: k must be positive, got {k}.")
    return metric, k


def _compute_single(metric: str, k: int, result: QueryResult) -> float:
    """Dispatch one parsed ``(metric, k)`` against a single :class:`QueryResult`."""
    if metric == "recall":
        return recall_at_k(result.retrieved, set(result.relevant), k)
    if metric == "mrr":
        return mrr(result.retrieved, set(result.relevant), k)
    if metric == "ndcg":
        return ndcg_at_k(result.retrieved, result.graded_relevance(), k)
    if metric == "map":
        return map_score(result.retrieved, set(result.relevant), k)
    # Unreachable: parse_metric_spec already constrains the metric name.
    raise ValueError(f"Unsupported metric {metric!r}.")


def aggregate_retrieval_metrics(
    results: Sequence[QueryResult],
    metric_specs: Sequence[str],
) -> dict[str, float]:
    """Compute mean metrics across a query set.

    Parses each ``"<metric>@<k>"`` spec, evaluates it per query, and returns the
    mean value across all queries (e.g. ``map@k`` averaged over queries is MAP).

    Args:
        results: Per-query retrieval outcomes.
        metric_specs: Canonical spec strings, e.g. ``["recall@10", "ndcg@10"]``.

    Returns:
        Mapping each original spec string to its mean value across queries.
        Returns ``0.0`` for every spec when ``results`` is empty.

    Raises:
        ValueError: If any spec is malformed (see :func:`parse_metric_spec`).
    """
    # Parse up front so a malformed spec fails fast before any computation.
    parsed = [(spec, *parse_metric_spec(spec)) for spec in metric_specs]

    if not results:
        return {spec: 0.0 for spec, _metric, _k in parsed}

    aggregated: dict[str, float] = {}
    for spec, metric, k in parsed:
        total = sum(_compute_single(metric, k, result) for result in results)
        aggregated[spec] = total / len(results)
    return aggregated


def _require_positive_k(k: int) -> None:
    """Validate that ``k`` is a positive integer, raising ``ValueError`` if not."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}.")
