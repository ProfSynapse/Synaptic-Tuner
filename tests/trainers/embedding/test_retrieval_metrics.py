"""Retrieval-metric correctness — recall@k / MRR / nDCG(graded) / MAP.

CONTEXT: ``shared/ml/retrieval_metrics.py`` provides the pure IR-metric
functions consumed by the retrieval verifier and the scenario YAML. The CODE
smoke only asserted "finite values in [0,1]". This suite asserts CONTINUOUS
CORRECTNESS against hand-computed expected values on tiny fixtures, plus the
edge cases the architect flagged (empty-relevant, malformed metric@k spec).

Hand-computations are spelled out in each test so a future reader can re-derive
the expected number without trusting the implementation under test.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.ml.retrieval_metrics import (  # noqa: E402
    QueryResult,
    aggregate_retrieval_metrics,
    map_score,
    mrr,
    ndcg_at_k,
    parse_metric_spec,
    recall_at_k,
)

APPROX = 1e-9


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------

def test_recall_at_k_partial():
    # relevant = {d1, d2, d3}; top-2 retrieved = [d1, dX] -> 1 of 3 found.
    retrieved = ["d1", "dX", "d2", "d3"]
    assert recall_at_k(retrieved, {"d1", "d2", "d3"}, k=2) == pytest.approx(1 / 3)


def test_recall_at_k_full():
    retrieved = ["d1", "d2", "d3"]
    assert recall_at_k(retrieved, {"d1", "d2", "d3"}, k=3) == pytest.approx(1.0)


def test_recall_at_k_cutoff_excludes_late_hit():
    # d3 is at rank 4, outside k=3 -> only d1,d2 counted -> 2/3.
    retrieved = ["d1", "d2", "dX", "d3"]
    assert recall_at_k(retrieved, {"d1", "d2", "d3"}, k=3) == pytest.approx(2 / 3)


def test_recall_empty_relevant_is_zero():
    # Edge case (architect-flagged): no relevant ids -> 0.0, not a div-by-zero.
    assert recall_at_k(["d1", "d2"], set(), k=10) == 0.0


def test_recall_non_positive_k_raises():
    with pytest.raises(ValueError):
        recall_at_k(["d1"], {"d1"}, k=0)
    with pytest.raises(ValueError):
        recall_at_k(["d1"], {"d1"}, k=-1)


def test_recall_rejects_bool_k():
    # bool is an int subclass; the validator must reject True/False as k.
    with pytest.raises(ValueError):
        recall_at_k(["d1"], {"d1"}, k=True)


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

def test_mrr_first_relevant_at_rank_3():
    # First relevant id at rank 3 -> 1/3.
    retrieved = ["dX", "dY", "d1", "d2"]
    assert mrr(retrieved, {"d1", "d2"}) == pytest.approx(1 / 3)


def test_mrr_rank_1_is_one():
    assert mrr(["d1", "dX"], {"d1"}) == pytest.approx(1.0)


def test_mrr_no_relevant_in_window_is_zero():
    # Relevant id exists but only at rank 3, window k=2 -> 0.0.
    assert mrr(["dX", "dY", "d1"], {"d1"}, k=2) == 0.0


def test_mrr_empty_relevant_is_zero():
    assert mrr(["d1"], set()) == 0.0


# ---------------------------------------------------------------------------
# nDCG@k (graded)
# ---------------------------------------------------------------------------

def test_ndcg_graded_known_value():
    # Retrieved ranks: d1(grade3), d2(grade2), dX(grade0)
    # gain = 2^g - 1; discount = log2(rank+1)
    # DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + 0
    #     = 7/1 + 3/1.5849625... = 7 + 1.892789...
    retrieved = ["d1", "d2", "dX"]
    relevance = {"d1": 3, "d2": 2}
    dcg = 7 / math.log2(2) + 3 / math.log2(3)
    # IDCG: ideal order is grade3 then grade2 (same as retrieved here) -> same.
    idcg = 7 / math.log2(2) + 3 / math.log2(3)
    expected = dcg / idcg
    assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(expected)
    assert expected == pytest.approx(1.0)  # perfectly ranked -> 1.0


def test_ndcg_imperfect_ranking_below_one():
    # Swapped order: lower-grade doc first penalizes nDCG below 1.
    # Retrieved: d2(grade2) at rank1, d1(grade3) at rank2.
    retrieved = ["d2", "d1"]
    relevance = {"d1": 3, "d2": 2}
    dcg = 3 / math.log2(2) + 7 / math.log2(3)       # 3 + 7/1.58496 = 3 + 4.41653
    idcg = 7 / math.log2(2) + 3 / math.log2(3)      # 7 + 3/1.58496 = 7 + 1.89279
    expected = dcg / idcg
    assert ndcg_at_k(retrieved, relevance, k=2) == pytest.approx(expected)
    assert 0.0 < expected < 1.0


def test_ndcg_no_graded_relevant_is_zero():
    # All grades 0 -> idcg 0 -> 0.0 (no div-by-zero).
    assert ndcg_at_k(["d1", "d2"], {"d1": 0, "d2": 0}, k=2) == 0.0


def test_ndcg_empty_relevance_is_zero():
    assert ndcg_at_k(["d1"], {}, k=5) == 0.0


# ---------------------------------------------------------------------------
# MAP (single-query average precision)
# ---------------------------------------------------------------------------

def test_map_average_precision_known_value():
    # relevant = {d1, d3}; retrieved = [d1, dX, d3, dY]
    # rank1 d1 relevant -> precision 1/1 = 1.0
    # rank3 d3 relevant -> precision 2/3
    # AP = (1.0 + 2/3) / |relevant(2)| = (1 + 0.6667) / 2 = 0.83333
    retrieved = ["d1", "dX", "d3", "dY"]
    expected = (1.0 + 2 / 3) / 2
    assert map_score(retrieved, {"d1", "d3"}) == pytest.approx(expected)


def test_map_perfect_is_one():
    # Both relevant at the top -> AP = (1/1 + 2/2)/2 = 1.0
    assert map_score(["d1", "d2", "dX"], {"d1", "d2"}) == pytest.approx(1.0)


def test_map_empty_relevant_is_zero():
    assert map_score(["d1"], set()) == 0.0


# ---------------------------------------------------------------------------
# Spec parsing (malformed-spec edge case, architect-flagged)
# ---------------------------------------------------------------------------

def test_parse_metric_spec_valid():
    assert parse_metric_spec("recall@10") == ("recall", 10)
    assert parse_metric_spec("NDCG@5") == ("ndcg", 5)       # case-insensitive
    assert parse_metric_spec("  mrr @ 3 ") == ("mrr", 3)    # whitespace-tolerant


@pytest.mark.parametrize(
    "bad",
    [
        "recall",          # missing @k
        "recall@",         # missing k
        "recall@0",        # non-positive k
        "recall@-3",       # negative k
        "precision@10",    # unsupported metric
        "ndcg@1.5",        # non-integer k
        "@10",             # missing metric
        "",                # empty
    ],
)
def test_parse_metric_spec_malformed_raises(bad):
    with pytest.raises(ValueError):
        parse_metric_spec(bad)


# ---------------------------------------------------------------------------
# Aggregator (mean across queries + MAP-as-mean-of-AP)
# ---------------------------------------------------------------------------

def test_aggregate_means_across_queries():
    # Two queries; recall@2 = 1.0 and 0.0 -> mean 0.5.
    q1 = QueryResult(retrieved=["d1", "d2"], relevant={"d1"})         # recall@2 = 1.0
    q2 = QueryResult(retrieved=["dX", "dY"], relevant={"d1"})         # recall@2 = 0.0
    out = aggregate_retrieval_metrics([q1, q2], ["recall@2"])
    assert out["recall@2"] == pytest.approx(0.5)


def test_aggregate_map_is_mean_of_ap():
    # MAP = mean of per-query AP.
    q1 = QueryResult(retrieved=["d1", "dX", "d3"], relevant={"d1", "d3"})  # AP=(1+2/3)/2
    q2 = QueryResult(retrieved=["d1", "d2"], relevant={"d1", "d2"})        # AP=1.0
    ap1 = (1.0 + 2 / 3) / 2
    out = aggregate_retrieval_metrics([q1, q2], ["map@3"])
    assert out["map@3"] == pytest.approx((ap1 + 1.0) / 2)


def test_aggregate_graded_ndcg_uses_relevance_map():
    # QueryResult.relevance (graded) is used for ndcg even though relevant set differs.
    q = QueryResult(retrieved=["d1", "d2"], relevant={"d1", "d2"}, relevance={"d1": 3, "d2": 2})
    out = aggregate_retrieval_metrics([q], ["ndcg@2"])
    assert out["ndcg@2"] == pytest.approx(1.0)  # perfectly ranked grades


def test_aggregate_empty_results_returns_zeros():
    out = aggregate_retrieval_metrics([], ["recall@10", "ndcg@10"])
    assert out == {"recall@10": 0.0, "ndcg@10": 0.0}


def test_aggregate_malformed_spec_fails_fast():
    q = QueryResult(retrieved=["d1"], relevant={"d1"})
    with pytest.raises(ValueError):
        aggregate_retrieval_metrics([q], ["recall@10", "bogus@k"])


def test_query_result_graded_relevance_derives_binary_when_absent():
    # When .relevance is None, graded_relevance() derives grade-1 from .relevant.
    q = QueryResult(retrieved=["d1"], relevant={"d1", "d2"})
    assert q.graded_relevance() == {"d1": 1, "d2": 1}
