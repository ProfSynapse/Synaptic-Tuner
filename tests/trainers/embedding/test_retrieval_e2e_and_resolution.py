"""Real-FAISS retrieval E2E + runner model-resolution seam + in-training parser.

CONTEXT (architect Test-Engineer Note #8): exercise the corpus-level
``evaluate_retrieval`` path with faiss ACTUALLY installed (the retrieval-coder
verified the ranking logic via a deterministic monkeypatch; THIS is the
real-faiss gate). Also confirm ``runner._resolve_retrieval_model``'s
``{registry_name}`` branch resolves against WU-A ``get_spec(name)`` for
``.hf_id`` / ``.query_prompt`` / ``.passage_prompt`` — the cross-WU concurrency
seam (CLAUDE.md note; CONTRACTS §1.2).

faiss gate: faiss-cpu is required; if absent the real-faiss tests skip (CI-gated)
while the resolution + parser tests still run. The ENCODER is a deterministic
fake (controlled unit vectors) so the test needs no model download — but FAISS
does the genuine top-k inner-product search over those vectors, so the ranking,
top-k truncation, and id-alignment in ``RetrievalVerifier._retrieve`` are
exercised for real.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolated_import import load_embedding_src  # noqa: E402

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed (CI-gated real-faiss E2E)")

import numpy as np  # noqa: E402

from shared.verifiers.builtins import retrieval_verifier as rv  # noqa: E402
from shared.verifiers.builtins.retrieval_verifier import (  # noqa: E402
    RetrievalConfig,
    RetrievalThresholds,
    RetrievalVerifier,
)


# ---------------------------------------------------------------------------
# Deterministic fake encoder: maps known texts to fixed unit vectors so FAISS
# inner-product retrieval has a hand-checkable ranking. Installed via a fake
# `sentence_transformers.SentenceTransformer` the verifier imports lazily.
# ---------------------------------------------------------------------------

# 3-d unit vectors. Queries are crafted to retrieve their gold doc first.
_VECTORS = {
    # corpus passages (verifier prefixes passage_prompt="" here, so raw text)
    "the cat sat on the mat": [1.0, 0.0, 0.0],
    "dogs are loyal companions": [0.0, 1.0, 0.0],
    "the sky is blue today": [0.0, 0.0, 1.0],
    "a feline rested on a rug": [0.95, 0.05, 0.0],   # near the cat passage
    # queries
    "where did the cat sit": [0.9, 0.1, 0.0],          # closest to cat passages
    "tell me about dogs": [0.1, 0.9, 0.0],             # closest to dogs passage
}


class _FakeSentenceTransformer:
    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id

    def encode(self, texts, **kwargs):
        rows = []
        for t in texts:
            if t not in _VECTORS:
                raise AssertionError(f"unexpected text into encoder: {t!r}")
            rows.append(_VECTORS[t])
        arr = np.array(rows, dtype=np.float32)
        if kwargs.get("normalize_embeddings"):
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


@pytest.fixture
def fake_encoder(monkeypatch):
    """Patch the SentenceTransformer the verifier imports lazily inside
    `_retrieve`. We patch the name in `sentence_transformers` so the verifier's
    `from sentence_transformers import SentenceTransformer` picks up the fake."""
    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer
    )
    return _FakeSentenceTransformer


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def corpus_files(tmp_path):
    corpus = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"id": "c1", "text": "the cat sat on the mat"},
            {"id": "c2", "text": "dogs are loyal companions"},
            {"id": "c3", "text": "the sky is blue today"},
            {"id": "c4", "text": "a feline rested on a rug"},
        ],
    )
    queries = _write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {"id": "q1", "text": "where did the cat sit"},
            {"id": "q2", "text": "tell me about dogs"},
        ],
    )
    qrels = _write_jsonl(
        tmp_path / "qrels.jsonl",
        [
            {"query_id": "q1", "doc_id": "c1", "relevance": 1},
            {"query_id": "q1", "doc_id": "c4", "relevance": 1},  # also relevant
            {"query_id": "q2", "doc_id": "c2", "relevance": 1},
        ],
    )
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Real-FAISS E2E
# ---------------------------------------------------------------------------

def test_evaluate_retrieval_real_faiss_ranks_correctly(fake_encoder, corpus_files):
    corpus, queries, qrels = corpus_files
    cfg = RetrievalConfig(
        corpus=str(corpus),
        queries=str(queries),
        qrels=str(qrels),
        metrics=["recall@2", "mrr@2", "ndcg@2"],
        model="fake-model-id",
        thresholds=RetrievalThresholds(min={"mrr@2": 0.5}, warn_margin=0.0),
        normalize=True,
    )

    result = RetrievalVerifier().evaluate_retrieval(cfg)

    # q1's gold doc c1 (and c4) are the nearest passages; q2's gold doc c2 is
    # nearest. Both queries retrieve a relevant doc at rank 1 -> mrr 1.0 each.
    assert result.metrics["mrr@2"] == pytest.approx(1.0)
    assert result.passed is True
    assert result.detail["num_corpus"] == 4
    assert result.detail["num_queries"] == 2
    assert result.detail["top_k"] == 2  # min(max_k=2, corpus=4)
    assert result.primary_metric_name == "recall@2"  # first metric by default


def test_evaluate_retrieval_top_k_capped_by_corpus_size(fake_encoder, corpus_files):
    """Requesting recall@10 on a 4-doc corpus caps retrieval depth at 4."""
    corpus, queries, qrels = corpus_files
    cfg = RetrievalConfig(
        corpus=str(corpus), queries=str(queries), qrels=str(qrels),
        metrics=["recall@10"], model="fake", normalize=True,
    )
    result = RetrievalVerifier().evaluate_retrieval(cfg)
    assert result.detail["top_k"] == 4  # min(10, corpus_size=4)


def test_evaluate_retrieval_empty_metrics_raises(corpus_files):
    corpus, queries, qrels = corpus_files
    cfg = RetrievalConfig(
        corpus=str(corpus), queries=str(queries), qrels=str(qrels),
        metrics=[], model="fake",
    )
    with pytest.raises(ValueError):
        RetrievalVerifier().evaluate_retrieval(cfg)


def test_evaluate_retrieval_malformed_spec_raises_before_load(corpus_files):
    """A malformed metric spec fails fast (before encoding the corpus)."""
    corpus, queries, qrels = corpus_files
    cfg = RetrievalConfig(
        corpus=str(corpus), queries=str(queries), qrels=str(qrels),
        metrics=["recall@notanint"], model="fake",
    )
    with pytest.raises(ValueError):
        RetrievalVerifier().evaluate_retrieval(cfg)


def test_evaluate_retrieval_missing_corpus_file_raises(tmp_path, corpus_files):
    _corpus, queries, qrels = corpus_files
    cfg = RetrievalConfig(
        corpus=str(tmp_path / "nope.jsonl"), queries=str(queries), qrels=str(qrels),
        metrics=["recall@2"], model="fake",
    )
    with pytest.raises(FileNotFoundError):
        RetrievalVerifier().evaluate_retrieval(cfg)


def test_verify_per_completion_entrypoint_refuses():
    """The per-completion verify() must refuse — retrieval is corpus-level (R9)."""
    with pytest.raises(NotImplementedError):
        RetrievalVerifier().verify(sample=object())  # type: ignore[arg-type]


def test_retrieval_verifier_is_registered():
    """The verifier registers under the 'retrieval' type key for build_verifier."""
    from shared.verifiers.registry import build_verifier

    verifier = build_verifier({"type": "retrieval"})
    assert isinstance(verifier, RetrievalVerifier)
    assert verifier.name == "retrieval"


# ---------------------------------------------------------------------------
# runner._resolve_retrieval_model — the cross-WU {registry_name} seam (CONTRACTS
# §1.2). Resolution does NOT need faiss, so these run even without the E2E gate.
# ---------------------------------------------------------------------------

from Evaluator.runner import _build_retrieval_config, _resolve_retrieval_model  # noqa: E402


def test_resolve_registry_name_uses_get_spec():
    """A {registry_name} block resolves against WU-A get_spec(name) for
    hf_id + query/passage prompts — the seam the CLAUDE.md note flagged."""
    model_id, qp, pp = _resolve_retrieval_model({"registry_name": "bge-base-en"})
    # These come straight from Trainers/embedding/configs/model_registry.yaml.
    assert model_id == "BAAI/bge-base-en-v1.5"
    assert qp == "Represent this sentence for searching relevant passages: "
    assert pp == ""


def test_resolve_registry_name_e5_threads_both_prompts():
    """E5 requires query:/passage: prefixes; both must thread through."""
    model_id, qp, pp = _resolve_retrieval_model({"registry_name": "e5-base"})
    assert model_id == "intfloat/e5-base-v2"
    assert qp == "query: "
    assert pp == "passage: "


def test_resolve_path_block_bypasses_registry():
    model_id, qp, pp = _resolve_retrieval_model(
        {"path": "/runs/embedding/final", "query_prompt": "Q: ", "passage_prompt": "P: "}
    )
    assert model_id == "/runs/embedding/final"
    assert qp == "Q: "
    assert pp == "P: "


def test_resolve_bare_string_is_id_with_empty_prompts():
    assert _resolve_retrieval_model("some/model-id") == ("some/model-id", "", "")


def test_resolve_empty_model_block_raises():
    with pytest.raises(ValueError):
        _resolve_retrieval_model({})


def test_resolve_unknown_registry_name_raises():
    with pytest.raises(KeyError):
        _resolve_retrieval_model({"registry_name": "no-such-model"})


def test_build_retrieval_config_full_roundtrip(corpus_files):
    """The full scenario -> RetrievalConfig translation resolves the registry
    name and carries thresholds/prompts through."""
    corpus, queries, qrels = corpus_files
    raw = {
        "corpus": str(corpus),
        "queries": str(queries),
        "qrels": str(qrels),
        "metrics": ["recall@10", "ndcg@10"],
        "model": {"registry_name": "bge-base-en"},
        "thresholds": {"min": {"ndcg@10": 0.30}, "warn_margin": 0.05},
        "primary_metric": "ndcg@10",
    }
    cfg = _build_retrieval_config(raw)
    assert cfg.model == "BAAI/bge-base-en-v1.5"
    assert cfg.query_prompt == "Represent this sentence for searching relevant passages: "
    assert cfg.thresholds.min == {"ndcg@10": 0.30}
    assert cfg.thresholds.warn_margin == 0.05
    assert cfg.resolved_primary_metric() == "ndcg@10"


def test_build_retrieval_config_missing_key_raises(corpus_files):
    corpus, queries, _qrels = corpus_files
    raw = {  # missing 'qrels'
        "corpus": str(corpus), "queries": str(queries),
        "metrics": ["recall@10"], "model": "id",
    }
    with pytest.raises(ValueError) as exc:
        _build_retrieval_config(raw)
    assert "qrels" in str(exc.value)


# ---------------------------------------------------------------------------
# In-training IR-evaluator metric-spec parser (Trainers/embedding/src/evaluation.py)
# ---------------------------------------------------------------------------

def test_evaluation_parse_metric_specs_maps_to_ir_kwargs():
    evaluation = load_embedding_src("evaluation")
    out = evaluation.parse_metric_specs(["recall@10", "mrr@10", "ndcg@10", "map@5"])
    assert out == {
        "precision_recall_at_k": [10],
        "mrr_at_k": [10],
        "ndcg_at_k": [10],
        "map_at_k": [5],
    }


def test_evaluation_parse_metric_specs_dedups_and_sorts():
    evaluation = load_embedding_src("evaluation")
    out = evaluation.parse_metric_specs(["recall@10", "recall@5", "recall@10"])
    assert out == {"precision_recall_at_k": [5, 10]}


@pytest.mark.parametrize("bad", ["recall", "recall@0", "bogus@10", "recall@x"])
def test_evaluation_parse_metric_specs_malformed_raises(bad):
    evaluation = load_embedding_src("evaluation")
    with pytest.raises(ValueError):
        evaluation.parse_metric_specs([bad])
