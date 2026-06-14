"""
shared/verifiers/builtins/retrieval_verifier.py

Corpus-level retrieval verifier. Embeds a document corpus and a query set with a
SentenceTransformer-compatible model, runs FAISS top-k nearest-neighbour
retrieval, aggregates IR metrics (recall@k / MRR / nDCG@k / MAP) against qrels,
and applies a pass/warn/fail threshold ladder.

Registration vs invocation (R9 resolution):
- This module registers a factory under the registry ``type`` key ``"retrieval"``
  via ``@register("retrieval")`` so the verifier is discoverable/buildable
  through ``shared.verifiers.registry.build_verifier`` like every other builtin.
- BUT retrieval is corpus-level, not per-completion. The existing
  ``Verifier.verify(VerifierInput) -> VerifierOutput`` contract maps one
  completion to a scalar; retrieval embeds a whole corpus and scores a query set.
  So the verifier is invoked through a DEDICATED corpus-level entry point,
  :meth:`RetrievalVerifier.evaluate_retrieval`, which consumes a
  :class:`RetrievalConfig` (NOT a ``VerifierInput``). ``verify()`` is implemented
  only to satisfy the ``Verifier`` protocol and is intentionally unused for
  retrieval — calling it raises ``NotImplementedError`` pointing at the
  corpus-level entry point.

shared/ purity (NON-NEGOTIABLE): this module imports nothing from ``Evaluator/``
or ``Trainers/``. The embedding/FAISS mechanics live here; the registry spec
needed for eval-time embedding is passed in via ``RetrievalConfig.model`` already
resolved to a model id / path by the Evaluator caller (which may import
``Trainers/``). Heavy deps (``sentence_transformers``, ``faiss``, ``numpy``) are
imported lazily inside :meth:`evaluate_retrieval` so importing this module for
registration stays cheap and dependency-free.

Used by:
- ``shared/verifiers/builtins/__init__.py`` imports this module for its
  ``@register`` side-effect.
- ``Evaluator/runner.py`` resolves a scenario's ``retrieval_config`` into a
  :class:`RetrievalConfig` and calls :meth:`evaluate_retrieval` once per
  retrieval scenario (a sibling to the per-completion loop).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from shared.ml.retrieval_metrics import QueryResult, aggregate_retrieval_metrics, parse_metric_spec
from shared.verifiers.contract import VerifierInput, VerifierOutput
from shared.verifiers.registry import register

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalThresholds:
    """Threshold block (R4) controlling the pass/warn/fail ladder.

    Attributes:
        min: Mapping metric-spec -> minimum value. A metric below its ``min`` is
            a hard fail. Metrics without a ``min`` entry are reported but never
            gate the result.
        warn_margin: A metric that meets its ``min`` but lands in
            ``[min, min + warn_margin)`` triggers a (passing) warning.
    """

    min: Mapping[str, float] = field(default_factory=dict)
    warn_margin: float = 0.0


@dataclass(frozen=True)
class RetrievalConfig:
    """Inputs for a corpus-level retrieval evaluation.

    Attributes:
        corpus: JSONL path; each line ``{"id": ..., "text": ...}``.
        queries: JSONL path; each line ``{"id": ..., "text": ...}``.
        qrels: JSONL path; each line ``{"query_id": ..., "doc_id": ...,
            "relevance": <int, optional, default 1>}``.
        metrics: Canonical ``"<metric>@<k>"`` spec strings to compute.
        model: A SentenceTransformer-loadable model id or local path. The
            ``registry_name -> hf_id`` resolution is the Evaluator caller's job;
            this module stays ``shared/``-pure and never reads the registry.
        thresholds: The R4 threshold block.
        primary_metric: Spec naming the headline metric (defaults to the first
            entry in ``metrics``).
        query_prompt: Optional prefix prepended to every query before encoding
            (e.g. E5 ``"query: "``). Passed in by the caller from the model spec.
        passage_prompt: Optional prefix prepended to every corpus passage.
        normalize: If True, L2-normalize embeddings and use inner-product
            (cosine) similarity; otherwise use raw inner product.
        batch_size: Encoder batch size.
    """

    corpus: str
    queries: str
    qrels: str
    metrics: Sequence[str]
    model: str
    thresholds: RetrievalThresholds = field(default_factory=RetrievalThresholds)
    primary_metric: str | None = None
    query_prompt: str = ""
    passage_prompt: str = ""
    normalize: bool = True
    batch_size: int = 32

    def resolved_primary_metric(self) -> str:
        """Return the primary metric spec, defaulting to the first in ``metrics``."""
        if self.primary_metric:
            return self.primary_metric
        if not self.metrics:
            raise ValueError("RetrievalConfig.metrics is empty; cannot pick a primary metric.")
        return self.metrics[0]


@dataclass(frozen=True)
class RetrievalValidationResult:
    """Outcome of a corpus-level retrieval evaluation.

    Attributes:
        metrics: Mapping metric-spec -> mean value across the query set.
        passed: True iff every gated metric meets its ``min`` threshold.
        warned: True iff ``passed`` and at least one gated metric lands within
            ``[min, min + warn_margin)``.
        primary_metric_name: Spec of the headline metric (e.g. ``"ndcg@10"``).
        primary_metric: Value of the headline metric.
        detail: Diagnostics (per-metric gating, corpus/query counts, etc.).
    """

    metrics: dict[str, float]
    passed: bool
    warned: bool
    primary_metric_name: str
    primary_metric: float
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registration + verifier
# ---------------------------------------------------------------------------

@register("retrieval")
def _build_retrieval_verifier(spec: Mapping) -> "RetrievalVerifier":
    """Factory for the ``retrieval`` verifier type.

    The spec mapping is accepted for registry-uniformity; the retrieval verifier
    is stateless and configured per-call via :class:`RetrievalConfig`, so no
    spec fields are consumed here today.
    """
    return RetrievalVerifier()


class RetrievalVerifier:
    """Corpus-level retrieval verifier (see module docstring for the R9 split)."""

    name = "retrieval"

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        """Per-completion entry point — NOT used for retrieval.

        Retrieval is corpus-level; use :meth:`evaluate_retrieval`. This method
        exists only to satisfy the ``Verifier`` protocol.
        """
        raise NotImplementedError(
            "RetrievalVerifier is corpus-level; call evaluate_retrieval(RetrievalConfig), "
            "not the per-completion verify(VerifierInput) entry point."
        )

    def evaluate_retrieval(self, cfg: RetrievalConfig) -> RetrievalValidationResult:
        """Embed corpus+queries, FAISS top-k retrieve, aggregate metrics, gate.

        Args:
            cfg: The retrieval configuration (paths, model, metrics, thresholds).

        Returns:
            A :class:`RetrievalValidationResult` with mean metrics and the
            pass/warn/fail decision.

        Raises:
            ValueError: If ``cfg.metrics`` is empty or a metric spec is malformed.
            FileNotFoundError: If a corpus/queries/qrels path does not exist.
        """
        if not cfg.metrics:
            raise ValueError("RetrievalConfig.metrics is empty; nothing to evaluate.")
        # Fail fast on malformed specs before loading data or a model.
        max_k = max(parse_metric_spec(spec)[1] for spec in cfg.metrics)

        corpus = _load_id_text(cfg.corpus)
        queries = _load_id_text(cfg.queries)
        qrels = _load_qrels(cfg.qrels)

        if not corpus:
            raise ValueError(f"Corpus {cfg.corpus!r} is empty.")
        if not queries:
            raise ValueError(f"Queries {cfg.queries!r} is empty.")

        corpus_ids = list(corpus.keys())
        query_ids = list(queries.keys())
        # Retrieval depth: at least max_k, but never more than the corpus size.
        top_k = min(max_k, len(corpus_ids))

        ranked = self._retrieve(
            corpus_texts=[corpus[cid] for cid in corpus_ids],
            corpus_ids=corpus_ids,
            query_texts=[queries[qid] for qid in query_ids],
            query_ids=query_ids,
            cfg=cfg,
            top_k=top_k,
        )

        results = []
        for qid in query_ids:
            grades = qrels.get(qid, {})
            results.append(
                QueryResult(
                    retrieved=ranked[qid],
                    relevant={did for did, grade in grades.items() if grade > 0},
                    relevance=grades or None,
                )
            )

        metrics = aggregate_retrieval_metrics(results, list(cfg.metrics))
        passed, warned, gating = _apply_thresholds(metrics, cfg.thresholds)

        primary_name = cfg.resolved_primary_metric()
        primary_value = metrics.get(primary_name, 0.0)

        return RetrievalValidationResult(
            metrics=metrics,
            passed=passed,
            warned=warned,
            primary_metric_name=primary_name,
            primary_metric=primary_value,
            detail={
                "gating": gating,
                "num_corpus": len(corpus_ids),
                "num_queries": len(query_ids),
                "top_k": top_k,
                "model": cfg.model,
            },
        )

    def _retrieve(
        self,
        *,
        corpus_texts: Sequence[str],
        corpus_ids: Sequence[str],
        query_texts: Sequence[str],
        query_ids: Sequence[str],
        cfg: RetrievalConfig,
        top_k: int,
    ) -> dict[str, list[str]]:
        """Encode + FAISS top-k retrieve. Returns query_id -> ranked corpus ids.

        Heavy deps are imported lazily here so module import (for registration)
        stays cheap and free of ``sentence_transformers``/``faiss``.
        """
        import faiss  # type: ignore[import-not-found]
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        model = SentenceTransformer(cfg.model)

        passages = [f"{cfg.passage_prompt}{text}" for text in corpus_texts]
        prefixed_queries = [f"{cfg.query_prompt}{text}" for text in query_texts]

        corpus_emb = model.encode(
            passages,
            batch_size=cfg.batch_size,
            normalize_embeddings=cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        query_emb = model.encode(
            prefixed_queries,
            batch_size=cfg.batch_size,
            normalize_embeddings=cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Inner product on normalized vectors == cosine similarity.
        index = faiss.IndexFlatIP(corpus_emb.shape[1])
        index.add(corpus_emb)
        _distances, neighbour_idx = index.search(query_emb, top_k)

        # FAISS rows align 1:1 with the encoder input order, which is query_ids.
        ranked: dict[str, list[str]] = {}
        for row, qid in enumerate(query_ids):
            ranked[qid] = [corpus_ids[i] for i in neighbour_idx[row] if i >= 0]
        return ranked


# ---------------------------------------------------------------------------
# Threshold gating + data loading helpers
# ---------------------------------------------------------------------------

def _apply_thresholds(
    metrics: Mapping[str, float],
    thresholds: RetrievalThresholds,
) -> tuple[bool, bool, dict[str, dict[str, Any]]]:
    """Apply the R4 pass/warn/fail ladder.

    ``passed = all(value >= min[m] for m in min)``;
    ``warned = passed and any(min[m] <= value < min[m] + warn_margin)``.
    Metrics without a ``min`` entry are reported but never gate.

    Returns:
        ``(passed, warned, gating)`` where ``gating`` maps each gated metric to
        ``{"value", "min", "ok", "warn"}`` for diagnostics.
    """
    passed = True
    warned = False
    gating: dict[str, dict[str, Any]] = {}
    for spec, minimum in thresholds.min.items():
        value = metrics.get(spec, 0.0)
        ok = value >= minimum
        in_warn_band = ok and value < minimum + thresholds.warn_margin
        gating[spec] = {"value": value, "min": minimum, "ok": ok, "warn": in_warn_band}
        if not ok:
            passed = False
        if in_warn_band:
            warned = True
    # A failing result is never reported as a (passing) warning.
    if not passed:
        warned = False
    return passed, warned, gating


def _load_id_text(path: str) -> dict[str, str]:
    """Load an ``{id, text}`` JSONL into an ordered id -> text mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line lacks ``id`` or ``text``.
    """
    out: dict[str, str] = {}
    for line_num, record in _iter_jsonl(path):
        if "id" not in record or "text" not in record:
            raise ValueError(f"{path}:{line_num} missing required 'id'/'text' field.")
        out[str(record["id"])] = str(record["text"])
    return out


def _load_qrels(path: str) -> dict[str, dict[str, int]]:
    """Load qrels JSONL into query_id -> {doc_id: relevance_grade}.

    Each line: ``{"query_id", "doc_id", "relevance"?}``. ``relevance`` defaults
    to ``1`` (binary relevant) when omitted.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line lacks ``query_id`` or ``doc_id``.
    """
    out: dict[str, dict[str, int]] = {}
    for line_num, record in _iter_jsonl(path):
        if "query_id" not in record or "doc_id" not in record:
            raise ValueError(
                f"{path}:{line_num} missing required 'query_id'/'doc_id' field."
            )
        qid = str(record["query_id"])
        did = str(record["doc_id"])
        grade = int(record.get("relevance", 1))
        out.setdefault(qid, {})[did] = grade
    return out


def _iter_jsonl(path: str):
    """Yield ``(line_number, parsed_record)`` for each non-empty JSONL line.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Retrieval data file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_num, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {file_path}:{line_num}: {exc}") from exc
