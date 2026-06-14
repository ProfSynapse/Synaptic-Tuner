"""
Embedding training data loader — JSONL triplets/pairs -> ST Dataset.

Location: Trainers/embedding/src/data_loader.py
Purpose:  Read a local JSONL of triplets ({query, positive, negative(s)}) or
          pairs ({query, positive}) and materialize a datasets.Dataset whose
          columns match what the sentence-transformers loss expects, applying the
          spec's query/passage prompt prefixes. Optionally carves an eval split.
Used by:  Trainers/embedding/train_embedding.py.

Contract: docs/architecture/embedding-reranker-phase1/01_CONTRACTS.md §6.

Column convention (sentence-transformers): for MultipleNegativesRankingLoss the
dataset columns are positional — (anchor, positive) or (anchor, positive,
negative). We emit canonical column names "anchor"/"positive"/"negative"; ST maps
columns positionally to the loss, so the names are for readability. A `negatives`
list with multiple entries is exploded into one row per negative (each row reuses
the same anchor/positive), which is the standard ST hard-negative shape.

Prompt prefixing: the spec's query_prompt is prepended to anchors and
passage_prompt to positives/negatives. E5-style models REQUIRE these prefixes
(spec.prompt_required); bge's query prompt is optional (R7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from registry import EmbeddingModelSpec


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: a non-blank line is not valid JSON (names the line number).
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"Dataset file {path} contained no records")
    return rows


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one record to {query, positive, negatives: list[str]}.

    Accepts the common field aliases: query/anchor/question for the anchor;
    positive/pos for the positive; negative/negatives/neg for negatives (scalar
    or list). Pair records (no negatives) yield an empty negatives list.

    Raises:
        ValueError: a record is missing a query or positive.
    """
    query = record.get("query") or record.get("anchor") or record.get("question")
    positive = record.get("positive") or record.get("pos")
    if not query or not positive:
        raise ValueError(
            f"Record missing required 'query'/'positive' fields: {record!r}"
        )

    raw_negatives = record.get("negatives")
    if raw_negatives is None:
        single = record.get("negative") or record.get("neg")
        raw_negatives = [single] if single else []
    if isinstance(raw_negatives, str):
        raw_negatives = [raw_negatives]

    negatives = [str(n) for n in raw_negatives if n]
    return {"query": str(query), "positive": str(positive), "negatives": negatives}


def _prefix(text: str, prompt: str) -> str:
    """Prepend a prompt prefix to text (no-op when prompt is empty)."""
    return f"{prompt}{text}" if prompt else text


def load_embedding_dataset(
    local_file: str | Path,
    spec: EmbeddingModelSpec,
    *,
    eval_split: float = 0.0,
    seed: int = 42,
):
    """Load a JSONL triplet/pairs file into a (train, eval) datasets.Dataset pair.

    Applies spec.query_prompt to anchors and spec.passage_prompt to
    positives/negatives. Multi-negative records are exploded into one row per
    negative. Columns: anchor, positive[, negative].

    Args:
        local_file: path to the JSONL dataset.
        spec:       the model spec (supplies prompt prefixes).
        eval_split: fraction in [0, 1) carved off for eval (0 -> no eval set).
        seed:       split shuffle seed.

    Returns:
        (train_dataset, eval_dataset_or_None).
    """
    from datasets import Dataset

    path = Path(local_file)
    raw_rows = _read_jsonl(path)

    anchors: list[str] = []
    positives: list[str] = []
    negatives: list[str] = []
    has_negatives = False

    for raw in raw_rows:
        rec = _normalize_record(raw)
        anchor_text = _prefix(rec["query"], spec.query_prompt)
        positive_text = _prefix(rec["positive"], spec.passage_prompt)
        if rec["negatives"]:
            has_negatives = True
            for neg in rec["negatives"]:
                anchors.append(anchor_text)
                positives.append(positive_text)
                negatives.append(_prefix(neg, spec.passage_prompt))
        else:
            anchors.append(anchor_text)
            positives.append(positive_text)
            negatives.append("")  # placeholder; dropped below if no record had negatives

    columns: dict[str, list[str]] = {"anchor": anchors, "positive": positives}
    if has_negatives:
        # If ANY record had negatives, the triplet shape is used. Rows that lacked
        # a negative carry an empty placeholder — drop them so the loss never sees
        # an empty negative (mixing pair/triplet rows in one MNRL dataset is unsafe).
        keep = [i for i, neg in enumerate(negatives) if neg]
        columns = {
            "anchor": [anchors[i] for i in keep],
            "positive": [positives[i] for i in keep],
            "negative": [negatives[i] for i in keep],
        }

    dataset = Dataset.from_dict(columns)

    if eval_split and 0.0 < eval_split < 1.0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=eval_split, seed=seed)
        return split["train"], split["test"]

    return dataset, None
