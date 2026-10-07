"""Train/eval contamination check via word n-gram containment.

Location: ``shared/contamination.py``
Used by: ``tuner/handlers/contamination_handler.py`` (``tuner.py check-contamination``)

For every eval item (a prompt or a reference text extracted with the
Evaluator's own loaders) the check reports the highest *containment* against
any single training row::

    containment(item, row) = |ngrams(item) ∩ ngrams(row)| / |ngrams(item)|

over unique word n-grams (default n=8) after normalization (unicode NFKC,
lowercase, every non-alphanumeric character treated as a word boundary,
whitespace collapsed). Training rows are indexed in an inverted index from
n-gram hash to row ids; only n-grams that occur in some eval item are kept, so
memory and time scale with the eval set and the matching rows, never with
eval x train pairs.

Items shorter than n words but at least ``min_item_tokens`` words are scored
with a single whole-item n-gram (containment is then 0 or 1). Shorter items
are only checked for exact duplicates.

Training rows are read with the trainers' own format detection and shape
normalization (:func:`shared.sft_preprocessing.detect_sft_record_format`,
:func:`~shared.sft_preprocessing.normalize_sft_messages` and
:func:`~shared.sft_preprocessing.sanitize_messages_for_chat_template`), so every
format a trainer accepts is supported: SFT ``messages``/``conversations``,
``prompt``/``completion`` (SFT and KTO), prepared ``raw_text`` rows, DPO
``prompt``/``chosen``/``rejected`` and prompt-only GRPO rows. Nothing about any one scenario or dataset is
hardcoded here; sources, roles and reference fields come from config.
"""

from __future__ import annotations

import heapq
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from shared.sft_preprocessing import (
    RAW_TEXT_FORMAT,
    detect_sft_record_format,
    normalize_sft_messages,
    sanitize_messages_for_chat_template,
)
from shared.validation.rollout_filters import MISSING, get_path

DEFAULT_NGRAM = 8
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_ITEM_TOKENS = 4
DEFAULT_TOP_K = 20
DEFAULT_TRAIN_ROLES = ("system", "user", "assistant", "tool")

_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)


# ---------------------------------------------------------------------------
# Normalization and n-grams
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """NFKC, lowercase, strip punctuation/symbols to word boundaries, collapse whitespace."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return _NON_ALNUM.sub(" ", normalized).strip()


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def _gram_hash(tokens: Sequence[str]) -> int:
    return hash(" ".join(tokens))


def ngram_hashes(tokens: Sequence[str], n: int) -> Set[int]:
    """Unique hashes of the word n-grams in ``tokens`` (empty when shorter than n)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return {_gram_hash(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def containment(item_grams: Set[int], row_grams: Set[int]) -> float:
    """Fraction of the item's n-grams that occur in the row (0.0 for an empty item)."""
    if not item_grams:
        return 0.0
    return len(item_grams & row_grams) / len(item_grams)


# ---------------------------------------------------------------------------
# Training rows
# ---------------------------------------------------------------------------

@dataclass
class TrainingRow:
    row_id: str
    dataset: str
    line_number: int
    row_format: str
    text: str
    prompt_texts: List[str]


def detect_row_format(row: Mapping[str, Any]) -> str:
    """Classify a raw training row into the shapes our trainers accept.

    DPO pairs and prompt-only (GRPO) rows are recognized here; everything else
    goes through the SFT trainer's own :func:`detect_sft_record_format`
    (``messages`` / ``prompt_completion`` / prepared ``raw_text``), which raises
    on unsupported shapes. A ``label`` column (KTO) is reported as ``+label``.
    """
    if row.get("chosen") is not None and row.get("rejected") is not None and row.get("prompt") is not None:
        return "dpo"
    if (
        row.get("prompt") is not None
        and row.get("completion") is None
        and not row.get("messages")
        and not row.get("conversations")
        and row.get("format") is None
        and row.get("schema_version") is None
    ):
        return "prompt_only"
    sft_format = detect_sft_record_format(dict(row))
    return f"{sft_format}+label" if "label" in row else sft_format


def row_messages(row: Mapping[str, Any], row_format: str) -> List[Dict[str, Any]]:
    """Canonical message list for a row, via the SFT preprocessing normalizer."""
    if row_format == "dpo":
        messages, _ = normalize_sft_messages({"prompt": row["prompt"], "completion": row["chosen"]})
        rejected, _ = normalize_sft_messages({"prompt": [], "completion": row["rejected"]})
        return list(messages) + list(rejected)
    if row_format == "prompt_only":
        messages, _ = normalize_sft_messages({"prompt": row["prompt"], "completion": []})
        return list(messages)
    messages, _ = normalize_sft_messages(dict(row))
    return list(messages)


def _message_text(message: Mapping[str, Any]) -> str:
    rendered = sanitize_messages_for_chat_template([dict(message)])
    return str(rendered[0].get("content") or "") if rendered else ""


def build_training_row(
    row: Mapping[str, Any],
    *,
    dataset: str,
    line_number: int,
    train_roles: Sequence[str] = DEFAULT_TRAIN_ROLES,
) -> TrainingRow:
    row_format = detect_row_format(row)
    roles = {str(role).lower() for role in train_roles}
    texts: List[str] = []
    prompt_texts: List[str] = []
    if row_format.split("+", 1)[0] == RAW_TEXT_FORMAT:
        # Prepared raw-text rows carry one rendered document and no roles.
        return TrainingRow(
            row_id=f"{dataset}:{line_number}",
            dataset=dataset,
            line_number=line_number,
            row_format=row_format,
            text=str(row["text"]),
            prompt_texts=[],
        )
    for message in row_messages(row, row_format):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "")).lower()
        text = _message_text(message)
        if role == "user" and text:
            prompt_texts.append(text)
        if role in roles and text:
            texts.append(text)
    return TrainingRow(
        row_id=f"{dataset}:{line_number}",
        dataset=dataset,
        line_number=line_number,
        row_format=row_format,
        text="\n".join(texts),
        prompt_texts=prompt_texts,
    )


def iter_jsonl(path: Path) -> Iterator[Tuple[int, str, Dict[str, Any]]]:
    """Yield ``(line_number, raw_line, record)`` for non-blank lines; fail loudly on bad JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield line_number, raw, record


def load_training_rows(
    path: Path,
    *,
    label: Optional[str] = None,
    train_roles: Sequence[str] = DEFAULT_TRAIN_ROLES,
) -> List[TrainingRow]:
    dataset = label or str(path)
    rows: List[TrainingRow] = []
    for line_number, _, record in iter_jsonl(path):
        try:
            rows.append(
                build_training_row(record, dataset=dataset, line_number=line_number, train_roles=train_roles)
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


# ---------------------------------------------------------------------------
# Eval items
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    source: str
    item_id: str
    kind: str  # "prompt" | "reference" | "text"
    text: str
    exact_text: Optional[str] = None  # text compared for exact-duplicate prompts


def _string_leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _string_leaves(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _string_leaves(nested)


def items_from_prompt_cases(
    cases: Iterable[Any],
    *,
    source: str,
    include_system: bool = False,
    reference_fields: Sequence[str] = (),
) -> List[EvalItem]:
    """Turn Evaluator ``PromptCase`` objects into eval items.

    The prompt text is every user turn of ``case.chat_messages()`` (plus the
    system turn when ``include_system``); reference text is every string leaf
    under each configured dot-path into ``case.metadata``.
    """
    items: List[EvalItem] = []
    for index, case in enumerate(cases):
        case_id = str(getattr(case, "case_id", "") or f"case_{index + 1:04d}")
        roles = {"user", "system"} if include_system else {"user"}
        prompt_parts = [
            str(message.get("content") or "")
            for message in case.chat_messages()
            if str(message.get("role", "")).lower() in roles
        ]
        prompt_text = "\n".join(part for part in prompt_parts if part.strip())
        if prompt_text.strip():
            items.append(
                EvalItem(
                    source=source,
                    item_id=case_id,
                    kind="prompt",
                    text=prompt_text,
                    exact_text=str(getattr(case, "question", "") or "") or prompt_text,
                )
            )
        metadata = getattr(case, "metadata", {}) or {}
        reference_parts: List[str] = []
        for dotted in reference_fields:
            value = get_path(metadata, dotted)
            if value is MISSING:
                continue
            reference_parts.extend(_string_leaves(value))
        if reference_parts:
            items.append(
                EvalItem(source=source, item_id=case_id, kind="reference", text="\n".join(reference_parts))
            )
    return items


def load_eval_source(
    path: Path,
    *,
    config_dir: Optional[Path] = None,
    include_system: bool = False,
    reference_fields: Sequence[str] = (),
) -> List[EvalItem]:
    """Load an Evaluator scenario YAML or prompt set (JSON/JSONL) with the Evaluator's loaders."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval source not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        from Evaluator.config_loader import ConfigLoader

        resolved = path.resolve()
        if config_dir is None:
            config_dir = resolved.parent.parent if resolved.parent.name == "scenarios" else resolved.parent
        cases = ConfigLoader(config_dir).load_all_scenarios([str(resolved)])
    elif suffix in {".json", ".jsonl"}:
        from Evaluator.prompt_sets import load_prompt_cases

        cases = load_prompt_cases(path)
    else:
        raise ValueError(f"Unsupported eval source type {suffix!r}: {path} (expected .yaml/.yml/.json/.jsonl)")
    return items_from_prompt_cases(
        cases, source=str(path), include_system=include_system, reference_fields=reference_fields
    )


def load_text_source(path: Path, *, text_field: str = "text", id_field: str = "id") -> List[EvalItem]:
    """Load a plain JSONL of eval text (one object per line with a text field)."""
    items: List[EvalItem] = []
    for line_number, _, record in iter_jsonl(Path(path)):
        value = get_path(record, text_field)
        if value is MISSING or not isinstance(value, str):
            raise ValueError(f"{path}:{line_number}: missing string field {text_field!r}")
        item_id = get_path(record, id_field)
        items.append(
            EvalItem(
                source=str(path),
                item_id=str(item_id) if item_id is not MISSING and item_id is not None else f"line_{line_number}",
                kind="text",
                text=value,
                exact_text=value,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------

class NgramIndex:
    """Inverted index: n-gram hash -> ids of training rows containing it.

    ``lengths`` are the gram sizes to index (the main n plus any whole-item
    sizes for short eval items). With ``restrict_to`` only hashes in that set
    are stored, which keeps the index proportional to the eval set.
    """

    def __init__(self, lengths: Iterable[int], restrict_to: Optional[Set[int]] = None):
        self.lengths = sorted({int(n) for n in lengths})
        if not self.lengths or self.lengths[0] <= 0:
            raise ValueError("NgramIndex needs at least one positive gram length")
        self.restrict_to = restrict_to
        self.postings: Dict[int, List[int]] = {}
        self.row_count = 0

    def add(self, row_index: int, text: str) -> None:
        tokens = tokenize(text)
        grams: Set[int] = set()
        for n in self.lengths:
            grams |= ngram_hashes(tokens, n)
        if self.restrict_to is not None:
            grams &= self.restrict_to
        for gram in grams:
            self.postings.setdefault(gram, []).append(row_index)
        self.row_count += 1

    def match_counts(self, item_grams: Iterable[int]) -> Counter:
        """Per-row count of the item's grams present in that row."""
        counts: Counter = Counter()
        for gram in item_grams:
            rows = self.postings.get(gram)
            if rows:
                counts.update(rows)
        return counts


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

@dataclass
class ContaminationConfig:
    ngram: int = DEFAULT_NGRAM
    threshold: float = DEFAULT_THRESHOLD
    min_item_tokens: int = DEFAULT_MIN_ITEM_TOKENS
    top_k: int = DEFAULT_TOP_K
    train_roles: List[str] = field(default_factory=lambda: list(DEFAULT_TRAIN_ROLES))

    def validate(self) -> None:
        if self.ngram <= 0:
            raise ValueError(f"ngram must be positive, got {self.ngram}")
        if not 0 < self.threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.min_item_tokens <= 0:
            raise ValueError(f"min_item_tokens must be positive, got {self.min_item_tokens}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")


@dataclass
class ItemResult:
    source: str
    item_id: str
    kind: str
    n_tokens: int
    gram_size: Optional[int]
    total_ngrams: int
    max_containment: float
    best_row_id: Optional[str]
    matched_ngrams: int
    exact_duplicate_row_ids: List[str]
    flagged: bool
    excerpt: str


def _excerpt(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _item_grams(tokens: Sequence[str], cfg: ContaminationConfig) -> Tuple[Optional[int], Set[int]]:
    if len(tokens) >= cfg.ngram:
        return cfg.ngram, ngram_hashes(tokens, cfg.ngram)
    if len(tokens) >= cfg.min_item_tokens:
        return len(tokens), ngram_hashes(tokens, len(tokens))
    return None, set()


def run_check(
    training_rows: Sequence[TrainingRow],
    eval_items: Sequence[EvalItem],
    cfg: ContaminationConfig,
) -> Dict[str, Any]:
    """Score every eval item against the training rows and build the report dict."""
    cfg.validate()

    prepared: List[Tuple[EvalItem, List[str], Optional[int], Set[int]]] = []
    restrict: Set[int] = set()
    lengths: Set[int] = {cfg.ngram}
    exact_targets: Dict[str, List[int]] = {}
    for item_index, item in enumerate(eval_items):
        tokens = tokenize(item.text)
        gram_size, grams = _item_grams(tokens, cfg)
        if gram_size is not None:
            lengths.add(gram_size)
        restrict |= grams
        prepared.append((item, tokens, gram_size, grams))
        if item.exact_text:
            key = normalize_text(item.exact_text)
            if key:
                exact_targets.setdefault(key, []).append(item_index)

    index = NgramIndex(lengths, restrict_to=restrict)
    exact_hits: Dict[int, List[int]] = {}
    for row_index, row in enumerate(training_rows):
        index.add(row_index, row.text)
        for prompt in row.prompt_texts:
            key = normalize_text(prompt)
            for item_index in exact_targets.get(key, ()):
                hits = exact_hits.setdefault(item_index, [])
                if not hits or hits[-1] != row_index:
                    hits.append(row_index)

    results: List[ItemResult] = []
    # Bounded min-heap of the top_k (eval item, training row) pairs; ordering key
    # is (containment, matched n-grams, -item index, -row index).
    top_pairs: List[Tuple[float, int, int, int]] = []
    flagged_rows: Dict[int, List[Dict[str, Any]]] = {}
    for item_index, (item, tokens, gram_size, grams) in enumerate(prepared):
        counts = index.match_counts(grams) if grams else Counter()
        best_row, best_hits = (None, 0)
        if counts:
            best_row, best_hits = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        max_cont = best_hits / len(grams) if grams else 0.0
        exact_rows = exact_hits.get(item_index, [])
        flagged = bool(exact_rows) or (bool(grams) and max_cont >= cfg.threshold)

        for row_index, hits in counts.items():
            value = hits / len(grams)
            if cfg.top_k:
                entry = (value, hits, -item_index, -row_index)
                if len(top_pairs) < cfg.top_k:
                    heapq.heappush(top_pairs, entry)
                elif entry > top_pairs[0]:
                    heapq.heapreplace(top_pairs, entry)
            if value >= cfg.threshold:
                flagged_rows.setdefault(row_index, []).append(
                    {
                        "eval_source": item.source,
                        "eval_item_id": item.item_id,
                        "kind": item.kind,
                        "reason": "containment",
                        "containment": round(value, 4),
                    }
                )
        for row_index in exact_rows:
            flagged_rows.setdefault(row_index, []).append(
                {
                    "eval_source": item.source,
                    "eval_item_id": item.item_id,
                    "kind": item.kind,
                    "reason": "exact_duplicate_prompt",
                }
            )

        results.append(
            ItemResult(
                source=item.source,
                item_id=item.item_id,
                kind=item.kind,
                n_tokens=len(tokens),
                gram_size=gram_size,
                total_ngrams=len(grams),
                max_containment=round(max_cont, 4),
                best_row_id=training_rows[best_row].row_id if best_row is not None else None,
                matched_ngrams=best_hits,
                exact_duplicate_row_ids=[training_rows[i].row_id for i in exact_rows],
                flagged=flagged,
                excerpt=_excerpt(item.text),
            )
        )

    pairs = []
    for value, hits, neg_item, neg_row in sorted(top_pairs, reverse=True):
        item = eval_items[-neg_item]
        pairs.append(
            {
                "eval_source": item.source,
                "eval_item_id": item.item_id,
                "kind": item.kind,
                "containment": round(value, 4),
                "matched_ngrams": hits,
                "total_ngrams": len(prepared[-neg_item][3]),
                "train_row_id": training_rows[-neg_row].row_id,
            }
        )
    scored = [r for r in results if r.total_ngrams > 0]
    buckets = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    histogram: Dict[str, int] = {}
    for low, high in zip(buckets, buckets[1:]):
        label = f"[{low:.2f},{high:.2f})" if high < 1.0 else f"[{low:.2f},1.00]"
        histogram[label] = sum(
            1 for r in scored if low <= r.max_containment < high or (high == 1.0 and r.max_containment == 1.0)
        )

    format_counts: Dict[str, Dict[str, int]] = {}
    for row in training_rows:
        per_dataset = format_counts.setdefault(row.dataset, {})
        per_dataset[row.row_format] = per_dataset.get(row.row_format, 0) + 1

    flagged_items = [r for r in results if r.flagged]
    summary = {
        "training_rows": len(training_rows),
        "eval_items": len(results),
        "scored_items": len(scored),
        "too_short_items": len(results) - len(scored),
        "items_with_any_overlap": sum(1 for r in scored if r.matched_ngrams > 0),
        "items_over_threshold": sum(1 for r in scored if r.max_containment >= cfg.threshold),
        "exact_duplicate_prompts": sum(1 for r in results if r.exact_duplicate_row_ids),
        "flagged_items": len(flagged_items),
        "flagged_training_rows": len(flagged_rows),
        "max_containment": max((r.max_containment for r in scored), default=0.0),
        "mean_containment": round(sum(r.max_containment for r in scored) / len(scored), 4) if scored else 0.0,
        "indexed_ngrams": len(index.postings),
    }
    return {
        "config": asdict(cfg),
        "passed": not flagged_items,
        "summary": summary,
        "histogram": histogram,
        "training_formats": format_counts,
        "flagged_items": [asdict(r) for r in sorted(flagged_items, key=lambda r: -r.max_containment)],
        "top_pairs": pairs,
        "items": [asdict(r) for r in results],
        "flagged_training_rows": {
            training_rows[i].row_id: reasons for i, reasons in sorted(flagged_rows.items())
        },
    }


# ---------------------------------------------------------------------------
# Decontaminated output
# ---------------------------------------------------------------------------

def write_decontaminated(
    dataset_path: Path,
    dataset_label: str,
    flagged_training_rows: Mapping[str, List[Dict[str, Any]]],
    output_path: Path,
) -> Dict[str, Any]:
    """Write ``dataset_path`` minus flagged rows (raw lines preserved) plus a sidecar.

    The sidecar ``<output>.removed.json`` lists every removed row id, its source
    line number and the reasons it was flagged.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    removed: List[Dict[str, Any]] = []
    kept = 0
    with output_path.open("w", encoding="utf-8") as out:
        for line_number, raw, _ in iter_jsonl(Path(dataset_path)):
            row_id = f"{dataset_label}:{line_number}"
            reasons = flagged_training_rows.get(row_id)
            if reasons:
                removed.append({"row_id": row_id, "line_number": line_number, "reasons": reasons})
                continue
            out.write(raw if raw.endswith("\n") else raw + "\n")
            kept += 1
    sidecar = output_path.with_name(output_path.name + ".removed.json")
    payload = {
        "source_dataset": str(dataset_path),
        "output": str(output_path),
        "kept_rows": kept,
        "removed_rows": len(removed),
        "removed": removed,
    }
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"output": str(output_path), "sidecar": str(sidecar), "kept_rows": kept, "removed_rows": len(removed)}


def dataset_labels(paths: Sequence[Path]) -> List[str]:
    """Short unique labels for training datasets (file names, disambiguated by parent dirs)."""
    labels: List[str] = []
    for path in paths:
        parts = Path(path).parts
        depth = 1
        label = "/".join(parts[-depth:])
        while any(
            "/".join(Path(other).parts[-depth:]) == label for other in paths if Path(other) != Path(path)
        ) and depth < len(parts):
            depth += 1
            label = "/".join(parts[-depth:])
        labels.append(label)
    return labels
