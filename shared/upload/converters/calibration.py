"""
Calibration text rendering for llama.cpp importance matrices (imatrix).

``llama-imatrix`` reads a plain-text file, tokenizes it and runs it through the
base GGUF to measure which weights matter most. Low-bit quantizations
(Q4 and below, and the IQ* types) use that importance matrix to keep the
important weights precise.

This module turns a JSONL training dataset into that text file. It is
format-agnostic: rows are normalized through the repo's canonical SFT row
normalizer (``shared.sft_preprocessing``), so any row shape the trainers accept
(``messages``, ``conversations``, ``prompt``/``completion``) works here too.

Rendering prefers the merged model's own tokenizer chat template, so the
calibration text matches the prompt format the quantized model will see at
inference time. Without a usable chat template, the message contents are joined
as plain text.

Usage:
    python -m shared.upload.converters.calibration dataset.jsonl calibration.txt \
        --tokenizer-dir path/to/merged_model --max-rows 512
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.sft_preprocessing import (
    normalize_sft_messages,
    sanitize_messages_for_chat_template,
)

DEFAULT_MAX_ROWS = 512
DEFAULT_MAX_CHARS = 4_000_000
DEFAULT_SEED = 0
DEFAULT_IMATRIX_CHUNKS = 128
DEFAULT_IMATRIX_CTX = 512

RENDER_CHAT_TEMPLATE = "chat_template"
RENDER_PLAIN = "plain"

ROW_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class CalibrationSpec:
    """What calibration data to use and how to compute the imatrix from it.

    Attributes:
        dataset_path: Local JSONL dataset to render.
        source: Human-readable origin recorded in the GGUF manifest
            (e.g. ``hf://datasets/org/repo/train.jsonl``). Defaults to the path.
        max_rows: Maximum number of rows rendered into the calibration text.
        max_chars: Character budget for the rendered calibration text.
        seed: Seed for the deterministic row sample.
        chunks: Maximum number of ``ctx_size`` token chunks llama-imatrix
            processes (None processes everything).
        ctx_size: Context size per imatrix chunk.
        label_field: Optional row field that marks a row desirable (truthy) or
            undesirable (falsy). When set, rows whose value is falsy are left
            out. Unset means every renderable row is used.
    """

    dataset_path: Path
    source: Optional[str] = None
    max_rows: int = DEFAULT_MAX_ROWS
    max_chars: int = DEFAULT_MAX_CHARS
    seed: int = DEFAULT_SEED
    chunks: Optional[int] = DEFAULT_IMATRIX_CHUNKS
    ctx_size: int = DEFAULT_IMATRIX_CTX
    label_field: Optional[str] = None

    def __post_init__(self):
        if self.max_rows < 1:
            raise ValueError("Calibration max_rows must be at least 1.")
        if self.max_chars < 1:
            raise ValueError("Calibration max_chars must be at least 1.")
        if self.chunks is not None and self.chunks < 1:
            raise ValueError("Calibration chunks must be at least 1 (or None for all).")
        if self.ctx_size < 1:
            raise ValueError("Calibration ctx_size must be at least 1.")

    @property
    def source_label(self) -> str:
        # Local paths are recorded by file name only: the manifest is published
        # with the GGUFs and must not expose private directory layouts.
        return self.source or Path(self.dataset_path).name


@dataclass
class CalibrationRenderResult:
    """Outcome of rendering a dataset into calibration text."""

    output_path: Path
    source: str
    text_sha256: str
    render_mode: str
    rows_total: int
    rows_used: int
    rows_skipped: int
    rows_filtered: int
    chars: int
    seed: int
    max_rows: int
    max_chars: int
    label_field: Optional[str]

    def to_manifest(self) -> Dict[str, Any]:
        """Stable snake_case keys recorded in gguf_manifest.json."""
        return {
            "source": self.source,
            "text_sha256": self.text_sha256,
            "render_mode": self.render_mode,
            "rows_total": self.rows_total,
            "rows_used": self.rows_used,
            "rows_skipped": self.rows_skipped,
            "rows_filtered": self.rows_filtered,
            "chars": self.chars,
            "seed": self.seed,
            "max_rows": self.max_rows,
            "max_chars": self.max_chars,
            "label_field": self.label_field,
        }


def load_chat_tokenizer(tokenizer_dir: Path) -> Optional[Any]:
    """Load a tokenizer that has a chat template, or None if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    except Exception:
        return None
    if not getattr(tokenizer, "chat_template", None):
        return None
    return tokenizer


def render_row(record: Dict[str, Any], tokenizer: Optional[Any]) -> str:
    """Render one dataset row into calibration text.

    Raises:
        ValueError: If the row cannot be normalized or renders to nothing.
    """
    messages, _ = normalize_sft_messages(record)
    messages = sanitize_messages_for_chat_template(messages)
    if not messages:
        raise ValueError("Row has no messages.")

    if tokenizer is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = "\n\n".join(
            str(message.get("content") or "").strip() for message in messages
        ).strip()

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Row rendered to empty text.")
    return text


def _is_filtered(record: Dict[str, Any], label_field: Optional[str]) -> bool:
    if not label_field or label_field not in record:
        return False
    return not bool(record[label_field])


def render_calibration_text(
    dataset_path: Path,
    output_path: Path,
    *,
    tokenizer_dir: Optional[Path] = None,
    tokenizer: Optional[Any] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_chars: int = DEFAULT_MAX_CHARS,
    seed: int = DEFAULT_SEED,
    label_field: Optional[str] = None,
    source: Optional[str] = None,
) -> CalibrationRenderResult:
    """Render a JSONL dataset into a plain-text calibration file.

    Rows are sampled deterministically: all rows are shuffled with ``seed`` and
    rendered in that order until ``max_rows`` rows are used or the next row
    would exceed ``max_chars``. Rows that are not valid JSON objects or cannot
    be normalized/rendered are skipped and counted.

    Args:
        dataset_path: JSONL dataset.
        output_path: Where to write the calibration text.
        tokenizer_dir: Model directory whose tokenizer chat template is used.
        tokenizer: Pre-loaded tokenizer (takes precedence over tokenizer_dir).
        max_rows: Row cap.
        max_chars: Character budget.
        seed: Sampling seed.
        label_field: Optional desirability field; rows with a falsy value are
            left out (see CalibrationSpec).
        source: Origin label for the manifest (defaults to dataset_path).

    Returns:
        CalibrationRenderResult with counts and the text sha256.

    Raises:
        ValueError: If no row could be rendered.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if tokenizer is None and tokenizer_dir is not None:
        tokenizer = load_chat_tokenizer(Path(tokenizer_dir))
    if tokenizer is not None and not getattr(tokenizer, "chat_template", None):
        tokenizer = None
    render_mode = RENDER_CHAT_TEMPLATE if tokenizer is not None else RENDER_PLAIN

    lines = [
        line for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)

    parts: List[str] = []
    chars = 0
    rows_skipped = 0
    rows_filtered = 0

    for index in order:
        if len(parts) >= max_rows:
            break
        try:
            record = json.loads(lines[index])
        except json.JSONDecodeError:
            rows_skipped += 1
            continue
        if not isinstance(record, dict):
            rows_skipped += 1
            continue
        if _is_filtered(record, label_field):
            rows_filtered += 1
            continue
        try:
            text = render_row(record, tokenizer)
        except Exception:
            rows_skipped += 1
            continue

        added = len(text) + (len(ROW_SEPARATOR) if parts else 0)
        if parts and chars + added > max_chars:
            break
        parts.append(text)
        chars += added

    if not parts:
        raise ValueError(
            f"No calibration rows could be rendered from {dataset_path} "
            f"({len(lines)} rows, {rows_skipped} skipped, {rows_filtered} filtered)."
        )

    text = ROW_SEPARATOR.join(parts) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    return CalibrationRenderResult(
        output_path=output_path,
        source=source or Path(dataset_path).name,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        render_mode=render_mode,
        rows_total=len(lines),
        rows_used=len(parts),
        rows_skipped=rows_skipped,
        rows_filtered=rows_filtered,
        chars=len(text),
        seed=seed,
        max_rows=max_rows,
        max_chars=max_chars,
        label_field=label_field,
    )


def main():
    """CLI entry point to preview the rendered calibration text."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Render a JSONL dataset into llama-imatrix calibration text"
    )
    parser.add_argument("dataset", help="JSONL dataset path")
    parser.add_argument("output", help="Output text file")
    parser.add_argument("--tokenizer-dir", help="Model dir whose chat template is applied")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--label-field", help="Leave out rows where this field is falsy")
    args = parser.parse_args()

    result = render_calibration_text(
        Path(args.dataset),
        Path(args.output),
        tokenizer_dir=Path(args.tokenizer_dir) if args.tokenizer_dir else None,
        max_rows=args.max_rows,
        max_chars=args.max_chars,
        seed=args.seed,
        label_field=args.label_field,
    )
    print(json.dumps(result.to_manifest(), indent=2))


if __name__ == "__main__":
    main()
