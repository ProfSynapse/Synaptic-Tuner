"""End-to-end `tool_call_mode` thread through the SFT DATA LOADER.

The shared layer (`materialize_sft_example` / `sanitize_messages_for_chat_template`)
already honored `tool_call_mode` (see test_native_tool_calls.py). The DEFECT this
covers was higher up: `load_and_prepare_tokenized_dataset` (and the
`load_and_prepare_sft_dataset` / `prepare_sft_dataset` it calls) did NOT accept or
thread `tool_call_mode`, so every loader-driven run silently defaulted to
"render_text" and folded native `tool_calls` into prose — wasting a native dataset.

These tests drive the PUBLIC loader entry point end-to-end (a temp JSONL of rows
carrying structured `tool_calls`, through `load_and_prepare_tokenized_dataset`)
and assert:

  * tool_call_mode="native"  -> native `<tool_call>` markup lands in the TRAINED
    label span (loss_mask_mode="assistant_only"), NOT a "tool_call:" prose fold.
  * tool_call_mode="render_text" (the OLD default) -> the structured call is folded
    to prose and the native `<tool_call>` markup is ABSENT.

The render_text counter-case is what makes the native assertion meaningful: it
proves the loader actually routes the flag, not that the markup would appear
regardless. A fake Qwen-like tokenizer keeps this hermetic (no model download).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

from data_loader import load_and_prepare_tokenized_dataset  # noqa: E402

# Reuse the Qwen-like fake that renders native <tool_call>NAME</tool_call> when a
# structured tool_calls field is present (and injects the empty-think block the
# stock template would). Single source of truth with the shared-layer test.
from test_native_tool_calls import _EmptyThinkInjectingTokenizer  # noqa: E402


def _trained_decoded(prepared, tok):
    return tok.decode([t for t, l in zip(prepared["input_ids"], prepared["labels"]) if l != -100])


_NATIVE_ROW = {
    "messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "ReadX", "arguments": {"file_path": "x.py"}}}]},
        {"role": "tool", "content": "TOOLOUT body"},
        {"role": "assistant", "content": "done"},
    ]
}


def _write_rows(tmp_path: Path, n: int = 2) -> str:
    p = tmp_path / "native_rows.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for _ in range(n):
            fh.write(json.dumps(_NATIVE_ROW, ensure_ascii=False) + "\n")
    return str(p)


def test_loader_native_mode_trains_native_tool_call_markup(tmp_path):
    """Through the PUBLIC loader: native + assistant_only puts the native
    <tool_call> markup in the trained label span (the defect was the loader
    dropping tool_call_mode so this folded to prose)."""
    local_file = _write_rows(tmp_path)
    tok = _EmptyThinkInjectingTokenizer()
    train, _ = load_and_prepare_tokenized_dataset(
        local_file=local_file,
        tokenizer=tok,
        max_seq_length=10_000,
        loss_mask_mode="assistant_only",
        tool_call_mode="native",
        split_dataset=False,
    )
    prepared = train[0]
    trained = _trained_decoded(prepared, tok)
    assert "<tool_call>ReadX</tool_call>" in trained, "native tool-call markup must be trained"
    assert "tool_call: ReadX" not in trained, "native mode must NOT prose-fold the call"
    assert "TOOLOUT" not in trained, "tool output stays masked"
    assert "done" in trained


def test_loader_render_text_mode_folds_to_prose_no_native_markup(tmp_path):
    """COUNTER-CASE: the OLD default (render_text) folds the structured call into
    prose and emits NO native <tool_call> markup — proving the loader genuinely
    routes the flag (the native assertion above is not vacuous)."""
    local_file = _write_rows(tmp_path)
    tok = _EmptyThinkInjectingTokenizer()
    train, _ = load_and_prepare_tokenized_dataset(
        local_file=local_file,
        tokenizer=tok,
        max_seq_length=10_000,
        loss_mask_mode="assistant_only",
        tool_call_mode="render_text",
        split_dataset=False,
    )
    prepared = train[0]
    full = tok.decode(prepared["input_ids"])
    # render_text pops the structured field and folds it to prose, so the fake
    # tokenizer (which only emits <tool_call> for a STRUCTURED field) renders none.
    assert "<tool_call>" not in full, "render_text must not emit native tool-call markup"
    assert "tool_call: ReadX" in full, "render_text folds the call into assistant prose"


def test_loader_defaults_to_render_text_when_mode_unset(tmp_path):
    """Back-compat: omitting tool_call_mode keeps the legacy render_text behavior
    so every existing loader call site is unchanged."""
    local_file = _write_rows(tmp_path)
    tok = _EmptyThinkInjectingTokenizer()
    train, _ = load_and_prepare_tokenized_dataset(
        local_file=local_file,
        tokenizer=tok,
        max_seq_length=10_000,
        loss_mask_mode="assistant_only",
        split_dataset=False,
    )
    full = tok.decode(train[0]["input_ids"])
    assert "<tool_call>" not in full
    assert "tool_call: ReadX" in full
