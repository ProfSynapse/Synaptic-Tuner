"""Native structured tool-call rendering + empty-think LABEL masking.

Two shared-code behaviors that the agentic tool-trajectory SFT rows depend on:

1. ``tool_call_mode`` in ``sanitize_messages_for_chat_template`` /
   ``materialize_sft_example`` (default "render_text", so every existing prose
   caller is byte-identical):
     - "render_text" (default, LEGACY): fold structured tool_calls into assistant
       PROSE text and POP the key — unchanged historical behavior.
     - "native": PASS structured tool_calls THROUGH to the chat template so it
       renders native ``<tool_call>`` markup, and coerce ``arguments`` from a
       JSON STRING (Codex) to an OBJECT (the Qwen template calls ``.items()`` on
       it and crashes on a string).
   SAFETY: rows WITHOUT a structured tool_calls field (every old prose dataset)
   are identical under both modes — the fold/passthrough only fires when a
   structured tool_calls field is present.

2. EMPTY-THINK LABEL MASKING (unconditional in ``loss_mask_mode="assistant_only"``):
   we render the STOCK chat template (no train/inference skew) and, within each
   supervised assistant span, re-mask a LEADING empty ``<think></think>`` block
   (whitespace-only body) back to -100 — empty think is never worth training. A
   FILLED think block (real reasoning) stays TRAINED. No flag; it is the default
   behavior of assistant_only.

A network-gated check (RUN_LIVE_HUB=1) proves the end-to-end contract against the
real Qwen3.5-4B tokenizer: native ``<tool_call>`` tokens are TRAINED and ORIGINATE
from the structured field (not prose), tool_response is MASKED, real reasoning is
TRAINED, and ZERO empty-think tokens are trained.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

from shared.sft_preprocessing import (
    materialize_sft_example,
    sanitize_messages_for_chat_template,
)


# ---------------------------------------------------------------------------
# tool_call_mode — sanitize-level unit tests (no tokenizer needed).
# ---------------------------------------------------------------------------
_TOOL_MSG = {
    "role": "assistant",
    "content": "thinking out loud",
    "tool_calls": [
        {"type": "function", "function": {"name": "Read", "arguments": {"file_path": "x.py"}}}
    ],
}


def test_render_text_folds_tool_calls_to_prose_and_pops_key():
    out = sanitize_messages_for_chat_template([dict(_TOOL_MSG)], tool_call_mode="render_text")
    assert "tool_calls" not in out[0], "legacy mode must POP the structured key"
    assert "tool_call: Read" in out[0]["content"], "legacy mode folds the call into prose"


def test_native_preserves_structured_tool_calls():
    out = sanitize_messages_for_chat_template([dict(_TOOL_MSG)], tool_call_mode="native")
    assert "tool_calls" in out[0], "native mode must PRESERVE the structured key"
    # content is NOT prose-folded in native mode.
    assert "tool_call: Read" not in out[0]["content"]
    assert out[0]["tool_calls"][0]["function"]["name"] == "Read"


def test_native_normalizes_json_string_arguments_to_object():
    """Codex emits arguments as a JSON STRING; the Qwen template requires an
    OBJECT (it calls .items()). Native mode must coerce string→dict."""
    codex_style = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "Read", "arguments": '{"file_path": "x.py"}'}}
        ],
    }
    out = sanitize_messages_for_chat_template([dict(codex_style)], tool_call_mode="native")
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict) and args == {"file_path": "x.py"}


def test_tool_call_mode_is_noop_for_prose_data_safety():
    """SAFETY (required): a row with NO structured tool_calls is byte-identical
    under both modes — so flipping the default never changes existing datasets."""
    prose = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Tool calls: this is legacy prose, not structured"},
    ]
    rt = sanitize_messages_for_chat_template([dict(m) for m in prose], tool_call_mode="render_text")
    nat = sanitize_messages_for_chat_template([dict(m) for m in prose], tool_call_mode="native")
    assert rt == nat == prose


# ---------------------------------------------------------------------------
# empty-think LABEL masking — STOCK-template render with a Qwen-like fake.
# ---------------------------------------------------------------------------
class _EmptyThinkInjectingTokenizer:
    """Fake that mimics the Qwen3.5 template's behavior of injecting
    ``<think>\\n\\n</think>\\n\\n`` into EVERY assistant turn that has no
    reasoning_content, and ``<think>\\n{reasoning}\\n</think>\\n\\n`` when it does.
    Renders native <tool_call> as a simple sentinel when tool_calls are present.
    Char-level encode with atomic ChatML control + role tokens (so assistant_only
    masking works)."""

    _SPECIALS = {
        "<|im_start|>": -1, "<|im_end|>": -2,
        "assistant": -10, "system": -11, "user": -12, "tool": -13,
    }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        parts = []
        for m in messages:
            body = m.get("content") or ""
            if m["role"] == "assistant":
                reasoning = m.get("reasoning_content")
                think = f"<think>\n{reasoning}\n</think>\n\n" if reasoning else "<think>\n\n</think>\n\n"
                tc = ""
                if m.get("tool_calls"):
                    name = m["tool_calls"][0]["function"]["name"]
                    tc = f"<tool_call>{name}</tool_call>"
                body = f"{think}{body}{tc}"
            parts.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids, i = [], 0
        while i < len(text):
            for marker, mid in self._SPECIALS.items():
                if text.startswith(marker, i):
                    ids.append(mid)
                    i += len(marker)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids):
        rev = {v: k for k, v in self._SPECIALS.items()}
        return "".join(rev[i] if i in rev else chr(i) for i in ids)


def _trained_decoded(prepared, tok):
    return tok.decode([t for t, l in zip(prepared.input_ids, prepared.labels) if l != -100])


def test_assistant_only_masks_empty_think_keeps_real_reasoning():
    """assistant_only renders the STOCK template (empty think stays IN the text)
    but MASKS the empty-think tokens at the label level, while a filled think
    block (real reasoning) is TRAINED. The answer + tool-call tokens are trained
    either way."""
    tok = _EmptyThinkInjectingTokenizer()
    record = {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "step one", "reasoning_content": "REASONING"},
        {"role": "assistant", "content": "step two"},  # traceless → empty think
    ]}
    prepared = materialize_sft_example(
        tokenizer=tok, record=record, max_seq_length=10_000,
        loss_mask_mode="assistant_only",
    )
    full = tok.decode(prepared.input_ids)
    # STOCK render: BOTH the empty think and the real reasoning are present in the
    # text (no stripping — zero train/inference skew).
    assert "<think>\n\n</think>" in full, "stock template keeps empty think in the text"
    assert "<think>\nREASONING\n</think>" in full, "real reasoning is in the text"

    trained = _trained_decoded(prepared, tok)
    # Real reasoning TRAINED; the empty-think block NOT trained; answers trained.
    assert "REASONING" in trained
    assert "<think>\n\n</think>" not in trained, "empty think must be masked out of the loss"
    assert "step one" in trained and "step two" in trained


def test_full_sequence_does_not_mask_empty_think():
    """Empty-think masking is scoped to assistant_only. Under full_sequence every
    token is supervised — including the empty think the template injects — so the
    masking change does not touch the train-everything mode."""
    tok = _EmptyThinkInjectingTokenizer()
    record = {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "answer"},  # traceless → empty think
    ]}
    prepared = materialize_sft_example(
        tokenizer=tok, record=record, max_seq_length=10_000,
        loss_mask_mode="full_sequence",
    )
    trained = _trained_decoded(prepared, tok)
    assert "<think>\n\n</think>" in trained, "full_sequence trains everything (incl. empty think)"


def test_assistant_only_masks_empty_think_on_tool_call_turn():
    """A tool-call turn with no reasoning still gets the injected empty think;
    its think tokens are masked while the native <tool_call> stays trained."""
    tok = _EmptyThinkInjectingTokenizer()
    record = {"messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "ReadX", "arguments": {"file_path": "x.py"}}}]},
        {"role": "tool", "content": "TOOLOUT body"},
        {"role": "assistant", "content": "done"},  # traceless final
    ]}
    prepared = materialize_sft_example(
        tokenizer=tok, record=record, max_seq_length=10_000,
        loss_mask_mode="assistant_only", tool_call_mode="native",
    )
    trained = _trained_decoded(prepared, tok)
    assert "<tool_call>ReadX</tool_call>" in trained, "native tool call trained"
    assert "<think>\n\n</think>" not in trained, "empty think masked on the tool-call turn"
    assert "TOOLOUT" not in trained, "tool output masked"
    assert "done" in trained


# ---------------------------------------------------------------------------
# Network-gated end-to-end proof against the real Qwen3.5-4B tokenizer.
# ---------------------------------------------------------------------------
_LIVE_HUB = os.environ.get("RUN_LIVE_HUB") == "1"
_QWEN_MODEL = "Qwen/Qwen3.5-4B"


@pytest.mark.skipif(
    not _LIVE_HUB,
    reason="network-gated; set RUN_LIVE_HUB=1 to download the Qwen3.5 tokenizer",
)
def test_native_tool_call_trained_from_structured_field_real_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_QWEN_MODEL)
    # Codex-style JSON-string arguments to also exercise the normalize path.
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug"},
        {"role": "assistant", "content": "", "reasoning_content": "REASONMARKER read first",
         "tool_calls": [{"type": "function",
                         "function": {"name": "ReadTool", "arguments": '{"file_path": "x.py"}'}}]},
        {"role": "tool", "content": "TOOLRESPMARKER file body"},
        {"role": "assistant", "content": "FINALMARKER fixed"},  # traceless
    ]
    prepared = materialize_sft_example(
        tokenizer=tokenizer, record={"messages": messages}, max_seq_length=4096,
        loss_mask_mode="assistant_only", tool_call_mode="native",
    )
    full = tokenizer.decode(prepared.input_ids)

    # (a) native <tool_call> markup is present (NOT a "tool_call: Read" prose fold)
    # and the argument object rendered (proving JSON-string→object normalize).
    assert "<tool_call>" in full and "<function=ReadTool>" in full
    assert "tool_call: ReadTool" not in full  # no prose fold
    assert "<parameter=file_path>" in full
    # STOCK template: the empty think the template injects on the traceless final
    # turn IS in the rendered text (no stripping → zero skew). (d) below proves it
    # is MASKED out of the loss rather than removed from the text.
    assert "<think>\n\n</think>" in full

    enc = getattr(tokenizer, "tokenizer", tokenizer)
    trained_ids = [t for t, l in zip(prepared.input_ids, prepared.labels) if l != -100]
    trained_text = tokenizer.decode(trained_ids)
    # (d) ZERO empty-think tokens are TRAINED (masked at the label level).
    assert "<think>\n\n</think>" not in trained_text

    def span(marker):
        a = full.find(marker)
        assert a >= 0, f"{marker!r} not in render"
        pre = len(enc.encode(full[:a], add_special_tokens=False))
        seg = len(enc.encode(marker, add_special_tokens=False))
        n = len(prepared.labels)
        b = min(pre + seg, n)
        return sum(1 for l in prepared.labels[min(pre, n):b] if l != -100), (b - min(pre, n))

    fn_tr, fn_tot = span("<function=ReadTool>")
    reason_tr, reason_tot = span("REASONMARKER")
    final_tr, final_tot = span("FINALMARKER")
    resp_tr, _ = span("TOOLRESPMARKER")
    assert fn_tr == fn_tot and fn_tot > 0          # (a) tool-call trained
    assert resp_tr == 0                            # (b) tool_response masked
    assert reason_tr == reason_tot and reason_tot > 0  # (c) reasoning trained
    assert final_tr == final_tot and final_tot > 0
