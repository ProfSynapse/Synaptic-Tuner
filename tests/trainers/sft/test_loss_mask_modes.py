"""Per-mode SFT loss-mask label spans on a multi-turn tool trajectory.

Covers the three loss-mask modes in ``shared.sft_preprocessing`` (task: make
SFT loss masking config-driven for multi-turn agentic tool-use):

  full_sequence   – every token supervised.
  completion_only – ONLY the final assistant turn (prompt-prefix mask). This is
                    the historical behavior formerly named "assistant_only".
                    Intermediate assistant/tool-call turns are MASKED.
  assistant_only  – EVERY assistant turn supervised (incl. intermediate
                    tool-call turns); system/user/tool spans MASKED.

The decisive property for agentic tool-use SFT: under ``assistant_only`` the
assistant tool-call turns land in the loss while the tool-RESPONSE (tool output)
turns are masked. ``completion_only`` masks every turn except the last — which
is exactly what made the prior trajectory rebuild a no-op in the loss.

A deterministic fake tokenizer keeps this in the normal (no-network) suite. It
renders an append-only ChatML-style transcript so render(messages[:k]) is a
token prefix of render(messages[:k+1]) — the invariant the per-span mask relies
on. A network-gated check against the real Qwen3.5 tokenizer (RUN_LIVE_HUB=1)
converts verified-by-fake into verified-by-execution.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

from shared.sft_preprocessing import materialize_sft_example


class _ChatMLFakeTokenizer:
    """Append-only ChatML fake. Each turn renders to
    ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` and the generation prompt to
    ``<|im_start|>assistant\\n``.

    encode() tokenizes the ChatML control strings ``<|im_start|>``/``<|im_end|>``
    AND the role words (``assistant``/``user``/``system``/``tool``) as ATOMIC
    single tokens (negative ids, disjoint from char ids), and every other
    character as one token (``ord(ch)``). This mirrors the real Qwen tokenizer
    (where ``assistant`` is a single token) so the assistant header renders as
    exactly the 3-token run ``[<|im_start|>, assistant, '\\n']`` that
    ``_mask_assistant_only_spans`` scans for, while plain text stays 1 char =
    1 token so a marker substring maps to a contiguous token run.
    """

    _IM_START = -1
    _IM_END = -2
    # Role words are atomic so the header is exactly [IM_START, ROLE, '\n'].
    _SPECIALS = {
        "<|im_start|>": _IM_START,
        "<|im_end|>": _IM_END,
        "assistant": -10,
        "system": -11,
        "user": -12,
        "tool": -13,
    }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids: list[int] = []
        i = 0
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


def _tok_index(input_ids, marker, tokenizer=None):
    """First token index where `marker`'s char-tokens begin in `input_ids`.

    Plain text is 1 char = 1 token in the fake, so marker chars form a
    contiguous token run; find it by scanning for the ord-sequence."""
    target = [ord(c) for c in marker]
    for start in range(len(input_ids) - len(target) + 1):
        if input_ids[start:start + len(target)] == target:
            return start
    raise AssertionError(f"{marker!r} not found in token stream")


# A multi-turn tool trajectory: 2 tool-call assistant turns + 1 final answer.
# Distinct markers per region so we can locate trained-vs-masked spans by string.
_TRAJECTORY = {
    "messages": [
        {"role": "system", "content": "SYSMARKER agent"},
        {"role": "user", "content": "USERMARKER fix it"},
        {"role": "assistant", "content": "CALLONE read the file"},
        {"role": "tool", "content": "RESPONE bug here"},
        {"role": "assistant", "content": "CALLTWO edit the file"},
        {"role": "tool", "content": "RESPTWO edited ok"},
        {"role": "assistant", "content": "FINAL fixed the bug"},
    ]
}


def _prepare(mode):
    return materialize_sft_example(
        tokenizer=_ChatMLFakeTokenizer(),
        record=_TRAJECTORY,
        max_seq_length=10_000,
        loss_mask_mode=mode,
    )


def _trained_str(prepared, tokenizer=_ChatMLFakeTokenizer()):
    """The decoded concatenation of the supervised (label != -100) tokens."""
    trained_ids = [t for t, l in zip(prepared.input_ids, prepared.labels) if l != -100]
    return tokenizer.decode(trained_ids)


def _marker_supervised(prepared, marker):
    """Is every token of `marker` within the supervised (label != -100) span?"""
    start = _tok_index(prepared.input_ids, marker)
    return all(l != -100 for l in prepared.labels[start:start + len(marker)])


# ---------------------------------------------------------------------------
# full_sequence: everything supervised.
# ---------------------------------------------------------------------------
def test_full_sequence_supervises_every_token():
    prepared = _prepare("full_sequence")
    assert prepared.loss_mask_mode == "full_sequence"
    assert all(l != -100 for l in prepared.labels)
    assert prepared.labels == prepared.input_ids


# ---------------------------------------------------------------------------
# completion_only: ONLY the final assistant turn.
# ---------------------------------------------------------------------------
def test_completion_only_supervises_only_final_turn():
    prepared = _prepare("completion_only")
    assert prepared.loss_mask_mode == "completion_only"

    # Final answer trained; everything before it masked.
    assert _marker_supervised(prepared, "FINAL fixed the bug")
    for masked in ("SYSMARKER", "USERMARKER", "CALLONE", "RESPONE", "CALLTWO", "RESPTWO"):
        assert not _marker_supervised(prepared, masked), f"{masked} should be masked"

    # The supervised text is exactly the final turn (plus its closing control).
    trained = _trained_str(prepared)
    assert "FINAL fixed the bug" in trained
    assert "CALLONE" not in trained and "CALLTWO" not in trained


# ---------------------------------------------------------------------------
# assistant_only: EVERY assistant turn; tool RESPONSES masked.  (the new mode)
# ---------------------------------------------------------------------------
def test_assistant_only_supervises_every_assistant_turn_masks_tool_responses():
    prepared = _prepare("assistant_only")
    assert prepared.loss_mask_mode == "assistant_only"

    # All three assistant turns (both tool-call turns + final) are supervised.
    for trained in ("CALLONE read the file", "CALLTWO edit the file", "FINAL fixed the bug"):
        assert _marker_supervised(prepared, trained), f"{trained} must be in the loss"

    # System, user, and BOTH tool-response (tool output) turns are masked.
    for masked in ("SYSMARKER", "USERMARKER", "RESPONE bug here", "RESPTWO edited ok"):
        assert not _marker_supervised(prepared, masked), f"{masked} must be masked"


def test_assistant_only_trains_strictly_more_than_completion_only():
    """The whole point: assistant_only puts the intermediate tool-call turns into
    the loss that completion_only throws away."""
    comp = _prepare("completion_only")
    asst = _prepare("assistant_only")
    comp_trained = sum(1 for l in comp.labels if l != -100)
    asst_trained = sum(1 for l in asst.labels if l != -100)
    assert asst_trained > comp_trained


def test_single_turn_completion_and_assistant_only_supervise_same_answer():
    """For a single user→assistant exchange the two modes supervise the same
    answer span — both train the answer content plus the closing <|im_end|> stop
    token (completion_only additionally trains the one trailing structural
    newline between turns; assistant_only stops at <|im_end|> inclusive). The
    meaningful invariant is that the answer + stop token are trained and the
    prompt stays masked under both — so swapping modes never silently drops or
    adds the answer for the common single-turn case."""
    single = {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world answer"},
    ]}
    comp = materialize_sft_example(
        tokenizer=_ChatMLFakeTokenizer(), record=single,
        max_seq_length=10_000, loss_mask_mode="completion_only",
    )
    asst = materialize_sft_example(
        tokenizer=_ChatMLFakeTokenizer(), record=single,
        max_seq_length=10_000, loss_mask_mode="assistant_only",
    )
    tok = _ChatMLFakeTokenizer()
    # Both train the answer + the <|im_end|> stop token; neither trains "hello".
    assert "world answer<|im_end|>" in tok.decode(
        [t for t, l in zip(comp.input_ids, comp.labels) if l != -100]
    )
    assert "world answer<|im_end|>" in tok.decode(
        [t for t, l in zip(asst.input_ids, asst.labels) if l != -100]
    )
    assert not _marker_supervised(comp, "hello")
    assert not _marker_supervised(asst, "hello")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="Unsupported loss_mask_mode"):
        materialize_sft_example(
            tokenizer=_ChatMLFakeTokenizer(), record=_TRAJECTORY,
            max_seq_length=10_000, loss_mask_mode="bogus_mode",
        )


# ---------------------------------------------------------------------------
# Network-gated runtime check against the real Qwen3.5 tokenizer.
# ---------------------------------------------------------------------------
_LIVE_HUB = os.environ.get("RUN_LIVE_HUB") == "1"
_QWEN_MODEL = "Qwen/Qwen3.5-4B"


@pytest.mark.skipif(
    not _LIVE_HUB,
    reason="network-gated; set RUN_LIVE_HUB=1 to download the Qwen3.5 tokenizer",
)
def test_assistant_only_tool_call_trained_tool_response_masked_real_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_QWEN_MODEL)
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "ReadMARKER", "arguments": {"file_path": "x.py"}}}]},
        {"role": "tool", "content": "TOOLRESPMARKER def add a minus b"},
        {"role": "assistant", "content": "FINALMARKER fixed the bug."},
    ]
    prepared = materialize_sft_example(
        tokenizer=tokenizer, record={"messages": messages},
        max_seq_length=4096, loss_mask_mode="assistant_only",
    )
    full = tokenizer.decode(prepared.input_ids)

    def span_trained(marker):
        a = full.find(marker)
        assert a >= 0, f"{marker!r} not in render"
        pre = len(tokenizer.encode(full[:a], add_special_tokens=False))
        seg = len(tokenizer.encode(marker, add_special_tokens=False))
        n = len(prepared.labels)
        b = min(pre + seg, n)
        return sum(1 for l in prepared.labels[min(pre, n):b] if l != -100), (b - min(pre, n))

    # tool-call name + final answer are supervised; tool RESPONSE is masked.
    read_tr, read_tot = span_trained("ReadMARKER")
    final_tr, final_tot = span_trained("FINALMARKER")
    resp_tr, resp_tot = span_trained("TOOLRESPMARKER")
    assert read_tr == read_tot and read_tot > 0
    assert final_tr == final_tot and final_tot > 0
    assert resp_tr == 0
