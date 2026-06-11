"""chat_template_kwargs passthrough — SFT preprocessing.

Covers the generic ``chat_template_kwargs`` parameter added to
``shared.sft_preprocessing.materialize_sft_example`` and threaded through the
SFT preprocessing wrappers (arch ruling #43). The kwarg is forwarded verbatim
into BOTH ``apply_chat_template`` calls (full-sequence and the prompt-only
loss-mask prefix), so a recipe can set e.g. ``{enable_thinking: false}`` for a
thinking-capable model without any model-specific code in shared/.

Two layers:

1. Fast, no-network forwarding/byte-identity tests with a recording fake
   tokenizer (run in the normal suite). These pin the contract: the kwargs reach
   both call sites, and the default (None) forwards NOTHING (byte-identical
   behavior for every existing caller).

2. A network-gated runtime render assertion (RUN_LIVE_HUB=1) against the real
   Qwen3-0.6B tokenizer — the same chat-template family as the hybrid Qwen3
   4B/8B presets (architect-verified). It converts 'verified-by-source' into
   'verified-by-execution' per arch open-risk #1.

   GROUND TRUTH (tester #49, live Qwen3-0.6B template): the empty think-off
   marker '<think>\\n\\n</think>\\n\\n' is injected UNCONDITIONALLY into the
   assistant turn — it appears in the full-sequence render identically for
   enable_thinking False / True / default. So the contract is NOT "no <think> in
   the full render" (that is false against the live template). The real contract
   is a MASKING one: the marker lands in the prompt PREFIX, which the
   assistant-only loss mask zeroes out, so the unmasked TRAINING TARGET decodes
   to exactly the clean completion 'I don't know.<|im_end|>\\n' with no <think>.
   The prompt-only render (== eval-time prompt) carries the marker, aligning the
   train prompt prefix with the eval prompt. This test asserts BOTH: the unmasked
   target is <think>-free, and the prompt render carries the marker.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

preprocessing = pytest.importorskip("preprocessing")

from shared.sft_preprocessing import materialize_sft_example


class _RecordingTokenizer:
    """Fake tokenizer that records every apply_chat_template kwargs payload.

    Accepts arbitrary **kwargs so the forwarded chat_template_kwargs land in
    ``self.calls`` for assertion. encode() is a deterministic char-based stub.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append(
            {"add_generation_prompt": add_generation_prompt, "kwargs": dict(kwargs)}
        )
        rendered = "\n".join(f"{m['role']}::{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\nassistant::"
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 for char in text]


_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "What is the capital of Atlantis?"},
        {"role": "assistant", "content": "I don't know."},
    ]
}


def test_chat_template_kwargs_forwarded_to_both_call_sites():
    tokenizer = _RecordingTokenizer()

    materialize_sft_example(
        tokenizer=tokenizer,
        record=_EXAMPLE,
        max_seq_length=128,
        assistant_only_loss=True,
        chat_template_kwargs={"enable_thinking": False},
    )

    # Two render calls: full-sequence (add_generation_prompt=False) and the
    # prompt-only loss-mask prefix (add_generation_prompt=True). BOTH must carry
    # the forwarded kwarg so the loss-mask prefix matches the eval-time prompt.
    assert len(tokenizer.calls) == 2
    full_call = next(c for c in tokenizer.calls if c["add_generation_prompt"] is False)
    prompt_call = next(c for c in tokenizer.calls if c["add_generation_prompt"] is True)
    assert full_call["kwargs"] == {"enable_thinking": False}
    assert prompt_call["kwargs"] == {"enable_thinking": False}


def test_default_none_forwards_no_kwargs_byte_identical():
    tokenizer = _RecordingTokenizer()

    materialize_sft_example(
        tokenizer=tokenizer,
        record=_EXAMPLE,
        max_seq_length=128,
        assistant_only_loss=True,
        # chat_template_kwargs omitted ⇒ default None ⇒ no extra kwargs.
    )

    assert len(tokenizer.calls) == 2
    for call in tokenizer.calls:
        assert call["kwargs"] == {}


def test_preprocessing_wrappers_thread_chat_template_kwargs():
    """The SFT-facing wrappers forward the kwarg down to materialize."""
    tokenizer = _RecordingTokenizer()

    normalized = preprocessing.normalize_sft_example(_EXAMPLE)
    preprocessing.materialize_sft_features(
        normalized,
        tokenizer=tokenizer,
        max_seq_length=128,
        loss_mask_mode="assistant_only",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert tokenizer.calls
    assert all(call["kwargs"] == {"enable_thinking": False} for call in tokenizer.calls)


# ---------------------------------------------------------------------------
# Network-gated runtime render assertion (arch open-risk #1).
# ---------------------------------------------------------------------------

_LIVE_HUB = os.environ.get("RUN_LIVE_HUB") == "1"

# Qwen3-0.6B ships the same chat-template family as the hybrid 4B/8B presets and
# is the smallest model the architect verified the template logic against.
_QWEN3_TEMPLATE_MODEL = "Qwen/Qwen3-0.6B"
_THINK_OFF_MARKER = "<think>\n\n</think>"


@pytest.mark.skipif(
    not _LIVE_HUB,
    reason="network-gated; set RUN_LIVE_HUB=1 to download the Qwen3 tokenizer and render",
)
def test_enable_thinking_false_masks_marker_out_of_target_keeps_it_in_prompt():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_QWEN3_TEMPLATE_MODEL)

    # The unmasked TRAINING TARGET (the only span that contributes loss) must
    # decode to the clean completion with NO <think> scaffolding. The Qwen3
    # template injects the empty think-off marker UNCONDITIONALLY, but it lands
    # in the prompt prefix, which the assistant-only loss mask zeroes out — so it
    # never reaches the supervised target. We exercise the real preprocessing
    # path (not a raw render) so the mask is applied exactly as in training.
    prepared = materialize_sft_example(
        tokenizer=tokenizer,
        record=_EXAMPLE,
        max_seq_length=512,
        assistant_only_loss=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert prepared.loss_mask_mode == "assistant_only"

    target_ids = [
        tok
        for tok, label in zip(prepared.input_ids, prepared.labels)
        if label != -100
    ]
    assert target_ids, "expected a non-empty unmasked training target span"
    target_str = tokenizer.decode(target_ids)
    assert "<think>" not in target_str, (
        "Unmasked training target must not contain a <think> block — the "
        "unconditional think-off marker belongs to the masked prompt prefix; "
        f"got target:\n{target_str}"
    )
    assert "I don't know." in target_str, (
        "Unmasked training target must carry the clean IDK completion; "
        f"got target:\n{target_str}"
    )

    # Prompt-only (the loss-mask prefix == eval-time prompt): with
    # enable_thinking=False the Qwen3 template injects the empty think-off marker
    # at the generation prompt. This is what aligns train and eval prompts.
    prompt_str = tokenizer.apply_chat_template(
        _EXAMPLE["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert _THINK_OFF_MARKER in prompt_str, (
        "Prompt-only render must carry the think-off marker "
        f"{_THINK_OFF_MARKER!r} at the generation prompt; got:\n{prompt_str}"
    )
