"""Tests for the prompt_completion render mode in SFT preprocessing.

Location: tests/trainers/sft/test_aux_head_prompt_completion_render.py

The default "full_conversation" render masks the assistant-only region by a
prefix match against the add_generation_prompt=True prompt render. When a chat
template renders the assistant scaffold differently with vs. without
add_generation_prompt (e.g. one fewer newline), that masked boundary is NOT the
generation-anchor token. The "prompt_completion" mode builds input_ids from the
add_generation_prompt=True prompt render directly, so the boundary token is the
generation anchor exactly.

These tests use a tokenizer that deliberately DIVERGES between the two render
modes (mirroring the Qwen3 scaffold divergence) to prove:
  * the prompt_completion boundary token IS the generation anchor (faithful),
  * the full_conversation boundary token is NOT (the bug the mode fixes),
  * the prompt segment is fully masked and every completion token (incl. the
    derived terminal) carries a real label,
  * the terminal is derived from the tokenizer (loud when undefined),
  * the default mode is byte-identical to the historical full_conversation path,
  * the mode threads through the prepare_sft_dataset wrapper hop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

from shared.sft_preprocessing import materialize_sft_example  # noqa: E402

preprocessing = pytest.importorskip("preprocessing")


class _DivergingTokenizer:
    """Renders the assistant scaffold differently in the two render modes.

    add_generation_prompt=True ends the prompt with a double-newline generation
    anchor (``<assistant>\\n\\n``); the full-conversation render (the assistant
    turn with add_generation_prompt=False) emits ``<assistant>{content}`` with no
    such scaffold, so the two renders diverge right at the boundary — exactly the
    structural divergence that breaks the prefix-match mask on real templates.
    """

    eos_token_id = 999

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            body += "<assistant>\n\n"
        return body

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


class _NoEosTokenizer(_DivergingTokenizer):
    eos_token_id = None


def _row():
    return {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "there"},
        ]
    }


def _materialize(tokenizer, *, prompt_render, max_seq_length=128):
    return materialize_sft_example(
        tokenizer=tokenizer,
        record=_row(),
        max_seq_length=max_seq_length,
        assistant_only_loss=True,
        prompt_render=prompt_render,
    )


def test_prompt_completion_boundary_is_the_generation_anchor():
    tokenizer = _DivergingTokenizer()
    prompt_str = tokenizer.apply_chat_template(
        _row()["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

    prepared = _materialize(tokenizer, prompt_render="prompt_completion")

    # The input_ids prefix is byte-for-byte the generation-prompt render, so the
    # boundary token (index len(prompt)-1) IS the generation anchor's last token.
    assert prepared.input_ids[: len(prompt_ids)] == prompt_ids
    assert prepared.input_ids[len(prompt_ids) - 1] == prompt_ids[-1]


def test_full_conversation_boundary_is_not_the_generation_anchor():
    # Contrast: on the same diverging template, the default render masks a prefix
    # that ends BEFORE the generation anchor, so its boundary token differs from
    # the prompt_completion (faithful) boundary token. This is the defect that
    # motivated the new mode — asserted here so a template change that silently
    # collapses the divergence is caught.
    tokenizer = _DivergingTokenizer()
    prompt_str = tokenizer.apply_chat_template(
        _row()["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    anchor_last = tokenizer.encode(prompt_str, add_special_tokens=False)[-1]

    full = _materialize(tokenizer, prompt_render="full_conversation")
    masked_len = sum(1 for label in full.labels if label == -100)
    full_boundary = full.input_ids[masked_len - 1]

    assert full_boundary != anchor_last


def test_prompt_completion_masks_prompt_and_labels_completion():
    tokenizer = _DivergingTokenizer()
    prompt_str = tokenizer.apply_chat_template(
        _row()["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    completion_ids = tokenizer.encode("there", add_special_tokens=False) + [tokenizer.eos_token_id]

    prepared = _materialize(tokenizer, prompt_render="prompt_completion")

    # Every prompt token masked; every completion token (incl. the terminal) kept.
    assert prepared.labels[: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert prepared.labels[len(prompt_ids) :] == completion_ids
    assert prepared.input_ids[len(prompt_ids) :] == completion_ids
    assert prepared.attention_mask == [1] * len(prepared.input_ids)
    assert prepared.loss_mask_mode == "assistant_only"


def test_prompt_completion_terminal_is_derived_eos():
    tokenizer = _DivergingTokenizer()
    prepared = _materialize(tokenizer, prompt_render="prompt_completion")
    assert prepared.input_ids[-1] == tokenizer.eos_token_id
    assert prepared.labels[-1] == tokenizer.eos_token_id


def test_prompt_completion_requires_eos_token_id():
    with pytest.raises(ValueError, match="eos_token_id"):
        _materialize(_NoEosTokenizer(), prompt_render="prompt_completion")


def test_prompt_completion_truncates_to_max_seq_length():
    tokenizer = _DivergingTokenizer()
    prepared = _materialize(tokenizer, prompt_render="prompt_completion", max_seq_length=4)
    assert len(prepared.input_ids) == 4
    assert len(prepared.labels) == 4
    assert len(prepared.attention_mask) == 4
    assert prepared.truncation_applied is True


def test_default_mode_is_byte_identical_to_full_conversation():
    # The new parameter defaults to "full_conversation"; an explicit pass and the
    # default must produce identical output, and neither may touch the historical
    # contract (input_ids/labels/attention_mask/loss_mask_mode).
    tokenizer = _DivergingTokenizer()
    explicit = _materialize(tokenizer, prompt_render="full_conversation")
    default = materialize_sft_example(
        tokenizer=_DivergingTokenizer(),
        record=_row(),
        max_seq_length=128,
        assistant_only_loss=True,
    )
    assert explicit.to_dict() == default.to_dict()


def test_non_assistant_final_turn_falls_through_to_full_conversation():
    # The branch is gated on an assistant final turn; a prompt-only (user-final)
    # row under prompt_completion must fall through to the default render, not
    # raise. Mirrors the inference-shaped-row carve-out.
    tokenizer = _DivergingTokenizer()
    record = {"messages": [{"role": "user", "content": "hello"}]}
    prepared = materialize_sft_example(
        tokenizer=tokenizer,
        record=record,
        max_seq_length=128,
        assistant_only_loss=True,
        prompt_render="prompt_completion",
    )
    # Full-conversation fallthrough: no assistant turn ⇒ full_sequence mask mode.
    assert prepared.loss_mask_mode == "full_sequence"


def test_prompt_render_threads_through_prepare_sft_dataset():
    # The wrapper hop must forward prompt_render to the seam: a prompt_completion
    # dataset row terminates with the derived eos, which the full_conversation
    # render never appends.
    dataset = Dataset.from_list([_row(), _row()])
    prepared = preprocessing.prepare_sft_dataset(
        dataset,
        tokenizer=_DivergingTokenizer(),
        max_seq_length=128,
        loss_mask_mode="assistant_only",
        prompt_render="prompt_completion",
    )
    assert prepared[0]["input_ids"][-1] == _DivergingTokenizer.eos_token_id
    assert prepared[0]["labels"][-1] == _DivergingTokenizer.eos_token_id


def test_aux_target_threads_through_under_prompt_completion_render():
    # The aux_target is read one layer up in prepare_sft_dataset (before the
    # original columns are dropped), independent of the render mode. This pins
    # that the prompt_completion render and the aux_target threading coexist: the
    # row carries its per-row target AND terminates with the derived eos.
    rows = [
        {**_row(), "target": 0.25},
        {**_row(), "target": 0.75},
    ]
    dataset = Dataset.from_list(rows)
    prepared = preprocessing.prepare_sft_dataset(
        dataset,
        tokenizer=_DivergingTokenizer(),
        max_seq_length=128,
        loss_mask_mode="assistant_only",
        aux_target_field="target",
        prompt_render="prompt_completion",
    )
    # aux_target survives the column drop and matches each row's value.
    assert prepared[0]["aux_target"] == pytest.approx(0.25)
    assert prepared[1]["aux_target"] == pytest.approx(0.75)
    # And the render mode still applied (derived-eos terminal on the same rows).
    assert prepared[0]["input_ids"][-1] == _DivergingTokenizer.eos_token_id
    assert prepared[0]["labels"][-1] == _DivergingTokenizer.eos_token_id
