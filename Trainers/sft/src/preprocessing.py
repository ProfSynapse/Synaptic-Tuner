"""
SFT-facing wrapper over the canonical repo-owned preprocessing contract.
"""

from __future__ import annotations

import math
from typing import Any

from datasets import Dataset

from shared.sft_preprocessing import (
    PreparedSFTExample,
    materialize_sft_example as _materialize_sft_example,
    normalize_sft_messages,
    sanitize_messages_for_chat_template,
)

ASSISTANT_ONLY = "assistant_only"
FULL_SEQUENCE = "full_sequence"


def sanitize_conversations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sanitize_messages_for_chat_template(messages)


def normalize_sft_example(example: dict[str, Any]) -> dict[str, Any]:
    messages, example_format = normalize_sft_messages(example)
    return {
        "messages": messages,
        "example_format": example_format,
    }


def render_chat_text(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    prepared = _materialize_sft_example(
        tokenizer=tokenizer,
        record={"messages": messages},
        max_seq_length=10**9,
        assistant_only_loss=False,
    )
    return tokenizer.decode(prepared.input_ids) if hasattr(tokenizer, "decode") else tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def materialize_sft_features(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    max_seq_length: int,
    loss_mask_mode: str = ASSISTANT_ONLY,
    tool_call_mode: str = "render_text",
    chat_template_kwargs: dict[str, Any] | None = None,
    prompt_render: str = "full_conversation",
) -> PreparedSFTExample:
    if tool_call_mode != "render_text":
        raise ValueError(f"Unsupported tool_call_mode: {tool_call_mode}")

    assistant_only_loss = loss_mask_mode == ASSISTANT_ONLY
    record = {"messages": example["messages"]} if "messages" in example else example
    return _materialize_sft_example(
        tokenizer=tokenizer,
        record=record,
        max_seq_length=max_seq_length,
        assistant_only_loss=assistant_only_loss,
        chat_template_kwargs=chat_template_kwargs,
        prompt_render=prompt_render,
    )


def prepare_sft_dataset(
    dataset: Dataset,
    *,
    tokenizer: Any,
    max_seq_length: int,
    loss_mask_mode: str = ASSISTANT_ONLY,
    backend: str = "trl_unsloth",
    chat_template_kwargs: dict[str, Any] | None = None,
    aux_target_field: str | None = None,
    prompt_render: str = "full_conversation",
) -> Dataset:
    del backend  # The contract is backend-agnostic; callers choose the trainer separately.

    # ``remove_columns=dataset.column_names`` (below) drops every original column
    # AFTER ``_materialize`` runs, so any per-row directive (e.g. the aux_head
    # target) must be READ HERE and threaded into the returned dict to survive —
    # extending only the collator is too late. When ``aux_target_field`` is None
    # the returned dict is exactly {input_ids, attention_mask, labels}, identical
    # to the feature-off behavior.
    def _materialize(example: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_sft_example(example)
        prepared = materialize_sft_features(
            normalized,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            loss_mask_mode=loss_mask_mode,
            chat_template_kwargs=chat_template_kwargs,
            prompt_render=prompt_render,
        )
        materialized = {
            "input_ids": prepared.input_ids,
            "attention_mask": prepared.attention_mask,
            "labels": prepared.labels,
        }
        if aux_target_field is not None:
            materialized["aux_target"] = _read_aux_target(example, aux_target_field)
        return materialized

    return dataset.map(
        _materialize,
        remove_columns=dataset.column_names,
        desc="Preparing tokenized SFT examples",
    )


def _read_aux_target(example: dict[str, Any], aux_target_field: str) -> float:
    """Read + validate a per-row aux_head target. Loud on missing/NaN (never default).

    Mirrors the subspan precedent's loud-fail discipline: every row must carry a
    finite target when the feature is enabled — there is no silent substitution.
    """
    raw_value = example.get(aux_target_field, None)
    if raw_value is None:
        raise ValueError(
            f"aux_head is enabled with target_field={aux_target_field!r} but a row is "
            f"missing it (or it is null). Every training row must carry a finite target."
        )
    try:
        target_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"aux_head target_field={aux_target_field!r} value {raw_value!r} is not numeric."
        ) from exc
    if not math.isfinite(target_value):
        raise ValueError(
            f"aux_head target_field={aux_target_field!r} value {raw_value!r} is not finite (NaN/inf)."
        )
    return target_value


def load_and_prepare_sft_dataset(
    *,
    dataset: Dataset,
    tokenizer: Any,
    max_seq_length: int,
    loss_mask_mode: str = ASSISTANT_ONLY,
    num_proc: int = 1,
    include_text: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    aux_target_field: str | None = None,
    prompt_render: str = "full_conversation",
) -> Dataset:
    del num_proc
    del include_text
    return prepare_sft_dataset(
        dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        loss_mask_mode=loss_mask_mode,
        chat_template_kwargs=chat_template_kwargs,
        aux_target_field=aux_target_field,
        prompt_render=prompt_render,
    )
