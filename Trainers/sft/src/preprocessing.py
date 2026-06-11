"""
SFT-facing wrapper over the canonical repo-owned preprocessing contract.
"""

from __future__ import annotations

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
    )


def prepare_sft_dataset(
    dataset: Dataset,
    *,
    tokenizer: Any,
    max_seq_length: int,
    loss_mask_mode: str = ASSISTANT_ONLY,
    backend: str = "trl_unsloth",
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Dataset:
    del backend  # The contract is backend-agnostic; callers choose the trainer separately.

    def _materialize(example: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_sft_example(example)
        prepared = materialize_sft_features(
            normalized,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            loss_mask_mode=loss_mask_mode,
            chat_template_kwargs=chat_template_kwargs,
        )
        return {
            "input_ids": prepared.input_ids,
            "attention_mask": prepared.attention_mask,
            "labels": prepared.labels,
        }

    return dataset.map(
        _materialize,
        remove_columns=dataset.column_names,
        desc="Preparing tokenized SFT examples",
    )


def load_and_prepare_sft_dataset(
    *,
    dataset: Dataset,
    tokenizer: Any,
    max_seq_length: int,
    loss_mask_mode: str = ASSISTANT_ONLY,
    num_proc: int = 1,
    include_text: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Dataset:
    del num_proc
    del include_text
    return prepare_sft_dataset(
        dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        loss_mask_mode=loss_mask_mode,
        chat_template_kwargs=chat_template_kwargs,
    )
