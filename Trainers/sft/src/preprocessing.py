"""
SFT-facing wrapper over the canonical repo-owned preprocessing contract.
"""

from __future__ import annotations

import math
from typing import Any

from datasets import Dataset

from shared.sft_preprocessing import (
    DEFAULT_MAX_DROPPED_ROW_FRACTION,
    DROP_MASK_PREFIX_MISMATCH,
    DROP_NO_SUPERVISED_TOKENS,
    DROP_REASONS,
    PreparedSFTExample,
    detect_sft_record_format,
    is_authoritative_preassigned_sft_record,
    materialize_sft_example as _materialize_sft_example,
    normalize_sft_messages,
    sanitize_messages_for_chat_template,
)

ASSISTANT_ONLY = "assistant_only"
FULL_SEQUENCE = "full_sequence"

# Original row index of each kept row, emitted by prepare_sft_dataset when
# keep_source_index=True so callers can realign per-row side data (validation
# group values) after untrainable rows are dropped. Remove it before training.
SOURCE_INDEX_COLUMN = "_source_index"
_DROP_REASON_COLUMN = "_drop_reason"
_FALLBACK_COLUMN = "_mask_fallback_full_sequence"


class DroppedRowsError(ValueError):
    """Too many rows were untrainable (see :func:`check_dropped_row_fraction`)."""


def dropped_row_fraction_exceeded(dropped: int, rows: int, max_fraction: float) -> bool:
    """Whether ``dropped`` of ``rows`` exceeds the configured maximum fraction."""
    return rows > 0 and dropped / rows > max_fraction


def check_dropped_row_fraction(counts: dict[str, int], rows: int, max_fraction: float) -> None:
    """Fail when untrainable rows exceed ``training.max_dropped_row_fraction``."""
    if not 0.0 <= max_fraction <= 1.0:
        raise ValueError(
            f"training.max_dropped_row_fraction must be within [0, 1], got {max_fraction!r}."
        )
    dropped = sum(counts.values())
    if dropped_row_fraction_exceeded(dropped, rows, max_fraction):
        detail = ", ".join(f"{reason}={count}" for reason, count in counts.items() if count)
        raise DroppedRowsError(
            f"SFT_DROPPED_ROWS_ABOVE_THRESHOLD: {dropped}/{rows} rows "
            f"({dropped / rows:.2%}) cannot be trained ({detail}); the maximum is "
            f"training.max_dropped_row_fraction={max_fraction}. Fix max_seq_length, the "
            "chat template kwargs or the data (inspect with `python tuner.py doctor "
            "sft-mask`) instead of training on a silently shrunken or mis-masked dataset."
        )


def sanitize_conversations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sanitize_messages_for_chat_template(messages)


def normalize_sft_example(example: dict[str, Any]) -> dict[str, Any]:
    detected = detect_sft_record_format(example)
    if detected == "raw_text":
        return {
            "schema_version": example["schema_version"],
            "format": example["format"],
            "text": example["text"],
            "split": example["split"],
        }
    messages, example_format = normalize_sft_messages(example)
    normalized = {
        "messages": messages,
        "example_format": example_format,
    }
    if is_authoritative_preassigned_sft_record(example):
        normalized.update(
            {
                "schema_version": example["schema_version"],
                "format": example["format"],
                "split": example["split"],
            }
        )
    return normalized


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
    record = example
    if "messages" in example and not is_authoritative_preassigned_sft_record(example):
        record = {"messages": example["messages"]}
    return _materialize_sft_example(
        tokenizer=tokenizer,
        record=record,
        max_seq_length=max_seq_length,
        assistant_only_loss=assistant_only_loss,
        chat_template_kwargs=chat_template_kwargs,
        prompt_render=prompt_render,
    )


def materialize_sft_row(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    max_seq_length: int,
    loss_mask_mode: str = ASSISTANT_ONLY,
    chat_template_kwargs: dict[str, Any] | None = None,
    prompt_render: str = "full_conversation",
) -> PreparedSFTExample:
    """Materialize one raw dataset row exactly as the SFT trainer does.

    This is the single per-row hop used by :func:`prepare_sft_dataset` (and so by
    every SFT training run). The SFT mask doctor calls it directly so it inspects
    the real masking path rather than a re-implementation.
    """
    normalized = normalize_sft_example(example)
    return materialize_sft_features(
        normalized,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        loss_mask_mode=loss_mask_mode,
        chat_template_kwargs=chat_template_kwargs,
        prompt_render=prompt_render,
    )


def validate_sft_dataset_contract(
    dataset: Dataset,
    *,
    loss_mask_mode: str = ASSISTANT_ONLY,
    prompt_render: str = "full_conversation",
    assistant_only_loss_requested: bool = False,
    aux_token_position: str | int | None = None,
    use_preassigned_splits: bool = False,
) -> None:
    """Enforce the dataset-level SFT row contract (raises ``ValueError``).

    :func:`prepare_sft_dataset` runs this before materializing any row; the SFT
    mask doctor runs it too, so a dataset the trainer would reject is reported.
    """
    dataset_formats = {detect_sft_record_format(dataset[index]) for index in range(len(dataset))}
    authoritative_rows = [
        is_authoritative_preassigned_sft_record(dataset[index]) for index in range(len(dataset))
    ]
    if any(authoritative_rows) and not all(authoritative_rows):
        raise ValueError("Prepared authoritative rows cannot mix with legacy SFT rows.")
    if use_preassigned_splits and (not authoritative_rows or not all(authoritative_rows)):
        raise ValueError(
            "dataset.use_preassigned_splits=true requires a uniform authoritative prepared dataset."
        )
    if "raw_text" in dataset_formats and len(dataset_formats) > 1:
        raise ValueError("SFT datasets cannot mix raw_text and conversational row formats.")
    dataset_format = next(iter(dataset_formats), None)
    if dataset_format == "raw_text":
        if not use_preassigned_splits:
            raise ValueError(
                "raw_text SFT rows require dataset.use_preassigned_splits=true."
            )
        if loss_mask_mode != FULL_SEQUENCE or assistant_only_loss_requested:
            raise ValueError(
                "raw_text SFT rows require full-sequence loss; disable both "
                "completion_only_loss and assistant_only_loss."
            )
        if prompt_render != "full_conversation":
            raise ValueError(
                "raw_text SFT rows bypass chat rendering and are incompatible with "
                "prompt_render='prompt_completion'."
            )
        if aux_token_position == "end_of_prompt":
            raise ValueError(
                "raw_text SFT rows have no prompt boundary and are incompatible "
                "with aux_head token_position='end_of_prompt'."
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
    assistant_only_loss_requested: bool = False,
    aux_token_position: str | int | None = None,
    use_preassigned_splits: bool = False,
    max_dropped_row_fraction: float = DEFAULT_MAX_DROPPED_ROW_FRACTION,
    keep_source_index: bool = False,
) -> Dataset:
    """Materialize every row into input_ids / attention_mask / labels.

    Rows that cannot be trained (every label masked, or assistant-only masking
    that stopped before the end of the prompt render) are dropped with a logged
    count per reason; the run fails when they exceed
    ``max_dropped_row_fraction``. Prompt tokens are never trained silently.
    ``keep_source_index`` keeps ``SOURCE_INDEX_COLUMN`` (each kept row's input
    index) for realigning per-row side data; remove it before training.
    """
    del backend  # The contract is backend-agnostic; callers choose the trainer separately.

    validate_sft_dataset_contract(
        dataset,
        loss_mask_mode=loss_mask_mode,
        prompt_render=prompt_render,
        assistant_only_loss_requested=assistant_only_loss_requested,
        aux_token_position=aux_token_position,
        use_preassigned_splits=use_preassigned_splits,
    )

    # ``remove_columns=dataset.column_names`` (below) drops every original column
    # AFTER ``_materialize`` runs, so any per-row directive (e.g. the aux_head
    # target) must be READ HERE and threaded into the returned dict to survive —
    # extending only the collator is too late. When ``aux_target_field`` is None
    # the returned dataset is exactly {input_ids, attention_mask, labels}; the
    # bookkeeping columns below are removed before it is returned.
    def _materialize(example: dict[str, Any], index: int) -> dict[str, Any]:
        prepared = materialize_sft_row(
            example,
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
        materialized[_DROP_REASON_COLUMN] = prepared.drop_reason or ""
        materialized[_FALLBACK_COLUMN] = prepared.mask_fallback_reason is not None
        materialized[SOURCE_INDEX_COLUMN] = index
        return materialized

    prepared_dataset = dataset.map(
        _materialize,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Preparing tokenized SFT examples",
    )
    rows = len(prepared_dataset)
    reasons = list(prepared_dataset[_DROP_REASON_COLUMN]) if rows else []
    counts = {reason: reasons.count(reason) for reason in DROP_REASONS}
    fallback = int(sum(prepared_dataset[_FALLBACK_COLUMN])) if rows else 0
    dropped = sum(counts.values())
    print(
        f"\nSFT loss-mask check ({loss_mask_mode}): dropped {dropped}/{rows} rows "
        f"({DROP_NO_SUPERVISED_TOKENS}={counts[DROP_NO_SUPERVISED_TOKENS]}, "
        f"{DROP_MASK_PREFIX_MISMATCH}={counts[DROP_MASK_PREFIX_MISMATCH]}); "
        f"{fallback} kept row(s) fell back to full-sequence loss"
    )
    if fallback:
        print(
            f"WARNING: {fallback} row(s) do not end with an assistant turn and are "
            "trained with full-sequence loss."
        )
    check_dropped_row_fraction(counts, rows, max_dropped_row_fraction)
    if dropped:
        print(
            f"WARNING: dropped {dropped} untrainable row(s); inspect them with "
            "`python tuner.py doctor sft-mask`."
        )
        prepared_dataset = prepared_dataset.filter(
            lambda reason: reason == "", input_columns=[_DROP_REASON_COLUMN]
        )
    bookkeeping = [_DROP_REASON_COLUMN, _FALLBACK_COLUMN]
    if not keep_source_index:
        bookkeeping.append(SOURCE_INDEX_COLUMN)
    return prepared_dataset.remove_columns(bookkeeping)


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
    assistant_only_loss_requested: bool = False,
    aux_token_position: str | int | None = None,
    use_preassigned_splits: bool = False,
    max_dropped_row_fraction: float = DEFAULT_MAX_DROPPED_ROW_FRACTION,
    keep_source_index: bool = False,
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
        assistant_only_loss_requested=assistant_only_loss_requested,
        aux_token_position=aux_token_position,
        use_preassigned_splits=use_preassigned_splits,
        max_dropped_row_fraction=max_dropped_row_fraction,
        keep_source_index=keep_source_index,
    )
