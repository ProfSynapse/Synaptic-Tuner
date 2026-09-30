from __future__ import annotations

import sys
from pathlib import Path

import pytest
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

preprocessing = pytest.importorskip("preprocessing")


class _FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = "\n".join(f"{message['role']}::{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "\nassistant::"
        return rendered

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) % 97 for char in text]


def _as_dict(example):
    if isinstance(example, dict):
        return example
    if hasattr(example, "model_dump"):
        return example.model_dump()
    if hasattr(example, "__dict__"):
        return dict(example.__dict__)
    raise TypeError(f"Unsupported example type: {type(example)!r}")


def test_normalize_sft_example_preserves_conversations_as_messages():
    raw = {
        "conversations": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }

    normalized = _as_dict(preprocessing.normalize_sft_example(raw))

    assert "messages" in normalized
    assert "conversations" not in normalized
    assert normalized["messages"][0]["role"] == "user"


def test_materialize_sft_features_emits_explicit_token_fields_for_conversational_rows():
    raw = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": "hi"},
        ]
    }

    normalized = preprocessing.normalize_sft_example(raw)
    prepared = _as_dict(
        preprocessing.materialize_sft_features(
            normalized,
            tokenizer=_FakeTokenizer(),
            max_seq_length=128,
            loss_mask_mode="assistant_only",
            tool_call_mode="render_text",
        )
    )

    assert {"input_ids", "attention_mask", "labels"}.issubset(prepared.keys())
    assert "text" not in prepared
    assert len(prepared["input_ids"]) == len(prepared["attention_mask"]) == len(prepared["labels"])
    assert any(label == -100 for label in prepared["labels"])
    assert any(label != -100 for label in prepared["labels"])


def test_prepare_sft_dataset_returns_dataset_with_explicit_token_columns():
    raw_dataset = Dataset.from_list(
        [
            {
                "conversations": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "there"},
                ]
            }
        ]
    )

    prepared_dataset = preprocessing.prepare_sft_dataset(
        raw_dataset,
        tokenizer=_FakeTokenizer(),
        max_seq_length=64,
        loss_mask_mode="assistant_only",
        backend="trl_unsloth",
    )

    assert prepared_dataset.column_names == ["input_ids", "attention_mask", "labels"]
    row = prepared_dataset[0]
    assert all(isinstance(token, int) for token in row["input_ids"])
    assert all(isinstance(token, int) for token in row["attention_mask"])
    assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])


def test_prepare_sft_dataset_truncates_overlong_examples_deterministically():
    raw_dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "x" * 80},
                    {"role": "assistant", "content": "y" * 80},
                ]
            }
        ]
    )

    prepared_dataset = preprocessing.prepare_sft_dataset(
        raw_dataset,
        tokenizer=_FakeTokenizer(),
        max_seq_length=32,
        loss_mask_mode="assistant_only",
        backend="trl_unsloth",
    )

    row = prepared_dataset[0]
    assert len(row["input_ids"]) <= 32
    assert len(row["input_ids"]) == len(row["labels"])


# ---------------------------------------------------------------------------
# Negative-path tests for error branches in shared/sft_preprocessing.py
# ---------------------------------------------------------------------------

def test_normalize_rejects_example_without_messages_or_prompt_completion():
    """Should raise ValueError when example has no messages, conversations, or prompt/completion."""
    from shared.sft_preprocessing import normalize_sft_messages

    with pytest.raises(ValueError, match="must provide messages/conversations or prompt/completion"):
        normalize_sft_messages({"some_other_key": "value"})


def test_normalize_rejects_unsupported_prompt_shape():
    """Should raise ValueError when prompt is an unsupported type (e.g., int)."""
    from shared.sft_preprocessing import normalize_sft_messages

    with pytest.raises(ValueError, match="Unsupported prompt shape"):
        normalize_sft_messages({"prompt": 42, "completion": "answer"})


def test_normalize_rejects_unsupported_completion_shape():
    """Should raise ValueError when completion is an unsupported type (e.g., int)."""
    from shared.sft_preprocessing import normalize_sft_messages

    with pytest.raises(ValueError, match="Unsupported completion shape"):
        normalize_sft_messages({"prompt": "question", "completion": 42})


def test_materialize_rejects_unsupported_tool_call_mode():
    """Should raise ValueError for unsupported tool_call_mode."""
    normalized = preprocessing.normalize_sft_example(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    )

    with pytest.raises(ValueError, match="Unsupported tool_call_mode"):
        preprocessing.materialize_sft_features(
            normalized,
            tokenizer=_FakeTokenizer(),
            max_seq_length=128,
            loss_mask_mode="assistant_only",
            tool_call_mode="unsupported_mode",
        )


def _raw_row(text="chapter text", **overrides):
    row = {
        "schema_version": "syntunia-sft-row/v1",
        "format": "raw_text",
        "text": text,
        "split": "train",
    }
    row.update(overrides)
    return row


def test_raw_text_bypasses_chat_template_and_uses_derived_eos_full_sequence():
    prepared = _as_dict(
        preprocessing.materialize_sft_features(
            _raw_row("abc"),
            tokenizer=_FakeTokenizer(),
            max_seq_length=32,
            loss_mask_mode="full_sequence",
        )
    )

    assert prepared["input_ids"] == [0, 1, 2, _FakeTokenizer.eos_token_id]
    assert prepared["labels"] == prepared["input_ids"]
    assert prepared["attention_mask"] == [1, 1, 1, 1]
    assert prepared["example_format"] == "raw_text"
    assert prepared["loss_mask_mode"] == "full_sequence"


def test_raw_text_truncates_after_appending_derived_eos():
    prepared = _as_dict(
        preprocessing.materialize_sft_features(
            _raw_row("abcd"),
            tokenizer=_FakeTokenizer(),
            max_seq_length=3,
            loss_mask_mode="full_sequence",
        )
    )

    assert prepared["input_ids"] == [0, 1, 2]
    assert prepared["truncation_applied"] is True


@pytest.mark.parametrize(
    "row",
    [
        {"format": "raw_text", "text": "x"},
        {"schema_version": "syntunia-sft-row/v1", "text": "x"},
        _raw_row("x", messages=[{"role": "user", "content": "mixed"}]),
        _raw_row(""),
        _raw_row("x", split=None),
    ],
)
def test_raw_text_requires_exact_authority_and_cannot_mix_shapes(row):
    with pytest.raises(ValueError):
        preprocessing.normalize_sft_example(row)


def test_arbitrary_text_column_is_not_raw_text_authority():
    with pytest.raises(ValueError, match="must provide messages/conversations"):
        preprocessing.normalize_sft_example({"text": "not authoritative"})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"loss_mask_mode": "assistant_only"}, "full-sequence loss"),
        ({"assistant_only_loss_requested": True}, "full-sequence loss"),
        ({"prompt_render": "prompt_completion"}, "bypass chat rendering"),
        ({"aux_token_position": "end_of_prompt"}, "no prompt boundary"),
    ],
)
def test_raw_text_rejects_incompatible_training_modes(kwargs, message):
    dataset = Dataset.from_list([_raw_row()])
    call_kwargs = {"loss_mask_mode": "full_sequence", **kwargs}
    with pytest.raises(ValueError, match=message):
        preprocessing.prepare_sft_dataset(
            dataset,
            tokenizer=_FakeTokenizer(),
            max_seq_length=64,
            use_preassigned_splits=True,
            **call_kwargs,
        )


def test_raw_text_requires_tokenizer_eos():
    tokenizer = _FakeTokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="eos_token_id"):
        preprocessing.materialize_sft_features(
            _raw_row(),
            tokenizer=tokenizer,
            max_seq_length=64,
            loss_mask_mode="full_sequence",
        )


def test_dataset_rejects_mixed_raw_and_conversational_rows():
    dataset = Dataset.from_dict(
        {
            "schema_version": ["syntunia-sft-row/v1", None],
            "format": ["raw_text", None],
            "text": ["chapter text", None],
            "split": ["train", None],
            "messages": [
                None,
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "there"},
                ],
            ],
        }
    )
    with pytest.raises(ValueError, match="cannot mix"):
        preprocessing.prepare_sft_dataset(
            dataset,
            tokenizer=_FakeTokenizer(),
            max_seq_length=64,
            loss_mask_mode="full_sequence",
            use_preassigned_splits=True,
        )


def test_raw_text_dataset_requires_explicit_preassigned_split_mode():
    dataset = Dataset.from_list([_raw_row()])
    with pytest.raises(ValueError, match="use_preassigned_splits=true"):
        preprocessing.prepare_sft_dataset(
            dataset,
            tokenizer=_FakeTokenizer(),
            max_seq_length=64,
            loss_mask_mode="full_sequence",
        )


def test_legacy_messages_and_prompt_completion_may_coexist():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "there"},
                ],
                "prompt": None,
                "completion": None,
            },
            {
                "messages": None,
                "prompt": "question",
                "completion": "answer",
            },
        ]
    )
    prepared = preprocessing.prepare_sft_dataset(
        dataset,
        tokenizer=_FakeTokenizer(),
        max_seq_length=128,
        loss_mask_mode="full_sequence",
    )
    assert len(prepared) == 2


def _authoritative_message_row():
    return {
        "schema_version": "syntunia-sft-row/v2",
        "format": "messages",
        "row_id": "row-" + "1" * 64,
        "target_item_id": "item-" + "2" * 64,
        "context_item_ids": ["item-" + "3" * 64],
        "group_id": "group-one",
        "split": "train",
        "messages": [
            {"role": "user", "content": "context prompt"},
            {"role": "assistant", "content": "exact projection"},
        ],
    }


def test_authoritative_messages_have_exact_two_turn_authority_and_prompt_completion_compatibility():
    row = _authoritative_message_row()
    normalized = preprocessing.normalize_sft_example(row)
    assert normalized["messages"] == row["messages"]
    assert [message["role"] for message in normalized["messages"]] == ["user", "assistant"]

    prepared = _as_dict(
        preprocessing.materialize_sft_features(
            row,
            tokenizer=_FakeTokenizer(),
            max_seq_length=256,
            loss_mask_mode="assistant_only",
            prompt_render="prompt_completion",
        )
    )
    assert prepared["example_format"] == "messages"
    assert prepared["loss_mask_mode"] == "assistant_only"
    assert any(label == -100 for label in prepared["labels"])
    assert any(label != -100 for label in prepared["labels"])


def test_authoritative_prompt_completion_rejects_over_budget_context_without_truncation():
    row = _authoritative_message_row()
    row["messages"][0]["content"] = "context " * 20
    with pytest.raises(ValueError, match="fit fully within max_seq_length"):
        preprocessing.materialize_sft_features(
            row,
            tokenizer=_FakeTokenizer(),
            max_seq_length=32,
            loss_mask_mode="assistant_only",
            prompt_render="prompt_completion",
        )


def test_authoritative_32k_prompt_completion_is_admitted_without_truncation():
    row = _authoritative_message_row()
    row["messages"][0]["content"] = "c" * 27000
    row["messages"][1]["content"] = "p" * 1500

    prepared = _as_dict(
        preprocessing.materialize_sft_features(
            row,
            tokenizer=_FakeTokenizer(),
            max_seq_length=32768,
            loss_mask_mode="assistant_only",
            prompt_render="prompt_completion",
        )
    )

    assert prepared["truncation_applied"] is False
    assert len(prepared["input_ids"]) <= 32768
    assert sum(label != -100 for label in prepared["labels"]) >= 1210


def test_legacy_prompt_completion_retains_existing_truncation_behavior():
    row = {
        "messages": [
            {"role": "user", "content": "context " * 20},
            {"role": "assistant", "content": "answer"},
        ]
    }
    prepared = _as_dict(
        preprocessing.materialize_sft_features(
            row,
            tokenizer=_FakeTokenizer(),
            max_seq_length=32,
            loss_mask_mode="assistant_only",
            prompt_render="prompt_completion",
        )
    )
    assert prepared["truncation_applied"] is True
    assert prepared["labels"] == [-100] * 32


@pytest.mark.parametrize(
    "mutation",
    [
        {"messages": [{"role": "assistant", "content": "wrong"}, {"role": "user", "content": "order"}]},
        {"messages": [{"role": "user", "content": "only one"}]},
        {"split": "test"},
        {"schema_version": "syntunia-sft-row/v1"},
    ],
)
def test_authoritative_messages_fail_closed_on_shape_or_authority_mismatch(mutation):
    row = _authoritative_message_row()
    row.update(mutation)
    with pytest.raises(ValueError):
        preprocessing.normalize_sft_example(row)
