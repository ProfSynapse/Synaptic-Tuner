from __future__ import annotations

import sys
from pathlib import Path

import pytest
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

preprocessing = pytest.importorskip("preprocessing")


class _FakeTokenizer:
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
