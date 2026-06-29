"""Tests for the aux_target preprocessing hop (R1 two-hop fix, part a).

Location: tests/trainers/sft/test_aux_head_preprocessing.py

The collator alone is insufficient: prepare_sft_dataset.map(remove_columns=...)
drops the target column before the collator runs, so the per-row target must be
READ inside _materialize. These tests verify the target survives, that a missing
target fails loud, and that the feature-off path is byte-identical (no extra
column).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

preprocessing = pytest.importorskip("preprocessing")


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "\n".join(f"{message['role']}::{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "\nassistant::"
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 for char in text]


def _row(target=None):
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "there"},
        ]
    }
    if target is not None:
        row["score"] = target
    return row


def test_feature_off_is_byte_identical_no_extra_column():
    dataset = Dataset.from_list([_row(0.7), _row(0.2)])
    prepared = preprocessing.prepare_sft_dataset(
        dataset,
        tokenizer=_FakeTokenizer(),
        max_seq_length=64,
        loss_mask_mode="assistant_only",
        # aux_target_field defaults to None ⇒ off
    )
    assert prepared.column_names == ["input_ids", "attention_mask", "labels"]


def test_enabled_carries_target_through_remove_columns():
    dataset = Dataset.from_list([_row(0.7), _row(0.2)])
    prepared = preprocessing.prepare_sft_dataset(
        dataset,
        tokenizer=_FakeTokenizer(),
        max_seq_length=64,
        loss_mask_mode="assistant_only",
        aux_target_field="score",
    )
    assert "aux_target" in prepared.column_names
    assert prepared[0]["aux_target"] == pytest.approx(0.7)
    assert prepared[1]["aux_target"] == pytest.approx(0.2)


def test_missing_target_fails_loud():
    dataset = Dataset.from_list([_row(0.7), _row(target=None)])
    with pytest.raises(ValueError, match="missing it"):
        preprocessing.prepare_sft_dataset(
            dataset,
            tokenizer=_FakeTokenizer(),
            max_seq_length=64,
            loss_mask_mode="assistant_only",
            aux_target_field="score",
        )


def test_non_numeric_target_fails_loud():
    dataset = Dataset.from_list([_row("not-a-number")])
    with pytest.raises(ValueError, match="not numeric"):
        preprocessing.prepare_sft_dataset(
            dataset,
            tokenizer=_FakeTokenizer(),
            max_seq_length=64,
            loss_mask_mode="assistant_only",
            aux_target_field="score",
        )


def test_nan_target_fails_loud():
    dataset = Dataset.from_list([_row(float("nan"))])
    with pytest.raises(ValueError, match="not finite"):
        preprocessing.prepare_sft_dataset(
            dataset,
            tokenizer=_FakeTokenizer(),
            max_seq_length=64,
            loss_mask_mode="assistant_only",
            aux_target_field="score",
        )


def test_read_aux_target_helper_validates_directly():
    assert preprocessing._read_aux_target({"score": 1}, "score") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        preprocessing._read_aux_target({}, "score")
