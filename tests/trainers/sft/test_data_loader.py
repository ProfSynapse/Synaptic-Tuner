import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

import data_loader


class _FakeTokenizer:
    eos_token_id = 77

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = "\n".join(f"{message['role']}::{message['content']}" for message in messages)
        return rendered + ("\nassistant::" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) % 97 for char in text]


def test_load_and_prepare_dataset_can_preformat_conversations_to_text(tmp_path, monkeypatch):
    dataset_path = tmp_path / "sample.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "useTools",
                                "arguments": "{\"calls\":[]}",
                            }
                        }
                    ],
                },
            ]
        }
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    monkeypatch.setattr(
        data_loader,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list(rows),
    )

    train_dataset, eval_dataset = data_loader.load_and_prepare_dataset(
        local_file=str(dataset_path),
        num_proc=None,
        tokenizer=_FakeTokenizer(),
        apply_chat_template=True,
    )

    assert eval_dataset is None
    assert train_dataset.column_names == ["text"]
    assert "user::hello" in train_dataset[0]["text"]
    assert "tool_call: useTools" in train_dataset[0]["text"]


def test_sanitize_conversations_normalizes_none_content_and_tool_calls():
    sanitized = data_loader.sanitize_conversations(
        [
            {"role": "user", "content": None},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"query\": \"abc\"}",
                        }
                    }
                ],
            },
        ]
    )

    assert sanitized[0]["content"] == ""
    assert "tool_call: lookup" in sanitized[1]["content"]
    assert "query" in sanitized[1]["content"]
    assert "tool_calls" not in sanitized[1]


def test_preprocessing_contract_module_is_importable_when_present():
    preprocessing = pytest.importorskip("preprocessing")
    assert hasattr(preprocessing, "normalize_sft_example")
    assert hasattr(preprocessing, "materialize_sft_features")
    assert hasattr(preprocessing, "prepare_sft_dataset")


def _raw_rows():
    return [
        {
            "schema_version": "syntunia-sft-row/v1",
            "format": "raw_text",
            "text": "train one",
            "split": "train",
        },
        {
            "schema_version": "syntunia-sft-row/v1",
            "format": "raw_text",
            "text": "validation one",
            "split": "validation",
        },
        {
            "schema_version": "syntunia-sft-row/v1",
            "format": "raw_text",
            "text": "train two",
            "split": "train",
        },
    ]


def test_tokenized_loader_consumes_preassigned_raw_text_splits(monkeypatch):
    rows = _raw_rows()
    monkeypatch.setattr(data_loader, "load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))
    metadata = {}

    train, validation = data_loader.load_and_prepare_tokenized_dataset(
        local_file="unused.jsonl",
        tokenizer=_FakeTokenizer(),
        loss_mask_mode="full_sequence",
        use_preassigned_splits=True,
        preparation_metadata=metadata,
    )

    assert len(train) == 2
    assert len(validation) == 1
    assert train[0]["input_ids"][:-1] == [ord(char) % 97 for char in "train one"]
    assert train[1]["input_ids"][:-1] == [ord(char) % 97 for char in "train two"]
    assert validation[0]["input_ids"][:-1] == [ord(char) % 97 for char in "validation one"]
    assert train.column_names == ["input_ids", "attention_mask", "labels"]
    assert metadata == {"dataset_format": "raw_text"}


def test_preassigned_splits_reject_random_split_at_same_time(monkeypatch):
    rows = _raw_rows()
    monkeypatch.setattr(data_loader, "load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))
    with pytest.raises(ValueError, match="cannot be combined"):
        data_loader.load_and_prepare_tokenized_dataset(
            local_file="unused.jsonl",
            tokenizer=_FakeTokenizer(),
            loss_mask_mode="full_sequence",
            use_preassigned_splits=True,
            split_dataset=True,
        )


@pytest.mark.parametrize("bad_split", [None, "test", ""])
def test_preassigned_splits_reject_unknown_or_null_values(monkeypatch, bad_split):
    rows = _raw_rows()
    rows[0]["split"] = bad_split
    monkeypatch.setattr(data_loader, "load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))
    with pytest.raises(ValueError, match="split='train' or split='validation'"):
        data_loader.load_and_prepare_tokenized_dataset(
            local_file="unused.jsonl",
            tokenizer=_FakeTokenizer(),
            loss_mask_mode="full_sequence",
            use_preassigned_splits=True,
        )


@pytest.mark.parametrize(
    "kept_split, message",
    [("validation", "non-empty train"), ("train", "non-empty declared validation")],
)
def test_preassigned_splits_require_both_nonempty_partitions(
    monkeypatch, kept_split, message
):
    rows = _raw_rows()
    for row in rows:
        row["split"] = kept_split
    monkeypatch.setattr(data_loader, "load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))
    with pytest.raises(ValueError, match=message):
        data_loader.load_and_prepare_tokenized_dataset(
            local_file="unused.jsonl",
            tokenizer=_FakeTokenizer(),
            loss_mask_mode="full_sequence",
            use_preassigned_splits=True,
        )


def test_split_column_requires_explicit_enable_instead_of_implicit_consumption(monkeypatch):
    rows = _raw_rows()
    monkeypatch.setattr(data_loader, "load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))
    with pytest.raises(ValueError, match="use_preassigned_splits=true"):
        data_loader.load_and_prepare_tokenized_dataset(
            local_file="unused.jsonl",
            tokenizer=_FakeTokenizer(),
            loss_mask_mode="full_sequence",
            use_preassigned_splits=False,
        )


def _message_rows():
    return [
        {
            "schema_version": "syntunia-sft-row/v2",
            "format": "messages",
            "row_id": f"row-{index:064x}",
            "target_item_id": f"item-{index + 10:064x}",
            "context_item_ids": [f"item-{index + 20:064x}"],
            "group_id": f"group-{index}",
            "split": split,
            "messages": [
                {"role": "user", "content": f"prompt {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
        }
        for index, split in enumerate(("train", "validation", "train"))
    ]


def test_tokenized_loader_consumes_authoritative_message_splits(monkeypatch):
    monkeypatch.setattr(
        data_loader,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list(_message_rows()),
    )
    metadata = {}

    train, validation = data_loader.load_and_prepare_tokenized_dataset(
        local_file="unused.jsonl",
        tokenizer=_FakeTokenizer(),
        loss_mask_mode="assistant_only",
        use_preassigned_splits=True,
        preparation_metadata=metadata,
    )

    assert len(train) == 2
    assert len(validation) == 1
    assert metadata == {"dataset_format": "messages"}


def test_authoritative_message_rows_never_fall_back_to_random_splitting(monkeypatch):
    monkeypatch.setattr(
        data_loader,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list(_message_rows()),
    )
    with pytest.raises(ValueError, match="use_preassigned_splits=true"):
        data_loader.load_and_prepare_tokenized_dataset(
            local_file="unused.jsonl",
            tokenizer=_FakeTokenizer(),
            loss_mask_mode="assistant_only",
            use_preassigned_splits=False,
            split_dataset=True,
        )
