import json
import sys
from pathlib import Path

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

import data_loader


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(f"{message['role']}::{message['content']}" for message in messages)


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
