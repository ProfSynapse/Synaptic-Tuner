"""Smoke tests for the DPO data loader (prompt/chosen/rejected schema).

These import only `datasets` (no trl/unsloth/torch), so they exercise the
config-validation dry-run contract without the ML stack.
"""

import json
from pathlib import Path

from tests.trainers._trainer_import import load_trainer_module

data_loader = load_trainer_module("dpo", "data_loader")


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _valid_pair(question="Who wrote Paradise Lost?"):
    return {
        "prompt": [
            {"role": "system", "content": "Answer if you know; otherwise say you do not."},
            {"role": "user", "content": question},
        ],
        "chosen": [{"role": "assistant", "content": "John Milton."}],
        "rejected": [{"role": "assistant", "content": "I don't know the answer."}],
    }


def test_load_and_validate_well_formed_dpo_pairs(tmp_path):
    path = tmp_path / "dpo_train.jsonl"
    _write_jsonl(path, [_valid_pair("q1"), _valid_pair("q2")])

    train, eval_ds = data_loader.load_and_prepare_dataset(local_file=str(path))

    assert eval_ds is None
    assert sorted(train.column_names) == sorted(data_loader.REQUIRED_DPO_COLUMNS)
    assert len(train) == 2
    assert data_loader.validate_dpo_dataset(train) is True


def test_extra_provenance_columns_are_dropped(tmp_path):
    path = tmp_path / "dpo_train.jsonl"
    row = _valid_pair()
    row["question_id"] = "tqa_train_000123"  # builder provenance, not a DPO column
    row["source_label"] = "unknown"
    _write_jsonl(path, [row])

    train, _ = data_loader.load_and_prepare_dataset(local_file=str(path))

    assert "question_id" not in train.column_names
    assert "source_label" not in train.column_names
    assert sorted(train.column_names) == sorted(data_loader.REQUIRED_DPO_COLUMNS)


def test_split_dataset_produces_eval(tmp_path):
    path = tmp_path / "dpo_train.jsonl"
    _write_jsonl(path, [_valid_pair(f"q{i}") for i in range(10)])

    train, eval_ds = data_loader.load_and_prepare_dataset(
        local_file=str(path), split_dataset=True, test_size=0.2,
    )

    assert eval_ds is not None
    assert len(train) + len(eval_ds) == 10


def test_validate_rejects_missing_columns(tmp_path):
    from datasets import Dataset

    bad = Dataset.from_dict({"prompt": [[{"role": "user", "content": "hi"}]],
                             "chosen": [[{"role": "assistant", "content": "yes"}]]})
    # rejected column missing
    assert data_loader.validate_dpo_dataset(bad) is False


def test_validate_rejects_malformed_message_lists(tmp_path):
    from datasets import Dataset

    bad = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "hi"}]],
        "chosen": [[{"role": "assistant", "content": "yes"}]],
        "rejected": [[]],  # empty -> not a valid message list
    })
    assert data_loader.validate_dpo_dataset(bad) is False


def test_tool_calls_and_tool_results_render_to_text(tmp_path):
    # Multi-turn agentic pairs carry OpenAI tool_calls (arguments as a JSON
    # string) and tool-result turns; chat templates reject both raw shapes.
    call = {"id": "c1", "type": "function", "function": {"name": "useTools", "arguments": "{\"tool\": \"search content x\"}"}}
    row = {
        "prompt": [
            {"role": "user", "content": "find x"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "content": "[\"docs/x.md\"]"},
        ],
        "chosen": [{"role": "assistant", "content": None, "tool_calls": [call]}],
        "rejected": [{"role": "assistant", "content": "It is in docs/x.md."}],
    }
    path = tmp_path / "dpo_train.jsonl"
    _write_jsonl(path, [row])

    train, _ = data_loader.load_and_prepare_dataset(local_file=str(path))

    prompt = train[0]["prompt"]
    assert [m["role"] for m in prompt] == ["user", "assistant", "user"]
    assert "useTools" in prompt[1]["content"] and "search content x" in prompt[1]["content"]
    assert "useTools" in train[0]["chosen"][0]["content"]
    assert train[0]["rejected"][0]["content"] == "It is in docs/x.md."
    assert all("tool_calls" not in m or m["tool_calls"] is None for m in prompt)
