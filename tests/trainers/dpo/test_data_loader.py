"""Smoke tests for the DPO data loader (prompt/chosen/rejected schema).

These import only `datasets` (no trl/unsloth/torch), so they exercise the
config-validation dry-run contract without the ML stack.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "dpo" / "src"))

import data_loader  # noqa: E402


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
