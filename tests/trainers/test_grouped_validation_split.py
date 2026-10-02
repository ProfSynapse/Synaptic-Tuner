"""validation_group_key wiring through the SFT, KTO and DPO data loaders.

Each trainer ships a module named ``data_loader``; they are loaded here under
unique names so this file does not collide with the per-trainer test modules.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]


def _load(trainer: str):
    src = ROOT / "Trainers" / trainer / "src"
    spec = importlib.util.spec_from_file_location(f"_grouped_{trainer}_data_loader", src / "data_loader.py")
    module = importlib.util.module_from_spec(spec)
    # The SFT loader imports its sibling ``preprocessing`` module.
    sys.path.insert(0, str(src))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(src))
    return module


class _FakeTokenizer:
    eos_token_id = 77

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "\n".join(f"{message['role']}::{message['content']}" for message in messages)
        return rendered + ("\nassistant::" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 for char in text]


def _conversation_rows(n_groups=6, per_group=3, label=None):
    rows = []
    for g in range(n_groups):
        for v in range(per_group):
            row = {
                "conversations": [
                    {"role": "user", "content": f"scenario {g} variant {v}"},
                    {"role": "assistant", "content": f"answer {g}/{v}"},
                ],
                "metadata": {"scenario": f"scn-{g}"},
            }
            if label is not None:
                row["label"] = (v % 2 == 0)
            rows.append(row)
    return rows


def _groups_of(dataset, column):
    return {value for value in dataset[column]}


def test_sft_text_loader_keeps_groups_whole(monkeypatch):
    loader = _load("sft")
    rows = _conversation_rows()
    monkeypatch.setattr(loader, "load_dataset", lambda *a, **k: Dataset.from_list(rows))

    train, validation = loader.load_and_prepare_dataset(
        local_file="unused.jsonl",
        split_dataset=True,
        test_size=0.34,
        validation_group_key="metadata.scenario",
    )
    train_groups = {row["metadata"]["scenario"] for row in train}
    validation_groups = {row["metadata"]["scenario"] for row in validation}
    assert train_groups.isdisjoint(validation_groups)
    assert len(validation_groups) == 3  # ceil(0.34 * 6)
    assert len(train) + len(validation) == len(rows)


def test_sft_tokenized_loader_reads_groups_before_columns_are_dropped(monkeypatch):
    loader = _load("sft")
    rows = _conversation_rows()
    monkeypatch.setattr(loader, "load_dataset", lambda *a, **k: Dataset.from_list(rows))

    kwargs = dict(
        local_file="unused.jsonl",
        tokenizer=_FakeTokenizer(),
        loss_mask_mode="full_sequence",
        split_dataset=True,
        test_size=0.34,
        validation_group_key="metadata.scenario",
    )
    train, validation = loader.load_and_prepare_tokenized_dataset(**kwargs)
    assert train.column_names == ["input_ids", "attention_mask", "labels"]
    # Each group is 3 rows; whole groups only, so both sides are multiples of 3.
    assert len(validation) == 9 and len(train) == 9

    again_train, again_validation = loader.load_and_prepare_tokenized_dataset(**kwargs)
    assert again_validation["input_ids"] == validation["input_ids"]
    assert again_train["input_ids"] == train["input_ids"]


def test_sft_missing_group_key_names_row(monkeypatch):
    loader = _load("sft")
    rows = _conversation_rows()
    rows[5]["metadata"] = {"other": "x"}
    monkeypatch.setattr(loader, "load_dataset", lambda *a, **k: Dataset.from_list(rows))
    with pytest.raises(ValueError, match=r"row index 5"):
        loader.load_and_prepare_tokenized_dataset(
            local_file="unused.jsonl",
            tokenizer=_FakeTokenizer(),
            loss_mask_mode="full_sequence",
            split_dataset=True,
            validation_group_key="metadata.scenario",
        )


def test_sft_group_key_rejects_preassigned_splits(monkeypatch):
    loader = _load("sft")
    monkeypatch.setattr(loader, "load_dataset", lambda *a, **k: Dataset.from_list(_conversation_rows()))
    with pytest.raises(ValueError, match="use_preassigned_splits"):
        loader.load_and_prepare_tokenized_dataset(
            local_file="unused.jsonl",
            tokenizer=_FakeTokenizer(),
            loss_mask_mode="full_sequence",
            use_preassigned_splits=True,
            validation_group_key="metadata.scenario",
        )


def test_sft_without_group_key_keeps_random_split(monkeypatch):
    loader = _load("sft")
    rows = _conversation_rows()
    monkeypatch.setattr(loader, "load_dataset", lambda *a, **k: Dataset.from_list(rows))
    train, validation = loader.load_and_prepare_dataset(
        local_file="unused.jsonl", split_dataset=True, test_size=0.2
    )
    expected = (
        Dataset.from_list(rows)
        .rename_column("conversations", "messages")
        .train_test_split(test_size=0.2, seed=42)
    )
    assert train["messages"] == expected["train"]["messages"]
    assert validation["messages"] == expected["test"]["messages"]


def test_kto_loader_keeps_groups_whole_after_projection(monkeypatch):
    loader = _load("kto")
    rows = _conversation_rows(label=True)
    # A row the KTO projection drops (no assistant turn) must not misalign groups.
    rows.insert(4, {"conversations": [{"role": "user", "content": "orphan"}], "label": True,
                    "metadata": {"scenario": "scn-orphan"}})
    monkeypatch.setattr(loader, "load_dataset", lambda *a, **k: Dataset.from_list(rows))

    train, validation = loader.load_and_prepare_dataset(
        local_file="unused.jsonl", split_dataset=True, test_size=0.34, validation_group_key="metadata.scenario"
    )
    def scenario(prompt):
        return prompt.split(" variant ")[0]
    train_groups = {scenario(p) for p in train["prompt"]}
    validation_groups = {scenario(p) for p in validation["prompt"]}
    assert train_groups.isdisjoint(validation_groups)
    assert len(train) + len(validation) == 18


def test_dpo_loader_reads_groups_before_dropping_provenance(monkeypatch):
    loader = _load("dpo")
    rows = [
        {
            "prompt": [{"role": "user", "content": f"q{g}"}],
            "chosen": [{"role": "assistant", "content": f"good {g}/{v}"}],
            "rejected": [{"role": "assistant", "content": f"bad {g}/{v}"}],
            "provenance": {"prompt_key": f"key-{g}"},
        }
        for g in range(5)
        for v in range(2)
    ]
    monkeypatch.setattr(loader, "load_dataset", lambda *a, **k: Dataset.from_list(rows))

    train, validation = loader.load_and_prepare_dataset(
        local_file="unused.jsonl", split_dataset=True, test_size=0.2, validation_group_key="provenance.prompt_key"
    )
    assert train.column_names == ["prompt", "chosen", "rejected"]
    train_prompts = {row["prompt"][0]["content"] for row in train}
    validation_prompts = {row["prompt"][0]["content"] for row in validation}
    assert train_prompts.isdisjoint(validation_prompts)
    assert len(validation) == 2


@pytest.mark.parametrize("trainer", ["sft", "kto", "dpo"])
def test_config_loaders_expose_validation_group_key(tmp_path, trainer):
    import yaml

    path = ROOT / "Trainers" / trainer / "configs" / "config_loader.py"
    spec = importlib.util.spec_from_file_location(f"_grouped_{trainer}_config_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    default = module.load_config()
    assert default.dataset.validation_group_key is None

    raw = yaml.safe_load((path.parent / "config.yaml").read_text(encoding="utf-8"))
    raw["dataset"]["validation_group_key"] = "metadata.scenario"
    custom = tmp_path / "config.yaml"
    custom.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert module.load_config(str(custom)).dataset.validation_group_key == "metadata.scenario"
