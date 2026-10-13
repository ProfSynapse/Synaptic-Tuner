"""Real local JSON -> Dataset -> SFT features, usable with an installed package."""
import json
from pathlib import Path

import pytest

pytest.importorskip("datasets")
from datasets import config as datasets_config
from Trainers.sft import runtime_v1
from tests.trainers import _trainer_import
from tests.training.test_conversation_sft_support import row, Tokenizer, ROW_SCHEMA


@pytest.fixture
def loader(tmp_path, monkeypatch):
    monkeypatch.setattr(_trainer_import, "TRAINERS_ROOT", Path(runtime_v1.__file__).resolve().parent.parent)
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(tmp_path / "dataset-cache"))
    result = _trainer_import.load_trainer_module("sft", "data_loader")
    assert Path(result.__file__).resolve().is_relative_to(Path(runtime_v1.__file__).resolve().parent.parent)
    return result


def write_rows(tmp_path, rows):
    path = tmp_path / "conversation.jsonl"
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")
    return path


def prepare(loader, path, tokenizer, **kwargs):
    return loader.load_and_prepare_tokenized_dataset(local_file=str(path), tokenizer=tokenizer,
        num_proc=1, use_preassigned_splits=True, prompt_render="prompt_completion",
        loss_mask_mode="assistant_only", expected_prepared_schema=ROW_SCHEMA, max_seq_length=1024,
        **kwargs)


def test_actual_dataset_preserves_new_identity_and_final_target_features(loader, tmp_path):
    path = write_rows(tmp_path, [row("train", 1), row("validation", 2)])
    tokenizer = Tokenizer()
    train, validation = prepare(loader, path, tokenizer, chat_template_kwargs={"explicit_option": False})
    assert len(train) == len(validation) == 1
    assert train.column_names == ["input_ids", "attention_mask", "labels"]
    labels = train[0]["labels"]
    expected = tokenizer.encode("only this target") + [tokenizer.eos_token_id]
    assert labels[-len(expected):] == expected
    assert labels[:-len(expected)] == [-100] * (len(labels) - len(expected))
    assert any(messages == row()["messages"][:-1] and add and options == {"explicit_option": False}
               for messages, add, options in tokenizer.renders)
    preprocessing = _trainer_import.load_trainer_module("sft", "preprocessing")
    assert preprocessing.normalize_sft_example(row()) == row()


@pytest.mark.parametrize("attack", ["v2_identity", "mixed_schema", "structured", "bool_index"])
def test_actual_dataset_rejects_identity_or_row_mismatch_before_tokenization(loader, tmp_path, attack):
    values = [row("train", 1), row("validation", 2)]
    expected = ROW_SCHEMA
    if attack == "v2_identity":
        expected = "syntunia-sft-row/v2"
    elif attack == "mixed_schema":
        values[1]["schema_version"] = "syntunia-sft-row/v2"
    elif attack == "structured":
        for value in values:
            value["messages"][-1]["content"] = {"text": "unadmitted content"}
    else:
        for value in values:
            value["source_row_number"] = True
    path = write_rows(tmp_path, values)
    tokenizer = Tokenizer()
    with pytest.raises(ValueError):
        loader.load_and_prepare_tokenized_dataset(local_file=str(path), tokenizer=tokenizer,
            num_proc=1, use_preassigned_splits=True, prompt_render="prompt_completion", loss_mask_mode="assistant_only",
            expected_prepared_schema=expected, max_seq_length=1024)
    assert tokenizer.renders == []


def test_actual_preprocessing_cannot_drop_authority_or_mix_new_and_old(loader):
    from datasets import Dataset
    preprocessing = _trainer_import.load_trainer_module("sft", "preprocessing")
    values = [row("train", 1), row("validation", 2)]
    values[1]["schema_version"] = "syntunia-sft-row/v2"
    with pytest.raises(ValueError, match="mix dataset schemas"):
        preprocessing.prepare_sft_dataset(Dataset.from_list(values), tokenizer=Tokenizer(),
            use_preassigned_splits=True, prompt_render="prompt_completion", loss_mask_mode="assistant_only", max_seq_length=1024)
