from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_train_sft_uses_runtime_compatible_trl_tokenizer_kwarg() -> None:
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    assert "load_and_prepare_tokenized_dataset" in source
    assert "collate_prepared_sft_batch" in source
    assert "Trainer(" in source
    assert '"dataset_representation": "tokenized"' in source
    assert "dataset_text_field" not in source


def test_train_sft_seed_override_is_wired_with_is_not_none() -> None:
    # train_sft imports unsloth at module load, so verify the --seed flag and its
    # is-not-None override (honoring seed=0) at the source level.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    assert '"--seed"' in source
    assert "if args.seed is not None:" in source
    assert "config.seed = args.seed" in source
