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


def test_train_sft_numeric_overrides_are_hardened_is_not_none() -> None:
    # The fifth silent-substitution instance (focus item 7), hardened in #41:
    # batch_size / num_epochs / max_seq_length / gradient_accumulation /
    # learning_rate override guards must use is not None so an explicit 0 forwards
    # to config rather than being dropped to the default.
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    for guard in (
        "if args.batch_size is not None:",
        "if args.num_epochs is not None:",
        "if args.max_seq_length is not None:",
        "if args.gradient_accumulation is not None:",
        "if args.learning_rate is not None:",
    ):
        assert guard in source, f"SFT trainer missing hardened guard: {guard}"
    # The pre-hardening truthy guards must be gone.
    for stale in (
        "if args.batch_size:",
        "if args.num_epochs:",
        "if args.max_seq_length:",
        "if args.gradient_accumulation:",
        "if args.learning_rate:",
    ):
        assert stale not in source, f"SFT trainer still has truthy guard: {stale}"
