"""Source-level tests for the KTO trainer's --seed CLI override.

train_kto imports unsloth at module load, so the --seed flag and its
is-not-None override (honoring seed=0) are verified at the source level,
mirroring tests/trainers/sft/test_train_sft_source.py.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_train_kto_seed_override_is_wired_with_is_not_none() -> None:
    source = (REPO_ROOT / "Trainers" / "kto" / "train_kto.py").read_text(encoding="utf-8")

    assert '"--seed"' in source
    assert "if args.seed is not None:" in source
    assert "config.seed = args.seed" in source


def test_train_kto_beta_override_is_wired_with_is_not_none() -> None:
    # The handler forwards an explicit --beta 0.0 (is-not-None at the boundary); the
    # trainer must honor it with the same is-not-None guard, not a truthy guard that
    # would silently swap 0.0 for the config default (provenance: no silent override).
    source = (REPO_ROOT / "Trainers" / "kto" / "train_kto.py").read_text(encoding="utf-8")

    assert "if args.beta is not None:" in source
    assert "config.training.beta = args.beta" in source
    assert "if args.beta:" not in source


def test_train_kto_lora_parity_flags_are_wired() -> None:
    # LoRA flag parity (§5.2 identical-budget confound control): the handler emits
    # --lora-* for dpo/kto, so the KTO trainer must accept them and thread the value
    # into config.lora.* (the SSOT), not silently fall back to config.yaml.
    source = (REPO_ROOT / "Trainers" / "kto" / "train_kto.py").read_text(encoding="utf-8")

    for flag in ('"--lora-r"', '"--lora-alpha"', '"--lora-dropout"', '"--lora-target-modules"'):
        assert flag in source, f"KTO trainer missing argparse for {flag}"

    assert "if args.lora_r is not None:" in source
    assert "config.lora.r = args.lora_r" in source
    assert "if args.lora_target_modules is not None:" in source
    assert "config.lora.target_modules = " in source
