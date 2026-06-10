"""Behavioral tests for the DPO trainer's --seed CLI override.

train_dpo's parser + apply_cli_overrides are import-light (config/presets only,
no trl/unsloth/torch), so the seed-override precedence can be exercised directly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "dpo"))

import train_dpo  # noqa: E402


def _config_after(argv):
    config = train_dpo.load_config()
    args = train_dpo.build_arg_parser().parse_args(argv)
    return train_dpo.apply_cli_overrides(config, args)


def test_seed_override_sets_config_seed():
    config = _config_after(["--seed", "1234"])
    assert config.seed == 1234


def test_seed_override_honors_zero():
    # `is not None` guard: seed=0 is a legitimate seed and must override the default.
    config = _config_after(["--seed", "0"])
    assert config.seed == 0


def test_seed_absent_preserves_default():
    default_seed = train_dpo.load_config().seed
    config = _config_after([])
    assert config.seed == default_seed


def test_beta_override_sets_config_beta():
    config = _config_after(["--beta", "0.05"])
    assert config.training.beta == 0.05


def test_beta_override_honors_zero():
    # `is not None` guard mirrors --seed: an explicit --beta 0.0 forwarded by the
    # handler must override the config default, not be silently swapped (provenance).
    # A truthy guard would drop it — the silent-substitution class this guards against.
    config = _config_after(["--beta", "0.0"])
    assert config.training.beta == 0.0


def test_beta_absent_preserves_default():
    default_beta = train_dpo.load_config().training.beta
    config = _config_after([])
    assert config.training.beta == default_beta


def test_lora_overrides_set_config_lora():
    # The recipe LoRA budget is the SSOT (§5.2 identical-budget confound control):
    # the --lora-* parity flags must override config.lora.* so a recipe pinning
    # r=64/alpha=128 trains at that budget, not the trainer config.yaml default.
    config = _config_after(
        [
            "--lora-r", "64",
            "--lora-alpha", "128",
            "--lora-dropout", "0.05",
            "--lora-target-modules", "q_proj,v_proj",
        ]
    )
    assert config.lora.r == 64
    assert config.lora.lora_alpha == 128
    assert config.lora.lora_dropout == 0.05
    assert config.lora.target_modules == ["q_proj", "v_proj"]


def test_lora_overrides_absent_preserve_default():
    default = train_dpo.load_config().lora
    config = _config_after([])
    assert config.lora.r == default.r
    assert config.lora.lora_alpha == default.lora_alpha
