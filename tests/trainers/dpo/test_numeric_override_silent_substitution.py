"""Regression tests for the FIFTH silent-substitution instance (now HARDENED).

TEST-phase finding (phase1-pipeline, focus item 7), hardened in #41. The DPO/KTO
trainers' `apply_cli_overrides` hardened seed/beta/lora to `is not None` guards
so that an explicit falsy value (seed=0, beta=0.0, lora-dropout=0.0) overrides
the config default instead of being silently dropped (the provenance discipline
of arch §(b.2): a run record must never claim a value the trainer didn't use).

The three sibling NUMERIC hyperparameters in the SAME function originally still
used the older truthy guard (`if args.X:`), so an explicit `--max-seq-length 0` /
`--batch-size 0` / `--num-epochs 0` was silently dropped to the config default —
the same silent-substitution SIGNATURE. #41 hardened them to `is not None` across
ALL THREE trainers:

    train_dpo.py  apply_cli_overrides:  max_seq_length / batch_size / num_epochs
    train_kto.py  apply_cli_overrides:  same three (elif batch_size keeps adaptive precedence)
    train_sft.py  same convention (pre-existing tuner-wide)

SEVERITY (TEST-phase calibration): was YELLOW, latent — NOT an active corruption
path for the pre-registered PROTOCOL v0.3 matrix (the Phase-1 recipes pin these
fields to non-zero values and the WS-5 materializer only overrides
seed / learning_rate / beta). The lead ruled HARDEN under the same no-silent-
override discipline as seed/beta: a silent-substitution hole is worth closing
regardless of whether today's matrix exercises the falsy value.

These tests now PIN the hardened contract: an explicit 0 reaches config rather
than being dropped. They exercise the DPO trainer directly (import-light); the
KTO/SFT hardening is verified by source-scan in the sibling trainer test files.
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


# --- Non-falsy values forward correctly (the guard is not broken in general) ---

def test_num_epochs_nonzero_overrides():
    config = _config_after(["--num-epochs", "3"])
    assert config.training.num_train_epochs == 3


def test_batch_size_nonzero_overrides():
    config = _config_after(["--batch-size", "8"])
    assert config.training.per_device_train_batch_size == 8


def test_max_seq_length_nonzero_overrides():
    config = _config_after(["--max-seq-length", "4096"])
    assert config.model.max_seq_length == 4096


# --- The fifth silent-substitution instance: falsy values now FORWARD (hardened) ---
# These pin the HARDENED behavior, matching test_seed_override_honors_zero /
# test_beta_override_honors_zero in the sibling test_seed_override.py.

def test_num_epochs_zero_overrides_hardened_guard():
    """`if args.num_epochs is not None:` forwards --num-epochs 0 to config.

    The provenance-correct guard sets num_train_epochs == 0 rather than silently
    keeping the config default (closes the fifth silent-substitution instance).
    """
    config = _config_after(["--num-epochs", "0"])
    assert config.training.num_train_epochs == 0


def test_batch_size_zero_overrides_hardened_guard():
    """`if args.batch_size is not None:` forwards --batch-size 0 to config."""
    config = _config_after(["--batch-size", "0"])
    assert config.training.per_device_train_batch_size == 0


def test_max_seq_length_zero_overrides_hardened_guard():
    """`if args.max_seq_length is not None:` forwards --max-seq-length 0 to config."""
    config = _config_after(["--max-seq-length", "0"])
    assert config.model.max_seq_length == 0
