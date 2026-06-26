"""P0 (CPU/CI) argv-contract tests for the ACE-STEP subprocess boundary.

These pin the config -> ``train.py`` argv translation in
``Trainers/ace_step/src/config_translation.py`` — the single highest silent-
failure surface in the pipeline. ``build_preprocess_argv`` / ``build_fixed_argv``
are PURE functions (config dict in, argv list out; stdlib + yaml only, no torch),
so this contract is verifiable on CPU TODAY — it does NOT belong behind the
@pytest.mark.gpu execution layer. A wrong flag here mistrains every run with NO
error surfaced until a human inspects an adapter on a GPU box.

The load-bearing case is the adapter-type branch (config_translation.py DELTA-2):
``--rank`` / ``--alpha`` are LoRA-ONLY upstream (a SEPARATE arg group from
``--lokr-linear-dim`` / ``--lokr-linear-alpha``); passing them under
``adapter.type=lokr`` is a SILENT no-op that trains with the DEFAULT rank/alpha.
``config.yaml`` defaults ``adapter.type=lokr``, so the lokr branch is the live
path — these tests fail loudly if the two scalars ever land on the wrong flag.

No source edits: this imports the pure functions by file path (hermetic, unique
module name to dodge the bare-``import`` shadow hazard) and asserts the argv.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_CT_PATH = REPO_ROOT / "Trainers" / "ace_step" / "src" / "config_translation.py"


def _load_config_translation():
    """Load config_translation.py under a UNIQUE module name (no sys.modules collision).

    A bare ``import config_translation`` after putting Trainers/ace_step/src on
    sys.path would risk shadowing a same-named module from another trainer dir
    (the documented ``data_loader``/``registry`` hazard). Loading by file path
    under ``ace_step_config_translation`` keeps this contract test hermetic.
    """
    spec = importlib.util.spec_from_file_location("ace_step_config_translation", _CT_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_CT_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ct = _load_config_translation()

# Synthetic absolute paths — the builders only str()/join these, no filesystem
# access, so non-existent dirs are fine and keep the assertions deterministic.
_REPO = Path("/repo")
_CKPT = Path("/ckpt")
_CACHE = Path("/cache")
_DATASET = Path("/dataset")
_OUT = Path("/out")


def _value_after(argv: list[str], flag: str) -> str:
    """Return the token immediately following ``flag`` (asserts present + has a value)."""
    assert flag in argv, f"{flag!r} not in argv: {argv}"
    i = argv.index(flag)
    assert i + 1 < len(argv), f"{flag!r} has no value in argv: {argv}"
    return argv[i + 1]


def _fixed_argv(adapter: dict | None = None, training: dict | None = None) -> list[str]:
    config: dict = {"model": {"variant": "base"}}
    if adapter is not None:
        config["adapter"] = adapter
    if training is not None:
        config["training"] = training
    return ct.build_fixed_argv(
        config, repo_root=_REPO, checkpoint_dir=_CKPT, dataset_dir=_DATASET, output_dir=_OUT
    )


# ---------------------------------------------------------------------------
# Adapter-type branch — the silent-no-op surface (DELTA-2)
# ---------------------------------------------------------------------------

def test_lokr_branch_emits_lokr_knobs_and_omits_lora_knobs():
    """adapter.type=lokr (config.yaml DEFAULT) -> --lokr-linear-dim/-alpha, NEVER --rank/--alpha."""
    argv = _fixed_argv(adapter={"type": "lokr", "rank": 16, "alpha": 32})

    assert _value_after(argv, "--adapter-type") == "lokr"
    assert _value_after(argv, "--lokr-linear-dim") == "16"
    assert _value_after(argv, "--lokr-linear-alpha") == "32"
    # The LoRA-only knobs must be ABSENT — passing them under lokr is a silent no-op.
    assert "--rank" not in argv, f"--rank leaked into a lokr run (silent no-op): {argv}"
    assert "--alpha" not in argv, f"--alpha leaked into a lokr run (silent no-op): {argv}"


def test_lora_branch_emits_lora_knobs_and_omits_lokr_knobs():
    """adapter.type=lora -> --rank/--alpha, NEVER the lokr knobs (the inverse branch)."""
    argv = _fixed_argv(adapter={"type": "lora", "rank": 8, "alpha": 16})

    assert _value_after(argv, "--adapter-type") == "lora"
    assert _value_after(argv, "--rank") == "8"
    assert _value_after(argv, "--alpha") == "16"
    assert "--lokr-linear-dim" not in argv, f"lokr knob leaked into a lora run: {argv}"
    assert "--lokr-linear-alpha" not in argv, f"lokr knob leaked into a lora run: {argv}"


def test_unknown_adapter_type_falls_back_to_lora_knobs():
    """No adapter.type -> lora-flag fallback (upstream's own default), and NO --adapter-type.

    Covers M-g: the line-349 ``normalized_type = ... if type is not None else 'lora'``
    fallback. A missing/typo'd type must NOT silently route the two scalars onto the
    lokr flags. --adapter-type is omitted entirely when type is unset (config-driven:
    absent key -> upstream default, never a hardcoded choice).
    """
    argv = _fixed_argv(adapter={"rank": 8, "alpha": 16})  # no "type" key

    assert "--adapter-type" not in argv, f"--adapter-type emitted with no configured type: {argv}"
    assert _value_after(argv, "--rank") == "8"
    assert _value_after(argv, "--alpha") == "16"
    assert "--lokr-linear-dim" not in argv
    assert "--lokr-linear-alpha" not in argv


# ---------------------------------------------------------------------------
# Preprocess diversion form — `fixed --preprocess`, not a `preprocess` subcommand
# ---------------------------------------------------------------------------

def test_preprocess_uses_fixed_preprocess_diversion_form():
    """Stage-1 is `train.py fixed --preprocess ...` (the FLAG diverts the `fixed` subparser)."""
    argv = ct.build_preprocess_argv(
        {"model": {"variant": "base"}}, checkpoint_dir=_CKPT, cache_dir=_CACHE, repo_root=_REPO
    )

    assert "fixed" in argv, f"preprocess missing the required `fixed` subparser token: {argv}"
    # --preprocess must IMMEDIATELY follow `fixed` (the diversion contract).
    assert argv[argv.index("fixed") + 1] == "--preprocess", (
        f"`fixed` must be immediately followed by --preprocess: {argv}"
    )
    assert _value_after(argv, "--tensor-output") == str(_CACHE)
    assert _value_after(argv, "--checkpoint-dir") == str(_CKPT)
    # No dataset config -> the default corpus dir as --audio-dir (one input source).
    assert _value_after(argv, "--audio-dir").endswith("Datasets/ace_step_corpus")
    # It must NOT also emit --dataset-json (exactly one input source).
    assert "--dataset-json" not in argv


def test_preprocess_prefers_dataset_json_over_audio_dir():
    """When dataset_json is set it WINS over data_dir (a labeled index is more specific)."""
    argv = ct.build_preprocess_argv(
        {"dataset": {"dataset_json": "/data/index.json", "data_dir": "/data/audio"}},
        checkpoint_dir=_CKPT,
        cache_dir=_CACHE,
        repo_root=_REPO,
    )
    assert _value_after(argv, "--dataset-json").endswith("index.json")
    assert "--audio-dir" not in argv, f"both input sources emitted: {argv}"


# ---------------------------------------------------------------------------
# Config-driven omission — absent keys emit NO flag (no hardcoded run-specific default)
# ---------------------------------------------------------------------------

def test_absent_training_keys_emit_no_flags():
    """The `if x is not None` guards mean an unset scalar emits NOTHING (SACROSANCT: config-driven).

    A regression that hardcoded a fallback (e.g. always --lr 1e-4) would flip these
    `not in` asserts. Required structural flags stay present.
    """
    argv = _fixed_argv()  # no training block, no adapter block

    for absent in ("--lr", "--epochs", "--batch-size", "--gradient-accumulation",
                   "--seed", "--save-every", "--precision", "--device",
                   "--adapter-type", "--rank", "--alpha",
                   "--lokr-linear-dim", "--lokr-linear-alpha", "--target-modules"):
        assert absent not in argv, f"{absent} emitted for an absent config key: {argv}"

    # The always-required structural flags remain.
    assert _value_after(argv, "--dataset-dir") == str(_DATASET)
    assert _value_after(argv, "--output-dir") == str(_OUT)
    assert _value_after(argv, "--checkpoint-dir") == str(_CKPT)


def test_fixed_argv_passes_through_training_scalars():
    """Present training scalars map to their flags (the positive of the omission test)."""
    argv = _fixed_argv(
        training={"learning_rate": 1e-4, "epochs": 3, "train_batch_size": 2,
                  "gradient_accumulation": 4, "seed": 7, "save_every": 100},
    )
    assert _value_after(argv, "--lr") == str(1e-4)
    assert _value_after(argv, "--epochs") == "3"
    assert _value_after(argv, "--batch-size") == "2"
    assert _value_after(argv, "--gradient-accumulation") == "4"
    assert _value_after(argv, "--seed") == "7"
    assert _value_after(argv, "--save-every") == "100"
    # EPOCHS, not steps — upstream has no --max-steps (contract §1.3).
    assert "--max-steps" not in argv
