"""
ACE-STEP config -> `train.py` argv translation (the §1.3 flag table, in code).

Location: Trainers/ace_step/src/config_translation.py
Purpose:  Translate our single config.yaml (§2) into the ACE-STEP `train.py` argv
          for BOTH stages, and resolve the three path inputs the stages share:
          the multi-file checkpoint dir (from model_registry.yaml), the .pt tensor
          cache dir, and the adapter output dir. This module is the single home of
          the §1.3 flag-name knowledge — so the build-time `--help` byte-confirm has
          exactly one place to correct if a spelling differs.
Used by:  Trainers/ace_step/train_ace_step.py (build_fixed_argv + the 3 resolvers),
          Trainers/ace_step/src/data_loader.py (build_preprocess_argv).

Contract: docs/architecture/ace-step-pipeline-contract.md §1.3 (flag table), §2
          (config schema), §3 (model registry), §5 (data/cache paths).

⚠️ PROVISIONAL: the flag spellings/defaults below were read by preparer-acestep via
raw-GitHub fetch of acestep/training_v2/cli/args.py, NOT by executing `--help`. The
§1.3 BUILD STEP-0 byte-confirm runs `python train.py fixed --help` (the `fixed`
subcommand) + `python train.py --help` (the root parser — `--preprocess` is a
root-level store_true FLAG, NOT a `preprocess` subcommand) against the provisioned
ACE-STEP repo to byte-confirm every flag before the translation is "locked". If any
flag differs, correct it HERE and ping the architect (design authority) before
locking. The wrapper `--dry-run` prints what this module produces.

Config-driven SACROSANCT: every value is read from config — nothing scenario-specific
is baked in. A value absent from config falls back to the ACE-STEP upstream default
(documented inline), never to a hardcoded run-specific choice.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# The ACE-STEP repo location differs by lane (§9): the build-host / dry-run lane finds
# it at the gitignored vendor/ clone under the repo root; the in-CONTAINER lane finds it
# wherever devops installs it in the Docker image. The ACE_STEP_HOME env var is the
# single seam that reconciles both: devops sets `ENV ACE_STEP_HOME=/opt/ace-step` in the
# image; when unset (build host / local dry-run) we fall back to repo_root/vendor/...
# This unbreaks the real container run (which would otherwise file-not-found on the
# vendor path — masked today only because the smoke recipe uses --dry-run). The env-var
# NAME must match devops's Dockerfile ENV exactly (coordinated: ACE_STEP_HOME).
ACE_STEP_HOME_ENV = "ACE_STEP_HOME"
ACE_STEP_REPO_SUBPATH = "vendor/ACE-Step-1.5"
ACE_STEP_TRAIN_SCRIPT = "train.py"

# Default gitignored landing dir for the .pt tensor cache when cache_dir is empty
# and no data_dir is set to anchor a default under (§5.1).
DEFAULT_CORPUS_SUBPATH = "Datasets/ace_step_corpus"
DEFAULT_CACHE_SUBPATH = "Datasets/ace_step_corpus/.cache"


def resolve_ace_step_home(repo_root: Path) -> Path:
    """Resolve the ACE-STEP repo root, honoring the ACE_STEP_HOME env-var seam.

    Priority:
        1. $ACE_STEP_HOME (set by devops in the container image, e.g. /opt/ace-step).
        2. repo_root/vendor/ACE-Step-1.5 (the gitignored build-host / dry-run clone).

    One var, both lanes — so the SAME wrapper code drives the build-host dry-run and
    the in-container real run without a path edit.
    """
    env_home = os.environ.get(ACE_STEP_HOME_ENV)
    if env_home:
        return Path(env_home)
    return repo_root / ACE_STEP_REPO_SUBPATH


def ace_step_train_argv_prefix(repo_root: Path) -> list[str]:
    """Return ["python", "<ace_step_home>/train.py"] — the shell prefix.

    The actual subcommand + flags are appended by the argv builders. ACE-STEP-repo
    location resolution (the ACE_STEP_HOME seam) is centralized in resolve_ace_step_home
    so the dual-lane path logic lives in exactly one place.
    """
    train_script = resolve_ace_step_home(repo_root) / ACE_STEP_TRAIN_SCRIPT
    return ["python", str(train_script)]


def _load_model_registry(repo_root: Path) -> dict[str, Any]:
    """Load Trainers/ace_step/configs/model_registry.yaml (the checkpoint SSOT)."""
    registry_path = repo_root / "Trainers" / "ace_step" / "configs" / "model_registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(f"model_registry.yaml not found: {registry_path}")
    with open(registry_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("models", {}) or {}


def _resolve_registry_entry(
    config: dict[str, Any], repo_root: Path
) -> tuple[str, dict[str, Any]]:
    """Resolve model.registry_name -> (registry_name, registry-entry dict).

    Shared by resolve_checkpoint_dir (the path namer) and fetch_checkpoint (the
    weights materializer) so the SSOT lookup + validation live in exactly one place.

    Raises:
        ValueError: model.registry_name is missing or not in the registry.
    """
    model_cfg = config.get("model", {})
    registry_name = model_cfg.get("registry_name")
    if not registry_name:
        raise ValueError("config.model.registry_name is required")

    models = _load_model_registry(repo_root)
    if registry_name not in models:
        raise ValueError(
            f"Unknown model.registry_name {registry_name!r}; "
            f"known: {sorted(models.keys())}"
        )
    return registry_name, (models[registry_name] or {})


def resolve_checkpoint_dir(config: dict[str, Any], repo_root: Path) -> Path:
    """Resolve model.registry_name -> the local base-checkpoint folder (--checkpoint-dir).

    PURE path namer — does NO network/IO beyond reading the registry YAML, so it is
    safe on the --dry-run / argv-build path (it runs BEFORE the dry-run early-return in
    train_ace_step.run, and build_fixed_argv needs the path to render the dry-run argv).
    The actual weight materialization (snapshot_download at the pinned revision) is the
    separate fetch_checkpoint() below, invoked only on a real run.

    Concrete folder = <repo>/Datasets/ace_step_models/<registry_name> — a documented,
    gitignored landing dir (mirrors the corpus default). fetch_checkpoint() populates it
    from the registry entry's hf_id @ revision; here we only NAME it deterministically.

    Raises:
        ValueError: model.registry_name is missing or not in the registry.
    """
    registry_name, _entry = _resolve_registry_entry(config, repo_root)
    return repo_root / "Datasets" / "ace_step_models" / registry_name


def resolve_dit_subfolder(config: dict[str, Any], repo_root: Path) -> str | None:
    """Return the DiT component subfolder for the selected registry entry (or None).

    Reads `components.dit` from the registry entry (§3). ACE-STEP's own train.py maps
    --model-variant -> the DiT subdir via its internal VARIANT_DIR_MAP, so this is
    informational/best-effort (used by fetch_checkpoint's idempotency check + surfaced
    for diagnostics). For the XL entry the subfolder is best-effort per the §3 XL
    DIR-MAP caution (VARIANT_DIR_MAP only carries the three 2B keys); a None/absent
    value simply disables the idempotency short-circuit (always (re)fetches).
    """
    _registry_name, entry = _resolve_registry_entry(config, repo_root)
    components = entry.get("components") or {}
    dit = components.get("dit")
    return str(dit) if dit else None


def fetch_checkpoint(
    config: dict[str, Any],
    repo_root: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> Path:
    """Materialize the base checkpoint locally at the registry-PINNED revision.

    This is the M-a wiring that makes the `revision` pin in model_registry.yaml load-
    bearing instead of decorative: it threads the entry's `hf_id` + `revision` into a
    `huggingface_hub.snapshot_download(repo_id=..., revision=...)` so a real run pulls
    the EXACT pinned commit (upstream model_downloader.py otherwise defaults to `main`).

    Returns the same local dir as resolve_checkpoint_dir() (the --checkpoint-dir target).

    Behavior:
      - dry_run=True: resolve + RETURN the dir WITHOUT any download (keeps --dry-run
        zero-network; the wrapper's dry-run path uses this).
      - already-populated (the DiT subfolder exists under the dir) and not force:
        SKIP the download (idempotent — re-pulling multi-GB weights is expensive),
        return the dir.
      - otherwise: lazy-import snapshot_download and pull hf_id @ revision into the dir.

    The live network call only happens on a real (non-dry-run) run with weights absent —
    deferred with the GPU/image smoke — but the revision THREADING + call wiring land now
    and are unit-tested (the download is mocked; no live network in CI).

    Raises:
        ValueError: model.registry_name missing/unknown, or the entry lacks hf_id.
    """
    registry_name, entry = _resolve_registry_entry(config, repo_root)
    checkpoint_dir = repo_root / "Datasets" / "ace_step_models" / registry_name

    if dry_run:
        return checkpoint_dir

    hf_id = entry.get("hf_id")
    if not hf_id:
        raise ValueError(
            f"registry entry {registry_name!r} is missing 'hf_id'; cannot fetch weights"
        )
    # revision may be absent -> None lets snapshot_download fall back to the repo default
    # (`main`); we pass whatever the registry pins so a pinned SHA is honored verbatim.
    revision = entry.get("revision")

    dit_subfolder = resolve_dit_subfolder(config, repo_root)
    already_present = bool(
        dit_subfolder and (checkpoint_dir / dit_subfolder).exists()
    )
    if already_present and not force:
        print(
            f"[ace_step:weights] {checkpoint_dir} already has '{dit_subfolder}' "
            f"-> SKIPPING snapshot_download (use force=True to re-pull)."
        )
        return checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Lazy import (repo convention: heavy HF dep imported at the call, not module load).
    from huggingface_hub import snapshot_download

    print(
        f"[ace_step:weights] snapshot_download(repo_id={hf_id!r}, "
        f"revision={revision!r}) -> {checkpoint_dir}"
    )
    snapshot_download(
        repo_id=str(hf_id),
        revision=revision,
        local_dir=str(checkpoint_dir),
    )
    return checkpoint_dir


def resolve_cache_dir(config: dict[str, Any], repo_root: Path) -> Path:
    """Resolve the .pt tensor cache dir (--tensor-output / --dataset-dir).

    Priority: dataset.cache_dir (explicit) -> a `.cache` dir under dataset.data_dir
    -> the documented gitignored default. Always an absolute path so the same value
    is valid for both preprocess (writes) and fixed (reads).
    """
    dataset_cfg = config.get("dataset", {})
    cache_dir = (dataset_cfg.get("cache_dir") or "").strip()
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()

    data_dir = (dataset_cfg.get("data_dir") or "").strip()
    if data_dir:
        return (Path(data_dir).expanduser().resolve() / ".cache")

    return (repo_root / DEFAULT_CACHE_SUBPATH).resolve()


def resolve_output_dir(config: dict[str, Any], repo_root: Path) -> Path:
    """Resolve the adapter output dir (--output-dir), in the ace_step_output/ layout.

    Priority: output.adapter_dir (explicit) -> a timestamped run dir under the
    canonical ace_step_output/ location. The canonical location is derived from the
    paths SSOT (get_primary_training_output_dir) so the `<method>_output` convention
    is not duplicated here.
    """
    from datetime import datetime

    output_cfg = config.get("output", {})
    adapter_dir = (output_cfg.get("adapter_dir") or "").strip()
    if adapter_dir:
        return Path(adapter_dir).expanduser().resolve()

    base = _canonical_ace_step_output_dir(repo_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / timestamp


def _canonical_ace_step_output_dir(repo_root: Path) -> Path:
    """Return the canonical ace_step output root, derived from the paths SSOT.

    Imports get_primary_training_output_dir from shared.utilities.paths (the SSOT for
    the `<method>_output` convention). The wrapper entry point already places repo_root
    on sys.path; this adds it defensively so config_translation is also importable
    standalone (e.g. by the test-engineer argv-contract test) without the bootstrap.
    """
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from shared.utilities.paths import get_primary_training_output_dir

    return get_primary_training_output_dir("ace_step", repo_root=repo_root)


def _resolve_audio_input_flags(config: dict[str, Any], repo_root: Path) -> list[str]:
    """Build the preprocess input-source flag(s): one of --audio-dir / --dataset-json.

    Exactly one of dataset.data_dir or dataset.dataset_json must be set (§4.1). If
    BOTH are set, dataset_json wins (a labeled index is more specific than a raw
    audio dir). If NEITHER is set, the documented gitignored default corpus dir
    (resolved repo-root-absolute) is used as --audio-dir.
    """
    dataset_cfg = config.get("dataset", {})
    dataset_json = (dataset_cfg.get("dataset_json") or "").strip()
    if dataset_json:
        return ["--dataset-json", str(Path(dataset_json).expanduser().resolve())]

    data_dir = (dataset_cfg.get("data_dir") or "").strip()
    if data_dir:
        return ["--audio-dir", str(Path(data_dir).expanduser().resolve())]

    # Neither set -> the documented gitignored default landing dir, resolved
    # repo-root-absolute so it is consistent with --tensor-output / --checkpoint-dir.
    return ["--audio-dir", str((repo_root / DEFAULT_CORPUS_SUBPATH).resolve())]


def build_preprocess_argv(
    config: dict[str, Any],
    *,
    checkpoint_dir: Path,
    cache_dir: Path,
    repo_root: Path,
) -> list[str]:
    """Translate config -> stage-1 preprocess argv (§1.3 stage 1).

    ⚠️ BYTE-CONFIRMED CORRECTION (vendor/ACE-Step-1.5@v0.1.8, args.py + train.py:79-109):
    preprocess is NOT a subcommand. The real subparsers are {vanilla, fixed, estimate}
    (required). Preprocessing is the `--preprocess` store_true FLAG; train.py:_dispatch
    early-returns `if getattr(args,"preprocess",False): return _run_preprocess(args)`
    BEFORE any training. So stage-1 is `train.py fixed --preprocess ...` — the `fixed`
    subcommand satisfies the required subparser, and `--preprocess` diverts into the
    two-pass preprocess pipeline and EXITS (no training). Confirmed by lead + devops.

    Flags emitted (BYTE-CONFIRMED against args.py):
        fixed                          (required subparser token; --preprocess diverts it)
        --preprocess                   (store_true FLAG that triggers _run_preprocess + exit)
        --audio-dir | --dataset-json   (input source; one required by _run_preprocess)
        --tensor-output <cache_dir>    (.pt output; required by _run_preprocess)
        --checkpoint-dir <ckpts>       (base model root)
        --model-variant <variant>
        --max-duration <s>
        --device <device>
        --precision <precision>

    Note: 48 kHz stereo resample is AUTOMATIC inside preprocess — there is NO
    --sr/--channels flag, so preprocess.target_sr/channels are NOT emitted (they are
    assertion/doc-only config keys, §4.1).
    """
    preprocess_cfg = config.get("preprocess", {})
    model_cfg = config.get("model", {})

    # `fixed` satisfies the required subparser; `--preprocess` diverts to preprocessing.
    argv = ace_step_train_argv_prefix(repo_root) + ["fixed", "--preprocess"]
    argv += _resolve_audio_input_flags(config, repo_root)
    argv += ["--tensor-output", str(cache_dir)]
    argv += ["--checkpoint-dir", str(checkpoint_dir)]

    variant = model_cfg.get("variant")
    if variant is not None:
        argv += ["--model-variant", str(variant)]

    max_duration = preprocess_cfg.get("max_duration")
    if max_duration is not None:
        argv += ["--max-duration", str(max_duration)]

    device = preprocess_cfg.get("device")
    if device is not None:
        argv += ["--device", str(device)]

    precision = preprocess_cfg.get("precision")
    if precision is not None:
        argv += ["--precision", str(precision)]

    return argv


def build_fixed_argv(
    config: dict[str, Any],
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Translate config -> `python train.py fixed ...` argv (§1.3 stage 2).

    BYTE-CONFIRMED against args.py. Flags emitted:
        --dataset-dir <cache>          (preprocessed .pt files)
        --output-dir <out>             (adapter output; remapped to ace_step_output/)
        --checkpoint-dir <ckpts>
        --model-variant <variant>
        --lr <learning_rate>
        --epochs <epochs>              (EPOCHS, not steps — there is NO --max-steps)
        --batch-size <bs>
        --gradient-accumulation <ga>
        --seed <seed>
        --save-every <n>
        --precision <precision>
        --device <device>
        --adapter-type <lora|lokr>
        --target-modules <m1> <m2> ... (nargs=+)
      ADAPTER-TYPE-DEPENDENT rank/alpha knobs (architect DELTA-2 ruling):
        type=lokr -> --lokr-linear-dim <rank> / --lokr-linear-alpha <alpha>  (DEFAULT)
        type=lora -> --rank <rank>            / --alpha <alpha>
      ⚠️ --rank/--alpha are LoRA-ONLY upstream (args.py:285 "used when --adapter-type=lora");
      passing them under lokr is a SILENT no-op. So we BRANCH on adapter.type from the same
      two config scalars (adapter.rank/adapter.alpha) — never emit lora knobs under lokr.
    """
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    # `adapter:` block (architect §2 rename, was `lora:`). adapter-agnostic.
    adapter_cfg = config.get("adapter", {})

    argv = ace_step_train_argv_prefix(repo_root) + ["fixed"]
    argv += ["--dataset-dir", str(dataset_dir)]
    argv += ["--output-dir", str(output_dir)]
    argv += ["--checkpoint-dir", str(checkpoint_dir)]

    variant = model_cfg.get("variant")
    if variant is not None:
        argv += ["--model-variant", str(variant)]

    learning_rate = training_cfg.get("learning_rate")
    if learning_rate is not None:
        argv += ["--lr", str(learning_rate)]

    epochs = training_cfg.get("epochs")
    if epochs is not None:
        argv += ["--epochs", str(epochs)]

    batch_size = training_cfg.get("train_batch_size")
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]

    grad_accum = training_cfg.get("gradient_accumulation")
    if grad_accum is not None:
        argv += ["--gradient-accumulation", str(grad_accum)]

    seed = training_cfg.get("seed")
    if seed is not None:
        argv += ["--seed", str(seed)]

    save_every = training_cfg.get("save_every")
    if save_every is not None:
        argv += ["--save-every", str(save_every)]

    precision = training_cfg.get("precision")
    if precision is not None:
        argv += ["--precision", str(precision)]

    device = training_cfg.get("device")
    if device is not None:
        argv += ["--device", str(device)]

    adapter_type = adapter_cfg.get("type")
    if adapter_type is not None:
        argv += ["--adapter-type", str(adapter_type)]

    # rank/alpha translate to DIFFERENT upstream flags by adapter type (DELTA-2 ruling):
    # lora uses --rank/--alpha; lokr uses --lokr-linear-dim/--lokr-linear-alpha. Branch
    # on the NORMALIZED type so the two scalars never land on the wrong (silently-ignored)
    # flag. An unknown/None type falls back to the lora flags (upstream's own default type).
    rank = adapter_cfg.get("rank")
    alpha = adapter_cfg.get("alpha")
    normalized_type = str(adapter_type).strip().lower() if adapter_type is not None else "lora"
    rank_flag, alpha_flag = (
        ("--lokr-linear-dim", "--lokr-linear-alpha")
        if normalized_type == "lokr"
        else ("--rank", "--alpha")
    )
    if rank is not None:
        argv += [rank_flag, str(rank)]
    if alpha is not None:
        argv += [alpha_flag, str(alpha)]

    target_modules = adapter_cfg.get("target_modules")
    if target_modules:
        argv += ["--target-modules", *[str(module) for module in target_modules]]

    return argv
