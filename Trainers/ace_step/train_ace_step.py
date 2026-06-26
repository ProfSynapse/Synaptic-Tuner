#!/usr/bin/env python3
"""
ACE-STEP v1.5 music fine-tuning entry point (thin subprocess wrapper).

Location: Trainers/ace_step/train_ace_step.py
Purpose:  Fine-tune an ACE-STEP v1.5 music model (DCAE + DiT) with a LoRA/LoKr
          adapter by driving ACE-STEP's OWN headless CLI as a subprocess (Option A
          — we do NOT reimplement training). Two stages, orchestrated in order:
            stage 1  `python train.py fixed --preprocess ...`  (audio -> .pt cache)
            stage 2  `python train.py fixed ...`              (LoRA/LoKr -> adapter)
          Translates our single config.yaml (§2) into the ACE-STEP `train.py` argv
          (§1.3 translation table), captures each subprocess's returncode, tees its
          stderr to the run log, and RAISES on a nonzero exit so a failed train.py
          can NEVER look like success. On success, remaps ACE-STEP's output dir into
          the canonical `ace_step_output/` layout and registers the run in the
          shared experiment-tracking registry (run_type="ace_step").
          A `--dry-run` prints the translated argv(s) and EXITs 0 without executing —
          this is the de-risk step that validates flag translation with zero GPU /
          audio cost (and is where the §1.3 build-time `--help` byte-confirm lands).
Used by:  The `ace_step` training method (CLI/recipe). Auto-resolved by the repo's
          `train_{method}.py` / `Trainers/{method}/` dispatch convention once
          `ace_step` is in the TRAINING_METHODS SSOT (shared/utilities/paths.py).

Contract: docs/architecture/ace-step-pipeline-contract.md §1 (wrapper API + flag
          translation), §2 (config schema), §3 (model registry), §5 (data/cache).
          Mirrors the Trainers/embedding/train_embedding.py bootstrap shape
          (sys.path + init_trainer_env before heavy imports; argparse --config +
          override merge via `is not None`).

Usage:
    python train_ace_step.py
    python train_ace_step.py --config configs/config.yaml --dry-run
    python train_ace_step.py --data-dir /host/path/to/audio_corpus
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Add this trainer's src + the repo root to sys.path BEFORE the env bootstrap
# (mirrors train_embedding.py / train_sft.py).
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Environment bootstrap — keep before any heavy imports. The ACE-STEP wrapper is a
# subprocess driver, so it does not import torch/transformers itself; we still call
# the bootstrap for .env loading (HF auth) + consistent logging/stdout setup.
from shared.env_bootstrap import init_trainer_env  # noqa: E402

init_trainer_env()

import yaml  # noqa: E402

from config_translation import (  # noqa: E402
    build_fixed_argv,
    fetch_checkpoint,
    resolve_cache_dir,
    resolve_checkpoint_dir,
    resolve_output_dir,
)
from data_loader import run_preprocess  # noqa: E402
from subprocess_runner import run_ace_step_subprocess  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse our CLI arguments. Flags override the loaded config values (§1.1)."""
    parser = argparse.ArgumentParser(
        description="ACE-STEP v1.5 music fine-tuning (subprocess wrapper)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "config.yaml"),
        help="Path to the ACE-STEP run config YAML (§2). Single source of all run params.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override dataset.data_dir (§5). CLI wins over YAML when both set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the translated ACE-STEP train.py argv(s) and EXIT 0 WITHOUT executing.",
    )
    return parser.parse_args(argv)


def _load_config(path: str) -> dict[str, Any]:
    """Load the run config YAML into a nested dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI overrides onto the loaded config in place.

    Uses `is not None` so an explicit value is honored and never silently dropped
    to the config default (the provenance discipline the SFT/embedding trainers use).
    """
    dataset = config.setdefault("dataset", {})
    if args.data_dir is not None:
        dataset["data_dir"] = args.data_dir


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute (or dry-run) an ACE-STEP fine-tuning run for the given CLI args.

    Sequence:
        1. Load config + apply CLI overrides.
        2. Resolve checkpoint dir (model_registry.yaml), cache dir, output dir.
        3. Materialize base weights at the registry-pinned revision (fetch_checkpoint,
           M-a) — real-run only; unreached on --dry-run (after the early-return below).
        4. Stage 1 — preprocess (delegated to data_loader.run_preprocess; builds +
           invokes `train.py fixed --preprocess`, returns the resolved .pt cache dir).
        5. Stage 2 — fixed train (build argv here, invoke, raise-on-nonzero).
        6. Remap output dir -> ace_step_output/ + register the run.

    A `--dry-run` short-circuits before any subprocess execution: it prints the
    fully translated argv for BOTH stages and returns. This is the single place the
    §1.3 flag-translation assertion lives (de-risk with zero GPU/audio cost).
    """
    config = _load_config(args.config)
    _apply_overrides(config, args)

    repo_root = Path(__file__).parent.parent.parent

    # Resolve the multi-file checkpoint folder from the model registry (the SSOT).
    # This validates model.registry_name before any subprocess is built.
    checkpoint_dir = resolve_checkpoint_dir(config, repo_root)
    cache_dir = resolve_cache_dir(config, repo_root)
    output_dir = resolve_output_dir(config, repo_root)

    _print_header(config, checkpoint_dir, cache_dir, output_dir, dry_run=args.dry_run)

    # Stage 2 argv is built up-front so --dry-run can show it alongside stage 1
    # WITHOUT running preprocess. cache_dir is the contract handoff between stages.
    fixed_argv = build_fixed_argv(
        config,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        dataset_dir=cache_dir,
        output_dir=output_dir,
    )

    if args.dry_run:
        # data_loader builds (but does NOT invoke) the preprocess argv in dry-run mode.
        preprocess_argv = run_preprocess(
            config,
            checkpoint_dir=checkpoint_dir,
            cache_dir=cache_dir,
            repo_root=repo_root,
            dry_run=True,
        )
        _print_dry_run(preprocess_argv, fixed_argv)
        return {
            "dry_run": True,
            "preprocess_argv": preprocess_argv,
            "fixed_argv": fixed_argv,
            "checkpoint_dir": str(checkpoint_dir),
            "cache_dir": str(cache_dir),
            "output_dir": str(output_dir),
        }

    # Materialize base checkpoint at the registry-pinned revision (M-a).
    # Unreached on --dry-run (named above, not downloaded); weights must be
    # present before stage-1 preprocess loads the VAE/text-encoder.
    fetch_checkpoint(config, repo_root, dry_run=False)

    # ----- Stage 1: preprocess (audio -> .pt cache). Skipped inside run_preprocess
    # when the cache is present and preprocess.force is false. -----
    run_preprocess(
        config,
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
        repo_root=repo_root,
        dry_run=False,
    )

    # ----- Stage 2: fixed LoRA/LoKr train (cache -> adapter). -----
    print("\n" + "=" * 80)
    print("ACE-STEP STAGE 2 — FIXED LoRA/LoKr TRAINING")
    print("=" * 80 + "\n")
    run_ace_step_subprocess(
        fixed_argv,
        repo_root=repo_root,
        stage="fixed",
        log_dir=output_dir,
    )

    # ----- Output remap + run registration. -----
    final_output = _finalize_output(config, output_dir, repo_root)

    print(f"\n[OK] ACE-STEP fine-tuning complete. Adapter: {final_output}")
    return {
        "dry_run": False,
        "checkpoint_dir": str(checkpoint_dir),
        "cache_dir": str(cache_dir),
        "output_dir": str(final_output),
    }


def _print_header(
    config: dict[str, Any],
    checkpoint_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool,
) -> None:
    """Print a human-readable run-configuration banner."""
    model_cfg = config.get("model", {})
    adapter_cfg = config.get("adapter", {})  # architect §2 rename (was "lora")
    training_cfg = config.get("training", {})
    print("\n" + "=" * 80)
    print("ACE-STEP v1.5 FINE-TUNING CONFIGURATION" + ("  (DRY RUN)" if dry_run else ""))
    print("=" * 80)
    print(f"Registry model:  {model_cfg.get('registry_name')} (variant={model_cfg.get('variant')})")
    print(f"Checkpoint dir:  {checkpoint_dir}")
    print(f"Cache dir:       {cache_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Adapter:         {adapter_cfg.get('type')} (rank={adapter_cfg.get('rank')}, alpha={adapter_cfg.get('alpha')})")
    print(f"Epochs:          {training_cfg.get('epochs')} (control by epochs, NOT max_steps)")
    print("=" * 80 + "\n")


def _print_dry_run(preprocess_argv: list[str], fixed_argv: list[str]) -> None:
    """Print both translated argv lists for --dry-run inspection.

    This is the de-risk surface: it shows EXACTLY what would be shelled out, so the
    §1.3 flag translation can be byte-confirmed against `train.py fixed --help`
    (preprocess is the `--preprocess` flag on `fixed`, not a separate subcommand)
    without spending any GPU/audio.
    """
    print("[DRY RUN] Stage 1 — preprocess argv:")
    print("    " + " ".join(preprocess_argv))
    print("\n[DRY RUN] Stage 2 — fixed argv:")
    print("    " + " ".join(fixed_argv))
    print("\n[DRY RUN] No subprocess executed. Exiting 0.")


def _finalize_output(config: dict[str, Any], output_dir: Path, repo_root: Path) -> Path:
    """Map the ACE-STEP output dir into ace_step_output/ + register the run.

    The wrapper already pointed `--output-dir` at the canonical ace_step_output/
    location (resolve_output_dir), so no copy/move is needed — this records the run
    in the shared experiment-tracking registry (run_type="ace_step", a free string
    reused as-is, no schema change).
    """
    _register_run(config, output_dir, repo_root)
    return output_dir


def _register_run(config: dict[str, Any], output_dir: Path, repo_root: Path) -> None:
    """Append a RunRecord for this run to the experiment-tracking registry.

    Best-effort: a registry failure must not fail a successful training run, so it
    is logged and swallowed (the adapter on disk is the source of truth).
    """
    try:
        from datetime import datetime, timezone

        from shared.experiment_tracking.registry import RunRegistry
        from shared.experiment_tracking.schema import RunRecord

        model_cfg = config.get("model", {})
        record = RunRecord(
            run_id=output_dir.name,
            run_type="ace_step",
            name=f"ace_step:{model_cfg.get('registry_name')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="completed",
            output_dir=str(output_dir),
            model_name=str(model_cfg.get("registry_name") or ""),
        )
        RunRegistry().register_run(record)
        print(f"[tracking] registered run {record.run_id} (run_type=ace_step)")
    except Exception as exc:  # noqa: BLE001 — best-effort tracking, never fail the run
        print(f"[tracking] WARNING: could not register run ({exc}); adapter is on disk regardless")


def main(argv: list[str] | None = None) -> None:
    """Main ACE-STEP fine-tuning entry point."""
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
