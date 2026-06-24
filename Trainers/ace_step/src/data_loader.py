"""
ACE-STEP dataset-prep shim — stage-1 preprocess orchestration.

Location: Trainers/ace_step/src/data_loader.py
Purpose:  Orchestrate ACE-STEP stage 1: turn an audio corpus (--audio-dir) or a
          labeled dataset JSON (--dataset-json) into the flat per-sample `.pt`
          tensor cache that stage-2 `train.py fixed` consumes via --dataset-dir.
          This shim BUILDS the `train.py preprocess` argv (via config_translation)
          and INVOKES it (via subprocess_runner), then points the trainer at the
          resolved cache dir. It does NOT re-encode audio itself — ACE-STEP's own
          preprocess does the two-pass VAE -> DiT-encoder work and writes the cache.
Used by:  Trainers/ace_step/train_ace_step.py (stage-1 of the run sequence).

Contract: docs/architecture/ace-step-pipeline-contract.md §4 (the .pt-cache
          contract) and §1.3 stage-1 (preprocess flag table).

The `.pt` cache schema (§4.2 — key NAMES byte-confirmed, shapes smoke-gated):
  flat dir of per-sample `<name>.pt`, NO central manifest.json. Each `.pt` carries
  top-level keys: target_latents, attention_mask, encoder_hidden_states,
  encoder_attention_mask, context_latents, metadata. `metadata` sub-dict:
  audio_path, filename, caption, lyrics, duration, bpm, keyscale, timesignature,
  genre, is_instrumental, custom_tag, prompt_override.

Cache-skip: when the cache dir already holds `.pt` files and preprocess.force is
false, stage-1 is SKIPPED (re-encoding a large corpus is expensive). force=true
re-runs preprocess regardless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config_translation import build_preprocess_argv
from subprocess_runner import run_ace_step_subprocess

# The contract-confirmed top-level keys every per-sample `.pt` must carry (§4.2).
# Exposed for the build-time shape smoke + the test-engineer golden fixture; this is
# the single in-code copy of the byte-confirmed key set.
EXPECTED_PT_TOP_LEVEL_KEYS = (
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
    "metadata",
)

# The `metadata` sub-dict keys (§4.2). audio_path/filename/duration are derived;
# the rest are the effective labeled-`dataset.json` input field surface (§4.1).
EXPECTED_PT_METADATA_KEYS = (
    "audio_path",
    "filename",
    "caption",
    "lyrics",
    "duration",
    "bpm",
    "keyscale",
    "timesignature",
    "genre",
    "is_instrumental",
    "custom_tag",
    "prompt_override",
)


def _cache_has_pt_files(cache_dir: Path) -> bool:
    """Return True if cache_dir exists and holds at least one `.pt` file (flat dir)."""
    if not cache_dir.exists():
        return False
    return any(cache_dir.glob("*.pt"))


def run_preprocess(
    config: dict[str, Any],
    *,
    checkpoint_dir: Path,
    cache_dir: Path,
    repo_root: Path,
    dry_run: bool,
) -> list[str]:
    """Build (and unless dry_run, invoke) the stage-1 `train.py preprocess` argv.

    Returns the translated preprocess argv in ALL modes (so the caller can show it in
    --dry-run and so the invocation path is the same argv that was displayed).

    Behavior:
        - dry_run=True:  build + RETURN the argv WITHOUT executing (de-risk display).
        - dry_run=False: build the argv, then SKIP if the cache is already populated
          and preprocess.force is false; otherwise invoke it (raise-on-nonzero via
          subprocess_runner).

    Args:
        config:         the loaded run config (§2).
        checkpoint_dir: resolved base-checkpoint folder (--checkpoint-dir).
        cache_dir:      resolved .pt tensor cache dir (--tensor-output).
        repo_root:      Synthetic-Conversations repo root (for the train.py path).
        dry_run:        when True, do not execute.

    Returns:
        The translated preprocess argv (list[str]).
    """
    preprocess_argv = build_preprocess_argv(
        config,
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
        repo_root=repo_root,
    )

    if dry_run:
        return preprocess_argv

    force = bool(config.get("preprocess", {}).get("force", False))
    if not force and _cache_has_pt_files(cache_dir):
        print(
            f"[ace_step:preprocess] cache present at {cache_dir} and preprocess.force=false "
            f"-> SKIPPING stage 1 (re-encoding skipped)."
        )
        return preprocess_argv

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("ACE-STEP STAGE 1 — PREPROCESS (audio -> .pt tensor cache)")
    print("=" * 80 + "\n")
    run_ace_step_subprocess(
        preprocess_argv,
        repo_root=repo_root,
        stage="preprocess",
        log_dir=cache_dir,
    )
    return preprocess_argv
