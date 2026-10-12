#!/usr/bin/env python3
"""
Image LoRA training method (ai-toolkit on Modal).

Location: Trainers/image_lora/train_image_lora.py
Purpose:  Train a diffusion-model LoRA (default: Qwen-Image-2512) by wrapping
          ostris/ai-toolkit on a Modal GPU. Like Trainers/ace_step, this wraps an
          outside trainer instead of reimplementing training. Unlike the packaged
          LLM Modal path, it owns a minimal resource lifecycle:
            - one shared base-weights cache Volume (reused across runs),
            - one per-run Volume (dataset + outputs), deleted only after the
              outputs are downloaded and verified,
            - the existing named ``hf-token`` Secret (never copied per run),
            - an ephemeral, detached app that stops when the call ends.
Used by:  The ``image_lora`` training method (TRAINING_METHODS in
          shared/utilities/paths.py). With no subcommand it only plans (no spend).

Subcommands:
  build-dataset  notes + images -> captioned dataset, manifest.json, tokens.md
  plan           render the ai-toolkit job and the cost estimate (no cloud calls)
  probe          cheap GPU smoke test of the training image
  launch         upload the dataset, start training detached, write run_state.json
  status         poll the call, tail the remote log, list samples
  wait           block until the call finishes (polls status)
  fetch          download samples, logs and chosen checkpoints; verify hashes
  cleanup        delete the per-run Volume and stop the app (after a verified fetch)
  run            launch + wait + fetch + cleanup

Contract: docs/architecture/image-lora-modal.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

TRAINER_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TRAINER_DIR / "configs" / "config.yaml"


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    from dataset_builder import build_dataset, contact_sheets

    out = Path(args.out)
    manifest = build_dataset(Path(args.recipe), out, write=not args.dry_run)
    counts = manifest["counts"]
    print(json.dumps({"included": counts["included"], "excluded": counts["excluded"],
                      "by_kind": counts["by_kind"], "exclusion_reasons": counts["exclusion_reasons"]},
                     indent=2, ensure_ascii=False))
    if args.dry_run:
        for item in manifest["included"][: args.show]:
            print(f"\n[{item['output_file']}] {item['caption']}")
    elif args.contact_sheets:
        for sheet in contact_sheets(out, manifest):
            print(f"contact sheet: {sheet}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from run_config import load_run_config, plan_run

    config = load_run_config(Path(args.config), overrides=_overrides(args))
    plan = plan_run(config, dataset_dir=Path(args.dataset) if args.dataset else None)
    print(json.dumps(plan.summary(), indent=2))
    if args.show_job:
        print(plan.job_yaml)
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from modal_runner import probe

    print(json.dumps(probe(gpu=args.gpu, timeout_seconds=args.timeout), indent=2))
    return 0


def _cmd_launch(args: argparse.Namespace) -> int:
    from modal_runner import launch
    from run_config import load_run_config

    config = load_run_config(Path(args.config), overrides=_overrides(args))
    state = launch(config, dataset_dir=Path(args.dataset), output_dir=Path(args.output_dir),
                   max_usd=args.max_usd, run_id=args.run_id)
    print(json.dumps(state, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from modal_runner import status

    print(json.dumps(status(Path(args.state), tail=args.tail), indent=2))
    return 0


def _cmd_wait(args: argparse.Namespace) -> int:
    from modal_runner import wait

    result = wait(Path(args.state), poll_seconds=args.poll)
    print(json.dumps({k: result.get(k) for k in ("call_state", "completion_status", "error", "samples")}))
    return 0 if result.get("completion_status") == "succeeded" else 1


def _cmd_fetch(args: argparse.Namespace) -> int:
    from modal_runner import fetch

    result = fetch(Path(args.state), checkpoints=args.checkpoint or [])
    print(json.dumps(result, indent=2))
    return 0 if result.get("verified") else 1


def _cmd_cleanup(args: argparse.Namespace) -> int:
    from modal_runner import cleanup

    result = cleanup(Path(args.state), force=args.force)
    print(json.dumps(result, indent=2))
    return 0 if result.get("cleaned") else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from modal_runner import cleanup, fetch, launch, wait
    from run_config import load_run_config

    config = load_run_config(Path(args.config), overrides=_overrides(args))
    state = launch(config, dataset_dir=Path(args.dataset), output_dir=Path(args.output_dir),
                   max_usd=args.max_usd, run_id=args.run_id)
    state_path = Path(state["state_path"])
    wait(state_path)
    result = fetch(state_path, checkpoints=[])
    if not result.get("verified"):
        print(json.dumps(result, indent=2))
        print("[image_lora] outputs did not verify; per-run resources were kept for inspection")
        return 1
    print(json.dumps(cleanup(state_path), indent=2))
    return 0


def _overrides(args: argparse.Namespace) -> dict:
    keys = ("steps", "rank", "learning_rate", "gpu", "sample_every", "save_every")
    return {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--gpu")
    parser.add_argument("--sample-every", dest="sample_every", type=int)
    parser.add_argument("--save-every", dest="save_every", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image LoRA training (ai-toolkit on Modal)")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("build-dataset", help="build a captioned dataset from a recipe")
    p.add_argument("--recipe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true", help="print decisions and captions; write nothing")
    p.add_argument("--show", type=int, default=20, help="captions to print in --dry-run")
    p.add_argument("--contact-sheets", action="store_true")
    p.set_defaults(func=_cmd_build_dataset)

    p = sub.add_parser("plan", help="render the ai-toolkit job and cost estimate")
    _add_overrides(p)
    p.add_argument("--dataset")
    p.add_argument("--show-job", action="store_true")
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("probe", help="GPU smoke test of the training image")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--timeout", type=int, default=900)
    p.set_defaults(func=_cmd_probe)

    for name, func, help_text in (("launch", _cmd_launch, "start a detached training run"),
                                  ("run", _cmd_run, "launch, wait, fetch, verify and clean up")):
        p = sub.add_parser(name, help=help_text)
        _add_overrides(p)
        p.add_argument("--dataset", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--max-usd", type=float, required=True,
                       help="spend cap for this run; sets the provider timeout")
        p.add_argument("--run-id")
        p.set_defaults(func=func)

    p = sub.add_parser("status", help="poll a run")
    p.add_argument("--state", required=True)
    p.add_argument("--tail", type=int, default=20)
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("wait", help="block until the run's call finishes")
    p.add_argument("--state", required=True)
    p.add_argument("--poll", type=int, default=120)
    p.set_defaults(func=_cmd_wait)

    p = sub.add_parser("fetch", help="download and verify outputs")
    p.add_argument("--state", required=True)
    p.add_argument("--checkpoint", action="append", type=int,
                   help="also download this intermediate checkpoint step (repeatable)")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("cleanup", help="delete per-run Modal resources after a verified fetch")
    p.add_argument("--state", required=True)
    p.add_argument("--force", action="store_true", help="clean up even without a verified fetch")
    p.set_defaults(func=_cmd_cleanup)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        # The local training menu runs this script with no arguments: plan only.
        args = parser.parse_args(["plan"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
