#!/usr/bin/env python3
"""Helper for canonical battle-of-the-models cloud-pipeline experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass(frozen=True)
class SweepModel:
    alias: str
    model_id: str
    family: str
    train_gpu: str
    batch_size: int
    image_profile: str = "stable"
    gradient_accumulation: int = 4
    notes: str = ""


SWEEP_MODELS: tuple[SweepModel, ...] = (
    SweepModel(
        alias="qwen35-0p8b",
        model_id="Qwen/Qwen3.5-0.8B",
        family="qwen3.5",
        train_gpu="a10g-small",
        batch_size=12,
        image_profile="next",
        notes="Cheap baseline for smoke tests and early comparison. Tries the newer Unsloth image profile, which still needs validation.",
    ),
    SweepModel(
        alias="qwen35-2b",
        model_id="Qwen/Qwen3.5-2B",
        family="qwen3.5",
        train_gpu="a10g-small",
        batch_size=8,
        image_profile="next",
    ),
    SweepModel(
        alias="qwen35-4b",
        model_id="Qwen/Qwen3.5-4B",
        family="qwen3.5",
        train_gpu="a10g-small",
        batch_size=6,
        image_profile="next",
    ),
    SweepModel(
        alias="qwen35-9b",
        model_id="Qwen/Qwen3.5-9B",
        family="qwen3.5",
        train_gpu="a100-large",
        batch_size=2,
        image_profile="next",
        gradient_accumulation=8,
        notes="Memory outlier; use A100 for stable full runs. Tries the newer Unsloth image profile, which still needs validation.",
    ),
    SweepModel(
        alias="smollm2-1p7b",
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        family="smollm2",
        train_gpu="a10g-small",
        batch_size=10,
    ),
    SweepModel(
        alias="nemotron-nano-4b",
        model_id="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        family="nemotron",
        train_gpu="a10g-small",
        batch_size=6,
    ),
)


def _resolve_models(requested: Iterable[str], query: str | None) -> list[SweepModel]:
    requested = [item.strip() for item in requested if item.strip()]
    models = list(SWEEP_MODELS)

    if requested and "all" not in requested:
        requested_set = set(requested)
        models = [model for model in models if model.alias in requested_set]

    if query:
        query_lower = query.lower()
        models = [
            model
            for model in models
            if query_lower in model.alias.lower()
            or query_lower in model.model_id.lower()
            or query_lower in model.family.lower()
        ]

    return models


def _command_for(
    model: SweepModel,
    *,
    smoke: bool,
    preset: str,
    image_profile_override: str | None,
    cloud_image_override: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "tuner.py",
        "cloud-pipeline",
        "--yes",
        "--method",
        "sft",
        "--preset",
        preset,
        "--train-model-name",
        model.model_id,
        "--train-gpu",
        model.train_gpu,
        "--train-batch-size",
        str(model.batch_size),
        "--train-gradient-accumulation",
        str(model.gradient_accumulation),
        "--train-no-load-in-4bit",
        "--train-lora-target-modules",
        DEFAULT_LORA_TARGETS,
    ]
    image_profile = image_profile_override or model.image_profile
    if image_profile:
        cmd.extend(["--train-image-profile", image_profile])
    if cloud_image_override:
        cmd.extend(["--train-cloud-image", cloud_image_override])
    if smoke:
        cmd.extend(["--train-max-steps", "20", "--train-timeout-hours", "3"])
    return cmd


def _print_models(models: list[SweepModel]) -> int:
    if not models:
        print("No matching sweep models found.")
        return 1

    print("Battle of the Models candidates")
    print("Canonical path: SFT cloud-pipeline -> cloud eval -> merge/publish winner -> KTO and env-GRPO promotion")
    for model in models:
        print(f"- {model.alias}")
        print(f"  model:  {model.model_id}")
        print(f"  family: {model.family}")
        print(f"  gpu:    {model.train_gpu}")
        print(f"  batch:  {model.batch_size}")
        print(f"  gacc:   {model.gradient_accumulation}")
        print(f"  image:  {model.image_profile}")
        if model.notes:
            print(f"  notes:  {model.notes}")
    return 0


def _print_commands(
    models: list[SweepModel],
    *,
    smoke: bool,
    preset: str,
    image_profile_override: str | None,
    cloud_image_override: str | None,
) -> int:
    if not models:
        print("No matching sweep models found.")
        return 1

    for model in models:
        cmd = " ".join(
            _command_for(
                model,
                smoke=smoke,
                preset=preset,
                image_profile_override=image_profile_override,
                cloud_image_override=cloud_image_override,
            )
        )
        print(f"# {model.alias} -> {model.model_id}")
        print(cmd)
    return 0


def _launch(
    models: list[SweepModel],
    *,
    smoke: bool,
    preset: str,
    image_profile_override: str | None,
    cloud_image_override: str | None,
    stagger_seconds: float,
) -> int:
    if not models:
        print("No matching sweep models found.")
        return 1

    for model in models:
        cmd = _command_for(
            model,
            smoke=smoke,
            preset=preset,
            image_profile_override=image_profile_override,
            cloud_image_override=cloud_image_override,
        )
        print(f"Launching {model.alias}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        if model != models[-1] and stagger_seconds > 0:
            print(f"Waiting {stagger_seconds:g}s before the next submission...")
            time.sleep(stagger_seconds)
    return 0


def _print_plan() -> int:
    print("Battle-of-models experiment flow:")
    print("1. Run canonical SFT cloud-pipeline jobs for each base model.")
    print("2. Compare cloud evaluation outputs and pick the strongest SFT winners.")
    print("3. Merge/publish the selected SFT winner as the source artifact.")
    print("4. Use that published model for KTO refinement.")
    print("5. Promote the best post-KTO/SFT winner into cloud env-GRPO.")
    print("6. Treat env-GRPO as the online optimization stage, not a replacement for SFT evaluation.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Battle-of-the-models canonical cloud-pipeline helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List the curated sweep model catalog")
    list_parser.add_argument("aliases", nargs="*", help="Optional aliases to include")
    list_parser.add_argument("--search", help="Filter models by alias, family, or model id")

    commands_parser = subparsers.add_parser("commands", help="Print exact cloud-pipeline commands")
    commands_parser.add_argument("aliases", nargs="*", help="Optional aliases to include")
    commands_parser.add_argument("--search", help="Filter models by alias, family, or model id")
    commands_parser.add_argument("--smoke", action="store_true", help="Print smoke-test pipeline commands with a short training budget")
    commands_parser.add_argument("--preset", default="full", help="Evaluation preset to use with cloud-pipeline")
    commands_parser.add_argument("--image-profile", help="Force one image profile for every selected model")
    commands_parser.add_argument("--cloud-image", help="Force one exact Docker image for every selected model")

    launch_parser = subparsers.add_parser("launch", help="Run canonical cloud-pipeline experiments through tuner.py")
    launch_parser.add_argument("aliases", nargs="*", help="Optional aliases to include")
    launch_parser.add_argument("--search", help="Filter models by alias, family, or model id")
    launch_parser.add_argument("--smoke", action="store_true", help="Launch smoke-test pipelines with a short training budget")
    launch_parser.add_argument("--preset", default="full", help="Evaluation preset to use with cloud-pipeline")
    launch_parser.add_argument("--image-profile", help="Force one image profile for every selected model")
    launch_parser.add_argument("--cloud-image", help="Force one exact Docker image for every selected model")
    launch_parser.add_argument("--stagger-seconds", type=float, default=5.0, help="Seconds to wait between submissions. Default: 5.")

    subparsers.add_parser("plan", help="Print the promotion path through KTO and env-GRPO")

    args = parser.parse_args()
    if args.command == "plan":
        return _print_plan()

    models = _resolve_models(getattr(args, "aliases", []), getattr(args, "search", None))

    if args.command == "list":
        return _print_models(models)
    if args.command == "commands":
        return _print_commands(
            models,
            smoke=args.smoke,
            preset=args.preset,
            image_profile_override=getattr(args, "image_profile", None),
            cloud_image_override=getattr(args, "cloud_image", None),
            stagger_seconds=max(0.0, getattr(args, "stagger_seconds", 5.0)),
        )
    if args.command == "launch":
        return _launch(
            models,
            smoke=args.smoke,
            preset=args.preset,
            image_profile_override=getattr(args, "image_profile", None),
            cloud_image_override=getattr(args, "cloud_image", None),
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
