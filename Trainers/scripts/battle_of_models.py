#!/usr/bin/env python3
"""Helper for the HF Jobs battle-of-the-models SFT sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SweepModel:
    alias: str
    model_id: str
    family: str
    flavor: str
    job_config: str
    notes: str = ""


SWEEP_MODELS: tuple[SweepModel, ...] = (
    SweepModel(
        alias="qwen35-0p8b",
        model_id="Qwen/Qwen3.5-0.8B",
        family="qwen3.5",
        flavor="a10g-small",
        job_config="Trainers/cloud/jobs/battle_of_models_qwen35_0p8b_sft.yaml",
        notes="Smallest Qwen3.5 checkpoint; keep as the cheap baseline.",
    ),
    SweepModel(
        alias="qwen35-2b",
        model_id="Qwen/Qwen3.5-2B",
        family="qwen3.5",
        flavor="a10g-small",
        job_config="Trainers/cloud/jobs/battle_of_models_qwen35_2b_sft.yaml",
    ),
    SweepModel(
        alias="qwen35-4b",
        model_id="Qwen/Qwen3.5-4B",
        family="qwen3.5",
        flavor="a10g-small",
        job_config="Trainers/cloud/jobs/battle_of_models_qwen35_4b_sft.yaml",
    ),
    SweepModel(
        alias="qwen35-9b",
        model_id="Qwen/Qwen3.5-9B",
        family="qwen3.5",
        flavor="a100-large",
        job_config="Trainers/cloud/jobs/battle_of_models_qwen35_9b_sft.yaml",
        notes="Uses A100 because the dense 9B run is the memory outlier in this sweep.",
    ),
    SweepModel(
        alias="smollm2-1p7b",
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        family="smollm2",
        flavor="a10g-small",
        job_config="Trainers/cloud/jobs/battle_of_models_smollm2_1p7b_sft.yaml",
    ),
    SweepModel(
        alias="nemotron-nano-4b",
        model_id="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        family="nemotron",
        flavor="a10g-small",
        job_config="Trainers/cloud/jobs/battle_of_models_nemotron_nano_4b_sft.yaml",
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


def _command_for(model: SweepModel) -> list[str]:
    return [
        sys.executable,
        "tuner.py",
        "cloud-run",
        "--job-config",
        model.job_config,
        "--yes",
    ]


def _print_models(models: list[SweepModel]) -> int:
    if not models:
        print("No matching sweep models found.")
        return 1

    print("Battle of the Models candidates:")
    for model in models:
        print(f"- {model.alias}")
        print(f"  model:  {model.model_id}")
        print(f"  family: {model.family}")
        print(f"  gpu:    {model.flavor}")
        print(f"  job:    {model.job_config}")
        if model.notes:
            print(f"  notes:  {model.notes}")
    return 0


def _print_commands(models: list[SweepModel]) -> int:
    if not models:
        print("No matching sweep models found.")
        return 1

    for model in models:
        cmd = " ".join(_command_for(model))
        print(f"# {model.alias} -> {model.model_id}")
        print(cmd)
    return 0


def _launch(models: list[SweepModel]) -> int:
    if not models:
        print("No matching sweep models found.")
        return 1

    for model in models:
        cmd = _command_for(model)
        print(f"Launching {model.alias}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Battle-of-the-models HF Jobs helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List the curated sweep model catalog")
    list_parser.add_argument("aliases", nargs="*", help="Optional aliases to include")
    list_parser.add_argument("--search", help="Filter models by alias, family, or model id")

    commands_parser = subparsers.add_parser("commands", help="Print exact cloud-run commands")
    commands_parser.add_argument("aliases", nargs="*", help="Optional aliases to include")
    commands_parser.add_argument("--search", help="Filter models by alias, family, or model id")

    launch_parser = subparsers.add_parser("launch", help="Submit matching jobs through tuner.py")
    launch_parser.add_argument("aliases", nargs="*", help="Optional aliases to include")
    launch_parser.add_argument("--search", help="Filter models by alias, family, or model id")

    args = parser.parse_args()
    models = _resolve_models(getattr(args, "aliases", []), getattr(args, "search", None))

    if args.command == "list":
        return _print_models(models)
    if args.command == "commands":
        return _print_commands(models)
    if args.command == "launch":
        return _launch(models)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
