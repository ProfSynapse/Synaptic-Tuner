#!/usr/bin/env python3
"""Inspect a private-input, config-driven Modal SFT job without creating it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_MODULE_DIR = REPO_ROOT / "tuner" / "backends" / "training" / "cloud"
sys.path.insert(0, str(PLAN_MODULE_DIR))

from modal_job_plan import build_modal_sft_plan  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local direct SFT YAML config to stage")
    parser.add_argument("--dataset", type=Path, required=True, help="Local private JSONL dataset to stage")
    parser.add_argument("--input-volume", required=True, help="Existing or planned Modal input Volume name")
    parser.add_argument("--input-prefix", required=True, help="Unique relative directory inside the input Volume")
    parser.add_argument("--runtime-image", required=True, help="Immutable OCI image with @sha256 digest")
    parser.add_argument("--pip-package", action="append", default=[], help="Exact runtime pip requirement; repeatable")
    parser.add_argument("--gpu", default="A10G")
    parser.add_argument("--timeout-hours", type=float, default=2.0)
    parser.add_argument("--output-volume", default="toolset-training-artifacts")
    parser.add_argument("--output-mount", default="/vol/artifacts")
    parser.add_argument("--input-mount", default="/vol/inputs")
    parser.add_argument("--cache-volume", default="toolset-model-cache")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = build_modal_sft_plan(
        repo_root=REPO_ROOT,
        config_path=args.config,
        dataset_path=args.dataset,
        input_volume_name=args.input_volume,
        input_prefix=args.input_prefix,
        runtime_image=args.runtime_image,
        pip_packages=args.pip_package,
        gpu=args.gpu,
        timeout_hours=args.timeout_hours,
        output_volume_name=args.output_volume,
        output_mount_path=args.output_mount,
        input_mount_path=args.input_mount,
        cache_volume_name=args.cache_volume,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
