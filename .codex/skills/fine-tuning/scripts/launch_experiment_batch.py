#!/usr/bin/env python3
"""Launch multiple experiment specs concurrently with a stagger between submissions."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]


def build_command(
    experiment_spec: str,
    *,
    auto_hardware: bool,
    optimize_for: str | None,
    price_cap: float | None,
    yes: bool,
) -> list[str]:
    cmd = [sys.executable, "tuner.py", "run-experiment", "--experiment-spec", experiment_spec]
    if auto_hardware:
        cmd.append("--auto-hardware")
    if optimize_for:
        cmd.extend(["--optimize-for", optimize_for])
    if price_cap is not None:
        cmd.extend(["--price-cap", str(price_cap)])
    if yes:
        cmd.append("--yes")
    return cmd


def launch_batch(
    experiment_specs: Sequence[str],
    *,
    stagger_seconds: float,
    auto_hardware: bool,
    optimize_for: str | None,
    price_cap: float | None,
    yes: bool,
    dry_run: bool,
) -> int:
    failures: list[tuple[str, str]] = []
    launched: list[tuple[str, subprocess.Popen[bytes]]] = []
    for index, experiment_spec in enumerate(experiment_specs):
        cmd = build_command(
            experiment_spec,
            auto_hardware=auto_hardware,
            optimize_for=optimize_for,
            price_cap=price_cap,
            yes=yes,
        )
        print(f"[{index + 1}/{len(experiment_specs)}] {' '.join(cmd)}")
        if not dry_run:
            try:
                process = subprocess.Popen(cmd, cwd=REPO_ROOT)
            except OSError as exc:
                failures.append((experiment_spec, str(exc)))
                print(f"  FAILED TO LAUNCH ({exc}), continuing batch...")
            else:
                launched.append((experiment_spec, process))
                print(f"  launched pid={process.pid}")
        if index < len(experiment_specs) - 1 and stagger_seconds > 0:
            print(f"Waiting {stagger_seconds:g}s before the next submission...")
            if not dry_run:
                time.sleep(stagger_seconds)
    for experiment_spec, process in launched:
        returncode = process.wait()
        if returncode != 0:
            failures.append((experiment_spec, f"exit code {returncode}"))
            print(f"  FAILED (exit code {returncode})")
    if failures:
        print(f"\n{len(failures)}/{len(experiment_specs)} experiment(s) failed:")
        for spec, reason in failures:
            print(f"  - {spec}: {reason}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch multiple run-experiment specs with a stagger.")
    parser.add_argument("experiment_specs", nargs="+", help="Experiment spec paths to submit in order.")
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait between submissions. Default: 5.",
    )
    parser.add_argument(
        "--auto-hardware",
        action="store_true",
        help="Resolve missing hardware and batch shape with the planner before each launch.",
    )
    parser.add_argument(
        "--optimize-for",
        choices=("cost", "balanced", "speed"),
        help="Optimization objective to pass through when using --auto-hardware.",
    )
    parser.add_argument(
        "--price-cap",
        type=float,
        help="Optional hourly price cap to pass through when using --auto-hardware.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts in run-experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching them.")
    args = parser.parse_args()
    return launch_batch(
        args.experiment_specs,
        stagger_seconds=max(0.0, args.stagger_seconds),
        auto_hardware=args.auto_hardware,
        optimize_for=args.optimize_for,
        price_cap=args.price_cap,
        yes=args.yes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
