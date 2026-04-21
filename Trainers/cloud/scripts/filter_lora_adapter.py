#!/usr/bin/env python3
"""Filter a LoRA adapter directory down to a subset of tensor keys."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Source adapter directory.")
    parser.add_argument("--dest-dir", required=True, help="Destination adapter directory.")
    parser.add_argument(
        "--include-substring",
        action="append",
        default=[],
        help="Keep tensors whose keys contain this substring. Repeatable.",
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=[],
        help="Drop tensors whose keys contain this substring. Repeatable.",
    )
    parser.add_argument(
        "--default-language-only",
        action="store_true",
        help="Equivalent to --include-substring .language_model.",
    )
    return parser


def _should_keep(
    key: str,
    *,
    include_substrings: Iterable[str],
    exclude_substrings: Iterable[str],
) -> bool:
    include_substrings = list(include_substrings)
    exclude_substrings = list(exclude_substrings)
    if include_substrings and not any(fragment in key for fragment in include_substrings):
        return False
    if exclude_substrings and any(fragment in key for fragment in exclude_substrings):
        return False
    return True


def filter_adapter_directory(
    *,
    source_dir: Path,
    dest_dir: Path,
    include_substrings: list[str],
    exclude_substrings: list[str],
) -> dict[str, object]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_dir = source_dir.resolve()
    dest_dir = dest_dir.resolve()
    adapter_path = source_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {source_dir}")

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)

    kept_tensors = {}
    counts = Counter()
    examples: dict[str, str] = {}

    with safe_open(str(adapter_path), framework="pt") as handle:
        for key in handle.keys():
            bucket = "kept" if _should_keep(
                key,
                include_substrings=include_substrings,
                exclude_substrings=exclude_substrings,
            ) else "dropped"
            counts[bucket] += 1
            examples.setdefault(bucket, key)
            if bucket == "kept":
                kept_tensors[key] = handle.get_tensor(key)

    if not kept_tensors:
        raise RuntimeError("Filtering removed every tensor. Refine the include/exclude rules.")

    save_file(kept_tensors, str(dest_dir / "adapter_model.safetensors"))

    summary = {
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "kept_tensors": counts["kept"],
        "dropped_tensors": counts["dropped"],
        "include_substrings": include_substrings,
        "exclude_substrings": exclude_substrings,
        "examples": examples,
    }
    (dest_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    include_substrings = list(args.include_substring)
    exclude_substrings = list(args.exclude_substring)
    if args.default_language_only and ".language_model." not in include_substrings:
        include_substrings.append(".language_model.")

    summary = filter_adapter_directory(
        source_dir=Path(args.source_dir),
        dest_dir=Path(args.dest_dir),
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
