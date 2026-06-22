#!/usr/bin/env python3
"""Read and summarize local or HF bucket-backed artifact files."""

from __future__ import annotations

import argparse
from shared.utilities.bucket_artifacts import (
    latest_jsonl_record as _latest_jsonl_record,
    read_artifact,
    tail_lines as _tail_lines,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read a local or hf:// bucket artifact file.")
    parser.add_argument("path", help="Local path or hf://buckets/... artifact path.")
    parser.add_argument("--tail", type=int, help="Print only the last N lines.")
    parser.add_argument(
        "--jsonl-latest",
        action="store_true",
        help="Parse the file as JSONL and print only the latest record.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    output = read_artifact(
        args.path,
        tail=args.tail,
        jsonl_latest=args.jsonl_latest,
        pretty=args.pretty,
    )
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
