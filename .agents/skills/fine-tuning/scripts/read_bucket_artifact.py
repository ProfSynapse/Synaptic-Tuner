#!/usr/bin/env python3
"""Read and summarize local or HF bucket-backed artifact files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, TextIO

from huggingface_hub import HfFileSystem


def _hf_token() -> str | None:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or "").strip()
    return token or None


def _open_path(path: str) -> TextIO:
    if path.startswith("hf://"):
        fs = HfFileSystem(token=_hf_token())
        return fs.open(path, "r", encoding="utf-8")
    return open(Path(path), "r", encoding="utf-8")


def _tail_lines(lines: Iterable[str], count: int) -> list[str]:
    if count <= 0:
        return list(lines)
    buffer: list[str] = []
    for line in lines:
        buffer.append(line)
        if len(buffer) > count:
            buffer.pop(0)
    return buffer


def _latest_jsonl_record(lines: Iterable[str]) -> dict:
    latest: dict | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        latest = json.loads(line)
    if latest is None:
        raise ValueError("No JSONL records found.")
    return latest


def read_artifact(
    path: str,
    *,
    tail: int | None = None,
    jsonl_latest: bool = False,
    pretty: bool = False,
) -> str:
    with _open_path(path) as handle:
        if jsonl_latest:
            record = _latest_jsonl_record(handle)
            return json.dumps(record, indent=2 if pretty else None, sort_keys=pretty)

        if tail is not None:
            lines = _tail_lines(handle, tail)
            return "".join(lines)

        contents = handle.read()
        if pretty:
            try:
                return json.dumps(json.loads(contents), indent=2, sort_keys=True)
            except json.JSONDecodeError:
                return contents
        return contents


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
