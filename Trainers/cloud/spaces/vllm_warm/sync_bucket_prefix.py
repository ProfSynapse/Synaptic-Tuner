#!/usr/bin/env python3
"""Download a bucket prefix into a local directory for warm Space runtimes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import sync_bucket


def parse_bucket_uri(uri: str) -> tuple[str, str]:
    prefix = "hf://buckets/"
    value = uri.strip()
    if not value.startswith(prefix):
        raise ValueError(f"Expected hf://buckets/... URI, got: {uri}")
    remainder = value[len(prefix):].strip("/")
    if "/" not in remainder:
        raise ValueError(f"Bucket URI must include a bucket id and prefix, got: {uri}")
    bucket_id, remote_prefix = remainder.split("/", 1)
    if "/" in remote_prefix:
        bucket_id = f"{bucket_id}/{remote_prefix.split('/', 1)[0]}"
        remote_prefix = remote_prefix.split("/", 1)[1]
    return bucket_id, remote_prefix.strip("/")


def sync_bucket_prefix(source_uri: str, dest_dir: Path) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    dest_dir.mkdir(parents=True, exist_ok=True)
    sync_bucket(source_uri, str(dest_dir), token=token)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="hf://buckets/<bucket>/<prefix> URI")
    parser.add_argument("--dest", required=True, help="Local destination directory")
    args = parser.parse_args()

    sync_bucket_prefix(args.source, Path(args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
