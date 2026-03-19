"""Helper entrypoint for syncing Hugging Face Buckets from an isolated Python env."""

from __future__ import annotations

import argparse
import os
import sys


def _normalize_token(token: str | None) -> str | None:
    if token is None:
        return None
    token = token.strip()
    return token or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync between a local path and a Hugging Face Bucket.")
    parser.add_argument("source_path")
    parser.add_argument("destination_path")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--skip-create", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import sync_bucket

    token = _normalize_token(os.environ.get("HF_TOKEN")) or _normalize_token(os.environ.get("HF_API_KEY"))
    sync_bucket(
        args.source_path,
        args.destination_path,
        token=token,
        delete=args.delete,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
