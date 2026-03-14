"""
Helper entrypoint for syncing Hugging Face Buckets from an isolated Python env.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync a local directory to a Hugging Face Bucket.")
    parser.add_argument("local_dir")
    parser.add_argument("bucket_uri")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--skip-create", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import sync_bucket

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    sync_bucket(
        args.local_dir,
        args.bucket_uri,
        token=token,
        delete=args.delete,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
