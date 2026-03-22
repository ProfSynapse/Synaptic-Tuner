#!/usr/bin/env python3
"""Publish a local dataset JSONL to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "shared").exists() and (parent / "Datasets").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.upload.dataset_publish import (  # noqa: E402
    build_upload_targets,
    format_publish_summary,
    publish_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a local dataset JSONL and optional metadata sidecar to a Hugging Face dataset repo."
    )
    parser.add_argument("dataset_path", help="Path to the local dataset JSONL file.")
    parser.add_argument("repo_id", help="Target Hugging Face dataset repo, e.g. namespace/dataset-name.")
    parser.add_argument(
        "--path-in-repo",
        help="Destination filename/path in the HF dataset repo. Defaults to the dataset filename.",
    )
    parser.add_argument(
        "--metadata-path",
        help="Optional metadata sidecar to upload. Defaults to <dataset>.metadata.json if present.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not upload a metadata sidecar even if one exists.",
    )
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Fail if the target dataset repo does not already exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it needs to be created.",
    )
    parser.add_argument(
        "--commit-message",
        help="Optional Hugging Face commit message.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without contacting Hugging Face.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    targets = build_upload_targets(
        args.dataset_path,
        path_in_repo=args.path_in_repo,
        metadata_path=args.metadata_path,
        include_metadata=not args.no_metadata,
    )

    result = publish_dataset(
        args.dataset_path,
        args.repo_id,
        path_in_repo=args.path_in_repo,
        metadata_path=args.metadata_path,
        include_metadata=not args.no_metadata,
        create_repo=not args.no_create_repo,
        private=args.private,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )

    print(format_publish_summary(result, targets=targets))
    if args.dry_run:
        print("Dry run only. No files were uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
