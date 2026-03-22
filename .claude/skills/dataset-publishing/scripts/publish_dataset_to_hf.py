#!/usr/bin/env python3
"""Publish a local JSONL dataset and optional metadata sidecar to a HF dataset repo."""

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
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import HfApi  # noqa: E402
from shared.utilities.env import get_hf_token, load_env_file  # noqa: E402


def _default_metadata_path(dataset_path: Path) -> Path:
    suffix = dataset_path.suffix
    if suffix:
        return dataset_path.with_suffix(".metadata.json")
    return dataset_path.parent / f"{dataset_path.name}.metadata.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a local JSONL dataset to a Hugging Face dataset repo.")
    parser.add_argument("--dataset-path", required=True, help="Local dataset JSONL path.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face dataset repo id.")
    parser.add_argument("--metadata-path", help="Optional metadata sidecar JSON path.")
    parser.add_argument("--path-in-repo", default=None, help="Optional dataset path within the repo.")
    parser.add_argument("--metadata-path-in-repo", default=None, help="Optional metadata path within the repo.")
    parser.add_argument("--private", action="store_true", help="Create or keep the repo private.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without uploading.")
    parser.add_argument(
        "--commit-message",
        default="Publish filtered dataset variant",
        help="Commit message for the upload.",
    )
    return parser


def main() -> int:
    load_env_file()
    args = _build_parser().parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    metadata_path = Path(args.metadata_path).expanduser().resolve() if args.metadata_path else _default_metadata_path(dataset_path)
    upload_metadata = metadata_path.exists()

    dataset_path_in_repo = args.path_in_repo or dataset_path.name
    metadata_path_in_repo = args.metadata_path_in_repo or metadata_path.name

    print("Dataset publish plan")
    print(f"  repo_id: {args.repo_id}")
    print(f"  dataset_path: {dataset_path}")
    print(f"  dataset_path_in_repo: {dataset_path_in_repo}")
    print(f"  metadata_path: {metadata_path if upload_metadata else '(none)'}")
    print(f"  metadata_path_in_repo: {metadata_path_in_repo if upload_metadata else '(none)'}")
    print(f"  private: {bool(args.private)}")
    print(f"  dry_run: {bool(args.dry_run)}")

    if args.dry_run:
        return 0

    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN or HF_API_KEY is required for dataset publishing.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)

    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo=dataset_path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
        token=token,
    )
    print(f"Uploaded dataset: {dataset_path.name}")

    if upload_metadata:
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo=metadata_path_in_repo,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
            token=token,
        )
        print(f"Uploaded metadata: {metadata_path.name}")
    else:
        print("No metadata sidecar found; skipped metadata upload.")

    print(f"Published to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
