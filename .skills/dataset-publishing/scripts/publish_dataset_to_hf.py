#!/usr/bin/env python3
"""Publish a local dataset JSONL to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "shared").exists() and (parent / "Datasets").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.utilities.env import get_hf_token, load_env_file  # noqa: E402


@dataclass(frozen=True)
class UploadTarget:
    local_path: Path
    path_in_repo: str


@dataclass(frozen=True)
class DatasetPublishResult:
    repo_id: str
    uploaded_files: list[str]
    url: str


def infer_metadata_path(dataset_path: Path) -> Optional[Path]:
    dataset_name = dataset_path.name
    if dataset_name.endswith(".jsonl"):
        candidate_name = f"{dataset_name[:-len('.jsonl')]}.metadata.json"
    else:
        candidate_name = f"{dataset_name}.metadata.json"
    candidate = dataset_path.with_name(candidate_name)
    if candidate.exists():
        return candidate
    alt_candidate = dataset_path.parent / f"{dataset_path.name}.metadata.json"
    if alt_candidate.exists():
        return alt_candidate
    return None


def build_upload_targets(
    dataset_path: str | Path,
    *,
    path_in_repo: Optional[str] = None,
    metadata_path: str | Path | None = None,
    include_metadata: bool = True,
) -> list[UploadTarget]:
    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset}")

    targets = [UploadTarget(local_path=dataset, path_in_repo=path_in_repo or dataset.name)]
    if not include_metadata:
        return targets

    metadata: Optional[Path]
    if metadata_path:
        metadata = Path(metadata_path).expanduser().resolve()
        if not metadata.exists() or not metadata.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata}")
    else:
        metadata = infer_metadata_path(dataset)

    if metadata is not None:
        targets.append(UploadTarget(local_path=metadata, path_in_repo=metadata.name))

    return targets


def publish_dataset(
    dataset_path: str | Path,
    repo_id: str,
    *,
    path_in_repo: Optional[str] = None,
    metadata_path: str | Path | None = None,
    include_metadata: bool = True,
    create_repo: bool = True,
    private: bool = False,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
    api=None,
    dry_run: bool = False,
) -> DatasetPublishResult:
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must be in '<namespace>/<dataset-repo>' format.")

    load_env_file()
    hf_token = (token or get_hf_token() or "").strip()
    if not dry_run and not hf_token:
        raise ValueError("HF_TOKEN not set. Required to publish datasets to Hugging Face.")

    targets = build_upload_targets(
        dataset_path,
        path_in_repo=path_in_repo,
        metadata_path=metadata_path,
        include_metadata=include_metadata,
    )
    uploaded_files = [target.path_in_repo for target in targets]
    url = f"https://huggingface.co/datasets/{repo_id}"

    if dry_run:
        return DatasetPublishResult(repo_id=repo_id, uploaded_files=uploaded_files, url=url)

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)

    if create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private,
            token=hf_token,
        )

    message = commit_message or f"Upload dataset {targets[0].path_in_repo}"
    for target in targets:
        api.upload_file(
            path_or_fileobj=str(target.local_path),
            path_in_repo=target.path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
            commit_message=message,
        )

    return DatasetPublishResult(repo_id=repo_id, uploaded_files=uploaded_files, url=url)


def format_publish_summary(result: DatasetPublishResult, *, targets: Optional[Iterable[UploadTarget]] = None) -> str:
    lines = [
        f"Repo: {result.repo_id}",
        f"URL: {result.url}",
        "Files:",
    ]
    if targets is not None:
        for target in targets:
            lines.append(f"  - {target.local_path} -> {target.path_in_repo}")
    else:
        for name in result.uploaded_files:
            lines.append(f"  - {name}")
    return "\n".join(lines)


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
