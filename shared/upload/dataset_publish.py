"""Helpers for publishing dataset artifacts to a Hugging Face dataset repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from shared.utilities.env import get_hf_token, load_env_file


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
    """Return the conventional metadata sidecar path if it exists."""
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
    """Upload a dataset JSONL and optional metadata sidecar to an HF dataset repo."""
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
