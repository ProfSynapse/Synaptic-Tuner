"""
Utilities for reading and listing local or HF bucket-backed artifacts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from posixpath import normpath
from typing import Any, Iterable, TextIO

from huggingface_hub import HfFileSystem

from shared.utilities.env import get_hf_token


def _load_sync_bucket():
    """Import sync_bucket lazily so list/read still work on older Hub builds."""
    try:
        from huggingface_hub import sync_bucket as _sync_bucket
    except ImportError:
        return None
    return _sync_bucket


def _strip_hf_scheme(path: str) -> str:
    normalized = str(path or "").strip()
    if normalized.startswith("hf://"):
        return normalized[len("hf://") :]
    return normalized


def _hf_relative_child(root: str, child: str) -> str:
    root_norm = normpath(_strip_hf_scheme(root)).rstrip("/")
    child_norm = normpath(_strip_hf_scheme(child)).rstrip("/")
    prefix = f"{root_norm}/"
    if child_norm.startswith(prefix):
        return child_norm[len(prefix) :]
    return Path(child_norm).name


def _copy_hf_tree(fs: HfFileSystem, artifact_path: str, target: Path) -> None:
    """Fallback pull implementation when sync_bucket is unavailable."""
    info = fs.info(artifact_path)
    entry_type = info.get("type", "file")

    if entry_type == "directory":
        target.mkdir(parents=True, exist_ok=True)
        raw_entries = fs.find(artifact_path, detail=True)
        iterator = raw_entries.items() if isinstance(raw_entries, dict) else []
        for remote_path, details in iterator:
            if details.get("type") == "directory":
                continue
            relative = _hf_relative_child(artifact_path, remote_path)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with fs.open(remote_path, "rb") as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with fs.open(artifact_path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)


def build_artifact_path(path: str, *, bucket_id: str | None = None) -> str:
    """Return a local path or fully-qualified HF bucket URI."""
    normalized = str(path or "").strip()
    if not normalized:
        raise ValueError("Artifact path is required.")
    if normalized.startswith("hf://"):
        return normalized
    if normalized.startswith("/") or normalized.startswith("./") or normalized.startswith("../"):
        return normalized
    if bucket_id:
        return f"hf://buckets/{bucket_id.strip('/')}/{normalized.lstrip('/')}"
    return normalized


def _artifact_relative_path(path: str, *, bucket_id: str | None = None) -> Path:
    artifact_path = build_artifact_path(path, bucket_id=bucket_id)
    if artifact_path.startswith("hf://buckets/"):
        remainder = artifact_path[len("hf://buckets/") :]
        normalized_bucket = str(bucket_id or "").strip("/")
        if normalized_bucket and remainder.startswith(f"{normalized_bucket}/"):
            relative = remainder[len(normalized_bucket) + 1 :]
        else:
            parts = remainder.split("/")
            if len(parts) >= 3:
                relative = "/".join(parts[2:])
            elif len(parts) >= 2:
                relative = "/".join(parts[1:])
            else:
                relative = ""
        return Path(relative)
    local_path = Path(path)
    if local_path.is_absolute():
        try:
            return local_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            return Path(local_path.name)
    return local_path


def _open_path(path: str) -> TextIO:
    if path.startswith("hf://"):
        fs = HfFileSystem(token=get_hf_token())
        return fs.open(path, "r", encoding="utf-8")
    return open(Path(path), "r", encoding="utf-8")


def tail_lines(lines: Iterable[str], count: int) -> list[str]:
    if count <= 0:
        return list(lines)
    buffer: list[str] = []
    for line in lines:
        buffer.append(line)
        if len(buffer) > count:
            buffer.pop(0)
    return buffer


def latest_jsonl_record(lines: Iterable[str]) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
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
    bucket_id: str | None = None,
    tail: int | None = None,
    jsonl_latest: bool = False,
    pretty: bool = False,
) -> str:
    artifact_path = build_artifact_path(path, bucket_id=bucket_id)
    with _open_path(artifact_path) as handle:
        if jsonl_latest:
            record = latest_jsonl_record(handle)
            return json.dumps(record, indent=2 if pretty else None, sort_keys=pretty)

        if tail is not None:
            lines = tail_lines(handle, tail)
            return "".join(lines)

        contents = handle.read()
        if pretty:
            try:
                return json.dumps(json.loads(contents), indent=2, sort_keys=True)
            except json.JSONDecodeError:
                return contents
        return contents


def list_artifacts(
    path: str,
    *,
    bucket_id: str | None = None,
    recursive: bool = False,
    files_only: bool = False,
    dirs_only: bool = False,
) -> list[dict[str, Any]]:
    artifact_path = build_artifact_path(path, bucket_id=bucket_id)
    if artifact_path.startswith("hf://"):
        fs = HfFileSystem(token=get_hf_token())
        raw_entries = fs.find(artifact_path, detail=True) if recursive else fs.ls(artifact_path, detail=True)
        if isinstance(raw_entries, dict):
            iterator = raw_entries.items()
        else:
            iterator = []
            for entry in raw_entries:
                if isinstance(entry, dict):
                    entry_path = entry.get("name") or entry.get("path")
                    iterator.append((entry_path, entry))
                else:
                    iterator.append((str(entry), {"name": str(entry), "type": "file"}))
        entries: list[dict[str, Any]] = []
        for entry_path, details in iterator:
            entry_type = details.get("type", "file")
            is_dir = entry_type == "directory"
            if files_only and is_dir:
                continue
            if dirs_only and not is_dir:
                continue
            entries.append(
                {
                    "path": str(entry_path),
                    "type": entry_type,
                    "size": details.get("size"),
                }
            )
        return sorted(entries, key=lambda item: item["path"])

    root = Path(artifact_path)
    if root.is_file():
        return [{"path": str(root), "type": "file", "size": root.stat().st_size}]
    if not root.exists():
        raise FileNotFoundError(str(root))
    walker = root.rglob("*") if recursive else root.iterdir()
    entries = []
    for entry in walker:
        is_dir = entry.is_dir()
        if files_only and is_dir:
            continue
        if dirs_only and not is_dir:
            continue
        entries.append(
            {
                "path": str(entry),
                "type": "directory" if is_dir else "file",
                "size": None if is_dir else entry.stat().st_size,
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def pull_artifacts(
    path: str,
    *,
    bucket_id: str | None = None,
    destination: str | Path = ".",
) -> Path:
    artifact_path = build_artifact_path(path, bucket_id=bucket_id)
    relative = _artifact_relative_path(path, bucket_id=bucket_id)
    destination_root = Path(destination).resolve()
    target = destination_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    if artifact_path.startswith("hf://"):
        sync_bucket = _load_sync_bucket()
        if sync_bucket is not None:
            sync_bucket(artifact_path, str(target), token=get_hf_token())
        else:
            fs = HfFileSystem(token=get_hf_token())
            _copy_hf_tree(fs, artifact_path, target)
        return target

    source = Path(artifact_path)
    if not source.exists():
        raise FileNotFoundError(str(source))
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)
    return target


def _default_remote_path(source: Path) -> str:
    if source.is_absolute():
        try:
            return str(source.resolve().relative_to(Path.cwd().resolve())).strip("/")
        except ValueError:
            return source.name
    return str(source).strip("/")


def _resolve_remote_target(source: Path, destination: str | None) -> str:
    default_target = _default_remote_path(source)
    if not destination:
        return default_target

    normalized = destination.strip().lstrip("/")
    if not normalized:
        return default_target

    if source.is_file():
        if destination.endswith("/"):
            return f"{normalized.rstrip('/')}/{source.name}"
        destination_name = Path(normalized).name
        if destination_name == source.name or Path(destination_name).suffix:
            return normalized
        return f"{normalized.rstrip('/')}/{source.name}"

    return normalized.rstrip("/")


def push_artifacts(
    path: str,
    *,
    bucket_id: str,
    destination: str | None = None,
) -> str:
    from shared.cloud_artifacts import sync_directory_to_hf_bucket, sync_file_to_hf_bucket

    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    normalized_bucket = bucket_id.strip("/")
    remote_target = _resolve_remote_target(source, destination)

    if source.is_dir():
        sync_directory_to_hf_bucket(source, normalized_bucket, remote_target, token=get_hf_token())
    else:
        sync_file_to_hf_bucket(source, normalized_bucket, remote_target, token=get_hf_token())
    return f"hf://buckets/{normalized_bucket}/{remote_target}"
