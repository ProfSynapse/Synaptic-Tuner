"""Pure, inspect-only planning for config-driven Modal SFT jobs.

This module deliberately does not import Modal.  Building a plan performs local
validation and hashing only; it cannot create an app, Volume, or job.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional
from urllib.parse import urlsplit

import yaml

_FULL_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_OCI_DIGEST = re.compile(r"^.+@sha256:[0-9a-fA-F]{64}$")
_EXACT_PIP = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^*<>=!~,;\s]+$")
_HASHED_URL_PIP = re.compile(
    r"^(?:[A-Za-z0-9_.-]+\s*@\s*)?https://\S+#sha256=[0-9a-fA-F]{64}$"
)
_PINNED_GIT_PIP = re.compile(r"^(?:[A-Za-z0-9_.-]+\s*@\s*)?git\+https://\S+@[0-9a-fA-F]{40}(?:#\S+)?$")


class ConfigurationError(ValueError):
    """A local Modal plan is incomplete, mutable, or internally inconsistent."""


@dataclass(frozen=True)
class RepoSource:
    url: str
    branch: str
    commit: str


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ConfigurationError(f"Failed to run git {' '.join(args)}: {exc}") from exc


def resolve_repo_source(repo_root: Path) -> RepoSource:
    """Resolve a fully clean named branch whose exact HEAD is already pushed."""
    url_result = _git(repo_root, "remote", "get-url", "origin")
    url = url_result.stdout.strip() if url_result.returncode == 0 else ""
    if not url:
        raise ConfigurationError("Cannot determine git remote 'origin' URL.")
    branch_result = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    branch = branch_result.stdout.strip()
    if branch_result.returncode != 0 or not branch or branch == "HEAD":
        raise ConfigurationError("Modal planning requires a named git branch.")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0 or status.stdout.strip():
        raise ConfigurationError("Modal planning requires a fully clean worktree.")
    commit_result = _git(repo_root, "rev-parse", "HEAD")
    commit = commit_result.stdout.strip()
    if commit_result.returncode != 0 or not _FULL_GIT_SHA.fullmatch(commit):
        raise ConfigurationError("Cannot resolve a full git HEAD commit SHA.")
    remote_ref = f"origin/{branch}"
    remote = _git(repo_root, "rev-parse", "--verify", remote_ref)
    if remote.returncode != 0:
        raise ConfigurationError(f"Pushed branch {remote_ref!r} does not exist.")
    remote_commit = remote.stdout.strip()
    if remote_commit != commit:
        raise ConfigurationError(
            f"Local HEAD {commit} must equal remote branch tip {remote_ref} ({remote_commit})."
        )
    return RepoSource(url=normalize_repo_url(url), branch=branch, commit=commit)


def normalize_repo_url(url: str) -> str:
    """Reject credential-bearing HTTP(S) URLs before they reach plans or logs."""
    value = url.strip()
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"} and (parsed.username or parsed.password):
        raise ConfigurationError(
            "Credential-bearing repository URLs are forbidden; use a credential-free remote URL."
        )
    if not value or any(char in value for char in ("\n", "\r", "\x00")):
        raise ConfigurationError("Repository URL is empty or contains control characters.")
    return value


def validate_exact_pip_requirement(requirement: str) -> str:
    value = requirement.strip()
    if re.search(r"https?://[^/@\s]+@", value):
        raise ConfigurationError("Credential-bearing pip requirement URLs are forbidden.")
    if not (
        _EXACT_PIP.fullmatch(value)
        or _HASHED_URL_PIP.fullmatch(value)
        or _PINNED_GIT_PIP.fullmatch(value)
    ):
        raise ConfigurationError(
            "Modal pip overlays must use name==version, a URL with #sha256=<64-hex>, "
            "or a git+https URL pinned to a full 40-character commit. Got "
            f"{value!r}."
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_readable_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ConfigurationError(f"{label} must be a readable regular file: {resolved}")
    try:
        with resolved.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read {label} {resolved}: {exc}") from exc
    return resolved


def _safe_volume_prefix(value: str) -> PurePosixPath:
    prefix = PurePosixPath(value)
    if not value or prefix.is_absolute() or any(part in {"", ".", ".."} for part in prefix.parts):
        raise ConfigurationError(
            "input_prefix must be a non-empty relative Modal Volume path without '.' or '..'."
        )
    return prefix


def build_modal_sft_plan(
    *,
    repo_root: Path,
    config_path: Path,
    dataset_path: Path,
    input_volume_name: str,
    input_prefix: str,
    runtime_image: str,
    pip_packages: Iterable[str],
    gpu: str,
    timeout_hours: float,
    output_volume_name: str = "toolset-training-artifacts",
    output_mount_path: str = "/vol/artifacts",
    input_mount_path: str = "/vol/inputs",
    cache_volume_name: str = "toolset-model-cache",
    source: Optional[RepoSource] = None,
) -> dict:
    """Validate inputs and return an exact, non-executing Modal job plan."""
    repo_root = repo_root.resolve(strict=True)
    config_path = _regular_readable_file(config_path, "SFT config")
    dataset_path = _regular_readable_file(dataset_path, "SFT dataset")

    if not input_volume_name.strip() or not output_volume_name.strip():
        raise ConfigurationError("Modal input and output Volume names must be non-empty.")
    if not _OCI_DIGEST.fullmatch(runtime_image):
        raise ConfigurationError(
            "runtime_image must be immutable and include a full @sha256:<64-hex> digest."
        )
    if not gpu.strip():
        raise ConfigurationError("gpu must be non-empty.")
    if timeout_hours <= 0:
        raise ConfigurationError("timeout_hours must be greater than zero.")

    packages = [validate_exact_pip_requirement(item) for item in pip_packages]
    if any(not isinstance(item, str) or not item.strip() for item in packages):
        raise ConfigurationError("Every pip package entry must be a non-empty string.")

    source = source or resolve_repo_source(repo_root)
    source = RepoSource(
        url=normalize_repo_url(source.url), branch=source.branch, commit=source.commit
    )
    if not _FULL_GIT_SHA.fullmatch(source.commit):
        raise ConfigurationError("Modal jobs require a full 40-character git commit SHA.")

    prefix = _safe_volume_prefix(input_prefix)
    mount = PurePosixPath(input_mount_path)
    if not mount.is_absolute():
        raise ConfigurationError("input_mount_path must be absolute.")
    remote_config_rel = prefix / config_path.name
    remote_dataset_rel = prefix / dataset_path.name
    remote_config = str(mount / remote_config_rel)
    remote_dataset = str(mount / remote_dataset_rel)

    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot parse SFT YAML config {config_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError("SFT YAML config must contain a top-level mapping.")
    dataset = document.get("dataset")
    if not isinstance(dataset, dict):
        raise ConfigurationError("SFT YAML config must contain a dataset mapping.")
    configured_dataset = dataset.get("local_file")
    if configured_dataset != remote_dataset:
        raise ConfigurationError(
            "dataset.local_file must name the staged Modal input exactly: "
            f"expected {remote_dataset!r}, got {configured_dataset!r}."
        )

    wrapper = repo_root / "Trainers" / "cloud" / "train_modal.py"
    if not wrapper.is_file():
        raise ConfigurationError(f"Modal wrapper not found: {wrapper}")

    config_sha = _sha256(config_path)
    dataset_sha = _sha256(dataset_path)
    launch_argv = [
        "modal", "run", "--detach", f"{wrapper}::run_training",
        "--trainer-type", "sft",
        "--repo-url", source.url,
        "--repo-branch", source.branch,
        "--repo-commit", source.commit,
        "--config-path", remote_config,
        "--config-sha256", config_sha,
        "--dataset-sha256", dataset_sha,
    ]
    environment = {
        "MODAL_TRAINING_IMAGE": runtime_image,
        "MODAL_TRAINING_PIP_PACKAGES_JSON": json.dumps(packages, separators=(",", ":")),
        "MODAL_GPU": gpu,
        "MODAL_TIMEOUT_SECONDS": str(int(timeout_hours * 3600)),
        "MODAL_INPUT_VOLUME_NAME": input_volume_name,
        "MODAL_INPUT_MOUNT_PATH": input_mount_path,
        "MODAL_CACHE_VOLUME_NAME": cache_volume_name,
        "MODAL_OUTPUT_VOLUME_NAME": output_volume_name,
        "MODAL_OUTPUT_MOUNT_PATH": output_mount_path,
    }

    return {
        "schema_version": 1,
        "inspection_only": True,
        "source": {
            "repo_url": source.url,
            "branch": source.branch,
            "commit": source.commit.lower(),
        },
        "runtime": {
            "image": runtime_image,
            "pip_packages": packages,
            "gpu": gpu,
            "timeout_hours": timeout_hours,
        },
        "inputs": {
            "volume_name": input_volume_name,
            "mount_path": input_mount_path,
            "config": {
                "local_path": str(config_path),
                "volume_path": str(remote_config_rel),
                "mounted_path": remote_config,
                "sha256": config_sha,
                "bytes": config_path.stat().st_size,
            },
            "dataset": {
                "local_path": str(dataset_path),
                "volume_path": str(remote_dataset_rel),
                "mounted_path": remote_dataset,
                "sha256": dataset_sha,
                "bytes": dataset_path.stat().st_size,
            },
        },
        "artifacts": {
            "backend": "modal_volume",
            "volume_name": output_volume_name,
            "mount_path": output_mount_path,
            "canonical_root": f"{output_mount_path}/outputs/runs/modal/sft",
            "publish_final_model": False,
        },
        "staging": {
            "create_volume_argv": ["modal", "volume", "create", input_volume_name],
            "config_put_argv": [
                "modal", "volume", "put", input_volume_name, str(config_path), str(remote_config_rel),
            ],
            "dataset_put_argv": [
                "modal", "volume", "put", input_volume_name, str(dataset_path), str(remote_dataset_rel),
            ],
        },
        "launch": {
            "environment": environment,
            "argv": launch_argv,
            "verification": {
                "require_nonempty_app_id_from_submission": True,
                "require_remote_function": "run_training",
                "require_running_or_completed_task": True,
                "commands": [
                    ["modal", "app", "list", "--json"],
                    ["modal", "app", "logs", "<app-id>"],
                ],
            },
        },
    }
