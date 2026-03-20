"""
Cloud artifact helpers shared by training scripts and cloud backends.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import TrainerCallback


_HF_BUCKET_CACHE: Dict[str, str] = {}


def _normalize_token_value(token: Optional[str]) -> Optional[str]:
    """Normalize optional auth tokens, treating blank strings as unset."""
    if token is None:
        return None
    token = token.strip()
    return token or None


def _bucket_sync_helper_python() -> Optional[str]:
    """Return the isolated Python interpreter for bucket sync, if configured."""
    helper_python = os.environ.get("HF_BUCKET_SYNC_PYTHON", "").strip()
    return helper_python or None


def _bucket_sync_helper_pythonpath() -> Optional[str]:
    """Return additional PYTHONPATH entries for the isolated bucket helper."""
    helper_pythonpath = os.environ.get("HF_BUCKET_SYNC_PYTHONPATH", "").strip()
    return helper_pythonpath or None


def _bucket_sync_helper_script() -> Path:
    """Return the helper script path used for isolated bucket sync."""
    return Path(__file__).with_name("hf_bucket_sync_helper.py")


@dataclass
class RunPaths:
    """Canonical run paths for cloud training outputs."""

    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    final_model_dir: Path
    lineage_path: Path
    manifest_path: Path
    per_example_losses_path: Path | None = None


def normalize_hf_bucket_id(bucket_id: str) -> str:
    """Normalize bucket identifiers to the canonical namespace/name form."""
    normalized = bucket_id.strip()
    for prefix in ("hf://buckets/", "buckets/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized.strip("/")


def build_run_paths(base_output_dir: Path, provider: str, method: str, timestamp: str, commit: str) -> RunPaths:
    """Build the canonical run layout used by cloud providers."""
    short_sha = (commit or "local")[:8]
    run_dir = base_output_dir / "runs" / provider / method / f"{timestamp}-{short_sha}"
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        logs_dir=run_dir / "logs",
        final_model_dir=run_dir / "final_model",
        lineage_path=run_dir / "training_lineage.json",
        manifest_path=run_dir / "manifest.json",
    )


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Write a manifest file using stable JSON formatting.

    After writing, attempts to register the run in the unified experiment
    tracking registry (best-effort — failure is logged, never raised).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Best-effort registration in unified tracking registry
    try:
        from shared.experiment_tracking.adapters import manifest_to_run_record
        from shared.experiment_tracking.registry import RunRegistry

        record = manifest_to_run_record(payload)
        RunRegistry().register_run(record)
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "Unified tracking registration from manifest failed (non-fatal)",
            exc_info=True,
        )


def build_manifest(
    *,
    provider: str,
    method: str,
    artifact_backend: str,
    artifact_identifier: Optional[str],
    run_paths: RunPaths,
    repo_branch: Optional[str],
    repo_commit: Optional[str],
    publish_final_model: bool,
    publish_target_repo: Optional[str],
    status: str,
) -> Dict[str, Any]:
    """Build the canonical cloud run manifest."""
    return {
        "provider": provider,
        "method": method,
        "status": status,
        "artifact_backend": artifact_backend,
        "artifact_identifier": artifact_identifier,
        "publish_final_model": publish_final_model,
        "publish_target_repo": publish_target_repo,
        "repo_branch": repo_branch,
        "repo_commit": repo_commit,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paths": {
            "run_dir": str(run_paths.run_dir),
            "checkpoints_dir": str(run_paths.checkpoints_dir),
            "logs_dir": str(run_paths.logs_dir),
            "final_model_dir": str(run_paths.final_model_dir),
            "training_lineage": str(run_paths.lineage_path),
            "per_example_losses": str(run_paths.per_example_losses_path) if run_paths.per_example_losses_path else None,
        },
    }


def ensure_hf_bucket(bucket_id: str, token: Optional[str] = None) -> str:
    """Best-effort bucket creation returning the normalized bucket identifier."""
    normalized_bucket_id = normalize_hf_bucket_id(bucket_id)
    cached_bucket_id = _HF_BUCKET_CACHE.get(normalized_bucket_id)
    if cached_bucket_id:
        return cached_bucket_id

    try:
        from huggingface_hub import create_bucket  # type: ignore
    except ImportError:
        return normalized_bucket_id
    except Exception:
        return normalized_bucket_id

    try:
        bucket_info = create_bucket(normalized_bucket_id, exist_ok=True, private=True, token=token)
    except TypeError:
        try:
            bucket_info = create_bucket(normalized_bucket_id, exist_ok=True, token=token)
        except TypeError:
            bucket_info = create_bucket(normalized_bucket_id, token=token)
    except Exception:
        return normalized_bucket_id

    resolved_bucket_id = (
        getattr(bucket_info, "bucket_id", None)
        or getattr(bucket_info, "id", None)
        or normalized_bucket_id
    )
    resolved_bucket_id = normalize_hf_bucket_id(str(resolved_bucket_id))
    _HF_BUCKET_CACHE[normalized_bucket_id] = resolved_bucket_id
    return resolved_bucket_id


def sync_directory_to_hf_bucket(local_dir: Path, bucket_id: str, prefix: str, token: Optional[str] = None) -> None:
    """Sync a local directory into a Hugging Face Bucket."""
    local_dir = Path(local_dir)
    bucket_id = normalize_hf_bucket_id(bucket_id)
    prefix = prefix.strip("/")
    bucket_uri = f"hf://buckets/{bucket_id}/{prefix}"
    token = _normalize_token_value(token)

    helper_python = _bucket_sync_helper_python()
    if helper_python:
        env = dict(os.environ)
        env_token = (
            token
            or _normalize_token_value(env.get("HF_TOKEN"))
            or _normalize_token_value(env.get("HF_API_KEY"))
        )
        if env_token:
            env["HF_TOKEN"] = env_token
            env["HF_API_KEY"] = env_token
        else:
            env.pop("HF_TOKEN", None)
            env.pop("HF_API_KEY", None)
        helper_pythonpath = _bucket_sync_helper_pythonpath()
        if helper_pythonpath:
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{helper_pythonpath}:{existing_pythonpath}"
                if existing_pythonpath
                else helper_pythonpath
            )
        try:
            subprocess.run(
                [
                    helper_python,
                    str(_bucket_sync_helper_script()),
                    str(local_dir),
                    bucket_uri,
                ],
                check=True,
                env=env,
            )
            return
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"HF bucket sync failed for {bucket_id}/{prefix}: {exc}"
            ) from exc

    bucket_id = ensure_hf_bucket(bucket_id, token=token)

    try:
        from huggingface_hub import sync_bucket  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub sync_bucket is unavailable; install huggingface_hub>=1.5.0."
        ) from exc

    try:
        sync_bucket(
            str(local_dir),
            f"hf://buckets/{bucket_id}/{prefix}",
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"HF bucket sync failed for {bucket_id}/{prefix}: {exc}"
        ) from exc


def sync_file_to_hf_bucket(local_path: Path, bucket_id: str, remote_path: str, token: Optional[str] = None) -> None:
    """Sync a single file into a Hugging Face Bucket."""
    local_path = Path(local_path)
    if not local_path.exists() or not local_path.is_file():
        return

    remote_parent = remote_path.strip("/").rsplit("/", 1)[0]
    sync_directory_to_hf_bucket(local_path.parent, bucket_id, remote_parent, token=token)


def publish_final_model_to_hub(final_model_dir: Path, repo_id: str, token: str, private: bool = True) -> None:
    """Upload a final model directory to a Hugging Face model repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(final_model_dir),
        repo_id=repo_id,
        commit_message="Upload final model from cloud training",
    )


class HFBucketSyncCallback(TrainerCallback):
    """Checkpoint-triggered sync for HF Jobs bucket persistence."""

    def __init__(
        self,
        run_dir: Path,
        bucket_id: str,
        prefix: str,
        token: Optional[str] = None,
        log_every_n_steps: int = 5,
    ):
        self.run_dir = Path(run_dir)
        self.bucket_id = bucket_id
        self.prefix = prefix
        self.token = token
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    def _latest_training_log(self) -> Optional[Path]:
        logs_dir = self.run_dir / "logs"
        candidates = sorted(logs_dir.glob("training_*.jsonl"))
        return candidates[-1] if candidates else None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state is None:
            return
        if state.global_step == 0 or state.global_step % self.log_every_n_steps != 0:
            return

        logs_dir = self.run_dir / "logs"
        if not logs_dir.exists():
            return

        sync_directory_to_hf_bucket(
            logs_dir,
            self.bucket_id,
            f"{self.prefix}/logs",
            token=self.token,
        )

    def on_save(self, args, state, control, **kwargs):
        sync_directory_to_hf_bucket(self.run_dir, self.bucket_id, self.prefix, token=self.token)
