"""
Bucket artifact read/list handler.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Optional

from shared.utilities.bucket_artifacts import list_artifacts, pull_artifacts, push_artifacts, read_artifact
from shared.utilities.env import get_hf_token
from tuner.backends.training.cloud.base_cloud import load_cloud_config
from tuner.cloud import load_huggingface_hub, resolve_hf_bucket_id
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler


class BucketHandler(BaseHandler):
    """Read or list local/HF bucket-backed artifacts from the CLI."""

    @property
    def name(self) -> str:
        return "bucket"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _cloud_config_path(self) -> Path:
        return self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"

    def _default_bucket_id(self) -> Optional[str]:
        settings = load_cloud_config(self._cloud_config_path()).get("hf_jobs", {})
        configured = str(settings.get("artifact_identifier", "")).strip()
        if not configured:
            return None
        if "/" in configured:
            return configured.strip("/")
        hf_token = get_hf_token()
        if not hf_token:
            return configured.strip("/")
        hub = load_huggingface_hub(require_apis=("create_bucket", "HfApi"))
        return resolve_hf_bucket_id(hub, configured, token=hf_token, private=True)

    def _selected_bucket_id(self) -> Optional[str]:
        explicit = str(getattr(self.args, "bucket", "") or "").strip()
        if explicit:
            return explicit.strip("/")
        return self._default_bucket_id()

    def _require_bucket_id(self) -> str:
        bucket_id = self._selected_bucket_id()
        if not bucket_id:
            raise CloudProviderError("Bucket ID is required. Use --bucket or configure HF artifact_identifier.")
        return bucket_id

    def _require_path(self) -> str:
        path = str(getattr(self.args, "path", "") or "").strip()
        if not path:
            raise CloudProviderError("Bucket path is required. Use --path <bucket-relative path or hf:// URI>.")
        return path

    def _bucket_for_path(self, path: str) -> Optional[str]:
        if path.startswith("hf://"):
            return None
        local_path = Path(path)
        if local_path.exists() or path.startswith("./") or path.startswith("../") or path.startswith("/"):
            return None
        return self._selected_bucket_id()

    def _handle_read(self) -> int:
        path = self._require_path()
        bucket_id = self._bucket_for_path(path)
        output = read_artifact(
            path,
            bucket_id=bucket_id,
            tail=getattr(self.args, "tail", None),
            jsonl_latest=bool(getattr(self.args, "jsonl_latest", False)),
            pretty=bool(getattr(self.args, "pretty", False)),
        )
        if self.json_mode:
            self.output(
                {
                    "path": path,
                    "bucket": bucket_id,
                    "content": output,
                }
            )
        elif output:
            print(output, end="" if output.endswith("\n") else "\n")
        return 0

    def _handle_list(self) -> int:
        path = self._require_path()
        bucket_id = self._bucket_for_path(path)
        entries = list_artifacts(
            path,
            bucket_id=bucket_id,
            recursive=bool(getattr(self.args, "recursive", False)),
            files_only=bool(getattr(self.args, "files_only", False)),
            dirs_only=bool(getattr(self.args, "dirs_only", False)),
        )
        limit = getattr(self.args, "limit", None)
        if limit:
            entries = entries[:limit]
        if self.json_mode:
            self.output(
                {
                    "path": path,
                    "bucket": bucket_id,
                    "entries": entries,
                }
            )
            return 0
        for entry in entries:
            size = entry.get("size")
            suffix = f" ({size} bytes)" if isinstance(size, int) else ""
            print(f"{entry['type']}: {entry['path']}{suffix}")
        return 0

    def _handle_pull(self) -> int:
        path = self._require_path()
        bucket_id = self._bucket_for_path(path)
        destination = str(getattr(self.args, "dest", "") or ".").strip() or "."
        local_path = pull_artifacts(path, bucket_id=bucket_id, destination=destination)
        payload = {
            "path": path,
            "bucket": bucket_id,
            "destination": str(Path(destination).resolve()),
            "local_path": str(local_path),
        }
        if self.json_mode:
            self.output(payload)
        else:
            print(f"Pulled to {local_path}")
        return 0

    def _handle_push(self) -> int:
        path = self._require_path()
        source = Path(path)
        if not source.exists():
            raise CloudProviderError(f"Local source path not found: {path}")
        bucket_id = self._require_bucket_id()
        destination = str(getattr(self.args, "dest", "") or "").strip() or None
        remote_uri = push_artifacts(path, bucket_id=bucket_id, destination=destination)
        payload = {
            "path": str(source.resolve()),
            "bucket": bucket_id,
            "destination": destination,
            "remote_uri": remote_uri,
        }
        if self.json_mode:
            self.output(payload)
        else:
            print(f"Pushed to {remote_uri}")
        return 0

    def handle(self) -> int:
        try:
            subcommand = str(getattr(self.args, "subcommand", "") or "").strip().lower()
            if subcommand == "read":
                return self._handle_read()
            if subcommand == "list":
                return self._handle_list()
            if subcommand == "pull":
                return self._handle_pull()
            if subcommand == "push":
                return self._handle_push()
            raise CloudProviderError("Bucket command requires subcommand 'read', 'list', 'pull', or 'push'.")
        except Exception as exc:
            self.output_error(str(exc), code="BUCKET_ERROR")
            return 1
