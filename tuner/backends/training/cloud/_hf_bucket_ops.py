"""
HF Jobs bucket operations mixin.

Location: tuner/backends/training/cloud/_hf_bucket_ops.py
Purpose: Manage HF bucket creation, syncing, and artifact downloads
Used by: HFJobsBackend (via mixin inheritance) in hf_jobs_backend.py

Handles all interactions with HuggingFace buckets: ensuring buckets exist,
building remote URIs, syncing artifacts between remote and local storage,
checking for completion artifacts, and recovering runs from bucket state
when job status reporting lags.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from shared.utilities.env import get_hf_token
from shared.utilities.paths import get_primary_training_output_dir
from tuner.cloud import resolve_hf_bucket_id
from tuner.core.config import CloudTrainingConfig
from tuner.core.exceptions import CloudProviderError

logger = logging.getLogger(__name__)


class HFBucketOpsMixin:
    """Methods for HF bucket creation, syncing, and artifact management."""

    def _ensure_hf_bucket(self, config: CloudTrainingConfig, huggingface_hub) -> None:
        """Resolve and create the configured HF bucket before launching the job."""
        if not config.artifact_identifier:
            raise CloudProviderError("HF Jobs requires an artifact bucket identifier.")

        hf_token = get_hf_token()
        if not hf_token:
            raise CloudProviderError(
                "HF_TOKEN not set. Required for HuggingFace Jobs bucket creation."
            )
        config.artifact_identifier = resolve_hf_bucket_id(
            huggingface_hub,
            config.artifact_identifier,
            token=hf_token,
            private=True,
        )
        logger.info("Using HF bucket: %s", config.artifact_identifier)

    def _build_remote_run_uri(self, config: CloudTrainingConfig, artifact_prefix: str) -> str:
        """Return the HF bucket URI for a completed run."""
        if not config.artifact_identifier:
            raise CloudProviderError("HF Jobs requires an artifact bucket identifier.")
        return f"hf://buckets/{config.artifact_identifier}/{artifact_prefix.strip('/')}"

    def _local_download_run_dir(self, config: CloudTrainingConfig, artifact_prefix: str) -> Path:
        """Return the local run directory used for downloaded cloud runs."""
        run_slug = artifact_prefix.strip("/").split("/")[-1]
        return get_primary_training_output_dir(config.method, self.repo_root) / run_slug

    def _sync_bucket_path(self, remote_uri: str, local_dir: Path, *, token: Optional[str] = None) -> None:
        """Sync a remote HF bucket path into a local directory."""
        from huggingface_hub import sync_bucket

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        sync_bucket(remote_uri, str(local_dir), token=token)

    def _run_dir_has_completion_artifacts(self, run_dir: Path) -> bool:
        """Return True when a run directory appears complete enough for next-step workflows."""
        final_model_dir = run_dir / "final_model"
        has_final_model = final_model_dir.exists() and any(final_model_dir.iterdir())
        has_lineage = (run_dir / "training_lineage.json").exists()
        has_grpo_logs = (run_dir / "logs" / "training_latest.jsonl").exists()
        return has_final_model and (has_lineage or has_grpo_logs)

    def _download_completed_run(
        self,
        *,
        config: CloudTrainingConfig,
        artifact_prefix: str,
        local_run_dir: Optional[Path] = None,
    ) -> Path:
        """Download a completed HF Jobs run into the local training outputs tree."""
        target_dir = Path(local_run_dir) if local_run_dir else self._local_download_run_dir(config, artifact_prefix)
        self._sync_bucket_path(
            self._build_remote_run_uri(config, artifact_prefix),
            target_dir,
            token=get_hf_token(),
        )
        return target_dir

    def _recover_completed_run_from_bucket(
        self,
        *,
        config: CloudTrainingConfig,
        artifact_prefix: str,
    ) -> bool:
        """Best-effort completion recovery when HF status lags behind uploaded artifacts."""
        recovery_dir = Path(tempfile.mkdtemp(prefix="hf-job-recovery-"))
        try:
            self._sync_bucket_path(
                self._build_remote_run_uri(config, artifact_prefix),
                recovery_dir,
                token=get_hf_token(),
            )
            return self._run_dir_has_completion_artifacts(recovery_dir)
        except Exception as exc:
            logger.warning("HF Jobs completion recovery failed: %s", exc)
            return False
        finally:
            shutil.rmtree(recovery_dir, ignore_errors=True)
