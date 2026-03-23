"""HF Jobs training stage runner for experiment orchestration.

Extracted from experiment_handler.py to keep each stage runner in its own
module.  Used by ExperimentHandler via ExperimentOrchestrator to execute the
training stage on Hugging Face Jobs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from shared.experiment_tracking import (
    Experiment,
    ExperimentSpec,
    StageResult,
    TrackingService,
)
from shared.experiment_tracking.schema import RunRecord
from shared.utilities.env import get_hf_token
from tuner.backends.registry import TrainingBackendRegistry
from tuner.cloud import load_huggingface_hub, resolve_hf_bucket_id
from tuner.core.config import apply_training_overrides
from tuner.core.exceptions import CloudProviderError


class HFTrainingStageRunner:
    """Run the training stage on HF Jobs and register its remote artifacts."""

    def __init__(self, *, repo_root: Path, tracking_service: TrackingService):
        self.repo_root = repo_root
        self.tracking_service = tracking_service

    def _resolve_bucket_id(self, bucket_id: str) -> str:
        normalized = str(bucket_id or "").strip()
        if not normalized:
            raise CloudProviderError("HF Jobs requires an artifact bucket identifier.")
        if "/" in normalized:
            return normalized
        huggingface_hub = load_huggingface_hub(require_apis=("run_job", "create_bucket", "HfApi"))
        hf_token = get_hf_token()
        if not hf_token:
            raise CloudProviderError("HF_TOKEN not set. Required for Hugging Face bucket resolution.")
        return resolve_hf_bucket_id(
            huggingface_hub,
            normalized,
            token=hf_token,
            private=True,
        )

    def _planned_training_state(self, *, experiment: Experiment, config) -> tuple[str, str]:
        timestamp = datetime.fromisoformat(experiment.created_at).astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifact_prefix = f"runs/{config.provider}/{config.method}/{timestamp}-{config.repo_commit[:8]}"
        artifact_root = f"hf://buckets/{config.artifact_identifier}/{artifact_prefix}"
        return artifact_prefix, artifact_root

    def _bucket_has_path(self, *, bucket_id: str, prefix: str, suffix: str) -> bool:
        huggingface_hub = load_huggingface_hub(require_apis=("HfApi",))
        api = huggingface_hub.HfApi(token=get_hf_token())
        target = f"{prefix.strip('/')}/{suffix}"
        for item in api.list_bucket_tree(bucket_id, prefix=prefix.strip("/"), recursive=True, token=get_hf_token()):
            if getattr(item, "path", "") == target:
                return True
        return False

    def _training_completion_suffixes(self, method: str) -> tuple[str, ...]:
        if method == "grpo":
            return (
                "final_model/adapter_config.json",
                "final_model/config.json",
                "logs/training_latest.jsonl",
            )
        return ("training_lineage.json",)

    def _recover_existing_training(self, *, experiment: Experiment) -> Optional[StageResult]:
        details = experiment.stage_details.get("training", {})
        status = details.get("status")
        bucket_id = details.get("bucket_id") or details.get("tags", {}).get("bucket_id")
        artifact_prefix = details.get("artifact_prefix") or details.get("tags", {}).get("artifact_prefix")
        artifact_root = details.get("artifact_root")
        if status not in {"running", "completed"} or not bucket_id or not artifact_prefix or not artifact_root:
            return None
        resolved_bucket_id = self._resolve_bucket_id(bucket_id)
        if resolved_bucket_id != bucket_id:
            artifact_root = artifact_root.replace(f"hf://buckets/{bucket_id}/", f"hf://buckets/{resolved_bucket_id}/", 1)
            self.tracking_service.update_stage_details(
                experiment,
                "training",
                bucket_id=resolved_bucket_id,
                artifact_root=artifact_root,
                tags={"bucket_id": resolved_bucket_id},
            )
            details = experiment.stage_details.get("training", {})
        if any(
            self._bucket_has_path(bucket_id=resolved_bucket_id, prefix=artifact_prefix, suffix=suffix)
            for suffix in self._training_completion_suffixes(experiment.method)
        ):
            self.tracking_service.update_stage_details(experiment, "training", status="completed")
            return StageResult(
                status="completed",
                run_record=RunRecord(
                    run_id=details.get("run_id") or f"{experiment.experiment_id}-training",
                    run_type=experiment.method,
                    name=f"{experiment.name} training",
                    timestamp=details.get("updated_at") or experiment.created_at,
                    status="completed",
                    output_dir=artifact_root,
                    model_name=experiment.base_model_name,
                    dataset_source=experiment.dataset_path,
                    provider=experiment.provider,
                    artifact_backend="hf_bucket",
                    artifact_root=artifact_root,
                    job_ref=details.get("job_ref"),
                    source_commit=details.get("source_commit"),
                    stage="training",
                    tags=dict(details.get("tags", {})),
                ),
                artifact_root=artifact_root,
            )
        if status == "running" and not details.get("job_ref"):
            self.tracking_service.update_stage_details(experiment, "training", status="failed")
            return None
        if status == "running":
            raise CloudProviderError(
                f"Experiment {experiment.experiment_id} already has a running training stage at {artifact_root}. "
                "Artifacts are not complete yet, so refusing to submit a duplicate training job."
            )
        return None

    def run(self, spec: ExperimentSpec, experiment: Experiment, previous: Optional[StageResult] = None) -> StageResult:
        recovered = self._recover_existing_training(experiment=experiment)
        if recovered is not None:
            return recovered
        backend = TrainingBackendRegistry.get("hf_jobs", repo_root=self.repo_root)
        is_valid, error = backend.validate_environment()
        if not is_valid:
            raise CloudProviderError(error or "HF Jobs environment validation failed.")

        config = backend.load_config(spec.method)
        apply_training_overrides(
            config,
            model_name=spec.training.model_name,
            dataset_name=spec.dataset.source,
            dataset_file=spec.dataset.file,
            batch_size=spec.training.batch_size,
            gradient_accumulation=spec.training.gradient_accumulation,
            learning_rate=spec.training.learning_rate,
            num_epochs=spec.training.num_epochs,
            max_steps=spec.training.max_steps,
            max_seq_length=spec.training.max_seq_length,
            load_in_4bit=spec.training.load_in_4bit,
            lora_target_modules=spec.training.lora_target_modules or None,
            gpu=spec.training.gpu,
            timeout_hours=spec.training.timeout_hours,
        )
        if spec.training.cloud_image:
            config.cloud_image = spec.training.cloud_image
            config.cloud_image_profile = None
        elif spec.training.image_profile:
            from tuner.backends.training.cloud.base_cloud import resolve_cloud_image

            cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
            config.cloud_image, config.cloud_image_profile = resolve_cloud_image(
                cloud_config_path,
                requested_profile=spec.training.image_profile,
                fallback_image=config.cloud_image,
            )
        config.artifact_identifier = self._resolve_bucket_id(config.artifact_identifier)

        artifact_prefix, artifact_root = self._planned_training_state(experiment=experiment, config=config)
        self.tracking_service.update_stage_details(
            experiment,
            "training",
            status="running",
            artifact_root=artifact_root,
            artifact_prefix=artifact_prefix,
            bucket_id=config.artifact_identifier,
            source_commit=config.repo_commit,
            tags={
                "provider": spec.provider,
                "artifact_prefix": artifact_prefix,
                "bucket_id": config.artifact_identifier or "",
                "image": config.cloud_image or "",
            },
        )

        backend.show_post_training_actions = False
        result = backend.execute(config, python_path="")

        artifact_prefix = result.artifact_prefix or artifact_prefix
        bucket_id = result.bucket_id
        job_id = result.job_id
        artifact_root = (
            f"hf://buckets/{bucket_id}/{artifact_prefix.strip('/')}"
            if artifact_prefix and bucket_id
            else artifact_root
        )
        status = "completed" if result == 0 else "failed"
        self.tracking_service.update_stage_details(
            experiment,
            "training",
            status=status,
            artifact_root=artifact_root,
            job_ref=job_id,
            source_commit=config.repo_commit,
            tags={
                "provider": spec.provider,
                "artifact_prefix": artifact_prefix or "",
                "bucket_id": bucket_id or config.artifact_identifier or "",
                "image": config.cloud_image or "",
            },
        )
        record = RunRecord(
            run_id=f"{experiment.experiment_id}-training",
            run_type=spec.method,
            name=f"{spec.name} training",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            output_dir=artifact_root or "",
            model_name=spec.training.model_name,
            dataset_source=spec.dataset.identifier,
            provider=spec.provider,
            artifact_backend="hf_bucket",
            artifact_root=artifact_root,
            job_ref=job_id,
            source_commit=config.repo_commit,
            stage="training",
            hardware=config.hf_flavor or config.gpu_type,
            tags={
                "provider": spec.provider,
                "artifact_prefix": artifact_prefix or "",
                "bucket_id": bucket_id or "",
                "image": config.cloud_image or "",
            },
        )
        return StageResult(status=status, run_record=record, artifact_root=artifact_root)
