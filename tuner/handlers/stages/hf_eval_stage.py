"""HF Jobs evaluation stage runner for the experiment lifecycle.

Located at tuner/handlers/stages/hf_eval_stage.py.
Runs cloud evaluation against a freshly trained HF bucketed run.
Used by ExperimentHandler (tuner/handlers/experiment_handler.py) via
ExperimentOrchestrator as the eval_runner.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from shared.experiment_tracking import (
    Experiment,
    ExperimentSpec,
    StageResult,
    TrackingService,
)
from shared.experiment_tracking.schema import RunRecord
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.cloud_eval_handler import CloudEvalHandler


class HFEvalStageRunner:
    """Run cloud evaluation against the freshly trained HF bucketed run."""

    def __init__(self, *, repo_root: Path, tracking_service: TrackingService):
        self.repo_root = repo_root
        self.tracking_service = tracking_service

    @staticmethod
    def _use_same_job_loss(spec: ExperimentSpec) -> bool:
        post_training = getattr(spec, "post_training", None)
        mode = getattr(post_training, "mode", "parallel")
        return mode == "same_job" and bool(spec.loss.enabled)

    def run(self, spec: ExperimentSpec, experiment: Experiment, previous: StageResult | None = None) -> StageResult:
        if previous is None or previous.run_record is None:
            raise CloudProviderError("Evaluation stage requires a completed training run.")
        artifact_prefix = previous.run_record.tags.get("artifact_prefix", "")
        bucket_id = previous.run_record.tags.get("bucket_id", "")
        if not artifact_prefix or not bucket_id:
            raise CloudProviderError("Training stage did not capture the HF artifact prefix and bucket.")
        self.tracking_service.update_stage_details(
            experiment,
            "evaluation",
            status="running",
            source_commit=previous.run_record.source_commit,
            tags={
                "provider": spec.provider,
                "bucket_id": bucket_id,
                "artifact_prefix": artifact_prefix,
            },
        )

        args = Namespace(
            json=False,
            run=artifact_prefix,
            method=spec.method,
            bucket=bucket_id,
            preset=spec.evaluation.preset,
            scenario=list(spec.evaluation.scenarios),
            tags=spec.evaluation.tags,
            eval_runtime=spec.evaluation.runtime,
            eval_image_profile=spec.evaluation.image_profile,
            eval_cloud_image=spec.evaluation.cloud_image,
            eval_pip_packages=list(getattr(spec.evaluation, "pip_packages", []) or []),
            env_backend="none",
            env_template=None,
            env_tool_schema=None,
            env_exec_config=None,
            upload_to_hf=None,
            update_model_card=False,
            gpu=spec.evaluation.gpu,
            timeout_hours=spec.evaluation.timeout_hours,
            with_loss=self._use_same_job_loss(spec),
            loss_dataset_name=spec.dataset.source if self._use_same_job_loss(spec) else None,
            loss_dataset_file=spec.dataset.file if self._use_same_job_loss(spec) else None,
            loss_max_seq_length=(spec.loss.max_seq_length or spec.training.max_seq_length) if self._use_same_job_loss(spec) else None,
            loss_no_completion_only=(not spec.loss.completion_only) if self._use_same_job_loss(spec) else False,
            auto_confirm=True,
        )
        handler = CloudEvalHandler(args=args)
        handler._repo_root = self.repo_root
        exit_code = handler.handle()
        status = "completed" if exit_code == 0 else "failed"
        self.tracking_service.update_stage_details(
            experiment,
            "evaluation",
            status=status,
            artifact_root=handler.last_results_uri,
            job_ref=handler.last_job_id,
            source_commit=previous.run_record.source_commit,
            tags={
                "provider": spec.provider,
                "bucket_id": bucket_id,
                "artifact_prefix": artifact_prefix,
            },
        )
        record = RunRecord(
            run_id=f"{experiment.experiment_id}-evaluation",
            run_type="evaluation",
            name=f"{spec.name} evaluation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            output_dir=handler.last_results_uri or "",
            model_name=spec.training.model_name,
            dataset_source=spec.dataset.identifier,
            provider=spec.provider,
            artifact_backend="hf_bucket",
            artifact_root=handler.last_results_uri,
            job_ref=handler.last_job_id,
            source_commit=previous.run_record.source_commit,
            stage="evaluation",
            tags={
                "provider": spec.provider,
                "bucket_id": bucket_id,
                "artifact_prefix": artifact_prefix,
            },
        )
        return StageResult(
            status=status,
            run_record=record,
            eval_payload=handler.last_eval_payload,
            artifact_root=handler.last_results_uri,
        )
