"""HF Jobs per-example loss stage runner for the experiment lifecycle.

Located at tuner/handlers/stages/hf_loss_stage.py.
Runs remote per-example loss computation for a bucketed training run.
Used by ExperimentHandler (tuner/handlers/experiment_handler.py) via
ExperimentOrchestrator as the loss_runner.
"""

from __future__ import annotations

import shlex
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from shared.experiment_tracking import (
    Experiment,
    ExperimentSpec,
    StageResult,
    TrackingService,
    load_losses,
)
from shared.experiment_tracking.schema import RunRecord
from shared.utilities.env import get_hf_token
from tuner.backends.registry import TrainingBackendRegistry
from tuner.cloud import (
    CloudJobSpec,
    HFJobExecutor,
    RepoCheckoutSpec,
    build_bash_command,
    build_hf_job_secrets,
    build_repo_checkout_steps,
    load_huggingface_hub,
)
from tuner.core.exceptions import CloudProviderError


class HFLossStageRunner:
    """Run remote per-example loss computation for a bucketed training run."""

    def __init__(self, *, repo_root: Path, tracking_service: TrackingService):
        self.repo_root = repo_root
        self.tracking_service = tracking_service

    def _build_command(
        self,
        *,
        spec: ExperimentSpec,
        training_run: RunRecord,
        results_prefix: str,
    ) -> str:
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        from tuner.backends.training.cloud.base_cloud import load_project_deps, resolve_repo_source

        project_deps = load_project_deps(cloud_config_path)
        repo_source = resolve_repo_source(self.repo_root)
        checkout_steps = build_repo_checkout_steps(
            RepoCheckoutSpec(
                url=repo_source.url,
                branch=repo_source.branch,
                commit=repo_source.commit,
            )
        )

        parts = [
            f"$(command -v python3 || command -v python) -m pip install --upgrade {' '.join(shlex.quote(dep) for dep in project_deps)}",
            "mkdir -p /tmp/hf-bucket-sync-site",
            "$(command -v python3 || command -v python) -m pip install --upgrade --target /tmp/hf-bucket-sync-site huggingface_hub>=1.5.0 hf_transfer",
            "export HF_BUCKET_SYNC_PYTHON=$(command -v python3 || command -v python)",
            "export HF_BUCKET_SYNC_PYTHONPATH=/tmp/hf-bucket-sync-site",
            "export HF_HUB_ENABLE_HF_TRANSFER=1",
            *checkout_steps,
        ]
        if spec.loss.pip_packages:
            parts.append(
                "$(command -v python3 || command -v python) -m pip install --upgrade "
                + " ".join(shlex.quote(pkg) for pkg in spec.loss.pip_packages)
            )
        loss_cmd = [
            "python3",
            "-m",
            "shared.experiment_tracking.cloud_loss_job",
            "--bucket-id",
            training_run.tags["bucket_id"],
            "--run-prefix",
            training_run.tags["artifact_prefix"],
            "--dataset-name",
            spec.dataset.source,
            "--dataset-file",
            spec.dataset.file,
            "--results-prefix",
            results_prefix,
        ]
        max_seq_length = spec.loss.max_seq_length or spec.training.max_seq_length
        if max_seq_length is not None:
            loss_cmd.extend(["--max-seq-length", str(max_seq_length)])
        if not spec.loss.completion_only:
            loss_cmd.append("--no-completion-only")
        parts.append("cd /workspace/repo && " + " ".join(shlex.quote(arg) for arg in loss_cmd))
        return " && ".join(parts)

    def _poll_job(self, huggingface_hub, *, job_id: str, timeout_hours: float) -> int:
        import time

        timeout_seconds = int(timeout_hours * 3600)
        elapsed = 0
        last_log_offset = 0
        while elapsed < timeout_seconds:
            job_info = huggingface_hub.inspect_job(job_id=job_id)
            status_obj = getattr(job_info, "status", None)
            status = status_obj.stage if status_obj and hasattr(status_obj, "stage") else str(status_obj or "UNKNOWN")
            try:
                logs = huggingface_hub.fetch_job_logs(job_id=job_id) or ""
                if len(logs) > last_log_offset:
                    print(logs[last_log_offset:], end="", flush=True)
                    last_log_offset = len(logs)
            except Exception:
                pass
            if status in ("completed", "COMPLETED"):
                return 0
            if status in ("error", "ERROR", "failed", "FAILED", "cancelled", "CANCELLED"):
                return 1
            time.sleep(30)
            elapsed += 30
        return 1

    def _download_results(self, *, bucket_id: str, results_prefix: str) -> Path | None:
        try:
            from huggingface_hub import sync_bucket
        except ImportError:
            return None

        local_root = Path(tempfile.mkdtemp(prefix="experiment-loss-results-"))
        try:
            sync_bucket(
                f"hf://buckets/{bucket_id}/{results_prefix.strip('/')}",
                str(local_root),
                token=get_hf_token(),
            )
        except Exception:
            for child in local_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            local_root.rmdir()
            return None
        return local_root

    def _inspect_job_stage(self, job_ref: str) -> str | None:
        try:
            huggingface_hub = load_huggingface_hub(require_apis=("inspect_job",))
            job_info = huggingface_hub.inspect_job(job_id=job_ref)
        except Exception:
            return None
        status_obj = getattr(job_info, "status", None)
        stage = status_obj.stage if status_obj and hasattr(status_obj, "stage") else status_obj
        return str(stage).lower() if stage else None

    def _recover_existing_loss(self, *, experiment: Experiment) -> StageResult | None:
        details = experiment.stage_details.get("loss", {})
        status = details.get("status")
        artifact_root = details.get("artifact_root")
        bucket_id = details.get("bucket_id") or details.get("tags", {}).get("bucket_id")
        artifact_prefix = details.get("artifact_prefix") or details.get("tags", {}).get("artifact_prefix")
        job_ref = details.get("job_ref")
        if not artifact_root or not bucket_id or not artifact_prefix or status not in {"running", "completed"}:
            return None

        results_prefix = artifact_root.replace(f"hf://buckets/{bucket_id}/", "", 1)
        results_dir = self._download_results(bucket_id=bucket_id, results_prefix=results_prefix)
        losses_path = results_dir / "per_example_losses.jsonl" if results_dir else None
        if losses_path and losses_path.exists():
            loss_results = load_losses(losses_path)
            if results_dir is not None:
                shutil.rmtree(results_dir, ignore_errors=True)
            self.tracking_service.update_stage_details(experiment, "loss", status="completed")
            return StageResult(
                status="completed",
                run_record=RunRecord(
                    run_id=details.get("run_id") or f"{experiment.experiment_id}-loss",
                    run_type="loss",
                    name=f"{experiment.name} per-example loss",
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
                    stage="loss",
                    tags=dict(details.get("tags", {})),
                ),
                loss_results=loss_results,
                artifact_root=artifact_root,
            )
        if results_dir is not None:
            shutil.rmtree(results_dir, ignore_errors=True)
        if status == "running":
            job_stage = self._inspect_job_stage(job_ref) if job_ref else None
            if job_stage in {"error", "failed", "cancelled", "canceled"}:
                self.tracking_service.update_stage_details(experiment, "loss", status="failed")
                return None
            if job_stage == "completed":
                self.tracking_service.update_stage_details(experiment, "loss", status="failed")
                return None
            raise CloudProviderError(
                f"Experiment {experiment.experiment_id} already has a running loss stage at {artifact_root}. "
                "Loss artifacts are not complete yet, so refusing to submit a duplicate loss job."
            )
        return None

    def _recover_loss_from_evaluation(self, *, experiment: Experiment) -> StageResult | None:
        eval_details = experiment.stage_details.get("evaluation", {})
        eval_status = eval_details.get("status")
        eval_artifact_root = eval_details.get("artifact_root")
        bucket_id = eval_details.get("bucket_id") or eval_details.get("tags", {}).get("bucket_id")
        if eval_status != "completed" or not eval_artifact_root or not bucket_id:
            return None

        embedded_root = f"{eval_artifact_root.rstrip('/')}/analysis"
        results_prefix = embedded_root.replace(f"hf://buckets/{bucket_id}/", "", 1)
        results_dir = self._download_results(bucket_id=bucket_id, results_prefix=results_prefix)
        losses_path = results_dir / "per_example_losses.jsonl" if results_dir else None
        if losses_path and losses_path.exists():
            loss_results = load_losses(losses_path)
            if results_dir is not None:
                shutil.rmtree(results_dir, ignore_errors=True)
            self.tracking_service.update_stage_details(
                experiment,
                "loss",
                status="completed",
                artifact_root=embedded_root,
                source_commit=eval_details.get("source_commit"),
                tags={
                    "provider": experiment.provider,
                    "bucket_id": bucket_id,
                    "artifact_prefix": eval_details.get("artifact_prefix") or eval_details.get("tags", {}).get("artifact_prefix", ""),
                },
            )
            return StageResult(
                status="completed",
                run_record=RunRecord(
                    run_id=f"{experiment.experiment_id}-loss",
                    run_type="loss",
                    name=f"{experiment.name} per-example loss",
                    timestamp=eval_details.get("updated_at") or experiment.created_at,
                    status="completed",
                    output_dir=embedded_root,
                    model_name=experiment.base_model_name,
                    dataset_source=experiment.dataset_path,
                    provider=experiment.provider,
                    artifact_backend="hf_bucket",
                    artifact_root=embedded_root,
                    source_commit=eval_details.get("source_commit"),
                    stage="loss",
                    tags={
                        "provider": experiment.provider,
                        "bucket_id": bucket_id,
                        "artifact_prefix": eval_details.get("artifact_prefix") or eval_details.get("tags", {}).get("artifact_prefix", ""),
                    },
                ),
                loss_results=loss_results,
                artifact_root=embedded_root,
            )
        if results_dir is not None:
            shutil.rmtree(results_dir, ignore_errors=True)
        return None

    def run(self, spec: ExperimentSpec, experiment: Experiment, previous: StageResult | None = None) -> StageResult:
        recovered = self._recover_existing_loss(experiment=experiment)
        if recovered is not None:
            return recovered
        recovered_from_eval = self._recover_loss_from_evaluation(experiment=experiment)
        if recovered_from_eval is not None:
            return recovered_from_eval
        if previous is None or previous.run_record is None:
            raise CloudProviderError("Loss stage requires a completed training run.")
        training_run = previous.run_record
        bucket_id = training_run.tags.get("bucket_id", "")
        artifact_prefix = training_run.tags.get("artifact_prefix", "")
        if not bucket_id or not artifact_prefix:
            raise CloudProviderError("Training stage did not capture the HF artifact prefix and bucket.")

        huggingface_hub = load_huggingface_hub(require_apis=("run_job", "inspect_job", "fetch_job_logs"))
        results_prefix = f"{artifact_prefix.strip('/')}/analysis/loss"
        command = self._build_command(spec=spec, training_run=training_run, results_prefix=results_prefix)
        timeout_hours = spec.loss.timeout_hours or spec.training.timeout_hours or 4.0
        flavor = spec.loss.gpu or spec.training.gpu or "a10g-small"
        image = spec.training.cloud_image or training_run.tags.get("image") or ""
        if not image:
            backend = TrainingBackendRegistry.get("hf_jobs", repo_root=self.repo_root)
            image = backend.load_config(spec.method).cloud_image

        submission = HFJobExecutor(huggingface_hub).submit(
            CloudJobSpec(
                provider="hf_jobs",
                image=image,
                command=build_bash_command([command]),
                flavor=flavor,
                timeout_hours=timeout_hours,
                secrets=build_hf_job_secrets(),
                labels={
                    "task": "per_example_loss",
                    "provider": "hf_jobs",
                    "experiment_id": experiment.experiment_id,
                },
            )
        )
        artifact_root = f"hf://buckets/{bucket_id}/{results_prefix}"
        self.tracking_service.update_stage_details(
            experiment,
            "loss",
            status="running",
            artifact_root=artifact_root,
            job_ref=submission.job_id,
            source_commit=training_run.source_commit,
            tags={
                "provider": spec.provider,
                "bucket_id": bucket_id,
                "artifact_prefix": artifact_prefix,
            },
        )

        exit_code = self._poll_job(huggingface_hub, job_id=submission.job_id, timeout_hours=timeout_hours)
        results_dir = self._download_results(bucket_id=bucket_id, results_prefix=results_prefix) if exit_code == 0 else None
        losses_path = results_dir / "per_example_losses.jsonl" if results_dir else None
        loss_results = load_losses(losses_path) if losses_path and losses_path.exists() else []
        if results_dir is not None:
            shutil.rmtree(results_dir, ignore_errors=True)
        self.tracking_service.update_stage_details(
            experiment,
            "loss",
            status="completed" if exit_code == 0 else "failed",
            artifact_root=artifact_root,
            job_ref=submission.job_id,
            source_commit=training_run.source_commit,
            tags={
                "provider": spec.provider,
                "bucket_id": bucket_id,
                "artifact_prefix": artifact_prefix,
            },
        )
        record = RunRecord(
            run_id=f"{experiment.experiment_id}-loss",
            run_type="loss",
            name=f"{spec.name} per-example loss",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="completed" if exit_code == 0 else "failed",
            output_dir=artifact_root,
            model_name=spec.training.model_name,
            dataset_source=spec.dataset.identifier,
            provider=spec.provider,
            artifact_backend="hf_bucket",
            artifact_root=artifact_root,
            job_ref=submission.job_id,
            source_commit=training_run.source_commit,
            stage="loss",
            tags={
                "provider": spec.provider,
                "bucket_id": bucket_id,
                "artifact_prefix": artifact_prefix,
            },
        )
        return StageResult(
            status="completed" if exit_code == 0 else "failed",
            run_record=record,
            loss_results=loss_results,
            artifact_root=artifact_root,
        )
