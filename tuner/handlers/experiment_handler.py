"""Run a provider-agnostic experiment lifecycle from a single config."""

from __future__ import annotations

import shlex
import shutil
import tempfile
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from shared.experiment_tracking import (
    Experiment,
    ExperimentOrchestrator,
    ExperimentSpec,
    StageResult,
    TrackingService,
    load_experiment_spec,
    load_losses,
)
from shared.experiment_tracking.schema import RunRecord
from tuner.cloud.hardware_planner import StagePlan, plan_experiment_hardware
from shared.utilities.env import get_hf_token, load_env_file
from tuner.backends.registry import TrainingBackendRegistry
from tuner.cloud import (
    CloudJobSpec,
    HFJobExecutor,
    RepoCheckoutSpec,
    build_bash_command,
    build_hf_job_secrets,
    build_repo_checkout_steps,
    load_huggingface_hub,
    resolve_hf_bucket_id,
)
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.handlers.cloud_eval_handler import CloudEvalHandler
from tuner.ui import confirm, print_config, print_error, print_header, print_info, print_success


def _optional_backend_value(value) -> str | None:
    """Return a backend metadata value only when it is a real non-empty string."""
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


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
        config.model_name = spec.training.model_name
        config.dataset_name = spec.dataset.source
        config.dataset_file = spec.dataset.file
        if spec.training.batch_size is not None:
            config.batch_size = spec.training.batch_size
        if spec.training.gradient_accumulation is not None:
            config.gradient_accumulation_steps = spec.training.gradient_accumulation
        if spec.training.learning_rate is not None:
            config.learning_rate = spec.training.learning_rate
        if spec.training.num_epochs is not None:
            config.epochs = spec.training.num_epochs
        if spec.training.max_steps is not None:
            config.max_steps = spec.training.max_steps
        if spec.training.max_seq_length is not None:
            config.max_seq_length = spec.training.max_seq_length
        if spec.training.load_in_4bit is not None:
            config.load_in_4bit = spec.training.load_in_4bit
        if spec.training.lora_r is not None:
            config.lora_r = spec.training.lora_r
        if spec.training.lora_alpha is not None:
            config.lora_alpha = spec.training.lora_alpha
        if spec.training.lora_dropout is not None:
            config.lora_dropout = spec.training.lora_dropout
        if spec.training.use_dora:
            config.use_dora = True
        if spec.training.use_rslora:
            config.use_rslora = True
        if spec.training.init_lora_weights is not None:
            config.init_lora_weights = spec.training.init_lora_weights
        if spec.training.lora_target_modules:
            config.lora_target_modules = (
                list(spec.training.lora_target_modules)
                if isinstance(spec.training.lora_target_modules, list)
                else spec.training.lora_target_modules
            )
        if spec.training.evolutionary.enabled:
            config.evolutionary_enabled = True
        if spec.training.evolutionary.candidates is not None:
            config.evolutionary_candidates = spec.training.evolutionary.candidates
        if spec.training.evolutionary.eval_batch_size is not None:
            config.evolutionary_eval_batch_size = spec.training.evolutionary.eval_batch_size
        if spec.training.evolutionary.validation_config is not None:
            config.evolutionary_validation_config = spec.training.evolutionary.validation_config
        if spec.training.evolutionary.strategy.type:
            config.evolutionary_strategy = spec.training.evolutionary.strategy.type
        if "noise_scale" in spec.training.evolutionary.strategy.params:
            config.evolutionary_noise_scale = float(spec.training.evolutionary.strategy.params["noise_scale"])
        if "max_grad_norm" in spec.training.evolutionary.strategy.params:
            config.evolutionary_max_grad_norm = float(spec.training.evolutionary.strategy.params["max_grad_norm"])
        if "scale_factors" in spec.training.evolutionary.strategy.params:
            config.evolutionary_scale_factors = [
                float(value) for value in spec.training.evolutionary.strategy.params["scale_factors"]
            ]
        if spec.training.evolutionary.selection.method:
            config.evolutionary_selection_method = spec.training.evolutionary.selection.method
        if spec.training.evolutionary.selection.min_improvement is not None:
            config.evolutionary_min_improvement = spec.training.evolutionary.selection.min_improvement
        if spec.training.evolutionary.eval_frequency is not None:
            config.evolutionary_eval_frequency = spec.training.evolutionary.eval_frequency
        if spec.training.evolutionary.warmup_steps is not None:
            config.evolutionary_warmup_steps = spec.training.evolutionary.warmup_steps
        config.evolutionary_cache_baseline = spec.training.evolutionary.cache_baseline
        config.evolutionary_log_candidates = spec.training.evolutionary.logging.candidates
        config.evolutionary_log_selected = spec.training.evolutionary.logging.selected
        if spec.training.gpu:
            config.gpu_type = spec.training.gpu
            config.hf_flavor = spec.training.gpu
        if spec.training.timeout_hours is not None:
            config.timeout_hours = spec.training.timeout_hours
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
        exit_code = backend.execute(config, python_path="")

        artifact_prefix = _optional_backend_value(getattr(backend, "last_artifact_prefix", None)) or artifact_prefix
        bucket_id = _optional_backend_value(getattr(backend, "last_bucket_id", None))
        job_id = _optional_backend_value(getattr(backend, "last_job_id", None))
        artifact_root = (
            f"hf://buckets/{bucket_id}/{artifact_prefix.strip('/')}"
            if artifact_prefix and bucket_id
            else artifact_root
        )
        status = "completed" if exit_code == 0 else "failed"
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

    def run(self, spec: ExperimentSpec, experiment: Experiment, previous: Optional[StageResult] = None) -> StageResult:
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

    def _download_results(self, *, bucket_id: str, results_prefix: str) -> Optional[Path]:
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
                    import shutil

                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            local_root.rmdir()
            return None
        return local_root

    def _inspect_job_stage(self, job_ref: str) -> Optional[str]:
        try:
            huggingface_hub = load_huggingface_hub(require_apis=("inspect_job",))
            job_info = huggingface_hub.inspect_job(job_id=job_ref)
        except Exception:
            return None
        status_obj = getattr(job_info, "status", None)
        stage = status_obj.stage if status_obj and hasattr(status_obj, "stage") else status_obj
        return str(stage).lower() if stage else None

    def _recover_existing_loss(self, *, experiment: Experiment) -> Optional[StageResult]:
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

    def _recover_loss_from_evaluation(self, *, experiment: Experiment) -> Optional[StageResult]:
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

    def run(self, spec: ExperimentSpec, experiment: Experiment, previous: Optional[StageResult] = None) -> StageResult:
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


class ExperimentHandler(BaseHandler):
    """Run train -> eval -> loss under one experiment config."""

    @property
    def name(self) -> str:
        return "run-experiment"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _load_spec(self) -> tuple[ExperimentSpec, Path]:
        spec_path = getattr(self.args, "experiment_spec", None) or getattr(self.args, "experiment_config", None)
        if not spec_path:
            raise CloudProviderError("run-experiment requires --experiment-spec <path>.")
        path = Path(spec_path).expanduser().resolve()
        if not path.exists():
            raise CloudProviderError(f"Experiment spec not found: {path}")
        return load_experiment_spec(path), path

    def _print_plan(self, spec: ExperimentSpec) -> None:
        print_config(
            {
                "Name": spec.name,
                "Provider": spec.provider,
                "Method": spec.method,
                "Model": spec.training.model_name,
                "Dataset": spec.dataset.identifier,
                "Train GPU": spec.training.gpu or "auto",
                "Eval GPU": spec.evaluation.gpu or "auto",
                "Loss GPU": spec.loss.gpu or "auto",
                "Eval": "enabled" if spec.evaluation.enabled else "disabled",
                "Loss": "enabled" if spec.loss.enabled else "disabled",
                "Objective": spec.objective or "-",
                "Stages": ", ".join(spec.execution.selected_stages()),
            },
            "Experiment Plan",
        )

    def _apply_stage_overrides(self, spec: ExperimentSpec) -> ExperimentSpec:
        only_stage = getattr(self.args, "only_stage", None)
        from_stage = getattr(self.args, "from_stage", None)
        skip_stages = getattr(self.args, "skip_stage", None) or []
        if only_stage and from_stage:
            raise CloudProviderError("--only-stage and --from-stage are mutually exclusive.")
        if only_stage:
            spec.execution.only_stage = only_stage
            spec.execution.from_stage = None
        elif from_stage:
            spec.execution.from_stage = from_stage
            spec.execution.only_stage = None
        if skip_stages:
            merged = list(spec.execution.skip_stages)
            for stage in skip_stages:
                if stage not in merged:
                    merged.append(stage)
            spec.execution.skip_stages = merged
        return spec

    def _apply_auto_hardware(self, spec: ExperimentSpec) -> tuple[ExperimentSpec, dict[str, StagePlan]]:
        plans = plan_experiment_hardware(
            spec=spec,
            optimize_for=getattr(self.args, "optimize_for", "balanced"),
            max_hourly_price=getattr(self.args, "max_hourly_price", None),
        )

        training_plan = plans.get("training")
        if training_plan and training_plan.recommendation:
            if spec.training.gpu is None:
                spec.training.gpu = training_plan.recommendation.flavor
            if spec.training.batch_size is None and training_plan.recommendation.recommended_batch_size:
                spec.training.batch_size = training_plan.recommendation.recommended_batch_size
            if spec.training.gradient_accumulation is None and training_plan.recommendation.recommended_gradient_accumulation:
                spec.training.gradient_accumulation = training_plan.recommendation.recommended_gradient_accumulation
        if spec.training.gpu is None:
            raise CloudProviderError("Auto-hardware could not find a feasible training GPU for this experiment.")

        evaluation_plan = plans.get("evaluation")
        if spec.evaluation.enabled and spec.evaluation.gpu is None and evaluation_plan and evaluation_plan.recommendation:
            spec.evaluation.gpu = evaluation_plan.recommendation.flavor
        if spec.evaluation.enabled and spec.evaluation.gpu is None:
            raise CloudProviderError("Auto-hardware could not find a feasible evaluation GPU for this experiment.")

        loss_plan = plans.get("loss")
        if spec.loss.enabled and spec.loss.gpu is None and loss_plan and loss_plan.recommendation:
            spec.loss.gpu = loss_plan.recommendation.flavor
        if spec.loss.enabled and spec.loss.gpu is None:
            raise CloudProviderError("Auto-hardware could not find a feasible loss GPU for this experiment.")

        return spec, plans

    def handle(self) -> int:
        load_env_file()
        hardware_plans: dict[str, StagePlan] = {}
        try:
            spec, spec_path = self._load_spec()
            spec = self._apply_stage_overrides(spec)
            if getattr(self.args, "auto_hardware", False):
                spec, hardware_plans = self._apply_auto_hardware(spec)
            issues = spec.validate()
            if issues:
                raise CloudProviderError("Experiment stage selection invalid: " + "; ".join(issues))
            if spec.provider != "hf_jobs":
                raise CloudProviderError(
                    f"Provider '{spec.provider}' is not wired yet. Current implementation supports hf_jobs."
                )
        except Exception as exc:
            if self.json_mode:
                self.output_error(str(exc), code="EXPERIMENT_CONFIG_ERROR")
                return 1
            print_error(str(exc))
            return 1

        tracking_service = TrackingService(getattr(self.args, "base_dir", ".tracking"))
        experiment = tracking_service.find_recoverable_experiment(
            spec_path=str(spec_path),
            provider=spec.provider,
            method=spec.method,
        )

        if not self.json_mode:
            print_header("RUN EXPERIMENT", "Train, evaluate, score losses, and register one experiment")
            self._print_plan(spec)
            if hardware_plans:
                for stage_name, plan in hardware_plans.items():
                    recommendation = plan.recommendation
                    if recommendation is None:
                        continue
                    print_info(
                        f"Auto hardware ({stage_name}): {recommendation.flavor} "
                        f"at ${recommendation.price_hr:.2f}/hr"
                    )
            if experiment is not None:
                print_info(f"Resuming experiment: {experiment.experiment_id}")
            if not getattr(self.args, "auto_confirm", False) and not confirm("Start experiment with this configuration?"):
                print_info("Experiment cancelled.")
                return 0

        orchestrator = ExperimentOrchestrator(
            tracking_service=tracking_service,
            training_runner=HFTrainingStageRunner(repo_root=self.repo_root, tracking_service=tracking_service),
            eval_runner=HFEvalStageRunner(repo_root=self.repo_root, tracking_service=tracking_service),
            loss_runner=HFLossStageRunner(repo_root=self.repo_root, tracking_service=tracking_service),
            base_dir=getattr(self.args, "base_dir", ".tracking"),
        )

        try:
            experiment = orchestrator.run(spec, spec_path=str(spec_path), experiment=experiment)
        except Exception as exc:
            if self.json_mode:
                self.output_error(str(exc), code="EXPERIMENT_RUN_ERROR")
                return 1
            print_error(f"Experiment failed: {exc}")
            return 1

        payload = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "run_ids": experiment.run_ids,
            "artifact_roots": experiment.artifact_roots,
            "derived_outputs": experiment.derived_outputs,
            "stage_statuses": experiment.stage_statuses,
        }
        if self.json_mode:
            self.output(payload)
            return 0

        print_success(f"Experiment registered: {experiment.experiment_id}")
        print_config(
            {
                "Status": experiment.status,
                "Training Run": experiment.training_run_id or "-",
                "Eval Run": experiment.evaluation_run_id or "-",
                "Loss Run": experiment.loss_run_id or "-",
                "Summary": experiment.derived_outputs.get("experiment_summary_json", "-"),
                "Features CSV": experiment.derived_outputs.get("feature_dataset_csv", "-"),
                "Hypothesis Context": experiment.derived_outputs.get("hypothesis_context_json", "-"),
            },
            "Experiment Outputs",
        )
        return 0
