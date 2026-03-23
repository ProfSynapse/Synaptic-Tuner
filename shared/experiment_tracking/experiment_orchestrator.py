from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .analysis_bundle import write_analysis_bundle
from .experiment import Experiment
from .experiment_spec import EXPERIMENT_STAGES, ExperimentSpec
from .schema import LossResult, RunRecord
from .service import TrackingService


@dataclass
class StageResult:
    status: str
    run_record: Optional[RunRecord] = None
    eval_payload: Optional[dict] = None
    loss_results: list[LossResult] = field(default_factory=list)
    artifact_root: Optional[str] = None


class StageRunner(Protocol):
    def run(self, spec: ExperimentSpec, experiment: Experiment, previous: Optional[StageResult] = None) -> StageResult: ...


class ExperimentOrchestrator:
    def __init__(
        self,
        *,
        tracking_service: TrackingService,
        training_runner: StageRunner,
        eval_runner: Optional[StageRunner] = None,
        loss_runner: Optional[StageRunner] = None,
        base_dir: str = ".tracking",
    ) -> None:
        self.tracking_service = tracking_service
        self.training_runner = training_runner
        self.eval_runner = eval_runner
        self.loss_runner = loss_runner
        self.base_dir = base_dir

    @staticmethod
    def _should_run_stage(
        *,
        spec: ExperimentSpec,
        stage_name: str,
        enabled: bool,
        runner: Optional[StageRunner],
        restored: Optional[StageResult],
    ) -> bool:
        if not enabled or runner is None or not ExperimentOrchestrator._stage_selected(spec, stage_name):
            return False
        return restored is None

    def _finalize_stage_result(
        self,
        *,
        experiment: Experiment,
        stage_name: str,
        result: Optional[StageResult],
        role: str,
    ) -> Optional[StageResult]:
        if result is None:
            return None
        if result.run_record:
            relationship = "parent" if role == "evaluation" else "derived_from"
            self.tracking_service.attach_run(
                experiment,
                result.run_record,
                role=role,
                relationship=relationship,
                parent_run_id=experiment.training_run_id,
            )
        if result.artifact_root:
            self.tracking_service.set_artifact_root(experiment, stage_name, result.artifact_root)
        self.tracking_service.mark_stage(experiment, stage_name, result.status)
        return result

    def _run_parallel_post_training(
        self,
        *,
        spec: ExperimentSpec,
        experiment: Experiment,
        training: StageResult,
        eval_result: Optional[StageResult],
        loss_result: Optional[StageResult],
    ) -> tuple[Optional[StageResult], Optional[StageResult]]:
        should_run_eval = self._should_run_stage(
            spec=spec,
            stage_name="evaluation",
            enabled=spec.evaluation.enabled,
            runner=self.eval_runner,
            restored=eval_result,
        )
        should_run_loss = self._should_run_stage(
            spec=spec,
            stage_name="loss",
            enabled=spec.loss.enabled,
            runner=self.loss_runner,
            restored=loss_result,
        )
        if not should_run_eval and not should_run_loss:
            return eval_result, loss_result

        with ThreadPoolExecutor(max_workers=2) as executor:
            eval_future = None
            loss_future = None
            if should_run_eval and self.eval_runner is not None:
                self.tracking_service.mark_stage(experiment, "evaluation", "running")
                eval_future = executor.submit(self.eval_runner.run, spec, experiment, training)
            if should_run_loss and self.loss_runner is not None:
                self.tracking_service.mark_stage(experiment, "loss", "running")
                loss_future = executor.submit(self.loss_runner.run, spec, experiment, training)

            if eval_future is not None:
                eval_result = eval_future.result()
            if loss_future is not None:
                loss_result = loss_future.result()
        return eval_result, loss_result

    def _restore_stage_result(self, experiment: Experiment, stage: str) -> Optional[StageResult]:
        run_id = getattr(experiment, f"{stage}_run_id", None)
        if run_id:
            record = self.tracking_service.registry.get_run(run_id)
            if record is not None:
                return StageResult(
                    status=record.status,
                    run_record=record,
                    artifact_root=record.artifact_root or record.output_dir,
                )

        details = experiment.stage_details.get(stage, {})
        status = details.get("status")
        if status not in {"completed", "failed"}:
            return None

        artifact_root = details.get("artifact_root", "")
        record = RunRecord(
            run_id=details.get("run_id") or f"{experiment.experiment_id}-{stage}",
            run_type=details.get("run_type") or ("evaluation" if stage == "evaluation" else stage if stage != "training" else experiment.method),
            name=f"{experiment.name} {stage}",
            timestamp=details.get("updated_at") or experiment.created_at,
            status=status,
            output_dir=artifact_root,
            model_name=experiment.base_model_name,
            dataset_source=experiment.dataset_path,
            provider=experiment.provider,
            artifact_backend="hf_bucket" if artifact_root.startswith("hf://") else None,
            artifact_root=artifact_root or None,
            job_ref=details.get("job_ref"),
            source_commit=details.get("source_commit"),
            stage=stage,
            tags=dict(details.get("tags", {})),
        )
        return StageResult(status=status, run_record=record, artifact_root=artifact_root or None)

    @staticmethod
    def _should_rerun_failed_stage(stage: str, restored: Optional[StageResult]) -> bool:
        """Allow lightweight post-training stages to be retried on resume."""
        if restored is None or restored.status != "failed":
            return False
        return stage in {"evaluation", "loss"}

    @staticmethod
    def _selected_stage_names(spec: ExperimentSpec) -> list[str]:
        selected = spec.execution.selected_stages()
        if not selected:
            return list(EXPERIMENT_STAGES)
        return selected

    @staticmethod
    def _stage_selected(spec: ExperimentSpec, stage: str) -> bool:
        return stage in spec.execution.selected_stages()

    def _resolve_or_create_experiment(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: str | None = None,
        experiment: Optional[Experiment] = None,
    ) -> Experiment:
        if experiment is not None:
            return experiment
        return self.tracking_service.create_experiment(
            name=spec.name,
            dataset_path=spec.dataset.identifier,
            dataset_hash=spec.dataset.hash,
            base_model_name=spec.training.model_name,
            provider=spec.provider,
            method=spec.method,
            objective=spec.objective,
            spec_path=spec_path,
        )

    def run(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: str | None = None,
        experiment: Optional[Experiment] = None,
    ) -> Experiment:
        experiment = self._resolve_or_create_experiment(spec, spec_path=spec_path, experiment=experiment)
        selected_stages = self._selected_stage_names(spec)
        run_analysis = "analysis" in selected_stages or "recommendation" in selected_stages
        training = self._restore_stage_result(experiment, "training")
        if self._stage_selected(spec, "training") and training is None:
            self.tracking_service.mark_stage(experiment, "training", "running")
            training = self.training_runner.run(spec, experiment)
        if training is None:
            raise RuntimeError("Training stage is required for this experiment run and no completed training run exists.")
        if training.run_record:
            self.tracking_service.attach_run(experiment, training.run_record, role="training")
        if training.artifact_root:
            self.tracking_service.set_artifact_root(experiment, "training", training.artifact_root)
        self.tracking_service.mark_stage(experiment, "training", training.status)
        if training.status != "completed":
            experiment.status = "failed"
            self.tracking_service.save_experiment(experiment)
            return experiment

        eval_result: Optional[StageResult] = None
        if spec.evaluation.enabled and self.eval_runner is not None and self._stage_selected(spec, "evaluation"):
            eval_result = self._restore_stage_result(experiment, "evaluation")
            if self._should_rerun_failed_stage("evaluation", eval_result):
                eval_result = None

        loss_result: Optional[StageResult] = None
        if spec.loss.enabled and self.loss_runner is not None and self._stage_selected(spec, "loss"):
            loss_result = self._restore_stage_result(experiment, "loss")
            if self._should_rerun_failed_stage("loss", loss_result):
                loss_result = None

        if spec.post_training.mode == "parallel":
            eval_result, loss_result = self._run_parallel_post_training(
                spec=spec,
                experiment=experiment,
                training=training,
                eval_result=eval_result,
                loss_result=loss_result,
            )
        else:
            if self._should_run_stage(
                spec=spec,
                stage_name="evaluation",
                enabled=spec.evaluation.enabled,
                runner=self.eval_runner,
                restored=eval_result,
            ):
                self.tracking_service.mark_stage(experiment, "evaluation", "running")
                eval_result = self.eval_runner.run(spec, experiment, previous=training)
            if self._should_run_stage(
                spec=spec,
                stage_name="loss",
                enabled=spec.loss.enabled,
                runner=self.loss_runner,
                restored=loss_result,
            ):
                self.tracking_service.mark_stage(experiment, "loss", "running")
                loss_result = self.loss_runner.run(spec, experiment, previous=training)

        if spec.evaluation.enabled and self.eval_runner is not None and self._stage_selected(spec, "evaluation"):
            eval_result = self._finalize_stage_result(
                experiment=experiment,
                stage_name="evaluation",
                result=eval_result,
                role="evaluation",
            )

        if spec.loss.enabled and self.loss_runner is not None and self._stage_selected(spec, "loss"):
            loss_result = self._finalize_stage_result(
                experiment=experiment,
                stage_name="loss",
                result=loss_result,
                role="loss",
            )

        final_status = "completed"
        for stage_name in ("training", "evaluation", "loss"):
            stage_status = experiment.stage_statuses.get(stage_name)
            if stage_status and stage_status != "completed":
                final_status = "partial"
        experiment.status = final_status

        if run_analysis:
            runs = [self.tracking_service.registry.get_run(run_id) for run_id in experiment.run_ids]
            resolved_runs = [run for run in runs if run is not None]
            analysis_outputs = write_analysis_bundle(
                experiment=experiment,
                runs=resolved_runs,
                base_dir=self.base_dir,
                eval_payload=eval_result.eval_payload if eval_result else None,
                loss_results=loss_result.loss_results if loss_result else None,
            )
            for key, value in analysis_outputs.items():
                self.tracking_service.set_derived_output(experiment, key, value)

        self.tracking_service.save_experiment(experiment)
        return experiment
