from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from .analysis_bundle import write_analysis_bundle
from .experiment import Experiment
from .experiment_spec import ExperimentSpec
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

    def run(self, spec: ExperimentSpec, *, spec_path: str | None = None) -> Experiment:
        experiment = self.tracking_service.create_experiment(
            name=spec.name,
            dataset_path=spec.dataset.identifier,
            dataset_hash=spec.dataset.hash,
            base_model_name=spec.training.model_name,
            provider=spec.provider,
            method=spec.method,
            objective=spec.objective,
            spec_path=spec_path,
        )
        self.tracking_service.mark_stage(experiment, "training", "running")
        training = self.training_runner.run(spec, experiment)
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
        if spec.evaluation.enabled and self.eval_runner is not None:
            self.tracking_service.mark_stage(experiment, "evaluation", "running")
            eval_result = self.eval_runner.run(spec, experiment, previous=training)
            if eval_result.run_record:
                self.tracking_service.attach_run(
                    experiment,
                    eval_result.run_record,
                    role="evaluation",
                    relationship="parent",
                    parent_run_id=experiment.training_run_id,
                )
            if eval_result.artifact_root:
                self.tracking_service.set_artifact_root(experiment, "evaluation", eval_result.artifact_root)
            self.tracking_service.mark_stage(experiment, "evaluation", eval_result.status)

        loss_result: Optional[StageResult] = None
        if spec.loss.enabled and self.loss_runner is not None:
            self.tracking_service.mark_stage(experiment, "loss", "running")
            loss_result = self.loss_runner.run(spec, experiment, previous=training)
            if loss_result.run_record:
                self.tracking_service.attach_run(
                    experiment,
                    loss_result.run_record,
                    role="loss",
                    relationship="derived_from",
                    parent_run_id=experiment.training_run_id,
                )
            if loss_result.artifact_root:
                self.tracking_service.set_artifact_root(experiment, "loss", loss_result.artifact_root)
            self.tracking_service.mark_stage(experiment, "loss", loss_result.status)

        final_status = "completed"
        for stage_name in ("evaluation", "loss"):
            stage_status = experiment.stage_statuses.get(stage_name)
            if stage_status and stage_status != "completed":
                final_status = "partial"
        experiment.status = final_status

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
