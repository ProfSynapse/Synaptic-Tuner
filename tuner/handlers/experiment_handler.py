"""Run a provider-agnostic experiment lifecycle from a single config.

Located at tuner/handlers/experiment_handler.py.
Contains ExperimentHandler which orchestrates the train -> eval -> loss
experiment pipeline. Stage runners are in tuner/handlers/stages/.
"""

from __future__ import annotations

from pathlib import Path

from shared.experiment_tracking import (
    ExperimentOrchestrator,
    ExperimentSpec,
    TrackingService,
    load_experiment_spec,
)
from tuner.cloud.hardware_planner import StagePlan, plan_experiment_hardware
from shared.utilities.env import load_env_file
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.handlers.stages import HFEvalStageRunner, HFLossStageRunner, HFTrainingStageRunner
from tuner.ui import confirm, print_config, print_error, print_header, print_info, print_success


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
