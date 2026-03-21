from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from .experiment import Experiment, create_experiment, load_experiment, save_experiment
from .registry import RunRegistry
from .schema import RunRecord


class TrackingService:
    """Canonical write API for experiments and runs."""

    def __init__(self, base_dir: str | Path = ".tracking", registry: Optional[RunRegistry] = None):
        self.base_dir = Path(base_dir)
        self.registry = registry or RunRegistry(self.base_dir / "registry.jsonl")

    def create_experiment(
        self,
        *,
        name: str,
        dataset_path: str,
        dataset_hash: str,
        base_model_name: str,
        provider: str = "",
        method: str = "",
        objective: str = "",
        spec_path: str | None = None,
    ) -> Experiment:
        return create_experiment(
            name=name,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            base_model_name=base_model_name,
            provider=provider,
            method=method,
            objective=objective,
            spec_path=spec_path,
            base_dir=self.base_dir,
        )

    def load_experiment(self, experiment_id: str) -> Experiment:
        return load_experiment(experiment_id, self.base_dir)

    def save_experiment(self, experiment: Experiment) -> None:
        save_experiment(experiment, self.base_dir)

    def mark_stage(self, experiment: Experiment, stage: str, status: str) -> Experiment:
        experiment.stage_statuses[stage] = status
        self.save_experiment(experiment)
        return experiment

    def set_artifact_root(self, experiment: Experiment, key: str, value: str) -> Experiment:
        experiment.artifact_roots[key] = value
        self.save_experiment(experiment)
        return experiment

    def set_derived_output(self, experiment: Experiment, key: str, value: str) -> Experiment:
        experiment.derived_outputs[key] = value
        if key in {"features_csv", "feature_dataset_csv"}:
            experiment.features_csv_path = value
        if key in {"base_losses", "feature_dataset_jsonl"}:
            experiment.base_losses_path = value
        if key == "judge_scores":
            experiment.judge_scores_path = value
        if key in {"hypothesis_context", "hypothesis_context_json"}:
            experiment.hypothesis_context_path = value
        if key in {"next_run_candidates", "next_run_candidates_json"}:
            experiment.next_run_candidates_path = value
        self.save_experiment(experiment)
        return experiment

    def attach_run(
        self,
        experiment: Experiment,
        record: RunRecord,
        *,
        role: str | None = None,
        relationship: str | None = None,
        parent_run_id: str | None = None,
    ) -> str:
        record = replace(record, experiment_id=experiment.experiment_id)
        run_id = self.registry.register_run(record)
        if run_id not in experiment.run_ids:
            experiment.run_ids.append(run_id)
        if role == "training":
            experiment.training_run_id = run_id
        elif role == "evaluation":
            experiment.evaluation_run_id = run_id
        elif role == "loss":
            experiment.loss_run_id = run_id
        elif role == "selected":
            experiment.selected_run_id = run_id
        if parent_run_id and relationship:
            self.registry.link_runs(run_id, parent_run_id, relationship=relationship)
        self.save_experiment(experiment)
        return run_id
