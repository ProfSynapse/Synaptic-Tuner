from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from .experiment import Experiment, create_experiment, load_experiment, save_experiment
from .registry import RunRegistry
from .schema import RunRecord


class TrackingService:
    """Canonical write API for experiments and runs."""

    def __init__(self, base_dir: str | Path = ".tracking", registry: Optional[RunRegistry] = None):
        self.base_dir = Path(base_dir)
        self.registry = registry or RunRegistry(self.base_dir / "registry.jsonl")
        self._lock = RLock()

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
        with self._lock:
            save_experiment(experiment, self.base_dir)

    def find_recoverable_experiment(
        self,
        *,
        spec_path: str | None = None,
        provider: str | None = None,
        method: str | None = None,
    ) -> Optional[Experiment]:
        experiments_root = self.base_dir / "experiments"
        if not experiments_root.exists():
            return None

        candidates: list[Experiment] = []
        for exp_dir in experiments_root.iterdir():
            if not exp_dir.is_dir():
                continue
            try:
                experiment = load_experiment(exp_dir.name, self.base_dir)
            except Exception:
                continue
            if experiment.status in {"completed", "failed"}:
                continue
            if spec_path and experiment.spec_path != spec_path:
                continue
            if provider and experiment.provider != provider:
                continue
            if method and experiment.method != method:
                continue
            candidates.append(experiment)

        if not candidates:
            return None
        candidates.sort(key=lambda item: item.created_at, reverse=True)
        return candidates[0]

    def mark_stage(self, experiment: Experiment, stage: str, status: str) -> Experiment:
        with self._lock:
            experiment.stage_statuses[stage] = status
            details = experiment.stage_details.setdefault(stage, {})
            details["status"] = status
            details["updated_at"] = datetime.now(timezone.utc).isoformat()
            if status == "running":
                details.setdefault("started_at", details["updated_at"])
            if status in {"completed", "failed"}:
                details["finished_at"] = details["updated_at"]
            save_experiment(experiment, self.base_dir)
        return experiment

    def update_stage_details(self, experiment: Experiment, stage: str, **details: Any) -> Experiment:
        with self._lock:
            stage_details = experiment.stage_details.setdefault(stage, {})
            tags = details.pop("tags", None)
            if tags:
                merged_tags = dict(stage_details.get("tags", {}))
                merged_tags.update(tags)
                stage_details["tags"] = merged_tags
            for key, value in details.items():
                if value is None:
                    continue
                stage_details[key] = value
            status = stage_details.get("status")
            if status:
                experiment.stage_statuses[stage] = status
            if "artifact_root" in stage_details and stage_details["artifact_root"]:
                experiment.artifact_roots[stage] = stage_details["artifact_root"]
            stage_details["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_experiment(experiment, self.base_dir)
        return experiment

    def set_artifact_root(self, experiment: Experiment, key: str, value: str) -> Experiment:
        with self._lock:
            experiment.artifact_roots[key] = value
            details = experiment.stage_details.setdefault(key, {})
            details["artifact_root"] = value
            details["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_experiment(experiment, self.base_dir)
        return experiment

    def set_derived_output(self, experiment: Experiment, key: str, value: str) -> Experiment:
        with self._lock:
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
            save_experiment(experiment, self.base_dir)
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
        with self._lock:
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
            stage_name = record.stage or role
            if stage_name:
                stage_details = experiment.stage_details.setdefault(stage_name, {})
                stage_details["run_id"] = run_id
                if record.job_ref is not None:
                    stage_details["job_ref"] = record.job_ref
                artifact_root = record.artifact_root or record.output_dir
                if artifact_root:
                    stage_details["artifact_root"] = artifact_root
                    experiment.artifact_roots[stage_name] = artifact_root
                if record.source_commit is not None:
                    stage_details["source_commit"] = record.source_commit
                stage_details["status"] = record.status
                merged_tags = dict(stage_details.get("tags", {}))
                merged_tags.update(record.tags)
                stage_details["tags"] = merged_tags
                stage_details["updated_at"] = datetime.now(timezone.utc).isoformat()
                experiment.stage_statuses[stage_name] = record.status
            if parent_run_id and relationship:
                self.registry.link_runs(run_id, parent_run_id, relationship=relationship)
            save_experiment(experiment, self.base_dir)
        return run_id
