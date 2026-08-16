from __future__ import annotations

import json
import hashlib
import stat
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from .experiment import (
    Experiment,
    _atomic_write_bytes,
    _canonical_json_bytes,
    create_experiment,
    load_experiment,
    save_experiment,
)
from .registry import RunRegistry
from .schema import RunRecord
from tuner.project import PathRef, ProjectContext, ResolvedConfig, SourceLock


class ProvenanceIntegrityError(ValueError):
    """Persisted experiment provenance is missing, unsafe, or inconsistent."""


def _contained(path: Path, root: Path) -> bool:
    try:
        return path.resolve(strict=False).is_relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        return False


class TrackingService:
    """Canonical write API for experiments and runs."""

    def __init__(
        self,
        base_dir: str | Path | None = None,
        registry: Optional[RunRegistry] = None,
        *,
        project_context: ProjectContext | None = None,
    ):
        self.project_context = project_context
        self.base_dir = self._resolve_base_dir(base_dir)
        self.registry = registry or RunRegistry(self.base_dir / "registry.jsonl")
        self._lock = RLock()

    def _resolve_base_dir(self, base_dir: str | Path | None) -> Path:
        if self.project_context is None:
            return Path(base_dir or ".tracking")
        if base_dir is None or str(base_dir) == ".tracking":
            return self.project_context.tracking_root
        if self.project_context.mode == "standalone":
            return Path(base_dir)
        candidate = Path(base_dir).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_context.tracking_root / candidate
        candidate = candidate.resolve(strict=False)
        if self.project_context.mode == "host" and not _contained(
            candidate, self.project_context.tracking_root
        ):
            raise ValueError("Host tracking base directory must remain below tracking_root")
        return candidate

    def tracking_uri(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        uri_root = (
            self.project_context.tracking_root
            if self.project_context is not None and self.project_context.mode == "host"
            else self.base_dir
        )
        relative = resolved.relative_to(uri_root.resolve(strict=False))
        return f"tracking://{relative.as_posix()}"

    def resolve_uri(self, value: str) -> Path:
        if value.startswith("tracking://"):
            uri_root = (
                self.project_context.tracking_root
                if self.project_context is not None and self.project_context.mode == "host"
                else self.base_dir
            )
            candidate = (uri_root / value.removeprefix("tracking://")).resolve(strict=False)
            if not _contained(candidate, uri_root):
                raise ValueError("Tracking URI escapes tracking root")
            return candidate
        if self.project_context is not None and value.startswith("artifact://"):
            candidate = (
                self.project_context.artifact_root / value.removeprefix("artifact://")
            ).resolve(strict=False)
            if not _contained(candidate, self.project_context.artifact_root):
                raise ValueError("Artifact URI escapes artifact root")
            return candidate
        return Path(value)

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
        source_lock_uri: str | None = None,
        source_lock_sha256: str | None = None,
        resolved_config_uri: str | None = None,
        resolved_config_sha256: str | None = None,
    ) -> Experiment:
        experiment = create_experiment(
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
        experiment.source_lock_uri = source_lock_uri
        experiment.source_lock_sha256 = source_lock_sha256
        experiment.resolved_config_uri = resolved_config_uri
        experiment.resolved_config_sha256 = resolved_config_sha256
        if any(
            value is not None
            for value in (
                source_lock_uri,
                source_lock_sha256,
                resolved_config_uri,
                resolved_config_sha256,
            )
        ):
            save_experiment(experiment, self.base_dir)
        return experiment

    def persist_resolved_config(
        self, experiment: Experiment, resolved_config: ResolvedConfig
    ) -> None:
        path = self.base_dir / "experiments" / experiment.experiment_id / "resolved-config.json"
        uri = self.tracking_uri(path)
        serialized = _canonical_json_bytes(resolved_config.to_dict())
        digest = hashlib.sha256(serialized).hexdigest()
        if experiment.resolved_config_uri or experiment.resolved_config_sha256:
            self._verify_provenance_artifact(
                kind="resolved config",
                uri=experiment.resolved_config_uri,
                expected_sha256=experiment.resolved_config_sha256,
                proposed_bytes=serialized,
            )
            return
        _atomic_write_bytes(path, serialized)
        experiment.resolved_config_uri = uri
        experiment.resolved_config_sha256 = digest
        save_experiment(experiment, self.base_dir)

    def persist_source_lock(self, experiment: Experiment, source_lock: SourceLock) -> None:
        if source_lock.run_id != experiment.experiment_id:
            raise ValueError("Source lock run_id must match the experiment_id")
        serialized = _canonical_json_bytes(source_lock.to_dict())
        digest = hashlib.sha256(serialized).hexdigest()
        path = self.base_dir / "experiments" / experiment.experiment_id / "source-lock.json"
        uri = self.tracking_uri(path)
        if experiment.source_lock_uri or experiment.source_lock_sha256:
            self._verify_provenance_artifact(
                kind="source lock",
                uri=experiment.source_lock_uri,
                expected_sha256=experiment.source_lock_sha256,
                proposed_bytes=serialized,
            )
            return
        _atomic_write_bytes(path, serialized)
        experiment.source_lock_uri = uri
        experiment.source_lock_sha256 = digest
        save_experiment(experiment, self.base_dir)

    def prepare_experiment_provenance(
        self,
        experiment: Experiment,
        *,
        resolved_config: ResolvedConfig | None = None,
        source_lock: SourceLock | None = None,
    ) -> None:
        """Persist new provenance or verify every stored reference before work."""

        if resolved_config is not None:
            self.persist_resolved_config(experiment, resolved_config)
        if source_lock is not None:
            self.persist_source_lock(experiment, source_lock)
        self.verify_experiment_provenance(experiment)

    def verify_experiment_provenance(self, experiment: Experiment) -> None:
        if experiment.resolved_config_uri or experiment.resolved_config_sha256:
            self._verify_provenance_artifact(
                kind="resolved config",
                uri=experiment.resolved_config_uri,
                expected_sha256=experiment.resolved_config_sha256,
            )
        if experiment.source_lock_uri or experiment.source_lock_sha256:
            self._verify_provenance_artifact(
                kind="source lock",
                uri=experiment.source_lock_uri,
                expected_sha256=experiment.source_lock_sha256,
            )

    def _verify_provenance_artifact(
        self,
        *,
        kind: str,
        uri: str | None,
        expected_sha256: str | None,
        proposed_bytes: bytes | None = None,
    ) -> bytes:
        if not uri or not expected_sha256:
            raise ProvenanceIntegrityError(
                f"{kind.title()} reference requires both URI and SHA-256"
            )
        path = self._strict_tracking_file(uri, kind=kind)
        try:
            stored_bytes = path.read_bytes()
            payload = json.loads(stored_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProvenanceIntegrityError(f"Stored {kind} is unreadable or malformed") from exc
        if not isinstance(payload, dict):
            raise ProvenanceIntegrityError(f"Stored {kind} must be a JSON object")
        canonical = _canonical_json_bytes(payload)
        if stored_bytes != canonical:
            raise ProvenanceIntegrityError(f"Stored {kind} is not canonically serialized")
        if kind == "source lock":
            try:
                SourceLock.from_dict(payload)
            except Exception as exc:
                raise ProvenanceIntegrityError("Stored source lock is invalid") from exc
        elif payload.get("schema_version") != "synaptic-resolved-config/v1":
            raise ProvenanceIntegrityError("Stored resolved config has an invalid schema version")
        actual_sha256 = hashlib.sha256(canonical).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ProvenanceIntegrityError(f"Stored {kind} SHA-256 does not match its record")
        if proposed_bytes is not None and hashlib.sha256(proposed_bytes).hexdigest() != actual_sha256:
            raise ProvenanceIntegrityError(f"Proposed {kind} does not match stored provenance")
        return canonical

    def _strict_tracking_file(self, uri: str, *, kind: str) -> Path:
        if not uri.startswith("tracking://"):
            raise ProvenanceIntegrityError(f"Stored {kind} must use a tracking:// URI")
        root = (
            self.project_context.tracking_root
            if self.project_context is not None and self.project_context.mode == "host"
            else self.base_dir
        ).resolve(strict=False)
        relative = Path(uri.removeprefix("tracking://"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ProvenanceIntegrityError(f"Stored {kind} URI escapes tracking root")
        candidate = root / relative
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ProvenanceIntegrityError(f"Stored {kind} cannot use symlinks")
        try:
            resolved = candidate.resolve(strict=True)
            mode = resolved.stat().st_mode
        except OSError as exc:
            raise ProvenanceIntegrityError(f"Stored {kind} file is missing") from exc
        if not _contained(resolved, root) or not stat.S_ISREG(mode):
            raise ProvenanceIntegrityError(f"Stored {kind} must be a contained regular file")
        return resolved

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
            if spec_path and not self._spec_paths_match(experiment.spec_path, spec_path):
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

    def _spec_paths_match(self, stored: str | None, requested: str) -> bool:
        if stored == requested:
            return True
        if not stored or self.project_context is None:
            return False
        try:
            return self._resolve_spec_identity(stored) == self._resolve_spec_identity(requested)
        except (OSError, ValueError):
            return False

    def _resolve_spec_identity(self, value: str) -> Path:
        if "://" in value:
            return PathRef.parse(value).resolve(self.project_context).resolve(strict=False)
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_context.invocation_cwd / candidate
        return candidate.resolve(strict=False)

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
            provenance: dict[str, str | None] = {}
            for field_name in (
                "source_lock_uri",
                "source_lock_sha256",
                "resolved_config_uri",
                "resolved_config_sha256",
            ):
                experiment_value = getattr(experiment, field_name)
                record_value = getattr(record, field_name)
                if experiment_value and record_value and experiment_value != record_value:
                    raise ValueError(
                        f"Run {field_name} does not match the experiment provenance"
                    )
                provenance[field_name] = experiment_value or record_value
                if experiment_value is None and record_value is not None:
                    setattr(experiment, field_name, record_value)
            record = replace(
                record,
                experiment_id=experiment.experiment_id,
                **provenance,
            )
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
