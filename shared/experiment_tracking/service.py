from __future__ import annotations

import json
import hashlib
import stat
from datetime import datetime, timezone
from dataclasses import fields, replace
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from .experiment import (
    Experiment,
    HF_SOURCE_TRANSPORT_STATES,
    _atomic_write_bytes,
    _canonical_json_bytes,
    _save_experiment_unlocked_after_validation,
    create_experiment,
    load_experiment,
)
from .registry import RunRegistry, _PathLock
from .schema import RunRecord
from tuner.project import PathRef, ProjectContext, ResolvedConfig, SourceLock


class ProvenanceIntegrityError(ValueError):
    """Persisted experiment provenance is missing, unsafe, or inconsistent."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceIntegrityError(f"Stored JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


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
        for kind, uri, digest in (
            ("Source lock", source_lock_uri, source_lock_sha256),
            ("Resolved config", resolved_config_uri, resolved_config_sha256),
        ):
            if (uri is None) != (digest is None):
                raise ValueError(f"{kind} reference requires both URI and SHA-256")
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
        self._stamp_snapshot(experiment, experiment)
        if any(
            value is not None
            for value in (
                source_lock_uri,
                source_lock_sha256,
                resolved_config_uri,
                resolved_config_sha256,
            )
        ):
            experiment_path = self._experiment_path(experiment.experiment_id)
            with self._lock:
                with _PathLock(experiment_path):
                    durable = load_experiment(experiment.experiment_id, self.base_dir)
                    if any(self._protected_identity(durable)):
                        raise ProvenanceIntegrityError(
                            "Initial experiment provenance is no longer neutral"
                        )
                    candidate = replace(
                        durable,
                        source_lock_uri=source_lock_uri,
                        source_lock_sha256=source_lock_sha256,
                        resolved_config_uri=resolved_config_uri,
                        resolved_config_sha256=resolved_config_sha256,
                    )
                    _save_experiment_unlocked_after_validation(
                        candidate, self.base_dir
                    )
                    self._copy_experiment(candidate, experiment)
                    self._stamp_snapshot(experiment, candidate)
        return experiment

    def persist_resolved_config(
        self, experiment: Experiment, resolved_config: ResolvedConfig
    ) -> None:
        path = self.base_dir / "experiments" / experiment.experiment_id / "resolved-config.json"
        uri = self.tracking_uri(path)
        serialized = _canonical_json_bytes(resolved_config.to_dict())
        digest = hashlib.sha256(serialized).hexdigest()
        self._persist_provenance_pair(
            experiment,
            kind="resolved config",
            uri_field="resolved_config_uri",
            sha256_field="resolved_config_sha256",
            uri=uri,
            sha256=digest,
            path=path,
            serialized=serialized,
        )

    def persist_source_lock(self, experiment: Experiment, source_lock: SourceLock) -> None:
        if source_lock.run_id != experiment.experiment_id:
            raise ValueError("Source lock run_id must match the experiment_id")
        serialized = _canonical_json_bytes(source_lock.to_dict())
        digest = hashlib.sha256(serialized).hexdigest()
        path = self.base_dir / "experiments" / experiment.experiment_id / "source-lock.json"
        uri = self.tracking_uri(path)
        self._persist_provenance_pair(
            experiment,
            kind="source lock",
            uri_field="source_lock_uri",
            sha256_field="source_lock_sha256",
            uri=uri,
            sha256=digest,
            path=path,
            serialized=serialized,
        )

    def _persist_provenance_pair(
        self,
        experiment: Experiment,
        *,
        kind: str,
        uri_field: str,
        sha256_field: str,
        uri: str,
        sha256: str,
        path: Path,
        serialized: bytes,
    ) -> None:
        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock:
            with _PathLock(experiment_path):
                if not experiment_path.exists():
                    experiment.__post_init__()
                    if any(self._protected_identity(experiment)):
                        raise ProvenanceIntegrityError(
                            "New provenance persistence requires neutral protected provenance"
                        )
                    durable = replace(experiment)
                else:
                    try:
                        durable = load_experiment(
                            experiment.experiment_id, self.base_dir
                        )
                    except (OSError, ValueError, TypeError) as exc:
                        raise ProvenanceIntegrityError(
                            "Durable experiment record is unavailable or invalid"
                        ) from exc
                for field_name in self._protected_field_names():
                    if field_name in {uri_field, sha256_field}:
                        continue
                    if getattr(durable, field_name) != getattr(experiment, field_name):
                        raise ProvenanceIntegrityError(
                            f"Experiment {field_name} conflicts with durable protected provenance"
                        )
                durable_uri = getattr(durable, uri_field)
                durable_sha256 = getattr(durable, sha256_field)
                caller_uri = getattr(experiment, uri_field)
                caller_sha256 = getattr(experiment, sha256_field)
                if durable_uri is not None or durable_sha256 is not None:
                    if (caller_uri, caller_sha256) not in {
                        (None, None),
                        (durable_uri, durable_sha256),
                    }:
                        raise ProvenanceIntegrityError(
                            f"{kind.title()} reference cannot be replaced"
                        )
                    self._verify_provenance_artifact(
                        kind=kind,
                        uri=durable_uri,
                        expected_sha256=durable_sha256,
                        proposed_bytes=serialized,
                    )
                    self.verify_experiment_provenance(durable)
                    self._copy_experiment_fields(
                        durable, experiment, set(self._protected_field_names())
                    )
                    self._stamp_snapshot(experiment, durable)
                    return
                if caller_uri is not None or caller_sha256 is not None:
                    raise ProvenanceIntegrityError(
                        f"{kind.title()} must be established from neutral durable state"
                    )
                self.verify_experiment_provenance(durable)
                _atomic_write_bytes(path, serialized)
                candidate = replace(
                    durable,
                    **{uri_field: uri, sha256_field: sha256},
                )
                self.verify_experiment_provenance(candidate)
                _save_experiment_unlocked_after_validation(candidate, self.base_dir)
                self._copy_experiment_fields(
                    candidate, experiment, set(self._protected_field_names())
                )
                self._stamp_snapshot(experiment, candidate)

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
        experiment.__post_init__()
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
        self.verify_source_transport_provenance(experiment)

    def verify_source_transport_provenance(self, experiment: Experiment) -> None:
        """Verify immutable HF descriptor/evidence references when present."""

        experiment.__post_init__()
        descriptor: dict[str, Any] | None = None
        if experiment.source_transport_uri or experiment.source_transport_sha256:
            descriptor = self._verify_hf_tracking_artifact(
                kind="source transport",
                uri=experiment.source_transport_uri,
                expected_sha256=experiment.source_transport_sha256,
                schema_version="synaptic-hf-source-transport/v1",
                experiment_id=experiment.experiment_id,
            )
            source_lock = descriptor["source_lock"]
            if (
                source_lock["uri"] != experiment.source_lock_uri
                or source_lock["sha256"] != experiment.source_lock_sha256
            ):
                raise ProvenanceIntegrityError(
                    "Source transport does not bind the experiment source lock"
                )
        if experiment.provisioning_evidence_uri or experiment.provisioning_evidence_sha256:
            evidence = self._verify_hf_tracking_artifact(
                kind="provisioning evidence",
                uri=experiment.provisioning_evidence_uri,
                expected_sha256=experiment.provisioning_evidence_sha256,
                schema_version="synaptic-hf-provisioning-evidence/v1",
                experiment_id=experiment.experiment_id,
            )
            if descriptor is None:
                raise ProvenanceIntegrityError(
                    "Provisioning evidence requires a verified source transport"
                )
            if evidence["descriptor"] != {
                "uri": experiment.source_transport_uri,
                "sha256": experiment.source_transport_sha256,
            }:
                raise ProvenanceIntegrityError(
                    "Provisioning evidence does not reference the experiment source transport"
                )
            if experiment.source_transport_state in {"CONSUMABLE", "SUBMITTED"}:
                try:
                    from tuner.cloud.hf_provisioning import validate_hf_evidence_binding

                    validate_hf_evidence_binding(
                        descriptor,
                        evidence,
                        descriptor_uri=experiment.source_transport_uri or "",
                        descriptor_sha256=experiment.source_transport_sha256,
                    )
                except Exception as exc:
                    raise ProvenanceIntegrityError(
                        "Provisioning evidence does not bind the experiment source transport"
                    ) from exc

    def require_consumable_hf_transport(self, experiment: Experiment) -> None:
        """Fail closed unless an HF run has a verified consumable transport."""

        self.verify_experiment_provenance(experiment)
        if experiment.source_transport_state not in {"CONSUMABLE", "SUBMITTED"}:
            raise ProvenanceIntegrityError(
                "HF source transport is not verified as CONSUMABLE"
            )

    def record_source_transport_prepared(
        self,
        experiment: Experiment,
        *,
        uri: str,
        sha256: str,
    ) -> Experiment:
        """Record the immutable local descriptor and monotonic PREPARED state."""

        return self._transition_source_transport(
            experiment,
            state="PREPARED",
            source_transport_uri=uri,
            source_transport_sha256=sha256,
        )

    def record_provisioning_acknowledged(
        self,
        experiment: Experiment,
        *,
        uri: str,
        sha256: str,
    ) -> Experiment:
        """Attach external evidence without provisioning or submitting anything."""

        return self._transition_source_transport(
            experiment,
            state="ACKNOWLEDGED",
            provisioning_evidence_uri=uri,
            provisioning_evidence_sha256=sha256,
        )

    def mark_source_transport_consumable(self, experiment: Experiment) -> Experiment:
        return self._transition_source_transport(experiment, state="CONSUMABLE")

    def _transition_source_transport(
        self,
        experiment: Experiment,
        *,
        state: str,
        source_transport_uri: str | None = None,
        source_transport_sha256: str | None = None,
        provisioning_evidence_uri: str | None = None,
        provisioning_evidence_sha256: str | None = None,
    ) -> Experiment:
        if state == "SUBMITTED":
            raise ProvenanceIntegrityError(
                "SUBMITTED requires a future separately approved submission boundary"
            )
        if state not in HF_SOURCE_TRANSPORT_STATES:
            raise ValueError("Unknown source transport lifecycle state")
        experiment_path = (
            self.base_dir / "experiments" / experiment.experiment_id / "experiment.json"
        )
        with self._lock:
            with _PathLock(experiment_path):
                try:
                    durable = load_experiment(experiment.experiment_id, self.base_dir)
                except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
                    raise ProvenanceIntegrityError(
                        "Durable experiment record is unavailable or invalid"
                    ) from exc
                self._require_same_immutable_tracking_identity(
                    durable,
                    experiment,
                    source_transport_uri=source_transport_uri,
                    source_transport_sha256=source_transport_sha256,
                    provisioning_evidence_uri=provisioning_evidence_uri,
                    provisioning_evidence_sha256=provisioning_evidence_sha256,
                )
                current = durable.source_transport_state
                expected_previous = {
                    "PREPARED": None,
                    "ACKNOWLEDGED": "PREPARED",
                    "CONSUMABLE": "ACKNOWLEDGED",
                }[state]
                if current not in {state, expected_previous}:
                    raise ProvenanceIntegrityError(
                        f"Source transport cannot transition from "
                        f"{current or 'unprepared'} to {state}"
                    )
                candidate = replace(
                    durable,
                    source_transport_uri=(
                        source_transport_uri or durable.source_transport_uri
                    ),
                    source_transport_sha256=(
                        source_transport_sha256 or durable.source_transport_sha256
                    ),
                    provisioning_evidence_uri=(
                        provisioning_evidence_uri or durable.provisioning_evidence_uri
                    ),
                    provisioning_evidence_sha256=(
                        provisioning_evidence_sha256
                        or durable.provisioning_evidence_sha256
                    ),
                    source_transport_state=state,
                )
                if current == state:
                    if self._transport_identity(candidate) != self._transport_identity(durable):
                        raise ProvenanceIntegrityError(
                            "Source transport transition would change immutable evidence"
                        )
                    self.verify_experiment_provenance(durable)
                    self._copy_transport_projection(durable, experiment)
                    self._stamp_snapshot(experiment, durable)
                    return experiment
                self.verify_experiment_provenance(candidate)
                _save_experiment_unlocked_after_validation(candidate, self.base_dir)
                self._copy_transport_projection(candidate, experiment)
                self._stamp_snapshot(experiment, candidate)
        return experiment

    @staticmethod
    def _transport_identity(experiment: Experiment) -> tuple[str | None, ...]:
        return (
            experiment.source_transport_uri,
            experiment.source_transport_sha256,
            experiment.provisioning_evidence_uri,
            experiment.provisioning_evidence_sha256,
            experiment.source_transport_state,
        )

    @classmethod
    def _protected_identity(cls, experiment: Experiment) -> tuple[str | None, ...]:
        return tuple(
            getattr(experiment, field_name)
            for field_name in cls._protected_field_names()
        )

    @staticmethod
    def _copy_transport_projection(source: Experiment, target: Experiment) -> None:
        for field_name in (
            "source_transport_uri",
            "source_transport_sha256",
            "provisioning_evidence_uri",
            "provisioning_evidence_sha256",
            "source_transport_state",
        ):
            setattr(target, field_name, getattr(source, field_name))

    @staticmethod
    def _require_same_immutable_tracking_identity(
        durable: Experiment,
        caller: Experiment,
        *,
        source_transport_uri: str | None,
        source_transport_sha256: str | None,
        provisioning_evidence_uri: str | None,
        provisioning_evidence_sha256: str | None,
    ) -> None:
        if durable.experiment_id != caller.experiment_id:
            raise ProvenanceIntegrityError("Experiment identity changed during transition")
        for field_name in (
            "source_lock_uri",
            "source_lock_sha256",
            "resolved_config_uri",
            "resolved_config_sha256",
        ):
            if getattr(durable, field_name) != getattr(caller, field_name):
                raise ProvenanceIntegrityError(
                    f"Stale experiment {field_name} does not match durable provenance"
                )
        proposed = {
            "source_transport_uri": source_transport_uri or caller.source_transport_uri,
            "source_transport_sha256": (
                source_transport_sha256 or caller.source_transport_sha256
            ),
            "provisioning_evidence_uri": (
                provisioning_evidence_uri or caller.provisioning_evidence_uri
            ),
            "provisioning_evidence_sha256": (
                provisioning_evidence_sha256 or caller.provisioning_evidence_sha256
            ),
        }
        for field_name, proposed_value in proposed.items():
            durable_value = getattr(durable, field_name)
            if durable_value is not None and proposed_value != durable_value:
                raise ProvenanceIntegrityError(
                    f"Stale experiment {field_name} conflicts with durable provenance"
                )

    def _verify_hf_tracking_artifact(
        self,
        *,
        kind: str,
        uri: str | None,
        expected_sha256: str | None,
        schema_version: str,
        experiment_id: str,
    ) -> dict[str, Any]:
        canonical = self._verify_provenance_artifact(
            kind=kind,
            uri=uri,
            expected_sha256=expected_sha256,
        )
        try:
            payload = json.loads(canonical, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProvenanceIntegrityError(f"Stored {kind} is malformed") from exc
        try:
            from tuner.cloud.hf_provisioning import (
                validate_hf_provisioning_evidence,
                validate_hf_source_transport_descriptor,
            )

            if schema_version == "synaptic-hf-source-transport/v1":
                payload = validate_hf_source_transport_descriptor(payload)
            else:
                payload = validate_hf_provisioning_evidence(payload)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                f"Stored {kind} does not match its exact schema"
            ) from exc
        if payload["run_id"] != experiment_id:
            raise ProvenanceIntegrityError(f"Stored {kind} belongs to another run")
        return payload

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
            payload = json.loads(stored_bytes, object_pairs_hook=_reject_duplicate_json_keys)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProvenanceIntegrityError(f"Stored {kind} is unreadable or malformed") from exc
        if not isinstance(payload, dict):
            raise ProvenanceIntegrityError(f"Stored {kind} must be a JSON object")
        if kind in {"source transport", "provisioning evidence"}:
            try:
                from tuner.cloud.hf_provisioning import canonical_json_bytes

                canonical = canonical_json_bytes(payload)
            except Exception as exc:
                raise ProvenanceIntegrityError(f"Stored {kind} is not canonical JSON") from exc
        else:
            canonical = _canonical_json_bytes(payload)
        if stored_bytes != canonical:
            raise ProvenanceIntegrityError(f"Stored {kind} is not canonically serialized")
        if kind == "source lock":
            try:
                SourceLock.from_dict(payload)
            except Exception as exc:
                raise ProvenanceIntegrityError("Stored source lock is invalid") from exc
        elif (
            kind == "resolved config"
            and payload.get("schema_version") != "synaptic-resolved-config/v1"
        ):
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
            if _is_link_or_reparse(current):
                raise ProvenanceIntegrityError(
                    f"Stored {kind} cannot use symlinks or reparse points"
                )
        try:
            resolved = candidate.resolve(strict=True)
            mode = resolved.stat().st_mode
        except OSError as exc:
            raise ProvenanceIntegrityError(f"Stored {kind} file is missing") from exc
        if not _contained(resolved, root) or not stat.S_ISREG(mode):
            raise ProvenanceIntegrityError(f"Stored {kind} must be a contained regular file")
        return resolved

    def load_experiment(self, experiment_id: str) -> Experiment:
        experiment = load_experiment(experiment_id, self.base_dir)
        self._stamp_snapshot(experiment, experiment)
        return experiment

    def save_experiment(self, experiment: Experiment) -> None:
        """Persist caller-owned fields without permitting transport projection changes."""

        experiment.__post_init__()
        experiment_path = self._experiment_path(experiment.experiment_id)
        if not experiment_path.exists():
            if any(self._protected_identity(experiment)):
                raise ProvenanceIntegrityError(
                    "New experiment saves require neutral protected provenance"
                )
            with self._lock:
                with _PathLock(experiment_path):
                    record_exists = experiment_path.exists()
                    if record_exists:
                        durable = load_experiment(experiment.experiment_id, self.base_dir)
                        candidate = self._merge_public_experiment(durable, experiment)
                        should_save = candidate.to_dict() != durable.to_dict()
                    else:
                        candidate = replace(experiment)
                        should_save = True
                    if should_save:
                        _save_experiment_unlocked_after_validation(
                            candidate, self.base_dir
                        )
                    self._copy_experiment(candidate, experiment)
                    self._stamp_snapshot(experiment, candidate)
            return

        def merge_public(durable: Experiment) -> bool:
            candidate = self._merge_public_experiment(durable, experiment)
            changed = candidate.to_dict() != durable.to_dict()
            self._copy_experiment(candidate, durable)
            return changed

        self._mutate_existing_experiment(experiment, merge_public, full_sync=True)

    def _experiment_path(self, experiment_id: str) -> Path:
        return self.base_dir / "experiments" / experiment_id / "experiment.json"

    def _merge_public_experiment(
        self, durable: Experiment, caller: Experiment
    ) -> Experiment:
        self._require_same_protected_projection(durable, caller)
        baseline = getattr(caller, "_tracking_snapshot", None)
        if baseline is None:
            if caller.to_dict() != durable.to_dict():
                raise ProvenanceIntegrityError(
                    "Existing experiment save requires a durable mutation baseline"
                )
            baseline = durable.to_dict()
        updates: dict[str, Any] = {}
        for item in fields(Experiment):
            if item.name in self._protected_field_names():
                continue
            caller_value = getattr(caller, item.name)
            baseline_value = baseline.get(item.name)
            durable_value = getattr(durable, item.name)
            if caller_value == baseline_value:
                updates[item.name] = durable_value
                continue
            if durable_value != baseline_value and durable_value != caller_value:
                raise ProvenanceIntegrityError(
                    f"Concurrent experiment mutation conflicts on {item.name}"
                )
            updates[item.name] = caller_value
        candidate = replace(durable, **updates)
        candidate.__post_init__()
        return candidate

    def _mutate_existing_experiment(
        self,
        caller: Experiment,
        mutate: Any,
        *,
        full_sync: bool = False,
    ) -> Experiment:
        experiment_path = self._experiment_path(caller.experiment_id)
        if not experiment_path.exists() and any(self._protected_identity(caller)):
            raise ProvenanceIntegrityError(
                "New experiment mutations require neutral protected provenance"
            )
        with self._lock:
            with _PathLock(experiment_path):
                is_new = not experiment_path.exists()
                if is_new:
                    caller.__post_init__()
                    if any(self._protected_identity(caller)):
                        raise ProvenanceIntegrityError(
                            "New experiment mutations require neutral protected provenance"
                        )
                    durable = replace(caller)
                else:
                    try:
                        durable = load_experiment(caller.experiment_id, self.base_dir)
                    except (OSError, ValueError, TypeError) as exc:
                        raise ProvenanceIntegrityError(
                            "Durable experiment record is unavailable or invalid"
                        ) from exc
                    self._require_same_protected_projection(durable, caller)
                    self.verify_experiment_provenance(durable)
                before = durable.to_dict()
                changed = bool(mutate(durable))
                durable.__post_init__()
                if is_new or changed:
                    _save_experiment_unlocked_after_validation(
                        durable, self.base_dir
                    )
                if full_sync:
                    self._copy_experiment(durable, caller)
                else:
                    changed_fields = {
                        item.name
                        for item in fields(Experiment)
                        if before[item.name] != getattr(durable, item.name)
                    }
                    changed_fields.update(self._protected_field_names())
                    self._copy_experiment_fields(durable, caller, changed_fields)
                self._stamp_snapshot(caller, durable)
        return caller

    @staticmethod
    def _protected_field_names() -> tuple[str, ...]:
        return (
            "source_lock_uri",
            "source_lock_sha256",
            "resolved_config_uri",
            "resolved_config_sha256",
            "source_transport_uri",
            "source_transport_sha256",
            "provisioning_evidence_uri",
            "provisioning_evidence_sha256",
            "source_transport_state",
        )

    @classmethod
    def _require_same_protected_projection(
        cls, durable: Experiment, caller: Experiment
    ) -> None:
        if durable.experiment_id != caller.experiment_id:
            raise ProvenanceIntegrityError("Experiment identity changed during mutation")
        for field_name in cls._protected_field_names():
            if getattr(durable, field_name) != getattr(caller, field_name):
                raise ProvenanceIntegrityError(
                    f"Experiment {field_name} conflicts with durable protected provenance"
                )

    @staticmethod
    def _copy_experiment(source: Experiment, target: Experiment) -> None:
        for item in fields(Experiment):
            setattr(target, item.name, getattr(source, item.name))

    @staticmethod
    def _copy_experiment_fields(
        source: Experiment, target: Experiment, field_names: set[str]
    ) -> None:
        for field_name in field_names:
            setattr(target, field_name, getattr(source, field_name))

    @staticmethod
    def _stamp_snapshot(target: Experiment, source: Experiment) -> None:
        target._tracking_snapshot = source.to_dict()

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
        recovered = candidates[0]
        self._stamp_snapshot(recovered, recovered)
        return recovered

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
        def mutate(durable: Experiment) -> bool:
            details = durable.stage_details.setdefault(stage, {})
            if durable.stage_statuses.get(stage) == status and details.get("status") == status:
                return False
            durable.stage_statuses[stage] = status
            details["status"] = status
            details["updated_at"] = datetime.now(timezone.utc).isoformat()
            if status == "running":
                details.setdefault("started_at", details["updated_at"])
            if status in {"completed", "failed"}:
                details["finished_at"] = details["updated_at"]
            return True

        return self._mutate_existing_experiment(experiment, mutate)

    def update_stage_details(self, experiment: Experiment, stage: str, **details: Any) -> Experiment:
        requested = dict(details)

        def mutate(durable: Experiment) -> bool:
            stage_details = durable.stage_details.setdefault(stage, {})
            before = dict(stage_details)
            before_status = durable.stage_statuses.get(stage)
            before_artifact = durable.artifact_roots.get(stage)
            tags = requested.get("tags")
            if tags:
                merged_tags = dict(stage_details.get("tags", {}))
                merged_tags.update(tags)
                stage_details["tags"] = merged_tags
            for key, value in requested.items():
                if key == "tags" or value is None:
                    continue
                stage_details[key] = value
            status = stage_details.get("status")
            if status:
                durable.stage_statuses[stage] = status
            if "artifact_root" in stage_details and stage_details["artifact_root"]:
                durable.artifact_roots[stage] = stage_details["artifact_root"]
            changed = (
                stage_details != before
                or durable.stage_statuses.get(stage) != before_status
                or durable.artifact_roots.get(stage) != before_artifact
            )
            if not changed:
                return False
            stage_details["updated_at"] = datetime.now(timezone.utc).isoformat()
            return True

        return self._mutate_existing_experiment(experiment, mutate)

    def set_artifact_root(self, experiment: Experiment, key: str, value: str) -> Experiment:
        def mutate(durable: Experiment) -> bool:
            details = durable.stage_details.setdefault(key, {})
            if durable.artifact_roots.get(key) == value and details.get("artifact_root") == value:
                return False
            durable.artifact_roots[key] = value
            details["artifact_root"] = value
            details["updated_at"] = datetime.now(timezone.utc).isoformat()
            return True

        return self._mutate_existing_experiment(experiment, mutate)

    def set_derived_output(self, experiment: Experiment, key: str, value: str) -> Experiment:
        def mutate(durable: Experiment) -> bool:
            before = durable.to_dict()
            durable.derived_outputs[key] = value
            if key in {"features_csv", "feature_dataset_csv"}:
                durable.features_csv_path = value
            if key in {"base_losses", "feature_dataset_jsonl"}:
                durable.base_losses_path = value
            if key == "judge_scores":
                durable.judge_scores_path = value
            if key in {"hypothesis_context", "hypothesis_context_json"}:
                durable.hypothesis_context_path = value
            if key in {"next_run_candidates", "next_run_candidates_json"}:
                durable.next_run_candidates_path = value
            return durable.to_dict() != before

        return self._mutate_existing_experiment(experiment, mutate)

    def attach_run(
        self,
        experiment: Experiment,
        record: RunRecord,
        *,
        role: str | None = None,
        relationship: str | None = None,
        parent_run_id: str | None = None,
    ) -> str:
        run_id: str | None = None

        def mutate(durable: Experiment) -> bool:
            nonlocal run_id
            provenance: dict[str, str | None] = {}
            for field_name in self._protected_field_names():
                experiment_value = getattr(durable, field_name)
                record_value = getattr(record, field_name)
                if record_value is not None and record_value != experiment_value:
                    raise ProvenanceIntegrityError(
                        f"Run {field_name} cannot change durable protected provenance"
                    )
                provenance[field_name] = experiment_value
            candidate = replace(durable, **provenance)
            self.verify_experiment_provenance(candidate)
            bound_record = replace(
                record,
                experiment_id=durable.experiment_id,
                **provenance,
            )
            run_id = self.registry.register_run(bound_record)
            before = durable.to_dict()
            for field_name, value in provenance.items():
                setattr(durable, field_name, value)
            if run_id not in durable.run_ids:
                durable.run_ids.append(run_id)
            if role == "training":
                durable.training_run_id = run_id
            elif role == "evaluation":
                durable.evaluation_run_id = run_id
            elif role == "loss":
                durable.loss_run_id = run_id
            elif role == "selected":
                durable.selected_run_id = run_id
            stage_name = bound_record.stage or role
            if stage_name:
                stage_details = durable.stage_details.setdefault(stage_name, {})
                stage_details["run_id"] = run_id
                if bound_record.job_ref is not None:
                    stage_details["job_ref"] = bound_record.job_ref
                artifact_root = bound_record.artifact_root or bound_record.output_dir
                if artifact_root:
                    stage_details["artifact_root"] = artifact_root
                    durable.artifact_roots[stage_name] = artifact_root
                if bound_record.source_commit is not None:
                    stage_details["source_commit"] = bound_record.source_commit
                stage_details["status"] = bound_record.status
                merged_tags = dict(stage_details.get("tags", {}))
                merged_tags.update(bound_record.tags)
                stage_details["tags"] = merged_tags
                if durable.to_dict() != before:
                    stage_details["updated_at"] = datetime.now(timezone.utc).isoformat()
                durable.stage_statuses[stage_name] = bound_record.status
            if parent_run_id and relationship:
                self.registry.link_runs(run_id, parent_run_id, relationship=relationship)
            return durable.to_dict() != before

        self._mutate_existing_experiment(experiment, mutate)
        assert run_id is not None
        return run_id
