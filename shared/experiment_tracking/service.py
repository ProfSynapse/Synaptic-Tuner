from __future__ import annotations

import json
import hashlib
import re
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, fields, replace
from pathlib import Path
from threading import RLock, local
from typing import TYPE_CHECKING, Any, Mapping, Optional

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
from .root_identity import ensure_tracking_root_identity, require_tracking_root_identity
from .schema import RunRecord
from tuner.project import PathRef, ProjectContext, ResolvedConfig, SourceLock

if TYPE_CHECKING:
    from tuner.cloud.hf_run_approval import HFRunApproval, HFSubmissionClaim


class ProvenanceIntegrityError(ValueError):
    """Persisted experiment provenance is missing, unsafe, or inconsistent."""


HF_CANCELLATION_SCHEMA_VERSION = "synaptic-hf-cancellation-attempt/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_CANONICAL_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)


@dataclass(frozen=True)
class HFCancellationClaimResult:
    """Immutable outcome of the durable cancellation-attempt CAS."""

    event_id: str
    event_uri: str
    event_sha256: str
    provider_attempt_authorized: bool
    _document_json: str

    @property
    def document(self) -> dict[str, object]:
        """Return a detached copy of the immutable canonical claim document."""

        return json.loads(self._document_json)


@dataclass(frozen=True)
class HFProvisioningClaimResult:
    """Immutable outcome/current head of the durable provisioning claim."""

    event_id: str
    event_uri: str
    event_sha256: str
    state: str
    provider_attempt_authorized: bool
    _document_json: str

    @property
    def document(self) -> dict[str, object]:
        return json.loads(self._document_json)


@dataclass(frozen=True)
class HFTrainingTransitionResult:
    """Detached immutable outcome of one protected training CAS transition."""

    state: str
    uri: str
    sha256: str
    provider_attempt_authorized: bool
    _document_json: str

    @property
    def document(self) -> dict[str, object]:
        return json.loads(self._document_json)


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
        self._provisioning_execution = local()
        self._source_preparation_execution = local()

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

    def persist_source_lock(
        self, experiment: Experiment, source_lock: SourceLock
    ) -> SourceLock:
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
            adopt_identical_orphan=True,
        )
        return self.load_source_lock(experiment)

    def load_source_lock(self, experiment: Experiment) -> SourceLock:
        """Load and authenticate the exact immutable SourceLock projection."""

        canonical = self._verify_provenance_artifact(
            kind="source lock",
            uri=experiment.source_lock_uri,
            expected_sha256=experiment.source_lock_sha256,
        )
        try:
            payload = json.loads(canonical, object_pairs_hook=_reject_duplicate_json_keys)
            source_lock = SourceLock.from_dict(payload)
        except Exception as exc:
            raise ProvenanceIntegrityError("Stored source lock is invalid") from exc
        if source_lock.run_id != experiment.experiment_id:
            raise ProvenanceIntegrityError("Stored source lock belongs to another experiment")
        return source_lock

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
        adopt_identical_orphan: bool = False,
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
                if adopt_identical_orphan and path.exists():
                    self._require_identical_immutable_orphan(
                        path, serialized, kind=kind
                    )
                else:
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

    def _require_identical_immutable_orphan(
        self, path: Path, serialized: bytes, *, kind: str
    ) -> None:
        root = self.base_dir.resolve(strict=False)
        try:
            relative = path.absolute().relative_to(root)
        except (OSError, ValueError):
            raise ProvenanceIntegrityError(f"{kind.title()} artifact escapes tracking root")
        if not _contained(path, root):
            raise ProvenanceIntegrityError(f"{kind.title()} artifact escapes tracking root")
        current = root
        for part in relative.parts:
            current = current / part
            if current.exists() and _is_link_or_reparse(current):
                raise ProvenanceIntegrityError(
                    f"{kind.title()} artifact cannot use symlinks or reparse points"
                )
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise ProvenanceIntegrityError(
                    f"Existing {kind} orphan must be a bounded regular file"
                )
            existing = path.read_bytes()
        except ProvenanceIntegrityError:
            raise
        except OSError as exc:
            raise ProvenanceIntegrityError(
                f"Existing {kind} orphan is unreadable"
            ) from exc
        if existing != serialized:
            raise ProvenanceIntegrityError(
                f"Existing {kind} orphan is not byte-identical"
            )

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
        self.verify_hf_provisioning_provenance(experiment)
        self.verify_hf_submission_provenance(experiment)
        self.verify_hf_training_provenance(experiment)

    def verify_hf_provisioning_provenance(self, experiment: Experiment) -> None:
        """Verify the durable provisioning event head and all predecessor bindings."""

        experiment.__post_init__()
        if experiment.hf_provisioning_state is None:
            return
        event = self._verify_hf_provisioning_event_artifact(
            uri=experiment.hf_provisioning_event_uri,
            expected_sha256=experiment.hf_provisioning_event_sha256,
        )
        if event["state"] != experiment.hf_provisioning_state:
            raise ProvenanceIntegrityError(
                "HF provisioning event state does not match the experiment"
            )
        if event["experiment_id"] != experiment.experiment_id:
            raise ProvenanceIntegrityError(
                "HF provisioning event belongs to another experiment"
            )
        descriptor = self._verify_hf_tracking_artifact(
            kind="source transport",
            uri=experiment.source_transport_uri,
            expected_sha256=experiment.source_transport_sha256,
            schema_version="synaptic-hf-source-transport/v1",
            experiment_id=experiment.experiment_id,
        )
        capsule = descriptor["capsule"]
        expected = {
            "descriptor": {
                "uri": experiment.source_transport_uri,
                "sha256": experiment.source_transport_sha256,
            },
            "source_lock": {
                "uri": experiment.source_lock_uri,
                "sha256": experiment.source_lock_sha256,
            },
            "volume": {
                key: descriptor["volume"][key]
                for key in ("source", "path", "type", "read_only")
            },
            "bundle_sha256": descriptor["bundle"]["content_sha256"],
            "capsule_manifest_sha256": capsule["manifest"]["sha256"],
            "checkout_policy_sha256": descriptor["checkout_policy"]["sha256"],
        }
        for field_name, expected_value in expected.items():
            if event[field_name] != expected_value:
                raise ProvenanceIntegrityError(
                    f"HF provisioning event does not bind durable {field_name}"
                )
        expected_evidence = (
            {
                "uri": experiment.provisioning_evidence_uri,
                "sha256": experiment.provisioning_evidence_sha256,
            }
            if experiment.hf_provisioning_state == "SUCCEEDED"
            else None
        )
        if event["evidence"] != expected_evidence:
            raise ProvenanceIntegrityError(
                "HF provisioning event evidence does not match the experiment"
            )

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
            if (
                experiment.source_transport_state in {"CONSUMABLE", "SUBMITTED"}
                or experiment.hf_provisioning_state == "SUCCEEDED"
            ):
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
        """Authorize protected use only after durable successful provisioning."""

        self.verify_experiment_provenance(experiment)
        self._require_authorized_hf_provisioning(experiment)

    def _require_authorized_hf_provisioning(self, experiment: Experiment) -> None:
        """Require the exact authenticated provisioning chain used by protected calls."""

        if experiment.source_transport_state != "CONSUMABLE":
            raise ProvenanceIntegrityError(
                "HF source transport is not verified as CONSUMABLE"
            )
        if experiment.hf_provisioning_state != "SUCCEEDED":
            raise ProvenanceIntegrityError(
                "HF protected operation requires durable SUCCEEDED provisioning"
            )
        self.verify_source_transport_provenance(experiment)
        self.verify_hf_provisioning_provenance(experiment)

    def verify_hf_submission_provenance(self, experiment: Experiment) -> None:
        """Verify the separate approval and append-only submission-event head."""

        experiment.__post_init__()
        if experiment.hf_run_approval_uri is None:
            return
        self._require_authorized_hf_provisioning(experiment)
        approval = self._verify_hf_run_approval_artifact(
            uri=experiment.hf_run_approval_uri,
            expected_sha256=experiment.hf_run_approval_sha256,
            experiment_id=experiment.experiment_id,
        )
        if approval.authorization_id != experiment.hf_authorization_id:
            raise ProvenanceIntegrityError(
                "HF approval authorization ID does not match the experiment"
            )
        self._require_hf_approval_bindings(experiment, approval.document)
        if experiment.hf_submission_state == "APPROVED":
            return
        event = self._verify_hf_submission_event_artifact(
            uri=experiment.hf_submission_event_uri,
            expected_sha256=experiment.hf_submission_event_sha256,
            approval=approval,
        )
        if event.authorization_id != experiment.hf_authorization_id:
            raise ProvenanceIntegrityError(
                "HF submission event authorization ID does not match the experiment"
            )
        if event.state.value != experiment.hf_submission_state:
            raise ProvenanceIntegrityError(
                "HF submission event state does not match the experiment"
            )
        if event.document["approval"] != {
            "uri": experiment.hf_run_approval_uri,
            "sha256": experiment.hf_run_approval_sha256,
        }:
            raise ProvenanceIntegrityError(
                "HF submission event does not bind the immutable approval"
            )
        if experiment.hf_submission_state in {"SUBMITTED", "AMBIGUOUS"}:
            previous_reference = event.document["previous_event"]
            previous = self._verify_hf_submission_event_artifact(
                uri=previous_reference["uri"],
                expected_sha256=previous_reference["sha256"],
                approval=approval,
            )
            if previous.state.value != "SUBMITTING":
                raise ProvenanceIntegrityError(
                    "Terminal HF submission event must follow SUBMITTING"
                )
            try:
                from tuner.cloud.hf_run_approval import validate_hf_submission_claim

                validate_hf_submission_claim(
                    event,
                    approval=approval,
                    previous_event=previous,
                )
            except Exception as exc:
                raise ProvenanceIntegrityError(
                    "Terminal HF submission event does not bind its claimed predecessor"
                ) from exc
        if experiment.hf_cancellation_state is not None:
            if experiment.hf_submission_state != "SUBMITTED":
                raise ProvenanceIntegrityError(
                    "HF cancellation claim requires a durable SUBMITTED event"
                )
            self._verify_hf_cancellation_event_artifact(
                uri=experiment.hf_cancellation_event_uri,
                expected_sha256=experiment.hf_cancellation_event_sha256,
                experiment=experiment,
                submitted_event=event,
            )

    def verify_hf_training_provenance(self, experiment: Experiment) -> None:
        """Authenticate the separate training-smoke projection and event heads."""

        if experiment.hf_training_root_id is None:
            return
        try:
            require_tracking_root_identity(self.base_dir, experiment.hf_training_root_id)
        except Exception as exc:
            raise ProvenanceIntegrityError("HF training tracking-root identity is invalid") from exc
        preflight = None
        if experiment.hf_training_preflight_uri is not None:
            preflight = self._verify_hf_training_document(
                experiment.hf_training_preflight_uri,
                experiment.hf_training_preflight_sha256,
            )
            self._require_hf_training_identity(experiment, preflight)
            if preflight["schema_version"] != "synaptic-hf-training-preflight/v1":
                raise ProvenanceIntegrityError("HF training preflight has wrong document kind")
        approval = None
        if experiment.hf_training_approval_uri is not None:
            approval = self._verify_hf_training_document(
                experiment.hf_training_approval_uri,
                experiment.hf_training_approval_sha256,
                preflight=preflight,
            )
            self._require_hf_training_identity(experiment, approval)
            if approval["authorization_id"] != experiment.hf_training_authorization_id:
                raise ProvenanceIntegrityError("HF training approval authorization is inconsistent")
            if approval["preflight"] != {
                "uri": experiment.hf_training_preflight_uri,
                "sha256": experiment.hf_training_preflight_sha256,
            }:
                raise ProvenanceIntegrityError("HF training approval does not bind durable preflight")
        submission = None
        if experiment.hf_training_submission_event_uri is not None:
            submission = self._verify_hf_training_document(
                experiment.hf_training_submission_event_uri,
                experiment.hf_training_submission_event_sha256,
                approval=approval,
            )
            self._require_hf_training_identity(experiment, submission)
            if submission["state"] != experiment.hf_training_submission_state:
                raise ProvenanceIntegrityError("HF training submission state is inconsistent")
            if submission["state"] != "SUBMITTING":
                previous = self._verify_hf_training_document(
                    submission["previous_event"]["uri"],
                    submission["previous_event"]["sha256"],
                    approval=approval,
                )
                from tuner.cloud.hf_training_smoke_contract import validate_submission_event

                if submission["sequence"] == 3:
                    claim = self._verify_hf_training_document(
                        previous["previous_event"]["uri"],
                        previous["previous_event"]["sha256"],
                        approval=approval,
                    )
                    validate_submission_event(
                        previous, approval=approval, previous_event=claim
                    )
                validate_submission_event(submission, approval=approval, previous_event=previous)
        if experiment.hf_training_cancellation_event_uri is not None:
            cancellation = self._verify_hf_training_document(
                experiment.hf_training_cancellation_event_uri,
                experiment.hf_training_cancellation_event_sha256,
            )
            self._require_hf_training_identity(experiment, cancellation)
            if cancellation["state"] != experiment.hf_training_cancellation_state:
                raise ProvenanceIntegrityError("HF training cancellation state is inconsistent")
            if cancellation["sequence"] == 2:
                previous = self._verify_hf_training_document(
                    cancellation["previous_event"]["uri"],
                    cancellation["previous_event"]["sha256"],
                )
                from tuner.cloud.hf_training_smoke_contract import validate_cancellation_event

                validate_cancellation_event(cancellation, previous_event=previous)
        if experiment.hf_training_observation_event_uri is not None:
            observation = self._verify_hf_training_document(
                experiment.hf_training_observation_event_uri,
                experiment.hf_training_observation_event_sha256,
            )
            self._require_hf_training_identity(experiment, observation)
            if observation["state"] != experiment.hf_training_observation_state:
                raise ProvenanceIntegrityError("HF training observation state is inconsistent")
            if observation["previous_event"] is not None:
                previous = self._verify_hf_training_document(
                    observation["previous_event"]["uri"],
                    observation["previous_event"]["sha256"],
                )
                from tuner.cloud.hf_training_smoke_contract import validate_observation_event

                validate_observation_event(observation, previous_event=previous)
        if experiment.hf_training_result_uri is not None:
            result = self._verify_hf_training_document(
                experiment.hf_training_result_uri,
                experiment.hf_training_result_sha256,
            )
            self._require_hf_training_identity(experiment, result)
            if result["state"] != experiment.hf_training_result_state:
                raise ProvenanceIntegrityError("HF training result state is inconsistent")
            if result["previous_result"] is not None:
                previous = self._verify_hf_training_document(
                    result["previous_result"]["uri"],
                    result["previous_result"]["sha256"],
                )
                from tuner.cloud.hf_training_smoke_contract import validate_result

                validate_result(result, previous_result=previous)

    def snapshot_hf_training_runtime_lock(
        self, experiment: Experiment, runtime_lock: Mapping[str, object]
    ) -> dict[str, str]:
        """Persist an authenticated canonical runtime lock for a future preflight."""

        from tuner.cloud.hf_training_smoke_contract import (
            canonical_json_bytes,
            validate_runtime_lock,
        )

        try:
            document = validate_runtime_lock(runtime_lock)
            serialized = canonical_json_bytes(document)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "HF training runtime lock is not a canonical reviewed contract"
            ) from exc
        digest = hashlib.sha256(serialized).hexdigest()
        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_same_protected_projection(durable, experiment)
            self._require_authorized_hf_provisioning(durable)
            if durable.hf_training_preflight_uri is not None or durable.hf_training_root_id is not None:
                raise ProvenanceIntegrityError(
                    "HF training runtime lock must be snapshotted before preflight"
                )
            path = (
                self.base_dir
                / "experiments"
                / durable.experiment_id
                / "cloud"
                / "hf"
                / "training-smoke"
                / "runtime-locks"
                / f"{document['lock_id']}.json"
            )
            self._persist_immutable_hf_artifact(path, serialized)
            reference = {"uri": self.tracking_uri(path), "sha256": digest}
            self._verify_hf_training_runtime_lock_reference(durable, reference)
            return reference

    def record_hf_training_preflight(
        self, experiment: Experiment, preflight: Mapping[str, object]
    ) -> Experiment:
        from tuner.cloud.hf_training_smoke_contract import validate_preflight

        document = validate_preflight(preflight)
        root = ensure_tracking_root_identity(self.base_dir)
        self._require_hf_training_input_identity(experiment, document, str(root["root_id"]))
        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_authorized_hf_provisioning(durable)
            if durable.hf_training_root_id is not None:
                raise ProvenanceIntegrityError("HF training preflight cannot be replayed or replaced")
            self._require_hf_training_input_identity(durable, document, str(root["root_id"]))
            self._verify_hf_training_runtime_lock_reference(
                durable, document["runtime_lock"]
            )
            uri, digest = self._persist_hf_training_document(
                durable, "preflight", str(document["preflight_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_root_id=str(root["root_id"]),
                hf_training_run_id=str(document["run_id"]),
                hf_training_preflight_uri=uri,
                hf_training_preflight_sha256=digest,
                hf_training_preflight_state="PASS",
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
        return experiment

    def record_hf_training_approval(
        self, experiment: Experiment, approval: Mapping[str, object]
    ) -> Experiment:
        from tuner.cloud.hf_training_smoke_contract import validate_approval

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_projection(durable, experiment)
            self._require_authorized_hf_provisioning(durable)
            if durable.hf_training_submission_state is not None:
                raise ProvenanceIntegrityError("HF training approval cannot be replayed or replaced")
            preflight = self._verify_hf_training_document(
                durable.hf_training_preflight_uri, durable.hf_training_preflight_sha256
            )
            document = validate_approval(approval, preflight=preflight)
            self._require_hf_training_identity(durable, document)
            if document["preflight"] != {
                "uri": durable.hf_training_preflight_uri,
                "sha256": durable.hf_training_preflight_sha256,
            }:
                raise ProvenanceIntegrityError("HF training approval references another preflight")
            uri, digest = self._persist_hf_training_document(
                durable, "approvals", str(document["authorization_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_approval_uri=uri,
                hf_training_approval_sha256=digest,
                hf_training_authorization_id=str(document["authorization_id"]),
                hf_training_submission_state="APPROVED",
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
        return experiment

    def claim_hf_training_submission(
        self, experiment: Experiment, event: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        from tuner.cloud.hf_training_smoke_contract import validate_submission_event

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_approval(durable, experiment)
            if durable.hf_training_submission_state == "SUBMITTING":
                head = self._verify_hf_training_document(
                    durable.hf_training_submission_event_uri,
                    durable.hf_training_submission_event_sha256,
                )
                return self._hf_training_result(durable, head, False)
            if durable.hf_training_submission_state != "APPROVED":
                raise ProvenanceIntegrityError("HF training submission authorization is unavailable")
            approval = self._verify_hf_training_document(
                durable.hf_training_approval_uri, durable.hf_training_approval_sha256
            )
            document = validate_submission_event(event, approval=approval)
            self._require_hf_training_identity(durable, document)
            if document["approval"] != {
                "uri": durable.hf_training_approval_uri,
                "sha256": durable.hf_training_approval_sha256,
            }:
                raise ProvenanceIntegrityError("HF training submission references another approval")
            uri, digest = self._persist_hf_training_document(
                durable, "submission", str(document["event_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_submission_event_uri=uri,
                hf_training_submission_event_sha256=digest,
                hf_training_submission_state="SUBMITTING",
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
            return self._hf_training_result(candidate, document, True)

    def record_hf_training_submission_terminal(
        self, experiment: Experiment, event: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        from tuner.cloud.hf_training_smoke_contract import validate_submission_event

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_approval(durable, experiment)
            if durable.hf_training_submission_state != "SUBMITTING":
                raise ProvenanceIntegrityError("HF training submission has no active claim")
            approval = self._verify_hf_training_document(
                durable.hf_training_approval_uri, durable.hf_training_approval_sha256
            )
            previous = self._verify_hf_training_document(
                durable.hf_training_submission_event_uri,
                durable.hf_training_submission_event_sha256,
            )
            document = validate_submission_event(event, approval=approval, previous_event=previous)
            self._require_hf_training_identity(durable, document)
            if document["previous_event"] != {
                "uri": durable.hf_training_submission_event_uri,
                "sha256": durable.hf_training_submission_event_sha256,
            }:
                raise ProvenanceIntegrityError("HF training terminal does not bind durable claim")
            uri, digest = self._persist_hf_training_document(
                durable, "submission", str(document["event_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_submission_event_uri=uri,
                hf_training_submission_event_sha256=digest,
                hf_training_submission_state=str(document["state"]),
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
            return self._hf_training_result(candidate, document, False)

    def recover_hf_training_submission(
        self, experiment: Experiment, event: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        """Persist the sole allowed recovery: AMBIGUOUS -> confirmed SUBMITTED."""

        from tuner.cloud.hf_training_smoke_contract import validate_submission_event

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_projection(durable, experiment)
            if durable.hf_training_submission_state != "AMBIGUOUS":
                raise ProvenanceIntegrityError(
                    "HF training submission recovery requires AMBIGUOUS"
                )
            approval = self._verify_hf_training_document(
                durable.hf_training_approval_uri, durable.hf_training_approval_sha256
            )
            previous = self._verify_hf_training_document(
                durable.hf_training_submission_event_uri,
                durable.hf_training_submission_event_sha256,
                approval=approval,
            )
            claim = self._verify_hf_training_document(
                previous["previous_event"]["uri"],
                previous["previous_event"]["sha256"],
                approval=approval,
            )
            validate_submission_event(previous, approval=approval, previous_event=claim)
            document = validate_submission_event(
                event, approval=approval, previous_event=previous
            )
            self._require_hf_training_identity(durable, document)
            if document["previous_event"] != {
                "uri": durable.hf_training_submission_event_uri,
                "sha256": durable.hf_training_submission_event_sha256,
            }:
                raise ProvenanceIntegrityError(
                    "HF training recovery does not bind durable ambiguous head"
                )
            uri, digest = self._persist_hf_training_document(
                durable, "submission", str(document["event_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_submission_event_uri=uri,
                hf_training_submission_event_sha256=digest,
                hf_training_submission_state="SUBMITTED",
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
            return self._hf_training_result(candidate, document, False)

    def record_hf_training_submission_recovery(
        self, experiment: Experiment, event: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        """Compatibility spelling for the explicit ambiguous-submission recovery boundary."""

        return self.recover_hf_training_submission(experiment, event)

    def claim_hf_training_cancellation(
        self, experiment: Experiment, event: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        from tuner.cloud.hf_training_smoke_contract import validate_cancellation_event

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_approval(durable, experiment)
            if durable.hf_training_submission_state != "SUBMITTED":
                raise ProvenanceIntegrityError("HF training cancellation requires SUBMITTED")
            if durable.hf_training_cancellation_state is not None:
                head = self._verify_hf_training_document(
                    durable.hf_training_cancellation_event_uri,
                    durable.hf_training_cancellation_event_sha256,
                )
                return self._hf_training_result(durable, head, False)
            document = validate_cancellation_event(event)
            self._require_hf_training_identity(durable, document)
            self._require_hf_training_downstream_bindings(durable, document)
            uri, digest = self._persist_hf_training_document(
                durable, "cancellation", str(document["event_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_cancellation_event_uri=uri,
                hf_training_cancellation_event_sha256=digest,
                hf_training_cancellation_state=str(document["state"]),
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
            return self._hf_training_result(candidate, document, document["state"] == "CLAIMED")

    def record_hf_training_cancellation_terminal(
        self, experiment: Experiment, event: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        from tuner.cloud.hf_training_smoke_contract import validate_cancellation_event

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_approval(durable, experiment)
            if durable.hf_training_cancellation_state != "CLAIMED":
                raise ProvenanceIntegrityError("HF training cancellation has no active claim")
            previous = self._verify_hf_training_document(
                durable.hf_training_cancellation_event_uri,
                durable.hf_training_cancellation_event_sha256,
            )
            document = validate_cancellation_event(event, previous_event=previous)
            self._require_hf_training_identity(durable, document)
            self._require_hf_training_downstream_bindings(durable, document)
            uri, digest = self._persist_hf_training_document(
                durable, "cancellation", str(document["event_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_cancellation_event_uri=uri,
                hf_training_cancellation_event_sha256=digest,
                hf_training_cancellation_state=str(document["state"]),
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
            return self._hf_training_result(candidate, document, False)

    def record_hf_training_observation(
        self, experiment: Experiment, event: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        from tuner.cloud.hf_training_smoke_contract import validate_observation_event

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_approval(durable, experiment)
            if durable.hf_training_submission_state != "SUBMITTED":
                raise ProvenanceIntegrityError("HF training observation requires SUBMITTED")
            previous = None
            if durable.hf_training_observation_state is not None:
                if durable.hf_training_observation_state != "STOPPED":
                    raise ProvenanceIntegrityError("Terminal HF training observation cannot be replaced")
                previous = self._verify_hf_training_document(
                    durable.hf_training_observation_event_uri,
                    durable.hf_training_observation_event_sha256,
                )
            document = validate_observation_event(event, previous_event=previous)
            if previous is not None and document["previous_event"] != {
                "uri": durable.hf_training_observation_event_uri,
                "sha256": durable.hf_training_observation_event_sha256,
            }:
                raise ProvenanceIntegrityError("HF training observation predecessor reference changed")
            self._require_hf_training_identity(durable, document)
            self._require_hf_training_downstream_bindings(durable, document)
            uri, digest = self._persist_hf_training_document(
                durable, "observation", str(document["event_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_observation_event_uri=uri,
                hf_training_observation_event_sha256=digest,
                hf_training_observation_state=str(document["state"]),
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
            return self._hf_training_result(candidate, document, False)

    def claim_hf_training_verification(
        self, experiment: Experiment, result: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        from tuner.cloud.hf_training_smoke_contract import validate_result

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_approval(durable, experiment)
            if durable.hf_training_observation_state != "COMPLETED":
                raise ProvenanceIntegrityError("Artifact verification requires completed observation")
            if durable.hf_training_result_state == "VERIFYING":
                head = self._verify_hf_training_document(
                    durable.hf_training_result_uri, durable.hf_training_result_sha256
                )
                return self._hf_training_result(durable, head, False)
            if durable.hf_training_result_state not in {None, "INCONCLUSIVE"}:
                raise ProvenanceIntegrityError("HF training artifact result is terminal")
            previous = None
            if durable.hf_training_result_state == "INCONCLUSIVE":
                previous = self._verify_hf_training_document(
                    durable.hf_training_result_uri, durable.hf_training_result_sha256
                )
            document = validate_result(result, previous_result=previous)
            if document["state"] != "VERIFYING":
                raise ProvenanceIntegrityError("Artifact verification claim must be VERIFYING")
            if previous is not None and document["previous_result"] != {
                "uri": durable.hf_training_result_uri,
                "sha256": durable.hf_training_result_sha256,
            }:
                raise ProvenanceIntegrityError("HF training result predecessor reference changed")
            self._require_hf_training_identity(durable, document)
            self._require_hf_training_downstream_bindings(durable, document)
            uri, digest = self._persist_hf_training_document(
                durable, "results", str(document["result_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_result_uri=uri,
                hf_training_result_sha256=digest,
                hf_training_result_state="VERIFYING",
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
            return self._hf_training_result(candidate, document, False)

    def record_hf_training_result(
        self, experiment: Experiment, result: Mapping[str, object]
    ) -> HFTrainingTransitionResult:
        from tuner.cloud.hf_training_smoke_contract import validate_result

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock, _PathLock(experiment_path):
            durable = load_experiment(experiment.experiment_id, self.base_dir)
            self._require_matching_hf_training_approval(durable, experiment)
            if durable.hf_training_result_state != "VERIFYING":
                raise ProvenanceIntegrityError("HF training result requires VERIFYING claim")
            previous = self._verify_hf_training_document(
                durable.hf_training_result_uri, durable.hf_training_result_sha256
            )
            document = validate_result(result, previous_result=previous)
            if document["previous_result"] != {
                "uri": durable.hf_training_result_uri,
                "sha256": durable.hf_training_result_sha256,
            }:
                raise ProvenanceIntegrityError("HF training result predecessor reference changed")
            self._require_hf_training_identity(durable, document)
            self._require_hf_training_downstream_bindings(durable, document)
            uri, digest = self._persist_hf_training_document(
                durable, "results", str(document["result_id"]), document
            )
            candidate = replace(
                durable,
                hf_training_result_uri=uri,
                hf_training_result_sha256=digest,
                hf_training_result_state=str(document["state"]),
            )
            self.verify_experiment_provenance(candidate)
            _save_experiment_unlocked_after_validation(candidate, self.base_dir)
            self._copy_hf_training_projection(candidate, experiment)
            self._stamp_snapshot(experiment, candidate)
            return self._hf_training_result(candidate, document, False)

    def _persist_hf_training_document(
        self,
        experiment: Experiment,
        group: str,
        document_id: str,
        document: Mapping[str, object],
    ) -> tuple[str, str]:
        from tuner.cloud.hf_training_smoke_contract import canonical_json_bytes

        serialized = canonical_json_bytes(document)
        digest = hashlib.sha256(serialized).hexdigest()
        path = (
            self.base_dir
            / "experiments"
            / experiment.experiment_id
            / "cloud"
            / "hf"
            / "training-smoke"
            / group
            / f"{document_id}.json"
        )
        self._persist_immutable_hf_artifact(path, serialized)
        return self.tracking_uri(path), digest

    def _verify_hf_training_document(
        self,
        uri: str | None,
        expected_sha256: str | None,
        *,
        preflight: Mapping[str, object] | None = None,
        approval: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if uri is None or expected_sha256 is None:
            raise ProvenanceIntegrityError("HF training document reference is incomplete")
        path = self._strict_tracking_file(uri, kind="HF training document")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ProvenanceIntegrityError("HF training document is unreadable") from exc
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ProvenanceIntegrityError("HF training document digest is invalid")
        try:
            payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
            if not isinstance(payload, dict):
                raise TypeError("document is not an object")
            from tuner.cloud.hf_training_smoke_contract import (
                APPROVAL_SCHEMA,
                canonical_json_bytes,
                validate_approval,
                validate_training_document_shape,
            )

            if raw != canonical_json_bytes(payload):
                raise ValueError("document bytes are not canonical")
            document = validate_training_document_shape(payload)
            if document["schema_version"] == APPROVAL_SCHEMA:
                document = validate_approval(document, preflight=preflight)
        except Exception as exc:
            if isinstance(exc, ProvenanceIntegrityError):
                raise
            raise ProvenanceIntegrityError("Stored HF training document is invalid") from exc
        return document

    def _verify_hf_training_runtime_lock_reference(
        self, experiment: Experiment, reference: object
    ) -> dict[str, object]:
        from tuner.cloud.hf_training_smoke_contract import (
            RUNTIME_LOCK_SCHEMA,
            validate_runtime_lock,
        )

        if not isinstance(reference, Mapping) or set(reference) != {"uri", "sha256"}:
            raise ProvenanceIntegrityError("HF training runtime-lock reference is not closed")
        document = self._verify_hf_training_document(
            str(reference["uri"]), str(reference["sha256"])
        )
        if document["schema_version"] != RUNTIME_LOCK_SCHEMA:
            raise ProvenanceIntegrityError("HF training runtime-lock document has wrong schema")
        try:
            document = validate_runtime_lock(document)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "HF training runtime-lock document is semantically invalid"
            ) from exc
        expected_path = (
            self.base_dir
            / "experiments"
            / experiment.experiment_id
            / "cloud"
            / "hf"
            / "training-smoke"
            / "runtime-locks"
            / f"{document['lock_id']}.json"
        )
        if reference != {
            "uri": self.tracking_uri(expected_path),
            "sha256": str(reference["sha256"]),
        }:
            raise ProvenanceIntegrityError(
                "HF training runtime-lock URI is outside its immutable snapshot slot"
            )
        return document

    @staticmethod
    def _require_hf_training_input_identity(
        experiment: Experiment, document: Mapping[str, object], root_id: str
    ) -> None:
        if document.get("experiment_id") != experiment.experiment_id:
            raise ProvenanceIntegrityError("HF training document belongs to another experiment")
        if document.get("tracking_root_id") != root_id:
            raise ProvenanceIntegrityError("HF training document belongs to another tracking root")

    @staticmethod
    def _require_hf_training_identity(
        experiment: Experiment, document: Mapping[str, object]
    ) -> None:
        expected = {
            "experiment_id": experiment.experiment_id,
            "run_id": experiment.hf_training_run_id,
            "tracking_root_id": experiment.hf_training_root_id,
        }
        if any(document.get(key) != value for key, value in expected.items()):
            raise ProvenanceIntegrityError("HF training document identity is inconsistent")
        if experiment.hf_training_authorization_id is not None and "authorization_id" in document:
            if document["authorization_id"] != experiment.hf_training_authorization_id:
                raise ProvenanceIntegrityError("HF training authorization identity is inconsistent")

    @staticmethod
    def _hf_training_projection_fields() -> tuple[str, ...]:
        return (
            "hf_training_root_id",
            "hf_training_run_id",
            "hf_training_preflight_uri",
            "hf_training_preflight_sha256",
            "hf_training_preflight_state",
            "hf_training_approval_uri",
            "hf_training_approval_sha256",
            "hf_training_authorization_id",
            "hf_training_submission_event_uri",
            "hf_training_submission_event_sha256",
            "hf_training_submission_state",
            "hf_training_cancellation_event_uri",
            "hf_training_cancellation_event_sha256",
            "hf_training_cancellation_state",
            "hf_training_observation_event_uri",
            "hf_training_observation_event_sha256",
            "hf_training_observation_state",
            "hf_training_result_uri",
            "hf_training_result_sha256",
            "hf_training_result_state",
        )

    @classmethod
    def _copy_hf_training_projection(cls, source: Experiment, target: Experiment) -> None:
        for field_name in cls._hf_training_projection_fields():
            setattr(target, field_name, getattr(source, field_name))

    def _require_matching_hf_training_projection(
        self, durable: Experiment, caller: Experiment
    ) -> None:
        self.verify_hf_training_provenance(durable)
        for field_name in self._hf_training_projection_fields():
            if getattr(durable, field_name) != getattr(caller, field_name):
                raise ProvenanceIntegrityError(
                    f"Stale HF training projection conflicts on {field_name}"
                )

    def _require_matching_hf_training_approval(
        self, durable: Experiment, caller: Experiment
    ) -> None:
        self.verify_hf_training_provenance(durable)
        fields_to_match = (
            "hf_training_root_id",
            "hf_training_run_id",
            "hf_training_preflight_uri",
            "hf_training_preflight_sha256",
            "hf_training_approval_uri",
            "hf_training_approval_sha256",
            "hf_training_authorization_id",
        )
        for field_name in fields_to_match:
            if getattr(durable, field_name) != getattr(caller, field_name):
                raise ProvenanceIntegrityError(
                    f"Stale HF training approval conflicts on {field_name}"
                )

    def _require_hf_training_downstream_bindings(
        self, durable: Experiment, document: Mapping[str, object]
    ) -> None:
        expected_approval = {
            "uri": durable.hf_training_approval_uri,
            "sha256": durable.hf_training_approval_sha256,
        }
        if document.get("approval") != expected_approval:
            raise ProvenanceIntegrityError("HF training event references another approval")
        if "submission" in document:
            expected_submission = {
                "uri": durable.hf_training_submission_event_uri,
                "sha256": durable.hf_training_submission_event_sha256,
            }
            if document["submission"] != expected_submission:
                raise ProvenanceIntegrityError("HF training event references another submission")
            submitted = self._verify_hf_training_document(
                durable.hf_training_submission_event_uri,
                durable.hf_training_submission_event_sha256,
            )
            if document.get("provider_job") != submitted.get("provider_job"):
                raise ProvenanceIntegrityError("HF training event provider identity is inconsistent")
        if "observation" in document:
            expected_observation = {
                "uri": durable.hf_training_observation_event_uri,
                "sha256": durable.hf_training_observation_event_sha256,
            }
            if document["observation"] != expected_observation:
                raise ProvenanceIntegrityError("HF training result references another observation")
        if "artifact_prefix" in document:
            approval = self._verify_hf_training_document(
                durable.hf_training_approval_uri,
                durable.hf_training_approval_sha256,
            )
            bindings = approval["bindings"]
            artifact = document["artifact_prefix"]
            expected_artifact = {
                "bucket_id": bindings["artifact_bucket_id"],
                "base_prefix": bindings["artifact_base_prefix"],
                "slot_id": bindings["artifact_slot_id"],
                "prefix": bindings["artifact_prefix"],
            }
            if any(artifact.get(key) != value for key, value in expected_artifact.items()):
                raise ProvenanceIntegrityError(
                    "HF training result artifact prefix differs from approval"
                )

    @staticmethod
    def _hf_training_result(
        experiment: Experiment,
        document: Mapping[str, object],
        provider_attempt_authorized: bool,
    ) -> HFTrainingTransitionResult:
        uri_fields = {
            "synaptic-hf-training-submission-event/v1": (
                experiment.hf_training_submission_event_uri,
                experiment.hf_training_submission_event_sha256,
            ),
            "synaptic-hf-training-cancellation-event/v1": (
                experiment.hf_training_cancellation_event_uri,
                experiment.hf_training_cancellation_event_sha256,
            ),
            "synaptic-hf-training-observation-event/v1": (
                experiment.hf_training_observation_event_uri,
                experiment.hf_training_observation_event_sha256,
            ),
            "synaptic-hf-training-result/v1": (
                experiment.hf_training_result_uri,
                experiment.hf_training_result_sha256,
            ),
        }
        uri, digest = uri_fields[str(document["schema_version"])]
        assert uri is not None and digest is not None
        return HFTrainingTransitionResult(
            state=str(document["state"]),
            uri=uri,
            sha256=digest,
            provider_attempt_authorized=provider_attempt_authorized,
            _document_json=json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )

    def record_hf_run_approval(
        self,
        experiment: Experiment,
        approval: Mapping[str, object] | HFRunApproval,
    ) -> Experiment:
        """Persist one immutable exact-run approval and project APPROVED.

        Approval does not change the source-transport lifecycle. The transport
        must already be exactly CONSUMABLE and remains CONSUMABLE afterward.
        """

        try:
            from tuner.cloud.hf_run_approval import validate_hf_run_approval

            validated = validate_hf_run_approval(approval)
        except Exception as exc:
            raise ProvenanceIntegrityError("HF run approval is invalid") from exc
        document = validated.to_dict()
        if document["experiment_id"] != experiment.experiment_id:
            raise ProvenanceIntegrityError("HF run approval belongs to another experiment")
        try:
            from tuner.cloud.hf_run_approval import canonical_json_bytes

            serialized = canonical_json_bytes(document)
        except Exception as exc:
            raise ProvenanceIntegrityError("HF run approval is not canonical") from exc
        digest = hashlib.sha256(serialized).hexdigest()
        relative = (
            Path("experiments")
            / experiment.experiment_id
            / "cloud"
            / "hf"
            / "submission"
            / "approvals"
            / f"{validated.approval_id}.json"
        )
        path = self.base_dir / relative
        uri = self.tracking_uri(path)
        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock:
            with _PathLock(experiment_path):
                durable = self._load_durable_experiment_for_hf_transition(experiment)
                self._require_authorized_hf_provisioning(durable)
                if durable.hf_submission_state is not None:
                    raise ProvenanceIntegrityError(
                        "HF run approval cannot be replayed or replaced"
                    )
                self._require_hf_approval_bindings(durable, document)
                self._persist_immutable_hf_artifact(path, serialized)
                candidate = replace(
                    durable,
                    hf_run_approval_uri=uri,
                    hf_run_approval_sha256=digest,
                    hf_authorization_id=validated.authorization_id,
                    hf_submission_state="APPROVED",
                )
                self.verify_experiment_provenance(candidate)
                _save_experiment_unlocked_after_validation(candidate, self.base_dir)
                self._copy_hf_submission_projection(candidate, experiment)
                self._stamp_snapshot(experiment, candidate)
        return experiment

    def claim_hf_submission(
        self,
        experiment: Experiment,
        submitting_event: Mapping[str, object] | HFSubmissionClaim,
    ) -> Experiment:
        """Irreversibly consume one authorization by persisting SUBMITTING."""

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock:
            with _PathLock(experiment_path):
                durable = self._load_durable_experiment_for_hf_transition(experiment)
                self._require_authorized_hf_provisioning(durable)
                if durable.hf_submission_state != "APPROVED":
                    raise ProvenanceIntegrityError(
                        "HF authorization is already claimed or is not approved"
                    )
                approval = self._verify_hf_run_approval_artifact(
                    uri=durable.hf_run_approval_uri,
                    expected_sha256=durable.hf_run_approval_sha256,
                    experiment_id=durable.experiment_id,
                )
                try:
                    from tuner.cloud.hf_run_approval import validate_hf_submission_claim

                    event = validate_hf_submission_claim(
                        submitting_event,
                        approval=approval,
                    )
                except Exception as exc:
                    raise ProvenanceIntegrityError(
                        "HF SUBMITTING event is invalid"
                    ) from exc
                if event.state.value != "SUBMITTING":
                    raise ProvenanceIntegrityError(
                        "Claim requires an HF SUBMITTING event"
                    )
                self._require_hf_event_identity(durable, event.document)
                uri, digest = self._persist_hf_submission_event(
                    durable,
                    event_id=event.event_id,
                    document=event.to_dict(),
                )
                candidate = replace(
                    durable,
                    hf_submission_event_uri=uri,
                    hf_submission_event_sha256=digest,
                    hf_submission_state="SUBMITTING",
                )
                self.verify_experiment_provenance(candidate)
                _save_experiment_unlocked_after_validation(candidate, self.base_dir)
                self._copy_hf_submission_projection(candidate, experiment)
                self._stamp_snapshot(experiment, candidate)
        return experiment

    def record_hf_submission_terminal(
        self,
        experiment: Experiment,
        terminal_event: Mapping[str, object] | HFSubmissionClaim,
    ) -> Experiment:
        """Record the only terminal outcome: SUBMITTED or AMBIGUOUS."""

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock:
            with _PathLock(experiment_path):
                durable = self._load_durable_experiment_for_hf_transition(experiment)
                self._require_authorized_hf_provisioning(durable)
                if durable.hf_submission_state != "SUBMITTING":
                    raise ProvenanceIntegrityError(
                        "HF submission terminal event requires an active SUBMITTING claim"
                    )
                approval = self._verify_hf_run_approval_artifact(
                    uri=durable.hf_run_approval_uri,
                    expected_sha256=durable.hf_run_approval_sha256,
                    experiment_id=durable.experiment_id,
                )
                previous = self._verify_hf_submission_event_artifact(
                    uri=durable.hf_submission_event_uri,
                    expected_sha256=durable.hf_submission_event_sha256,
                    approval=approval,
                )
                try:
                    from tuner.cloud.hf_run_approval import validate_hf_submission_claim

                    event = validate_hf_submission_claim(
                        terminal_event,
                        approval=approval,
                        previous_event=previous,
                    )
                except Exception as exc:
                    raise ProvenanceIntegrityError(
                        "HF terminal submission event is invalid"
                    ) from exc
                if event.state.value not in {"SUBMITTED", "AMBIGUOUS"}:
                    raise ProvenanceIntegrityError(
                        "HF terminal event must be SUBMITTED or AMBIGUOUS"
                    )
                self._require_hf_event_identity(durable, event.document)
                if event.document["previous_event"] != {
                    "uri": durable.hf_submission_event_uri,
                    "sha256": durable.hf_submission_event_sha256,
                }:
                    raise ProvenanceIntegrityError(
                        "HF terminal event does not bind the durable SUBMITTING event"
                    )
                uri, digest = self._persist_hf_submission_event(
                    durable,
                    event_id=event.event_id,
                    document=event.to_dict(),
                )
                candidate = replace(
                    durable,
                    hf_submission_event_uri=uri,
                    hf_submission_event_sha256=digest,
                    hf_submission_state=event.state.value,
                )
                self.verify_experiment_provenance(candidate)
                _save_experiment_unlocked_after_validation(candidate, self.base_dir)
                self._copy_hf_submission_projection(candidate, experiment)
                self._stamp_snapshot(experiment, candidate)
        return experiment

    def build_hf_cancellation_attempt_event(
        self,
        experiment: Experiment,
        *,
        occurred_at: str,
    ) -> dict[str, object]:
        """Build a cancellation event from durable SUBMITTED provider identity."""

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock:
            with _PathLock(experiment_path):
                durable = self._load_durable_experiment_for_hf_transition(experiment)
                if durable.hf_submission_state != "SUBMITTED":
                    raise ProvenanceIntegrityError(
                        "HF cancellation requires submission state SUBMITTED"
                    )
                approval = self._verify_hf_run_approval_artifact(
                    uri=durable.hf_run_approval_uri,
                    expected_sha256=durable.hf_run_approval_sha256,
                    experiment_id=durable.experiment_id,
                )
                submitted = self._verify_hf_submission_event_artifact(
                    uri=durable.hf_submission_event_uri,
                    expected_sha256=durable.hf_submission_event_sha256,
                    approval=approval,
                )
                return self._build_hf_cancellation_document(
                    durable,
                    submitted_event=submitted,
                    occurred_at=occurred_at,
                )

    @contextmanager
    def hf_source_preparation_execution_lock(self, experiment_id: str):
        """Serialize provider-free source preparation across threads/processes."""

        held = getattr(self._source_preparation_execution, "experiment_ids", set())
        if experiment_id in held:
            raise ProvenanceIntegrityError(
                "HF source preparation execution lock is not reentrant"
            )
        lock_path = (
            self.base_dir
            / "experiments"
            / experiment_id
            / "cloud"
            / "hf"
            / "source-preparation"
            / "execution.lock"
        )
        with _PathLock(lock_path):
            current = set(
                getattr(self._source_preparation_execution, "experiment_ids", set())
            )
            current.add(experiment_id)
            self._source_preparation_execution.experiment_ids = current
            try:
                yield
            finally:
                current = set(
                    getattr(self._source_preparation_execution, "experiment_ids", set())
                )
                current.discard(experiment_id)
                self._source_preparation_execution.experiment_ids = current

    @contextmanager
    def hf_provisioning_execution_lock(self, experiment_id: str):
        """Hold the kernel-backed lock spanning claim, provider call, and terminal CAS."""

        held = getattr(self._provisioning_execution, "experiment_ids", set())
        if experiment_id in held:
            raise ProvenanceIntegrityError("HF provisioning execution lock is not reentrant")
        lock_path = (
            self.base_dir
            / "experiments"
            / experiment_id
            / "cloud"
            / "hf"
            / "provisioning"
            / "execution.lock"
        )
        with _PathLock(lock_path):
            current = set(getattr(self._provisioning_execution, "experiment_ids", set()))
            current.add(experiment_id)
            self._provisioning_execution.experiment_ids = current
            try:
                yield
            finally:
                current = set(getattr(self._provisioning_execution, "experiment_ids", set()))
                current.discard(experiment_id)
                self._provisioning_execution.experiment_ids = current

    def claim_hf_provisioning(
        self,
        experiment: Experiment,
        claim: Mapping[str, object],
    ) -> HFProvisioningClaimResult:
        """Durably consume at-most-once provider authority for source provisioning."""

        self._require_provisioning_execution_lock(experiment.experiment_id)
        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock:
            with _PathLock(experiment_path):
                durable = self._load_durable_experiment_for_provisioning(experiment)
                accepted = self._validate_hf_provisioning_claim_identity(durable, claim)
                serialized = self._canonical_hf_provisioning_bytes(accepted)
                digest = hashlib.sha256(serialized).hexdigest()
                path = self._hf_provisioning_event_path(
                    durable.experiment_id, str(accepted["event_id"])
                )
                uri = self.tracking_uri(path)

                if durable.hf_provisioning_state is not None:
                    current = self._verify_hf_provisioning_event_artifact(
                        uri=durable.hf_provisioning_event_uri,
                        expected_sha256=durable.hf_provisioning_event_sha256,
                    )
                    original = current
                    original_uri = durable.hf_provisioning_event_uri or ""
                    original_sha256 = durable.hf_provisioning_event_sha256 or ""
                    if current["state"] != "CLAIMED":
                        previous = current.get("previous_event")
                        if not isinstance(previous, dict):
                            raise ProvenanceIntegrityError(
                                "HF provisioning terminal event has no claim predecessor"
                            )
                        original = self._verify_hf_provisioning_event_artifact(
                            uri=str(previous.get("uri") or ""),
                            expected_sha256=str(previous.get("sha256") or ""),
                        )
                        original_uri = str(previous.get("uri") or "")
                        original_sha256 = str(previous.get("sha256") or "")
                    if original != accepted or (original_uri, original_sha256) != (
                        uri,
                        digest,
                    ):
                        raise ProvenanceIntegrityError(
                            "HF provisioning claim cannot be replayed or replaced"
                        )
                    self._copy_hf_provisioning_projection(durable, experiment)
                    self._copy_transport_projection(durable, experiment)
                    self._stamp_snapshot(experiment, durable)
                    return self._hf_provisioning_claim_result(
                        current,
                        uri=durable.hf_provisioning_event_uri or "",
                        sha256=durable.hf_provisioning_event_sha256 or "",
                        provider_attempt_authorized=False,
                    )

                if durable.source_transport_state != "PREPARED":
                    raise ProvenanceIntegrityError(
                        "HF provisioning claim requires PREPARED source transport"
                    )

                self._persist_immutable_hf_artifact(path, serialized)
                candidate = replace(
                    durable,
                    hf_provisioning_event_uri=uri,
                    hf_provisioning_event_sha256=digest,
                    hf_provisioning_state="CLAIMED",
                )
                self.verify_experiment_provenance(candidate)
                _save_experiment_unlocked_after_validation(candidate, self.base_dir)
                self._copy_hf_provisioning_projection(candidate, experiment)
                self._stamp_snapshot(experiment, candidate)
                return self._hf_provisioning_claim_result(
                    accepted,
                    uri=uri,
                    sha256=digest,
                    provider_attempt_authorized=True,
                )

    def find_hf_provisioning_terminal(
        self, experiment: Experiment
    ) -> HFProvisioningClaimResult | None:
        """Discover at most one exact orphan terminal for a durable CLAIMED head."""

        self._require_provisioning_execution_lock(experiment.experiment_id)
        durable = self._load_durable_experiment_for_provisioning(experiment)
        if durable.hf_provisioning_state != "CLAIMED":
            return None
        claim = self._verify_hf_provisioning_event_artifact(
            uri=durable.hf_provisioning_event_uri,
            expected_sha256=durable.hf_provisioning_event_sha256,
        )
        events_dir = self._hf_provisioning_event_path(
            durable.experiment_id, str(claim["event_id"])
        ).parent
        if not events_dir.exists():
            return None
        if _is_link_or_reparse(events_dir) or not events_dir.is_dir():
            raise ProvenanceIntegrityError(
                "HF provisioning event directory must be a regular directory"
            )
        candidates: list[HFProvisioningClaimResult] = []
        count = 0
        try:
            entries = events_dir.iterdir()
            for path in entries:
                count += 1
                if count > 8:
                    raise ProvenanceIntegrityError(
                        "HF provisioning terminal discovery exceeded its fixed bound"
                    )
                if (
                    _is_link_or_reparse(path)
                    or not path.is_file()
                    or not re.fullmatch(r"[0-9a-f]{64}\.json", path.name)
                    or path.stat().st_size > 64 * 1024
                ):
                    raise ProvenanceIntegrityError(
                        "HF provisioning event directory contains an unknown artifact"
                    )
                uri = self.tracking_uri(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                document = self._verify_hf_provisioning_event_artifact(
                    uri=uri,
                    expected_sha256=digest,
                )
                if path.stem != document["event_id"]:
                    raise ProvenanceIntegrityError(
                        "HF provisioning event filename does not match its event ID"
                    )
                if document["state"] == "CLAIMED":
                    if document != claim or (uri, digest) != (
                        durable.hf_provisioning_event_uri,
                        durable.hf_provisioning_event_sha256,
                    ):
                        raise ProvenanceIntegrityError(
                            "HF provisioning event directory contains a conflicting claim"
                        )
                    continue
                if document["previous_event"] != {
                    "uri": durable.hf_provisioning_event_uri,
                    "sha256": durable.hf_provisioning_event_sha256,
                }:
                    raise ProvenanceIntegrityError(
                        "HF provisioning orphan terminal does not bind the durable claim"
                    )
                self._validate_hf_provisioning_terminal(
                    durable, document, previous_event=claim
                )
                candidates.append(
                    self._hf_provisioning_claim_result(
                        document,
                        uri=uri,
                        sha256=digest,
                        provider_attempt_authorized=False,
                    )
                )
        except ProvenanceIntegrityError:
            raise
        except OSError as exc:
            raise ProvenanceIntegrityError(
                "HF provisioning terminal discovery could not authenticate its directory"
            ) from exc
        if len(candidates) > 1:
            raise ProvenanceIntegrityError(
                "HF provisioning terminal discovery found multiple terminal events"
            )
        return candidates[0] if candidates else None

    def record_hf_provisioning_succeeded(
        self,
        experiment: Experiment,
        terminal_event: Mapping[str, object],
        *,
        evidence_uri: str,
        evidence_sha256: str,
    ) -> Experiment:
        """Atomically bind verified evidence, SUCCEEDED, and PREPARED→ACKNOWLEDGED."""

        return self._record_hf_provisioning_terminal(
            experiment,
            terminal_event,
            state="SUCCEEDED",
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha256,
        )

    def record_hf_provisioning_ambiguous(
        self,
        experiment: Experiment,
        terminal_event: Mapping[str, object],
    ) -> Experiment:
        """Record terminal provider ambiguity without evidence or retry authority."""

        return self._record_hf_provisioning_terminal(
            experiment,
            terminal_event,
            state="AMBIGUOUS",
            evidence_uri=None,
            evidence_sha256=None,
        )

    def _record_hf_provisioning_terminal(
        self,
        experiment: Experiment,
        terminal_event: Mapping[str, object],
        *,
        state: str,
        evidence_uri: str | None,
        evidence_sha256: str | None,
    ) -> Experiment:
        self._require_provisioning_execution_lock(experiment.experiment_id)
        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock:
            with _PathLock(experiment_path):
                durable = self._load_durable_experiment_for_provisioning(experiment)
                current = self._verify_hf_provisioning_event_artifact(
                    uri=durable.hf_provisioning_event_uri,
                    expected_sha256=durable.hf_provisioning_event_sha256,
                )
                if durable.hf_provisioning_state == state:
                    accepted = self._validate_hf_provisioning_terminal(
                        durable, terminal_event, previous_event=None
                    )
                    if current != accepted:
                        raise ProvenanceIntegrityError(
                            "HF provisioning terminal event cannot be replaced"
                        )
                    self._copy_transport_projection(durable, experiment)
                    self._copy_hf_provisioning_projection(durable, experiment)
                    self._stamp_snapshot(experiment, durable)
                    return experiment
                if durable.hf_provisioning_state != "CLAIMED":
                    raise ProvenanceIntegrityError(
                        "HF provisioning terminal event requires durable CLAIMED state"
                    )
                accepted = self._validate_hf_provisioning_terminal(
                    durable, terminal_event, previous_event=current
                )
                if accepted["state"] != state:
                    raise ProvenanceIntegrityError(
                        f"HF provisioning terminal event must be {state}"
                    )
                expected_previous = {
                    "uri": durable.hf_provisioning_event_uri,
                    "sha256": durable.hf_provisioning_event_sha256,
                }
                if accepted["previous_event"] != expected_previous:
                    raise ProvenanceIntegrityError(
                        "HF provisioning terminal event does not bind durable CLAIMED event"
                    )
                expected_evidence = (
                    {"uri": evidence_uri, "sha256": evidence_sha256}
                    if state == "SUCCEEDED"
                    else None
                )
                if accepted["evidence"] != expected_evidence:
                    raise ProvenanceIntegrityError(
                        "HF provisioning terminal event evidence binding is invalid"
                    )
                if state == "SUCCEEDED":
                    descriptor = self._verify_hf_tracking_artifact(
                        kind="source transport",
                        uri=durable.source_transport_uri,
                        expected_sha256=durable.source_transport_sha256,
                        schema_version="synaptic-hf-source-transport/v1",
                        experiment_id=durable.experiment_id,
                    )
                    evidence = self._verify_hf_tracking_artifact(
                        kind="provisioning evidence",
                        uri=evidence_uri,
                        expected_sha256=evidence_sha256,
                        schema_version="synaptic-hf-provisioning-evidence/v1",
                        experiment_id=durable.experiment_id,
                    )
                    try:
                        from tuner.cloud.hf_provisioning import validate_hf_evidence_binding

                        validate_hf_evidence_binding(
                            descriptor,
                            evidence,
                            descriptor_uri=durable.source_transport_uri or "",
                            descriptor_sha256=durable.source_transport_sha256 or "",
                        )
                    except Exception as exc:
                        raise ProvenanceIntegrityError(
                            "Provisioning evidence does not bind the experiment source transport"
                        ) from exc
                serialized = self._canonical_hf_provisioning_bytes(accepted)
                digest = hashlib.sha256(serialized).hexdigest()
                path = self._hf_provisioning_event_path(
                    durable.experiment_id, str(accepted["event_id"])
                )
                uri = self.tracking_uri(path)
                self._persist_immutable_hf_artifact(path, serialized)
                candidate = replace(
                    durable,
                    hf_provisioning_event_uri=uri,
                    hf_provisioning_event_sha256=digest,
                    hf_provisioning_state=state,
                    provisioning_evidence_uri=evidence_uri,
                    provisioning_evidence_sha256=evidence_sha256,
                    source_transport_state=(
                        "ACKNOWLEDGED" if state == "SUCCEEDED" else "PREPARED"
                    ),
                )
                self.verify_experiment_provenance(candidate)
                _save_experiment_unlocked_after_validation(candidate, self.base_dir)
                self._copy_transport_projection(candidate, experiment)
                self._copy_hf_provisioning_projection(candidate, experiment)
                self._stamp_snapshot(experiment, candidate)
        return experiment

    def claim_hf_cancellation(
        self,
        experiment: Experiment,
        event: Mapping[str, object],
    ) -> HFCancellationClaimResult:
        """Claim at-most-once authority to attempt provider cancellation.

        The first durable creator receives ``provider_attempt_authorized=True``.
        Identity-equal resumes receive the same immutable claim with ``False``.
        """

        experiment_path = self._experiment_path(experiment.experiment_id)
        with self._lock:
            with _PathLock(experiment_path):
                durable = self._load_durable_experiment_for_cancellation(experiment)
                if durable.hf_submission_state != "SUBMITTED":
                    raise ProvenanceIntegrityError(
                        "HF cancellation requires submission state SUBMITTED"
                    )
                approval = self._verify_hf_run_approval_artifact(
                    uri=durable.hf_run_approval_uri,
                    expected_sha256=durable.hf_run_approval_sha256,
                    experiment_id=durable.experiment_id,
                )
                submitted = self._verify_hf_submission_event_artifact(
                    uri=durable.hf_submission_event_uri,
                    expected_sha256=durable.hf_submission_event_sha256,
                    approval=approval,
                )
                validated = self._validate_hf_cancellation_document(
                    event,
                    experiment=durable,
                    submitted_event=submitted,
                )
                serialized = self._canonical_hf_authorization_bytes(validated)
                digest = hashlib.sha256(serialized).hexdigest()
                event_id = str(validated["event_id"])
                path = (
                    self.base_dir
                    / "experiments"
                    / durable.experiment_id
                    / "cloud"
                    / "hf"
                    / "submission"
                    / "events"
                    / f"{event_id}.json"
                )
                uri = self.tracking_uri(path)
                if durable.hf_cancellation_state == "CLAIMED":
                    stored = self._verify_hf_cancellation_event_artifact(
                        uri=durable.hf_cancellation_event_uri,
                        expected_sha256=durable.hf_cancellation_event_sha256,
                        experiment=durable,
                        submitted_event=submitted,
                    )
                    if stored != validated or (
                        durable.hf_cancellation_event_uri,
                        durable.hf_cancellation_event_sha256,
                    ) != (uri, digest):
                        raise ProvenanceIntegrityError(
                            "HF cancellation claim cannot be replayed or replaced"
                        )
                    self._copy_hf_cancellation_projection(durable, experiment)
                    self._stamp_snapshot(experiment, durable)
                    return self._cancellation_claim_result(
                        validated,
                        uri=uri,
                        sha256=digest,
                        provider_attempt_authorized=False,
                    )
                if durable.hf_cancellation_state is not None:
                    raise ProvenanceIntegrityError("Unknown durable HF cancellation state")
                self._persist_immutable_hf_artifact(path, serialized)
                candidate = replace(
                    durable,
                    hf_cancellation_event_uri=uri,
                    hf_cancellation_event_sha256=digest,
                    hf_cancellation_state="CLAIMED",
                )
                self.verify_experiment_provenance(candidate)
                _save_experiment_unlocked_after_validation(candidate, self.base_dir)
                self._copy_hf_cancellation_projection(candidate, experiment)
                self._stamp_snapshot(experiment, candidate)
                return self._cancellation_claim_result(
                    validated,
                    uri=uri,
                    sha256=digest,
                    provider_attempt_authorized=True,
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

    def _load_durable_experiment_for_hf_transition(
        self, caller: Experiment
    ) -> Experiment:
        try:
            durable = load_experiment(caller.experiment_id, self.base_dir)
        except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
            raise ProvenanceIntegrityError(
                "Durable experiment record is unavailable or invalid"
            ) from exc
        self._require_same_protected_projection(durable, caller)
        self.verify_experiment_provenance(durable)
        return durable

    def _load_durable_experiment_for_cancellation(
        self, caller: Experiment
    ) -> Experiment:
        try:
            durable = load_experiment(caller.experiment_id, self.base_dir)
        except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
            raise ProvenanceIntegrityError(
                "Durable experiment record is unavailable or invalid"
            ) from exc
        if durable.experiment_id != caller.experiment_id:
            raise ProvenanceIntegrityError("Experiment identity changed during cancellation")
        cancellation_fields = set(self._hf_cancellation_field_names())
        caller_cancellation = tuple(
            getattr(caller, field_name) for field_name in self._hf_cancellation_field_names()
        )
        durable_cancellation = tuple(
            getattr(durable, field_name) for field_name in self._hf_cancellation_field_names()
        )
        if caller_cancellation not in {
            (None, None, None),
            durable_cancellation,
        }:
            raise ProvenanceIntegrityError(
                "Experiment cancellation projection conflicts with durable claim"
            )
        for field_name in self._protected_field_names():
            durable_value = getattr(durable, field_name)
            caller_value = getattr(caller, field_name)
            if field_name in cancellation_fields:
                continue
            elif durable_value != caller_value:
                raise ProvenanceIntegrityError(
                    f"Experiment {field_name} conflicts with durable protected provenance"
                )
        self.verify_experiment_provenance(durable)
        return durable

    def _require_provisioning_execution_lock(self, experiment_id: str) -> None:
        held = getattr(self._provisioning_execution, "experiment_ids", set())
        if experiment_id not in held:
            raise ProvenanceIntegrityError(
                "HF provisioning transition requires its execution lock"
            )

    def _load_durable_experiment_for_provisioning(
        self, caller: Experiment
    ) -> Experiment:
        try:
            durable = load_experiment(caller.experiment_id, self.base_dir)
        except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
            raise ProvenanceIntegrityError(
                "Durable experiment record is unavailable or invalid"
            ) from exc
        if durable.experiment_id != caller.experiment_id:
            raise ProvenanceIntegrityError("Experiment identity changed during provisioning")
        provisioning_fields = set(self._hf_provisioning_field_names())
        caller_projection = tuple(
            getattr(caller, name) for name in self._hf_provisioning_field_names()
        )
        durable_projection = tuple(
            getattr(durable, name) for name in self._hf_provisioning_field_names()
        )
        if caller_projection not in {(None, None, None), durable_projection}:
            raise ProvenanceIntegrityError(
                "Experiment provisioning projection conflicts with durable claim"
            )
        for field_name in self._protected_field_names():
            if field_name in provisioning_fields:
                continue
            # A stale PREPARED caller is permitted to recover the terminal head;
            # all immutable descriptor/evidence identities remain checked below.
            if field_name in {
                "provisioning_evidence_uri",
                "provisioning_evidence_sha256",
                "source_transport_state",
            } and durable.hf_provisioning_state == "SUCCEEDED":
                continue
            if getattr(durable, field_name) != getattr(caller, field_name):
                raise ProvenanceIntegrityError(
                    f"Experiment {field_name} conflicts with durable protected provenance"
                )
        self.verify_experiment_provenance(durable)
        return durable

    @staticmethod
    def _hf_provisioning_field_names() -> tuple[str, ...]:
        return (
            "hf_provisioning_event_uri",
            "hf_provisioning_event_sha256",
            "hf_provisioning_state",
        )

    def _validate_hf_provisioning_claim_identity(
        self, experiment: Experiment, value: Mapping[str, object]
    ) -> dict[str, object]:
        try:
            from tuner.cloud.hf_provisioning_claim import validate_hf_provisioning_event

            document = validate_hf_provisioning_event(value)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "HF provisioning claim does not match its exact schema"
            ) from exc
        if document["state"] != "CLAIMED":
            raise ProvenanceIntegrityError("HF provisioning claim must be CLAIMED")
        descriptor = self._verify_hf_tracking_artifact(
            kind="source transport",
            uri=experiment.source_transport_uri,
            expected_sha256=experiment.source_transport_sha256,
            schema_version="synaptic-hf-source-transport/v1",
            experiment_id=experiment.experiment_id,
        )
        capsule = descriptor["capsule"]
        expected = {
            "experiment_id": experiment.experiment_id,
            "run_id": experiment.experiment_id,
            "descriptor": {
                "uri": experiment.source_transport_uri,
                "sha256": experiment.source_transport_sha256,
            },
            "source_lock": {
                "uri": experiment.source_lock_uri,
                "sha256": experiment.source_lock_sha256,
            },
            "volume": {
                key: descriptor["volume"][key]
                for key in ("source", "path", "type", "read_only")
            },
            "bundle_sha256": descriptor["bundle"]["content_sha256"],
            "capsule_manifest_sha256": capsule["manifest"]["sha256"],
            "checkout_policy_sha256": descriptor["checkout_policy"]["sha256"],
        }
        for field_name, expected_value in expected.items():
            if document[field_name] != expected_value:
                raise ProvenanceIntegrityError(
                    f"HF provisioning claim changed durable {field_name} identity"
                )
        return document

    def _validate_hf_provisioning_terminal(
        self,
        experiment: Experiment,
        value: Mapping[str, object],
        *,
        previous_event: Mapping[str, object] | None,
    ) -> dict[str, object]:
        try:
            from tuner.cloud.hf_provisioning_claim import validate_hf_provisioning_event

            if previous_event is None:
                # Idempotent terminal replay: recover and validate its predecessor.
                previous_ref = value.get("previous_event")
                if not isinstance(previous_ref, Mapping):
                    raise ValueError("missing previous event")
                previous_event = self._verify_hf_provisioning_event_artifact(
                    uri=str(previous_ref.get("uri") or ""),
                    expected_sha256=str(previous_ref.get("sha256") or ""),
                )
            document = validate_hf_provisioning_event(
                value, previous_event=previous_event
            )
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "HF provisioning terminal event does not match its exact transition"
            ) from exc
        self._validate_hf_provisioning_claim_identity(experiment, previous_event)
        return document

    @staticmethod
    def _canonical_hf_provisioning_bytes(value: Mapping[str, object]) -> bytes:
        try:
            from tuner.cloud.hf_provisioning_claim import canonical_json_bytes

            return canonical_json_bytes(value)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "HF provisioning event is not canonical JSON data"
            ) from exc

    def _hf_provisioning_event_path(self, experiment_id: str, event_id: str) -> Path:
        return (
            self.base_dir
            / "experiments"
            / experiment_id
            / "cloud"
            / "hf"
            / "provisioning"
            / "events"
            / f"{event_id}.json"
        )

    def _verify_hf_provisioning_event_artifact(
        self,
        *,
        uri: str | None,
        expected_sha256: str | None,
    ) -> dict[str, object]:
        canonical = self._verify_provenance_artifact(
            kind="HF provisioning event",
            uri=uri,
            expected_sha256=expected_sha256,
        )
        try:
            payload = json.loads(canonical, object_pairs_hook=_reject_duplicate_json_keys)
            from tuner.cloud.hf_provisioning_claim import validate_hf_provisioning_event

            # Stored terminal transitions are validated against their predecessor
            # by verify_hf_provisioning_provenance, where the exact ref is known.
            if payload.get("state") == "CLAIMED":
                return validate_hf_provisioning_event(payload)
            previous = payload.get("previous_event")
            if not isinstance(previous, Mapping):
                raise ValueError("missing predecessor")
            predecessor = self._verify_hf_provisioning_event_artifact(
                uri=str(previous.get("uri") or ""),
                expected_sha256=str(previous.get("sha256") or ""),
            )
            return validate_hf_provisioning_event(
                payload, previous_event=predecessor
            )
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "Stored HF provisioning event does not match its exact schema"
            ) from exc

    @staticmethod
    def _hf_provisioning_claim_result(
        document: Mapping[str, object],
        *,
        uri: str,
        sha256: str,
        provider_attempt_authorized: bool,
    ) -> HFProvisioningClaimResult:
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return HFProvisioningClaimResult(
            event_id=str(document["event_id"]),
            event_uri=uri,
            event_sha256=sha256,
            state=str(document["state"]),
            provider_attempt_authorized=provider_attempt_authorized,
            _document_json=canonical,
        )

    @staticmethod
    def _hf_cancellation_field_names() -> tuple[str, ...]:
        return (
            "hf_cancellation_event_uri",
            "hf_cancellation_event_sha256",
            "hf_cancellation_state",
        )

    @staticmethod
    def _canonical_hf_authorization_bytes(value: Mapping[str, object]) -> bytes:
        try:
            from tuner.cloud.hf_run_approval import canonical_json_bytes

            return canonical_json_bytes(dict(value))
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "HF cancellation event is not canonical JSON data"
            ) from exc

    def _build_hf_cancellation_document(
        self,
        experiment: Experiment,
        *,
        submitted_event: Any,
        occurred_at: str,
    ) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": HF_CANCELLATION_SCHEMA_VERSION,
            "event_id": "0" * 64,
            "authorization_id": experiment.hf_authorization_id,
            "approval": {
                "uri": experiment.hf_run_approval_uri,
                "sha256": experiment.hf_run_approval_sha256,
            },
            "submitted_event": {
                "uri": experiment.hf_submission_event_uri,
                "sha256": experiment.hf_submission_event_sha256,
            },
            "provider_job": dict(submitted_event.document["provider_job"]),
            "occurred_at": occurred_at,
        }
        document["event_id"] = self._hf_cancellation_event_id(document)
        return self._validate_hf_cancellation_document(
            document,
            experiment=experiment,
            submitted_event=submitted_event,
        )

    def _validate_hf_cancellation_document(
        self,
        value: Mapping[str, object],
        *,
        experiment: Experiment,
        submitted_event: Any,
    ) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise ProvenanceIntegrityError("HF cancellation event must be an object")
        document = json.loads(self._canonical_hf_authorization_bytes(value))
        expected_keys = {
            "schema_version",
            "event_id",
            "authorization_id",
            "approval",
            "submitted_event",
            "provider_job",
            "occurred_at",
        }
        if set(document) != expected_keys:
            raise ProvenanceIntegrityError("HF cancellation event has an unknown or missing field")
        if document["schema_version"] != HF_CANCELLATION_SCHEMA_VERSION:
            raise ProvenanceIntegrityError("HF cancellation event schema version is invalid")
        if not isinstance(document["event_id"], str) or not _SHA256_RE.fullmatch(
            document["event_id"]
        ):
            raise ProvenanceIntegrityError("HF cancellation event ID is invalid")
        if document["event_id"] != self._hf_cancellation_event_id(document):
            raise ProvenanceIntegrityError(
                "HF cancellation event ID does not match its canonical document"
            )
        expected_identity = {
            "authorization_id": experiment.hf_authorization_id,
            "approval": {
                "uri": experiment.hf_run_approval_uri,
                "sha256": experiment.hf_run_approval_sha256,
            },
            "submitted_event": {
                "uri": experiment.hf_submission_event_uri,
                "sha256": experiment.hf_submission_event_sha256,
            },
            "provider_job": submitted_event.document["provider_job"],
        }
        for field_name, expected in expected_identity.items():
            if document[field_name] != expected:
                raise ProvenanceIntegrityError(
                    f"HF cancellation event changed durable {field_name} identity"
                )
        provider_job = document["provider_job"]
        if not isinstance(provider_job, dict) or set(provider_job) != {
            "namespace",
            "job_id",
        }:
            raise ProvenanceIntegrityError("HF cancellation provider job is invalid")
        if any(
            not isinstance(provider_job[field_name], str)
            or not _PROVIDER_ID_RE.fullmatch(provider_job[field_name])
            for field_name in ("namespace", "job_id")
        ):
            raise ProvenanceIntegrityError("HF cancellation provider identity is invalid")
        occurred_at = document["occurred_at"]
        if not isinstance(occurred_at, str) or not _CANONICAL_UTC_RE.fullmatch(occurred_at):
            raise ProvenanceIntegrityError(
                "HF cancellation occurred_at must be canonical UTC"
            )
        try:
            occurred = datetime.fromisoformat(occurred_at[:-1] + "+00:00")
            submitted_at = datetime.fromisoformat(
                str(submitted_event.document["occurred_at"])[:-1] + "+00:00"
            )
        except ValueError as exc:
            raise ProvenanceIntegrityError("HF cancellation timestamp is invalid") from exc
        normalized = occurred.isoformat(
            timespec="microseconds" if occurred.microsecond else "seconds"
        ).replace("+00:00", "Z")
        if normalized != occurred_at:
            raise ProvenanceIntegrityError("HF cancellation timestamp is not canonical")
        if occurred < submitted_at:
            raise ProvenanceIntegrityError(
                "HF cancellation event predates the SUBMITTED event"
            )
        return document

    def _hf_cancellation_event_id(self, document: Mapping[str, object]) -> str:
        body = {key: value for key, value in document.items() if key != "event_id"}
        return hashlib.sha256(self._canonical_hf_authorization_bytes(body)).hexdigest()

    def _verify_hf_cancellation_event_artifact(
        self,
        *,
        uri: str | None,
        expected_sha256: str | None,
        experiment: Experiment,
        submitted_event: Any,
    ) -> dict[str, object]:
        canonical = self._verify_provenance_artifact(
            kind="HF cancellation event",
            uri=uri,
            expected_sha256=expected_sha256,
        )
        try:
            payload = json.loads(
                canonical,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProvenanceIntegrityError("Stored HF cancellation event is malformed") from exc
        return self._validate_hf_cancellation_document(
            payload,
            experiment=experiment,
            submitted_event=submitted_event,
        )

    @staticmethod
    def _cancellation_claim_result(
        document: Mapping[str, object],
        *,
        uri: str,
        sha256: str,
        provider_attempt_authorized: bool,
    ) -> HFCancellationClaimResult:
        canonical = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return HFCancellationClaimResult(
            event_id=str(document["event_id"]),
            event_uri=uri,
            event_sha256=sha256,
            provider_attempt_authorized=provider_attempt_authorized,
            _document_json=canonical,
        )

    def _require_hf_approval_bindings(
        self, durable: Experiment, document: Mapping[str, Any]
    ) -> None:
        expected_pairs = {
            "source_lock": (
                durable.source_lock_uri,
                durable.source_lock_sha256,
            ),
            "descriptor": (
                durable.source_transport_uri,
                durable.source_transport_sha256,
            ),
            "provisioning_evidence": (
                durable.provisioning_evidence_uri,
                durable.provisioning_evidence_sha256,
            ),
        }
        for key, (uri, sha256) in expected_pairs.items():
            value = document.get(key)
            if value is None:
                continue
            if value != {"uri": uri, "sha256": sha256}:
                raise ProvenanceIntegrityError(
                    f"HF run approval does not bind the durable {key.replace('_', ' ')}"
                )
        descriptor = self._verify_hf_tracking_artifact(
            kind="source transport",
            uri=durable.source_transport_uri,
            expected_sha256=durable.source_transport_sha256,
            schema_version="synaptic-hf-source-transport/v1",
            experiment_id=durable.experiment_id,
        )
        expected_digests = {
            "bundle_sha256": descriptor["bundle"]["content_sha256"],
            "capsule_manifest_sha256": descriptor["capsule"]["manifest"]["sha256"],
            "checkout_policy_sha256": descriptor["checkout_policy"]["sha256"],
        }
        for field_name, expected in expected_digests.items():
            if document.get(field_name) != expected:
                raise ProvenanceIntegrityError(
                    f"HF run approval does not bind the descriptor {field_name}"
                )

    @staticmethod
    def _require_hf_event_identity(
        durable: Experiment, document: Mapping[str, Any]
    ) -> None:
        if document.get("authorization_id") != durable.hf_authorization_id:
            raise ProvenanceIntegrityError(
                "HF submission event uses another authorization"
            )
        if document.get("experiment_id") != durable.experiment_id:
            raise ProvenanceIntegrityError(
                "HF submission event belongs to another experiment"
            )
        if document.get("approval") != {
            "uri": durable.hf_run_approval_uri,
            "sha256": durable.hf_run_approval_sha256,
        }:
            raise ProvenanceIntegrityError(
                "HF submission event does not bind the immutable approval"
            )

    def _persist_hf_submission_event(
        self,
        experiment: Experiment,
        *,
        event_id: str,
        document: Mapping[str, Any],
    ) -> tuple[str, str]:
        try:
            from tuner.cloud.hf_run_approval import canonical_json_bytes

            serialized = canonical_json_bytes(dict(document))
        except Exception as exc:
            raise ProvenanceIntegrityError("HF submission event is not canonical") from exc
        digest = hashlib.sha256(serialized).hexdigest()
        relative = (
            Path("experiments")
            / experiment.experiment_id
            / "cloud"
            / "hf"
            / "submission"
            / "events"
            / f"{event_id}.json"
        )
        path = self.base_dir / relative
        self._persist_immutable_hf_artifact(path, serialized)
        return self.tracking_uri(path), digest

    def _persist_immutable_hf_artifact(self, path: Path, serialized: bytes) -> None:
        root = self.base_dir.resolve(strict=False)
        if not _contained(path, root):
            raise ProvenanceIntegrityError("HF submission artifact escapes tracking root")
        relative = path.resolve(strict=False).relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.exists() and _is_link_or_reparse(current):
                raise ProvenanceIntegrityError(
                    "HF submission artifacts cannot use symlinks or reparse points"
                )
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise ProvenanceIntegrityError(
                    "Existing HF submission artifact is unreadable"
                ) from exc
            if existing != serialized:
                raise ProvenanceIntegrityError(
                    "Immutable HF submission artifact cannot be replaced"
                )
            return
        _atomic_write_bytes(path, serialized)

    def _verify_hf_run_approval_artifact(
        self,
        *,
        uri: str | None,
        expected_sha256: str | None,
        experiment_id: str,
    ) -> Any:
        canonical = self._verify_provenance_artifact(
            kind="HF run approval",
            uri=uri,
            expected_sha256=expected_sha256,
        )
        try:
            payload = json.loads(
                canonical,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            from tuner.cloud.hf_run_approval import validate_hf_run_approval

            approval = validate_hf_run_approval(
                payload,
                at=payload.get("issued_at"),
            )
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "Stored HF run approval does not match its exact schema"
            ) from exc
        if approval.document["experiment_id"] != experiment_id:
            raise ProvenanceIntegrityError(
                "Stored HF run approval belongs to another experiment"
            )
        return approval

    def _verify_hf_submission_event_artifact(
        self,
        *,
        uri: str | None,
        expected_sha256: str | None,
        approval: Any,
    ) -> Any:
        canonical = self._verify_provenance_artifact(
            kind="HF submission event",
            uri=uri,
            expected_sha256=expected_sha256,
        )
        try:
            payload = json.loads(
                canonical,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            from tuner.cloud.hf_run_approval import validate_hf_submission_claim

            return validate_hf_submission_claim(payload, approval=approval)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                "Stored HF submission event does not match its exact schema"
            ) from exc

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
    def _copy_hf_submission_projection(source: Experiment, target: Experiment) -> None:
        for field_name in (
            "hf_run_approval_uri",
            "hf_run_approval_sha256",
            "hf_authorization_id",
            "hf_submission_event_uri",
            "hf_submission_event_sha256",
            "hf_submission_state",
        ):
            setattr(target, field_name, getattr(source, field_name))

    @staticmethod
    def _copy_hf_provisioning_projection(source: Experiment, target: Experiment) -> None:
        for field_name in TrackingService._hf_provisioning_field_names():
            setattr(target, field_name, getattr(source, field_name))

    @staticmethod
    def _copy_hf_cancellation_projection(source: Experiment, target: Experiment) -> None:
        for field_name in TrackingService._hf_cancellation_field_names():
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
            "hf_provisioning_event_uri",
            "hf_provisioning_event_sha256",
            "hf_provisioning_state",
            "hf_run_approval_uri",
            "hf_run_approval_sha256",
            "hf_authorization_id",
            "hf_submission_event_uri",
            "hf_submission_event_sha256",
            "hf_submission_state",
            "hf_cancellation_event_uri",
            "hf_cancellation_event_sha256",
            "hf_cancellation_state",
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
        if kind in {
            "source transport",
            "provisioning evidence",
            "HF provisioning event",
            "HF run approval",
            "HF submission event",
            "HF cancellation event",
        }:
            try:
                if kind == "HF provisioning event":
                    from tuner.cloud.hf_provisioning_claim import canonical_json_bytes
                elif kind in {
                    "HF run approval",
                    "HF submission event",
                    "HF cancellation event",
                }:
                    from tuner.cloud.hf_run_approval import canonical_json_bytes
                else:
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
            "hf_provisioning_event_uri",
            "hf_provisioning_event_sha256",
            "hf_provisioning_state",
            "hf_run_approval_uri",
            "hf_run_approval_sha256",
            "hf_authorization_id",
            "hf_submission_event_uri",
            "hf_submission_event_sha256",
            "hf_submission_state",
            "hf_cancellation_event_uri",
            "hf_cancellation_event_sha256",
            "hf_cancellation_state",
            *TrackingService._hf_training_projection_fields(),
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
