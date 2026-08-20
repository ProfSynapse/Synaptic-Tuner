import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.utilities.unique_ids import unique_prefixed_id

logger = logging.getLogger(__name__)

HF_SOURCE_TRANSPORT_STATES = (
    "PREPARED",
    "ACKNOWLEDGED",
    "CONSUMABLE",
    "SUBMITTED",
)

HF_SUBMISSION_STATES = (
    "APPROVED",
    "SUBMITTING",
    "SUBMITTED",
    "AMBIGUOUS",
)

HF_PROVISIONING_STATES = (
    "CLAIMED",
    "SUCCEEDED",
    "AMBIGUOUS",
)


def _validate_reference_pair(*, kind: str, uri: str | None, sha256: str | None) -> None:
    if (uri is None) != (sha256 is None):
        raise ValueError(f"{kind} reference requires both URI and SHA-256")
    if sha256 is not None and (
        len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(f"{kind} SHA-256 must be 64 lowercase hexadecimal characters")


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace one complete file using a unique same-directory temporary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))

@dataclass
class Experiment:
    """Experiment definition matching the local .tracking/experiments/{id}/ schema."""
    experiment_id: str
    name: str
    created_at: str
    dataset_path: str
    dataset_hash: str
    base_model_name: str
    run_ids: list[str] = field(default_factory=list)
    base_losses_path: str | None = None
    features_csv_path: str | None = None
    judge_scores_path: str | None = None
    status: str = "partial"
    provider: str = ""
    method: str = ""
    objective: str = ""
    spec_path: str | None = None
    training_run_id: str | None = None
    evaluation_run_id: str | None = None
    loss_run_id: str | None = None
    selected_run_id: str | None = None
    artifact_roots: dict[str, str] = field(default_factory=dict)
    derived_outputs: dict[str, str] = field(default_factory=dict)
    stage_statuses: dict[str, str] = field(default_factory=dict)
    stage_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    hypothesis_context_path: str | None = None
    next_run_candidates_path: str | None = None
    source_lock_uri: str | None = None
    source_lock_sha256: str | None = None
    resolved_config_uri: str | None = None
    resolved_config_sha256: str | None = None
    source_transport_uri: str | None = None
    source_transport_sha256: str | None = None
    provisioning_evidence_uri: str | None = None
    provisioning_evidence_sha256: str | None = None
    source_transport_state: str | None = None
    hf_provisioning_event_uri: str | None = None
    hf_provisioning_event_sha256: str | None = None
    hf_provisioning_state: str | None = None
    hf_run_approval_uri: str | None = None
    hf_run_approval_sha256: str | None = None
    hf_authorization_id: str | None = None
    hf_submission_event_uri: str | None = None
    hf_submission_event_sha256: str | None = None
    hf_submission_state: str | None = None
    hf_cancellation_event_uri: str | None = None
    hf_cancellation_event_sha256: str | None = None
    hf_cancellation_state: str | None = None

    def __post_init__(self) -> None:
        _validate_reference_pair(
            kind="Source transport",
            uri=self.source_transport_uri,
            sha256=self.source_transport_sha256,
        )
        _validate_reference_pair(
            kind="Provisioning evidence",
            uri=self.provisioning_evidence_uri,
            sha256=self.provisioning_evidence_sha256,
        )
        _validate_reference_pair(
            kind="HF provisioning event",
            uri=self.hf_provisioning_event_uri,
            sha256=self.hf_provisioning_event_sha256,
        )
        _validate_reference_pair(
            kind="HF approval",
            uri=self.hf_run_approval_uri,
            sha256=self.hf_run_approval_sha256,
        )
        _validate_reference_pair(
            kind="HF submission event",
            uri=self.hf_submission_event_uri,
            sha256=self.hf_submission_event_sha256,
        )
        _validate_reference_pair(
            kind="HF cancellation event",
            uri=self.hf_cancellation_event_uri,
            sha256=self.hf_cancellation_event_sha256,
        )
        if self.source_transport_state is not None:
            if self.source_transport_state not in HF_SOURCE_TRANSPORT_STATES:
                raise ValueError("Unknown source transport lifecycle state")
            if self.source_transport_uri is None:
                raise ValueError("Source transport lifecycle state requires a descriptor reference")
        if self.provisioning_evidence_uri is not None:
            if self.source_transport_uri is None:
                raise ValueError("Provisioning evidence requires a source transport descriptor")
            if self.source_transport_state == "PREPARED":
                raise ValueError("PREPARED source transport cannot include provisioning evidence")
        if self.source_transport_state in {"ACKNOWLEDGED", "CONSUMABLE", "SUBMITTED"}:
            if self.provisioning_evidence_uri is None:
                raise ValueError(
                    f"{self.source_transport_state} source transport requires provisioning evidence"
                )
        if self.hf_provisioning_state is not None:
            if self.hf_provisioning_state not in HF_PROVISIONING_STATES:
                raise ValueError("Unknown HF provisioning state")
            if self.hf_provisioning_event_uri is None:
                raise ValueError("HF provisioning state requires an event")
            if self.hf_provisioning_state == "CLAIMED":
                if self.source_transport_state != "PREPARED":
                    raise ValueError("CLAIMED provisioning requires PREPARED source transport")
                if self.provisioning_evidence_uri is not None:
                    raise ValueError("CLAIMED provisioning cannot include evidence")
            elif self.hf_provisioning_state == "SUCCEEDED":
                if self.source_transport_state not in {"ACKNOWLEDGED", "CONSUMABLE", "SUBMITTED"}:
                    raise ValueError("SUCCEEDED provisioning requires acknowledged source transport")
                if self.provisioning_evidence_uri is None:
                    raise ValueError("SUCCEEDED provisioning requires evidence")
            elif self.provisioning_evidence_uri is not None:
                raise ValueError("AMBIGUOUS provisioning cannot include evidence")
        elif self.hf_provisioning_event_uri is not None:
            raise ValueError("HF provisioning event requires provisioning state")
        if self.hf_submission_state is not None:
            if self.hf_submission_state not in HF_SUBMISSION_STATES:
                raise ValueError("Unknown HF submission state")
            if self.hf_run_approval_uri is None or self.hf_authorization_id is None:
                raise ValueError("HF submission state requires an approval and authorization ID")
        if self.hf_run_approval_uri is not None:
            if self.hf_provisioning_state != "SUCCEEDED":
                raise ValueError("HF approval requires durable SUCCEEDED provisioning")
            if self.hf_authorization_id is None:
                raise ValueError("HF approval requires an authorization ID")
            if self.hf_submission_state is None:
                raise ValueError("HF approval requires an APPROVED submission state")
            if self.source_transport_state != "CONSUMABLE":
                raise ValueError("HF approval requires source transport state CONSUMABLE")
        elif self.hf_authorization_id is not None:
            raise ValueError("HF authorization ID requires an approval")
        if self.hf_submission_event_uri is not None:
            if self.hf_submission_state not in {"SUBMITTING", "SUBMITTED", "AMBIGUOUS"}:
                raise ValueError("HF submission event requires a claimed submission state")
        elif self.hf_submission_state in {"SUBMITTING", "SUBMITTED", "AMBIGUOUS"}:
            raise ValueError(f"{self.hf_submission_state} requires an HF submission event")
        if self.hf_cancellation_state is not None:
            if self.hf_cancellation_state != "CLAIMED":
                raise ValueError("Unknown HF cancellation state")
            if self.hf_submission_state != "SUBMITTED":
                raise ValueError("HF cancellation requires submission state SUBMITTED")
            if self.hf_cancellation_event_uri is None:
                raise ValueError("CLAIMED cancellation requires an HF cancellation event")
        elif self.hf_cancellation_event_uri is not None:
            raise ValueError("HF cancellation event requires CLAIMED state")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        """Deserialize from a dictionary, ignoring unknown fields."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

def create_experiment(
    name: str,
    dataset_path: str,
    dataset_hash: str,
    base_model_name: str,
    provider: str = "",
    method: str = "",
    objective: str = "",
    spec_path: str | None = None,
    base_dir: Path | str = ".tracking",
) -> Experiment:
    """Create a new experiment, write to disk, and return the metadata."""
    # Imported lazily because registry schema validation imports this module.
    from .registry import _PathLock

    now = datetime.now(timezone.utc)
    timestamp_id = unique_prefixed_id("exp_", now=now)
    
    experiment = Experiment(
        experiment_id=timestamp_id,
        name=name,
        created_at=now.isoformat(),
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        base_model_name=base_model_name,
        provider=provider,
        method=method,
        objective=objective,
        spec_path=spec_path,
    )
    
    exp_file = Path(base_dir) / "experiments" / timestamp_id / "experiment.json"
    with _PathLock(exp_file):
        if exp_file.exists():
            raise FileExistsError(f"Experiment file already exists: {exp_file}")
        _save_experiment_unlocked_after_validation(experiment, base_dir=base_dir)
    return experiment


def _save_experiment_unlocked_after_validation(
    experiment: Experiment, base_dir: Path | str = ".tracking"
) -> None:
    """Atomically write a record after the caller has locked and validated it.

    This is deliberately private. The caller must already hold the path lock
    for ``experiment.json`` and must have validated either neutral first
    creation or the durable CAS/provenance contract. Public callers must use
    :class:`TrackingService` instead.
    """
    exp_dir = Path(base_dir) / "experiments" / experiment.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    exp_file = exp_dir / "experiment.json"
    payload = json.dumps(experiment.to_dict(), indent=2).encode("utf-8")
    _atomic_write_bytes(exp_file, payload)

def load_experiment(experiment_id: str, base_dir: Path | str = ".tracking") -> Experiment:
    """Load an experiment.json from disk."""
    exp_file = Path(base_dir) / "experiments" / experiment_id / "experiment.json"
    
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_file}")
        
    with open(exp_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return Experiment.from_dict(data)
