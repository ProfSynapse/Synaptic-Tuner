"""
shared/experiment_tracking/schema.py

Unified run record schema and query filter for the experiment tracking registry.
All run types (SFT, KTO, ML, evaluation, cloud) share the same RunRecord structure.
Detailed run data stays in per-run lineage files; the registry is a lightweight index.

Used by: registry.py (storage), adapters.py (conversion), CLI list-runs (display)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .experiment import (
    HF_PROVISIONING_STATES,
    HF_SOURCE_TRANSPORT_STATES,
    HF_SUBMISSION_STATES,
    HF_TRAINING_CANCELLATION_STATES,
    HF_TRAINING_OBSERVATION_STATES,
    HF_TRAINING_RESULT_STATES,
    HF_TRAINING_SUBMISSION_STATES,
    _validate_reference_pair,
)

logger = logging.getLogger(__name__)

# Current schema version — bump when adding/removing fields.
_CURRENT_SCHEMA_VERSION = 2

@dataclass
class LossResult:
    """Per-example loss result for a single sequence in a dataset."""
    index: int                    # JSONL line index (0-based)
    loss: float                   # Mean cross-entropy on completion tokens only
    num_completion_tokens: int    # Non-masked token count
    num_total_tokens: int         # Total tokenized sequence length
    jsonl_hash: str               # First 8 chars of SHA-256 of raw JSONL line

@dataclass
class RunRecord:
    """Common fields across ALL run types. Stored in registry.jsonl.

    Each line in the registry JSONL file is one serialized RunRecord.
    The schema_version field allows forward-compatible evolution.

    Schema migration strategy:
        - Unknown fields are silently dropped (forward compat: old reader, new data).
        - When schema_version > _CURRENT_SCHEMA_VERSION, a debug message is logged
          but the record is still loaded (best-effort, using known fields only).
        - When schema_version < _CURRENT_SCHEMA_VERSION, future migrations can be
          applied in from_dict() before constructing the record. Currently v1 is
          the only version, so no migrations exist yet.
    """

    run_id: str
    run_type: str  # "sft" | "kto" | "grpo" | "dpo" | "ace_step" | "ml" | "evaluation" | "embedding" | "cloud_sft" | "cloud_kto" | "cloud_grpo" | "cloud_embedding"
    name: str
    timestamp: str  # ISO 8601 UTC
    status: str  # "completed" | "failed" | "running"
    output_dir: str
    parent_run_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    schema_version: int = 1

    # Common optional fields (not all types populate all)
    model_name: str | None = None
    dataset_source: str | None = None
    primary_metric: float | None = None
    primary_metric_name: str | None = None
    hardware: str | None = None
    per_example_losses_path: str | None = None
    experiment_id: str | None = None
    provider: str | None = None
    artifact_backend: str | None = None
    artifact_root: str | None = None
    job_ref: str | None = None
    source_commit: str | None = None
    stage: str | None = None
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
    hf_training_root_id: str | None = None
    hf_training_run_id: str | None = None
    hf_training_preflight_uri: str | None = None
    hf_training_preflight_sha256: str | None = None
    hf_training_preflight_state: str | None = None
    hf_training_approval_uri: str | None = None
    hf_training_approval_sha256: str | None = None
    hf_training_authorization_id: str | None = None
    hf_training_submission_event_uri: str | None = None
    hf_training_submission_event_sha256: str | None = None
    hf_training_submission_state: str | None = None
    hf_training_cancellation_event_uri: str | None = None
    hf_training_cancellation_event_sha256: str | None = None
    hf_training_cancellation_state: str | None = None
    hf_training_observation_event_uri: str | None = None
    hf_training_observation_event_sha256: str | None = None
    hf_training_observation_state: str | None = None
    hf_training_result_uri: str | None = None
    hf_training_result_sha256: str | None = None
    hf_training_result_state: str | None = None

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
        if self.provisioning_evidence_uri is not None and self.source_transport_uri is None:
            raise ValueError("Provisioning evidence requires a source transport descriptor")
        if self.source_transport_state == "PREPARED" and self.provisioning_evidence_uri is not None:
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
        for kind, uri, digest in (
            ("HF training preflight", self.hf_training_preflight_uri, self.hf_training_preflight_sha256),
            ("HF training approval", self.hf_training_approval_uri, self.hf_training_approval_sha256),
            ("HF training submission event", self.hf_training_submission_event_uri, self.hf_training_submission_event_sha256),
            ("HF training cancellation event", self.hf_training_cancellation_event_uri, self.hf_training_cancellation_event_sha256),
            ("HF training observation event", self.hf_training_observation_event_uri, self.hf_training_observation_event_sha256),
            ("HF training result", self.hf_training_result_uri, self.hf_training_result_sha256),
        ):
            _validate_reference_pair(kind=kind, uri=uri, sha256=digest)
        training_present = any(
            value is not None
            for value in (
                self.hf_training_preflight_uri,
                self.hf_training_approval_uri,
                self.hf_training_submission_event_uri,
                self.hf_training_cancellation_event_uri,
                self.hf_training_observation_event_uri,
                self.hf_training_result_uri,
            )
        )
        if training_present and self.hf_training_root_id is None:
            raise ValueError("HF training projection requires canonical tracking-root identity")
        if training_present and self.hf_training_run_id is None:
            raise ValueError("HF training projection requires protected run identity")
        if self.hf_training_root_id is not None and (
            len(self.hf_training_root_id) != 64
            or any(character not in "0123456789abcdef" for character in self.hf_training_root_id)
        ):
            raise ValueError("HF training root ID must be 64 lowercase hexadecimal characters")
        if self.hf_training_preflight_state not in {None, "PASS"}:
            raise ValueError("Unknown HF training preflight state")
        if self.hf_training_preflight_state == "PASS" and self.hf_training_preflight_uri is None:
            raise ValueError("HF training PASS requires preflight evidence")
        if self.hf_training_approval_uri is not None:
            if self.hf_training_preflight_state != "PASS" or self.hf_training_authorization_id is None or self.hf_training_submission_state is None:
                raise ValueError("HF training approval projection is incomplete")
        elif self.hf_training_authorization_id is not None:
            raise ValueError("HF training authorization ID requires approval")
        if self.hf_training_submission_state not in {None, *HF_TRAINING_SUBMISSION_STATES}:
            raise ValueError("Unknown HF training submission state")
        if self.hf_training_submission_state == "APPROVED":
            if self.hf_training_submission_event_uri is not None:
                raise ValueError("APPROVED training submission cannot include an event")
        elif self.hf_training_submission_state is not None and self.hf_training_submission_event_uri is None:
            raise ValueError("Claimed HF training submission requires an event")
        if self.hf_training_cancellation_state not in {None, *HF_TRAINING_CANCELLATION_STATES}:
            raise ValueError("Unknown HF training cancellation state")
        if self.hf_training_cancellation_state is not None and (
            self.hf_training_submission_state != "SUBMITTED" or self.hf_training_cancellation_event_uri is None
        ):
            raise ValueError("HF training cancellation projection is incomplete")
        if self.hf_training_observation_state not in {None, *HF_TRAINING_OBSERVATION_STATES}:
            raise ValueError("Unknown HF training observation state")
        if self.hf_training_observation_state is not None and (
            self.hf_training_submission_state != "SUBMITTED" or self.hf_training_observation_event_uri is None
        ):
            raise ValueError("HF training observation projection is incomplete")
        if self.hf_training_result_state not in {None, *HF_TRAINING_RESULT_STATES}:
            raise ValueError("Unknown HF training result state")
        if self.hf_training_result_state is not None and (
            self.hf_training_observation_state is None or self.hf_training_result_uri is None
        ):
            raise ValueError("HF training result projection is incomplete")

    def to_json_line(self) -> str:
        """Serialize to a single JSON line for JSONL storage."""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Deserialize from a dictionary, ignoring unknown fields.

        Unknown fields are silently dropped for forward compatibility
        (older reader, newer schema_version). When the record's
        schema_version exceeds _CURRENT_SCHEMA_VERSION, a debug log is
        emitted but the record is still loaded using known fields.
        """
        version = data.get("schema_version", 1)
        if version > _CURRENT_SCHEMA_VERSION:
            logger.debug(
                "RunRecord schema_version %d is newer than supported %d; "
                "loading with known fields only",
                version, _CURRENT_SCHEMA_VERSION,
            )
        # Future: apply migrations for version < _CURRENT_SCHEMA_VERSION here.

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json_line(cls, line: str) -> RunRecord:
        """Deserialize from a single JSON line."""
        return cls.from_dict(json.loads(line))


@dataclass
class RunFilter:
    """Filter criteria for querying the run registry.

    All fields are optional. When multiple fields are set, they are
    combined with AND logic (all must match).
    """

    run_type: str | list[str] | None = None
    status: str | None = None
    model_name: str | None = None
    since: str | None = None  # ISO 8601 — include runs at or after this timestamp
    until: str | None = None  # ISO 8601 — include runs at or before this timestamp
    tags: dict[str, str] | None = None

    def matches(self, record: RunRecord) -> bool:
        """Check whether a RunRecord satisfies this filter."""
        if self.run_type is not None:
            allowed = self.run_type if isinstance(self.run_type, list) else [self.run_type]
            if record.run_type not in allowed:
                return False

        if self.status is not None and record.status != self.status:
            return False

        if self.model_name is not None:
            if record.model_name is None:
                return False
            if self.model_name.lower() not in record.model_name.lower():
                return False

        # Timestamp comparison using parsed datetime objects for correctness
        # across timezone offsets and format variants (e.g. "Z" vs "+00:00").
        if self.since is not None:
            if _parse_ts(record.timestamp) < _parse_ts(self.since):
                return False

        if self.until is not None:
            if _parse_ts(record.timestamp) > _parse_ts(self.until):
                return False

        if self.tags is not None:
            for key, value in self.tags.items():
                if record.tags.get(key) != value:
                    return False

        return True


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a timezone-aware datetime.

    Handles both "+00:00" and "Z" suffixes. Timestamps without timezone info
    are assumed to be UTC.
    """
    normalized = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        # Last resort: return a sentinel that preserves lexicographic ordering
        return datetime.min.replace(tzinfo=timezone.utc)
