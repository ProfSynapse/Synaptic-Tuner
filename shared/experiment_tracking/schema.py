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
    run_type: str  # "sft" | "kto" | "grpo" | "ml" | "evaluation" | "cloud_sft" | "cloud_kto" | "cloud_grpo"
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
