"""
shared/experiment_tracking/schema.py

Unified run record schema and query filter for the experiment tracking registry.
All run types (SFT, KTO, ML, evaluation, cloud) share the same RunRecord structure.
Detailed run data stays in per-run lineage files; the registry is a lightweight index.

Used by: registry.py (storage), adapters.py (conversion), CLI list-runs (display)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunRecord:
    """Common fields across ALL run types. Stored in registry.jsonl.

    Each line in the registry JSONL file is one serialized RunRecord.
    The schema_version field allows forward-compatible evolution.
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

    def to_json_line(self) -> str:
        """Serialize to a single JSON line for JSONL storage."""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Deserialize from a dictionary, ignoring unknown fields.

        Unknown fields are silently dropped for forward compatibility
        (older reader, newer schema_version).
        """
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

        if self.since is not None and record.timestamp < self.since:
            return False

        if self.until is not None and record.timestamp > self.until:
            return False

        if self.tags is not None:
            for key, value in self.tags.items():
                if record.tags.get(key) != value:
                    return False

        return True
