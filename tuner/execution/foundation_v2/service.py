"""Mechanical compare-and-set lifecycle storage only."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock

from .canonical import safe_ref
from .lifecycle import LifecyclePhaseV2, LifecycleStateV2, transition


@dataclass(frozen=True, slots=True)
class LifecycleRecordV2:
    project_ref: str
    run_id: str
    revision: int
    state: LifecycleStateV2


class MechanicalLifecycleStoreV2:
    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[tuple[str, str], LifecycleRecordV2] = {}

    def create(self, project_ref: str, run_id: str) -> LifecycleRecordV2:
        project_ref = safe_ref(project_ref, "project_ref")
        run_id = safe_ref(run_id, "run_id")
        record = LifecycleRecordV2(
            project_ref, run_id, 1, LifecycleStateV2(LifecyclePhaseV2.PLANNED),
        )
        with self._lock:
            if (project_ref, run_id) in self._records:
                raise ValueError("lifecycle record already exists")
            self._records[(project_ref, run_id)] = record
        return record

    def apply(
        self, project_ref: str, run_id: str, *, expected_revision: int,
        target: LifecycleStateV2,
    ) -> LifecycleRecordV2:
        with self._lock:
            record = self._records[(project_ref, run_id)]
            if record.revision != expected_revision:
                raise ValueError("lifecycle revision conflict")
            updated = replace(
                record, revision=record.revision + 1,
                state=transition(record.state, target),
            )
            self._records[(project_ref, run_id)] = updated
            return updated
