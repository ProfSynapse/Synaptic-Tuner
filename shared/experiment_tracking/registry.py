"""
shared/experiment_tracking/registry.py

Append-only JSONL registry for unified run tracking. Stores RunRecord entries
and supports query/filter operations. Handles run linkage (e.g. eval -> training).

Used by: local_tracker.py (auto-registration), adapters.py (manual registration),
         CLI list-runs (query), Evaluator (parent linkage).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .schema import RunFilter, RunRecord

logger = logging.getLogger(__name__)

# Link records are stored as separate JSONL lines alongside RunRecords.
# They have a "__link__" marker to distinguish them from run entries.
_LINK_MARKER = "__link__"


def _default_registry_path() -> Path:
    """Resolve the default registry path: {repo_root}/.tracking/registry.jsonl."""
    # Walk up from this file to find the repo root (where .git lives)
    current = Path(__file__).resolve().parent
    for ancestor in [current] + list(current.parents):
        if (ancestor / ".git").exists() or (ancestor / ".git").is_file():
            return ancestor / ".tracking" / "registry.jsonl"
    # Fallback: use the shared/ parent
    return current.parent.parent / ".tracking" / "registry.jsonl"


class RunRegistry:
    """Central registry for experiment runs.

    Stores RunRecord entries in an append-only JSONL file. Each line is either
    a serialized RunRecord or a link record (for parent/child relationships).

    Args:
        registry_path: Path to the JSONL registry file. If None, uses the
                       default location at {repo_root}/.tracking/registry.jsonl.
    """

    def __init__(self, registry_path: Path | str | None = None) -> None:
        if registry_path is None:
            self._path = _default_registry_path()
        else:
            self._path = Path(registry_path)

    @property
    def path(self) -> Path:
        """Return the registry file path."""
        return self._path

    def register_run(self, record: RunRecord) -> str:
        """Append a RunRecord to the registry.

        Uses write-to-temp-then-rename for crash safety on the first write.
        Subsequent writes use direct append (JSONL append is atomic for
        reasonable line lengths on POSIX).

        Args:
            record: The run record to register.

        Returns:
            The run_id of the registered record.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = record.to_json_line() + "\n"

        if not self._path.exists():
            # First write: use atomic temp-file rename
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp"
            )
            try:
                os.write(fd, line.encode("utf-8"))
                os.close(fd)
                os.replace(tmp, str(self._path))
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        else:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)

        logger.info("Registered run %s (%s)", record.run_id, record.run_type)
        return record.run_id

    def link_runs(
        self, child_run_id: str, parent_run_id: str, relationship: str = "parent"
    ) -> None:
        """Record a link between two runs (e.g. evaluation -> training).

        Links are stored as special JSONL lines with a __link__ marker.

        Args:
            child_run_id: The dependent run (e.g. evaluation run).
            parent_run_id: The upstream run (e.g. training run).
            relationship: Label for the link type (default: "parent").
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        link = {
            _LINK_MARKER: True,
            "child_run_id": child_run_id,
            "parent_run_id": parent_run_id,
            "relationship": relationship,
        }
        line = json.dumps(link, ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def find_runs(self, filters: RunFilter | None = None) -> list[RunRecord]:
        """Query runs matching the given filter.

        Args:
            filters: Optional filter criteria. If None, returns all runs.

        Returns:
            List of matching RunRecord instances, ordered by file position.
        """
        records = self._load_records()
        if filters is None:
            return records
        return [r for r in records if filters.matches(r)]

    def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a single run by its ID.

        Args:
            run_id: The UUID of the run to find.

        Returns:
            The RunRecord if found, None otherwise.
        """
        for record in self._load_records():
            if record.run_id == run_id:
                return record
        return None

    def get_linked_runs(
        self, run_id: str, relationship: str | None = None
    ) -> list[RunRecord]:
        """Find runs linked to the given run_id.

        Searches both directions: returns runs where the given run_id appears
        as either parent or child in a link record.

        Args:
            run_id: The run to find links for.
            relationship: Optional filter by relationship type.

        Returns:
            List of linked RunRecord instances.
        """
        links = self._load_links()
        linked_ids: set[str] = set()

        for link in links:
            if relationship and link.get("relationship") != relationship:
                continue
            if link.get("parent_run_id") == run_id:
                linked_ids.add(link["child_run_id"])
            elif link.get("child_run_id") == run_id:
                linked_ids.add(link["parent_run_id"])

        if not linked_ids:
            return []

        return [r for r in self._load_records() if r.run_id in linked_ids]

    def _load_records(self) -> list[RunRecord]:
        """Load all RunRecord entries from the registry, skipping malformed lines."""
        if not self._path.exists():
            return []

        records: list[RunRecord] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if _LINK_MARKER in data:
                        continue  # Skip link records
                    records.append(RunRecord.from_dict(data))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    logger.warning(
                        "Skipping malformed line %d in %s: %s",
                        line_num, self._path, exc,
                    )
        return records

    def _load_links(self) -> list[dict[str, Any]]:
        """Load all link records from the registry."""
        if not self._path.exists():
            return []

        links: list[dict[str, Any]] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if _LINK_MARKER in data:
                        links.append(data)
                except (json.JSONDecodeError, TypeError):
                    continue
        return links
