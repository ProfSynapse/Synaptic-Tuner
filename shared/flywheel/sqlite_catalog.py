"""
shared/flywheel/sqlite_catalog.py

SQLiteLogCatalog: local SQLite implementation of LogCatalog using aiosqlite.
Uses WAL mode for concurrent read access from dashboard/CLI while the
pipeline writes. Single writer is fine for local use.

Used by: catalog.py (create_catalog factory)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .catalog import DatasetVersion, InferenceLogRecord, LogFilter

logger = logging.getLogger(__name__)

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_logs (
    log_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    adapter_name    TEXT,
    has_tool_calls  INTEGER NOT NULL DEFAULT 0,
    tools_requested INTEGER NOT NULL DEFAULT 0,
    fitness_score   REAL,
    is_valid        INTEGER,
    tag             TEXT,
    tag_source      TEXT,
    dataset_version TEXT,
    source_file     TEXT NOT NULL,
    line_number     INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_logs_tag ON inference_logs(tag);
CREATE INDEX IF NOT EXISTS idx_logs_score ON inference_logs(fitness_score);
CREATE INDEX IF NOT EXISTS idx_logs_unused ON inference_logs(dataset_version)
    WHERE dataset_version IS NULL;
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON inference_logs(timestamp);

CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id      TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    source_model_id TEXT NOT NULL,
    record_counts   TEXT NOT NULL,
    file_paths      TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    parent_version  TEXT,
    filter_criteria TEXT,
    training_run_id TEXT,
    FOREIGN KEY (parent_version) REFERENCES dataset_versions(version_id)
);
"""


class SQLiteLogCatalog:
    """Local SQLite implementation using aiosqlite.

    Uses WAL mode for concurrent read access from dashboard/CLI
    while the pipeline writes.

    Args:
        db_path: Path to SQLite database file (e.g., ".tracking/flywheel.db")
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._conn: Any = None

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        import aiosqlite

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.executescript(_SQLITE_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def insert_log(self, record: InferenceLogRecord) -> str:
        """Index a single inference log record."""
        await self._conn.execute(
            """INSERT OR IGNORE INTO inference_logs
               (log_id, timestamp, model_id, adapter_name,
                has_tool_calls, tools_requested,
                source_file, line_number)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.log_id, record.timestamp, record.model_id,
                record.adapter_name,
                1 if record.tool_calls else 0,
                1 if record.tools_requested else 0,
                record.source_file, record.line_number,
            ),
        )
        await self._conn.commit()
        return record.log_id

    async def insert_logs_batch(
        self, records: list[InferenceLogRecord],
    ) -> int:
        """Batch-insert inference log indexes."""
        rows = [
            (
                r.log_id, r.timestamp, r.model_id, r.adapter_name,
                1 if r.tool_calls else 0,
                1 if r.tools_requested else 0,
                r.source_file, r.line_number,
            )
            for r in records
        ]
        await self._conn.executemany(
            """INSERT OR IGNORE INTO inference_logs
               (log_id, timestamp, model_id, adapter_name,
                has_tool_calls, tools_requested,
                source_file, line_number)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        await self._conn.commit()
        return len(rows)

    async def find_logs(self, filters: LogFilter) -> list[InferenceLogRecord]:
        """Query logs matching filter criteria."""
        sql, params = self._build_query("*", filters)
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        columns = [d[0] for d in cursor.description]
        return [self._row_to_record(dict(zip(columns, row))) for row in rows]

    async def count_logs(self, filters: LogFilter) -> int:
        """Count logs matching filter criteria."""
        sql, params = self._build_query("COUNT(*)", filters)
        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def update_score(
        self, log_id: str, fitness_score: float, is_valid: bool, errors: list[str],
    ) -> None:
        """Update fitness score and validation status."""
        await self._conn.execute(
            """UPDATE inference_logs
               SET fitness_score = ?, is_valid = ?
               WHERE log_id = ?""",
            (fitness_score, 1 if is_valid else 0, log_id),
        )
        await self._conn.commit()

    async def update_tag(
        self, log_id: str, tag: str, tag_source: str,
    ) -> None:
        """Update the training tag for a log entry."""
        await self._conn.execute(
            """UPDATE inference_logs
               SET tag = ?, tag_source = ?
               WHERE log_id = ?""",
            (tag, tag_source, log_id),
        )
        await self._conn.commit()

    async def mark_used(
        self, log_ids: list[str], dataset_version: str,
    ) -> None:
        """Mark logs as consumed by a dataset version."""
        if not log_ids:
            return
        placeholders = ",".join("?" for _ in log_ids)
        await self._conn.execute(
            f"""UPDATE inference_logs
                SET dataset_version = ?
                WHERE log_id IN ({placeholders})""",
            [dataset_version] + list(log_ids),
        )
        await self._conn.commit()

    async def create_dataset_version(self, version: DatasetVersion) -> str:
        """Store a dataset version manifest."""
        await self._conn.execute(
            """INSERT INTO dataset_versions
               (version_id, created_at, source_model_id, record_counts,
                file_paths, content_hash, parent_version,
                filter_criteria, training_run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version.version_id, version.created_at,
                version.source_model_id,
                json.dumps(version.record_counts),
                json.dumps(version.file_paths),
                version.content_hash, version.parent_version,
                json.dumps(version.filter_criteria),
                version.training_run_id,
            ),
        )
        await self._conn.commit()
        return version.version_id

    async def get_dataset_version(
        self, version_id: str,
    ) -> DatasetVersion | None:
        """Retrieve a dataset version manifest by ID."""
        cursor = await self._conn.execute(
            "SELECT * FROM dataset_versions WHERE version_id = ?",
            (version_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        columns = [d[0] for d in cursor.description]
        return self._row_to_version(dict(zip(columns, row)))

    async def get_latest_dataset_version(self) -> DatasetVersion | None:
        """Retrieve the most recent dataset version."""
        cursor = await self._conn.execute(
            "SELECT * FROM dataset_versions ORDER BY created_at DESC LIMIT 1",
        )
        row = await cursor.fetchone()
        if not row:
            return None
        columns = [d[0] for d in cursor.description]
        return self._row_to_version(dict(zip(columns, row)))

    # -- Internal helpers ---------------------------------------------------

    def _build_query(
        self, select: str, filters: LogFilter,
    ) -> tuple[str, list]:
        """Build SQL query from LogFilter."""
        clauses: list[str] = []
        params: list[Any] = []

        if filters.since:
            clauses.append("timestamp >= ?")
            params.append(filters.since)
        if filters.until:
            clauses.append("timestamp <= ?")
            params.append(filters.until)
        if filters.model_id:
            clauses.append("model_id = ?")
            params.append(filters.model_id)
        if filters.tag is not None:
            if isinstance(filters.tag, list):
                placeholders = ",".join("?" for _ in filters.tag)
                clauses.append(f"tag IN ({placeholders})")
                params.extend(filters.tag)
            else:
                clauses.append("tag = ?")
                params.append(filters.tag)
        if filters.min_score is not None:
            clauses.append("fitness_score >= ?")
            params.append(filters.min_score)
        if filters.max_score is not None:
            clauses.append("fitness_score <= ?")
            params.append(filters.max_score)
        if filters.is_valid is not None:
            clauses.append("is_valid = ?")
            params.append(1 if filters.is_valid else 0)
        if filters.has_tool_calls is not None:
            clauses.append("has_tool_calls = ?")
            params.append(1 if filters.has_tool_calls else 0)
        if filters.unscored_only:
            clauses.append("fitness_score IS NULL")
        if filters.untagged_only:
            clauses.append("tag IS NULL")
        if filters.unused_only:
            clauses.append("dataset_version IS NULL")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        limit_sql = f" LIMIT {filters.limit}" if filters.limit else ""

        sql = f"SELECT {select} FROM inference_logs{where} ORDER BY timestamp{limit_sql}"
        return sql, params

    @staticmethod
    def _row_to_record(row: dict) -> InferenceLogRecord:
        """Convert a database row to an InferenceLogRecord."""
        return InferenceLogRecord(
            log_id=row["log_id"],
            timestamp=row["timestamp"],
            model_id=row["model_id"],
            adapter_name=row.get("adapter_name"),
            tools_requested=bool(row.get("tools_requested", 0)),
            tool_calls=[{}] if row.get("has_tool_calls") else [],
            fitness_score=row.get("fitness_score"),
            is_valid=bool(row["is_valid"]) if row.get("is_valid") is not None else None,
            tag=row.get("tag"),
            dataset_version=row.get("dataset_version"),
            source_file=row.get("source_file", ""),
            line_number=row.get("line_number", 0),
        )

    @staticmethod
    def _row_to_version(row: dict) -> DatasetVersion:
        """Convert a database row to a DatasetVersion."""
        return DatasetVersion(
            version_id=row["version_id"],
            created_at=row["created_at"],
            source_model_id=row["source_model_id"],
            record_counts=json.loads(row["record_counts"]),
            file_paths=json.loads(row["file_paths"]),
            content_hash=row["content_hash"],
            parent_version=row.get("parent_version"),
            filter_criteria=json.loads(row.get("filter_criteria") or "{}"),
            training_run_id=row.get("training_run_id"),
        )
