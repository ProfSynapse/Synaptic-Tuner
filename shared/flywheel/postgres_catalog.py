"""
shared/flywheel/postgres_catalog.py

PostgresLogCatalog: cloud Postgres implementation of LogCatalog using asyncpg.
Schema-per-tenant isolation for multi-tenant deployments. Connection pooling
via asyncpg.Pool for concurrent proxy requests.

Used by: catalog.py (create_catalog factory)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .catalog import DatasetVersion, InferenceLogRecord, LogFilter

logger = logging.getLogger(__name__)


class PostgresLogCatalog:
    """Cloud Postgres implementation using asyncpg.

    Schema-per-tenant isolation: all tables live under tenant_{id} schema.
    Connection pooling via asyncpg.Pool for concurrent proxy requests.

    Args:
        dsn: PostgreSQL connection string
        tenant_id: Tenant identifier for schema isolation
        pool_size: Connection pool size (default 10)
    """

    def __init__(
        self, dsn: str, tenant_id: str, pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._tenant_id = tenant_id
        self._pool_size = pool_size
        self._pool: Any = None  # asyncpg.Pool
        self._schema = f"tenant_{tenant_id}"

    async def initialize(self) -> None:
        """Create schema and tables if they don't exist."""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=self._pool_size,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._schema}.inference_logs (
                    log_id          TEXT PRIMARY KEY,
                    timestamp       TEXT NOT NULL,
                    model_id        TEXT NOT NULL,
                    adapter_name    TEXT,
                    has_tool_calls  BOOLEAN NOT NULL DEFAULT FALSE,
                    tools_requested BOOLEAN NOT NULL DEFAULT FALSE,
                    fitness_score   DOUBLE PRECISION,
                    is_valid        BOOLEAN,
                    tag             TEXT,
                    tag_source      TEXT,
                    dataset_version TEXT,
                    source_file     TEXT NOT NULL,
                    line_number     INTEGER NOT NULL,
                    tenant_id       TEXT NOT NULL DEFAULT '{self._tenant_id}',
                    created_at      TEXT NOT NULL DEFAULT NOW()::TEXT
                )
            """)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._schema}.dataset_versions (
                    version_id      TEXT PRIMARY KEY,
                    created_at      TEXT NOT NULL,
                    source_model_id TEXT NOT NULL,
                    record_counts   JSONB NOT NULL,
                    file_paths      JSONB NOT NULL,
                    content_hash    TEXT NOT NULL,
                    parent_version  TEXT,
                    filter_criteria JSONB,
                    training_run_id TEXT
                )
            """)
            # Create indexes
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_logs_tag "
                f"ON {self._schema}.inference_logs(tag)"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_logs_score "
                f"ON {self._schema}.inference_logs(fitness_score)"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_logs_timestamp "
                f"ON {self._schema}.inference_logs(timestamp)"
            )

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def insert_log(self, record: InferenceLogRecord) -> str:
        """Index a single inference log record."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {self._schema}.inference_logs
                    (log_id, timestamp, model_id, adapter_name,
                     has_tool_calls, tools_requested,
                     source_file, line_number, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (log_id) DO NOTHING""",
                record.log_id, record.timestamp, record.model_id,
                record.adapter_name,
                bool(record.tool_calls), record.tools_requested,
                record.source_file, record.line_number,
                self._tenant_id,
            )
        return record.log_id

    async def insert_logs_batch(
        self, records: list[InferenceLogRecord],
    ) -> int:
        """Batch-insert inference log indexes."""
        rows = [
            (
                r.log_id, r.timestamp, r.model_id, r.adapter_name,
                bool(r.tool_calls), r.tools_requested,
                r.source_file, r.line_number, self._tenant_id,
            )
            for r in records
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""INSERT INTO {self._schema}.inference_logs
                    (log_id, timestamp, model_id, adapter_name,
                     has_tool_calls, tools_requested,
                     source_file, line_number, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (log_id) DO NOTHING""",
                rows,
            )
        return len(rows)

    async def find_logs(self, filters: LogFilter) -> list[InferenceLogRecord]:
        """Query logs matching filter criteria."""
        sql, params = self._build_query("*", filters)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [self._row_to_record(dict(r)) for r in rows]

    async def count_logs(self, filters: LogFilter) -> int:
        """Count logs matching filter criteria."""
        sql, params = self._build_query("COUNT(*)", filters)
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(sql, *params)
        return row or 0

    async def update_score(
        self, log_id: str, fitness_score: float, is_valid: bool, errors: list[str],
    ) -> None:
        """Update fitness score and validation status."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""UPDATE {self._schema}.inference_logs
                    SET fitness_score = $1, is_valid = $2
                    WHERE log_id = $3""",
                fitness_score, is_valid, log_id,
            )

    async def update_tag(
        self, log_id: str, tag: str, tag_source: str,
    ) -> None:
        """Update the training tag for a log entry."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""UPDATE {self._schema}.inference_logs
                    SET tag = $1, tag_source = $2
                    WHERE log_id = $3""",
                tag, tag_source, log_id,
            )

    async def mark_used(
        self, log_ids: list[str], dataset_version: str,
    ) -> None:
        """Mark logs as consumed by a dataset version."""
        if not log_ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""UPDATE {self._schema}.inference_logs
                    SET dataset_version = $1
                    WHERE log_id = ANY($2::TEXT[])""",
                dataset_version, log_ids,
            )

    async def create_dataset_version(self, version: DatasetVersion) -> str:
        """Store a dataset version manifest."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {self._schema}.dataset_versions
                    (version_id, created_at, source_model_id, record_counts,
                     file_paths, content_hash, parent_version,
                     filter_criteria, training_run_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                version.version_id, version.created_at,
                version.source_model_id,
                json.dumps(version.record_counts),
                json.dumps(version.file_paths),
                version.content_hash, version.parent_version,
                json.dumps(version.filter_criteria),
                version.training_run_id,
            )
        return version.version_id

    async def get_dataset_version(
        self, version_id: str,
    ) -> DatasetVersion | None:
        """Retrieve a dataset version manifest by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._schema}.dataset_versions "
                f"WHERE version_id = $1",
                version_id,
            )
        if not row:
            return None
        return self._row_to_version(dict(row))

    async def get_latest_dataset_version(self) -> DatasetVersion | None:
        """Retrieve the most recent dataset version."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._schema}.dataset_versions "
                f"ORDER BY created_at DESC LIMIT 1",
            )
        if not row:
            return None
        return self._row_to_version(dict(row))

    # -- Internal helpers ---------------------------------------------------

    def _build_query(
        self, select: str, filters: LogFilter,
    ) -> tuple[str, list]:
        """Build Postgres query from LogFilter (uses $N placeholders)."""
        clauses: list[str] = []
        params: list[Any] = []
        idx = 1

        if filters.since:
            clauses.append(f"timestamp >= ${idx}")
            params.append(filters.since)
            idx += 1
        if filters.until:
            clauses.append(f"timestamp <= ${idx}")
            params.append(filters.until)
            idx += 1
        if filters.model_id:
            clauses.append(f"model_id = ${idx}")
            params.append(filters.model_id)
            idx += 1
        if filters.tag is not None:
            if isinstance(filters.tag, list):
                clauses.append(f"tag = ANY(${idx}::TEXT[])")
                params.append(filters.tag)
            else:
                clauses.append(f"tag = ${idx}")
                params.append(filters.tag)
            idx += 1
        if filters.min_score is not None:
            clauses.append(f"fitness_score >= ${idx}")
            params.append(filters.min_score)
            idx += 1
        if filters.max_score is not None:
            clauses.append(f"fitness_score <= ${idx}")
            params.append(filters.max_score)
            idx += 1
        if filters.is_valid is not None:
            clauses.append(f"is_valid = ${idx}")
            params.append(filters.is_valid)
            idx += 1
        if filters.has_tool_calls is not None:
            clauses.append(f"has_tool_calls = ${idx}")
            params.append(filters.has_tool_calls)
            idx += 1
        if filters.unscored_only:
            clauses.append("fitness_score IS NULL")
        if filters.untagged_only:
            clauses.append("tag IS NULL")
        if filters.unused_only:
            clauses.append("dataset_version IS NULL")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        limit_sql = f" LIMIT {filters.limit}" if filters.limit else ""

        table = f"{self._schema}.inference_logs"
        sql = f"SELECT {select} FROM {table}{where} ORDER BY timestamp{limit_sql}"
        return sql, params

    @staticmethod
    def _row_to_record(row: dict) -> InferenceLogRecord:
        """Convert a database row to an InferenceLogRecord."""
        return InferenceLogRecord(
            log_id=row["log_id"],
            timestamp=row["timestamp"],
            model_id=row["model_id"],
            adapter_name=row.get("adapter_name"),
            tools_requested=bool(row.get("tools_requested", False)),
            tool_calls=[{}] if row.get("has_tool_calls") else [],
            fitness_score=row.get("fitness_score"),
            is_valid=row["is_valid"] if row.get("is_valid") is not None else None,
            tag=row.get("tag"),
            dataset_version=row.get("dataset_version"),
            source_file=row.get("source_file", ""),
            line_number=row.get("line_number", 0),
        )

    @staticmethod
    def _row_to_version(row: dict) -> DatasetVersion:
        """Convert a database row to a DatasetVersion."""
        rc = row["record_counts"]
        fp = row["file_paths"]
        fc = row.get("filter_criteria")
        return DatasetVersion(
            version_id=row["version_id"],
            created_at=row["created_at"],
            source_model_id=row["source_model_id"],
            record_counts=rc if isinstance(rc, dict) else json.loads(rc),
            file_paths=fp if isinstance(fp, dict) else json.loads(fp),
            content_hash=row["content_hash"],
            parent_version=row.get("parent_version"),
            filter_criteria=(
                fc if isinstance(fc, dict)
                else json.loads(fc) if fc else {}
            ),
            training_run_id=row.get("training_run_id"),
        )
