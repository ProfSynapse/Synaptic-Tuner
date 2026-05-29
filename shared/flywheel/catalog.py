"""
shared/flywheel/catalog.py

LogCatalog Protocol and data structures for the inference log index.
Provides SQLiteLogCatalog (aiosqlite) for local use and PostgresLogCatalog
(asyncpg) for cloud multi-tenant deployments. Factory function create_catalog()
selects the backend based on config.

Used by: inference_logger.py, cleaner.py, tagger.py, stager.py, orchestrator.py
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InferenceLogRecord:
    """A single inference request/response pair captured by the proxy."""

    log_id: str
    timestamp: str
    model_id: str
    adapter_name: str | None = None

    # Request
    messages: list[dict[str, str]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 1024
    tools: list[dict] = field(default_factory=list)
    tools_requested: bool = False

    # Response
    response_content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Token-faithful rollout capture (optional; only populated when the proxy
    # requested logprobs and the response carried them — see capture_token_ids).
    # These ride in the JSONL only; the catalog index does not store them.
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    completion_logprobs: list[float] | None = None

    # Pipeline metadata (populated during processing)
    latency_ms: float = 0.0
    fitness_score: float | None = None
    is_valid: bool | None = None
    tag: str | None = None
    dataset_version: str | None = None
    errors: list[str] = field(default_factory=list)

    # Tenant (reserved for cloud multi-tenancy)
    tenant_id: str | None = None

    # Source file pointer (for content retrieval)
    source_file: str = ""
    line_number: int = 0

    # Optional token-capture fields are omitted from the JSONL when unset, so
    # records without rollout capture stay byte-compatible with existing readers.
    _OPTIONAL_TOKEN_FIELDS = (
        "prompt_token_ids",
        "completion_token_ids",
        "completion_logprobs",
    )

    def to_json(self) -> str:
        """Serialize to JSON string (omitting unset token-capture fields)."""
        data = asdict(self)
        for key in self._OPTIONAL_TOKEN_FIELDS:
            if data.get(key) is None:
                data.pop(key, None)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict) -> InferenceLogRecord:
        """Deserialize from dict, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class LogFilter:
    """Filter criteria for querying inference logs."""

    since: str | None = None
    until: str | None = None
    model_id: str | None = None
    tag: str | list[str] | None = None
    min_score: float | None = None
    max_score: float | None = None
    is_valid: bool | None = None
    dataset_version: str | None = None
    has_tool_calls: bool | None = None
    unscored_only: bool = False
    untagged_only: bool = False
    unused_only: bool = False
    limit: int | None = None


@dataclass
class DatasetVersion:
    """Metadata for a staged dataset version."""

    version_id: str
    created_at: str
    source_model_id: str
    record_counts: dict[str, int] = field(default_factory=dict)
    file_paths: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""
    parent_version: str | None = None
    filter_criteria: dict = field(default_factory=dict)
    training_run_id: str | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DatasetVersion:
        """Deserialize from dict, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LogCatalog(Protocol):
    """Async interface for the inference log catalog.

    Two implementations:
    - SQLiteLogCatalog: local, zero-config, aiosqlite
    - PostgresLogCatalog: cloud, multi-tenant, asyncpg
    """

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def insert_log(self, record: InferenceLogRecord) -> str: ...
    async def insert_logs_batch(self, records: list[InferenceLogRecord]) -> int: ...
    async def find_logs(self, filters: LogFilter) -> list[InferenceLogRecord]: ...
    async def count_logs(self, filters: LogFilter) -> int: ...
    async def avg_score(self, filters: LogFilter) -> float: ...
    async def update_score(
        self, log_id: str, fitness_score: float, is_valid: bool, errors: list[str],
    ) -> None: ...
    async def update_tag(self, log_id: str, tag: str, tag_source: str) -> None: ...
    async def mark_used(self, log_ids: list[str], dataset_version: str) -> None: ...
    async def create_dataset_version(self, version: DatasetVersion) -> str: ...
    async def get_dataset_version(self, version_id: str) -> DatasetVersion | None: ...
    async def get_latest_dataset_version(self) -> DatasetVersion | None: ...


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

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
        self._conn: Any = None  # aiosqlite.Connection

    async def initialize(self) -> None:
        """Create tables if they don't exist. Called once at startup."""
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
                record.log_id,
                record.timestamp,
                record.model_id,
                record.adapter_name,
                1 if record.tool_calls else 0,
                1 if record.tools_requested else 0,
                record.source_file,
                record.line_number,
            ),
        )
        await self._conn.commit()
        return record.log_id

    async def insert_logs_batch(
        self, records: list[InferenceLogRecord],
    ) -> int:
        """Batch-insert inference log indexes. Returns count inserted."""
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

    async def avg_score(self, filters: LogFilter) -> float:
        """Average fitness_score for logs matching filters (ignoring NULLs)."""
        # Build query without LIMIT so we average across all matching logs
        no_limit = LogFilter(
            since=filters.since, until=filters.until,
            model_id=filters.model_id, tag=filters.tag,
            min_score=filters.min_score, max_score=filters.max_score,
            is_valid=filters.is_valid, has_tool_calls=filters.has_tool_calls,
            unscored_only=filters.unscored_only,
            untagged_only=filters.untagged_only,
            unused_only=filters.unused_only,
            dataset_version=filters.dataset_version,
            limit=None,
        )
        sql, params = self._build_query(
            "AVG(fitness_score)", no_limit,
        )
        # Replace ORDER BY clause — AVG doesn't need ordering
        sql = sql.replace(" ORDER BY timestamp", "")
        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

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
                version.version_id,
                version.created_at,
                version.source_model_id,
                json.dumps(version.record_counts),
                json.dumps(version.file_paths),
                version.content_hash,
                version.parent_version,
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
        limit = f" LIMIT {filters.limit}" if filters.limit else ""

        sql = f"SELECT {select} FROM inference_logs{where} ORDER BY timestamp{limit}"
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


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------

class PostgresLogCatalog:
    """Cloud Postgres implementation using asyncpg.

    Schema-per-tenant isolation: all tables live under tenant_{id} schema.
    Connection pooling via asyncpg.Pool for concurrent proxy requests.

    Args:
        dsn: PostgreSQL connection string
        tenant_id: Tenant identifier for schema isolation
        pool_size: Connection pool size (default 10)
    """

    # Strict allowlist for tenant_id to prevent SQL injection in schema names.
    # Identifiers cannot be parameterized ($1) in Postgres, so we validate instead.
    _TENANT_ID_RE = re.compile(r"^[a-zA-Z0-9_]+$")

    def __init__(
        self, dsn: str, tenant_id: str, pool_size: int = 10,
    ) -> None:
        if not self._TENANT_ID_RE.match(tenant_id):
            raise ValueError(
                f"Invalid tenant_id: {tenant_id!r} "
                "(must match ^[a-zA-Z0-9_]+$)"
            )
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

    async def avg_score(self, filters: LogFilter) -> float:
        """Average fitness_score for logs matching filters (ignoring NULLs)."""
        no_limit = LogFilter(
            since=filters.since, until=filters.until,
            model_id=filters.model_id, tag=filters.tag,
            min_score=filters.min_score, max_score=filters.max_score,
            is_valid=filters.is_valid, has_tool_calls=filters.has_tool_calls,
            unscored_only=filters.unscored_only,
            untagged_only=filters.untagged_only,
            unused_only=filters.unused_only,
            dataset_version=filters.dataset_version,
            limit=None,
        )
        sql, params = self._build_query("AVG(fitness_score)", no_limit)
        sql = sql.replace(" ORDER BY timestamp", "")
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(sql, *params)
        return float(row) if row is not None else 0.0

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def create_catalog(
    backend: str = "sqlite",
    *,
    path: str | Path | None = None,
    url: str | None = None,
    tenant_id: str | None = None,
    pool_size: int = 10,
) -> LogCatalog:
    """Factory function to create the appropriate LogCatalog backend.

    Args:
        backend: "sqlite" (default) or "postgres"
        path: SQLite database file path (required for sqlite backend)
        url: PostgreSQL connection URL (required for postgres backend)
        tenant_id: Tenant ID for Postgres schema isolation
        pool_size: Postgres connection pool size

    Returns:
        Initialized LogCatalog instance

    Raises:
        ValueError: If required arguments are missing for the chosen backend
    """
    if backend == "sqlite":
        db_path = Path(path) if path else Path(".tracking/flywheel.db")
        catalog = SQLiteLogCatalog(db_path)
        await catalog.initialize()
        return catalog

    if backend == "postgres":
        if not url:
            raise ValueError("Postgres backend requires 'url' parameter")
        if not tenant_id:
            raise ValueError("Postgres backend requires 'tenant_id' parameter")
        catalog = PostgresLogCatalog(url, tenant_id, pool_size)
        await catalog.initialize()
        return catalog

    raise ValueError(f"Unknown catalog backend: {backend!r}")
