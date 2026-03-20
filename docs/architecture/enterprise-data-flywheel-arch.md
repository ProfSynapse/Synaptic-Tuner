# Enterprise Data Flywheel -- Architecture Specification

> **Date**: 2026-03-20
> **Phase**: ARCHITECT
> **Plan Reference**: `docs/plans/enterprise-data-flywheel-plan.md` (APPROVED)
> **Preparation Reference**: `docs/preparation/enterprise-data-flywheel-prep.md`

---

## 1. Executive Summary

The Enterprise Data Flywheel is a self-reinforcing fine-tuning loop that captures inference logs from a vLLM-served model, cleans and tags them by quality, stages versioned training datasets, and optionally triggers retraining. The architecture adds a `shared/flywheel/` pipeline package, a FastAPI logging proxy (`services/proxy/`), and a `tuner flywheel` CLI subcommand. It integrates deeply with existing components: `shared/validation/fitness.py` for quality scoring, `shared/judge/` for LLM-based tagging of ambiguous examples, and `shared/experiment_tracking/` for run lineage.

**Design principles applied**: Single Responsibility (each module owns one pipeline stage), Dependency Inversion (LogCatalog is a Protocol, backends are pluggable), Open/Closed (tagger rules are config-driven, new tag strategies can be added without modifying core), KISS (SQLite + JSONL for local, Postgres for cloud -- no new infrastructure services).

---

## 2. System Context (C4 Level 1)

```
                    +------------------+
                    |   User / App     |
                    | (OpenAI client)  |
                    +--------+---------+
                             | OpenAI API (port 8080)
                             v
+------------------+   +-----+------+   +------------------+
|  tuner CLI       |   |  Logging   |   |    vLLM          |
|  (flywheel cmds) |   |  Proxy     +-->|    :8000          |
+--------+---------+   |  :8080     |   |  (GPU, LoRA)     |
         |             +-----+------+   +------------------+
         |                   |
         v                   v
+--------+-------------------+--------+
|        shared/flywheel/             |
|  (pipeline: ingest, clean, tag,    |
|   stage, orchestrate)              |
+--------+---------------------------+
         |            |            |
         v            v            v
+--------+--+ +------+------+ +---+------------+
| LogCatalog | | RunRegistry | | Datasets/      |
| (SQLite/   | | (JSONL)     | | flywheel/vN/   |
|  Postgres) | +-------------+ +----------------+
+------------+
```

**External boundaries**:
- **Inbound**: Any OpenAI-compatible client sends requests to the proxy on port 8080
- **Outbound**: vLLM serves inference on port 8000; Trainers/ execute fine-tuning
- **Storage**: SQLite (local) or Postgres (cloud) for log catalog; JSONL files for datasets and RunRegistry

---

## 3. Component Architecture (C4 Level 3)

### 3.1 Package Layout

```
shared/flywheel/
    __init__.py
    config.py           # FlywheelConfig dataclass
    inference_logger.py  # InferenceLogger (async JSONL writer)
    catalog.py          # LogCatalog Protocol + SQLiteLogCatalog + PostgresLogCatalog + factory
    cleaner.py          # DataCleaner (FitnessEvaluator integration)
    tagger.py           # AutoTagger (rule-based + LLM judge fallback)
    stager.py           # DatasetStager (assemble + version + register)
    orchestrator.py     # FlywheelOrchestrator (pipeline + GPU mutex + retrain)

services/proxy/
    app.py              # FastAPI logging proxy
    __init__.py

tuner/handlers/
    flywheel_handler.py # CLI handler

configs/flywheel/
    default.yaml        # Default configuration
```

### 3.2 Component Interaction Diagram

```
[LoggingProxy]
    |
    | writes InferenceLogRecord
    v
[InferenceLogger] --> inference_logs/YYYY-MM-DD.jsonl
    |
    | inserts into
    v
[LogCatalog]  <-- queried by all downstream
    |
    | find_logs(unused, min_score)
    v
[DataCleaner]
    |  FitnessEvaluator.evaluate()
    |  updates score + is_valid in catalog
    v
[AutoTagger]
    |  rule-based: score thresholds
    |  fallback: JudgeService for ambiguous
    |  updates tag in catalog
    v
[DatasetStager]
    |  assembles JSONL by tag (sft/kto/grpo)
    |  creates DatasetVersion in catalog
    |  registers flywheel_cycle in RunRegistry
    v
[FlywheelOrchestrator]
    |  ReadinessChecker: enough examples?
    |  GPU mutex: stop vLLM -> train -> restart
    |  invokes Trainers/ via subprocess
    v
[vLLM hot-swap or restart]
```

---

## 4. Data Architecture

### 4.1 Core Data Structures

#### InferenceLogRecord

```python
@dataclass
class InferenceLogRecord:
    """A single inference request/response pair captured by the proxy."""

    log_id: str                          # UUID4
    timestamp: str                       # ISO 8601 UTC
    model_id: str                        # e.g. "org/model-7b"
    adapter_name: str | None = None      # LoRA adapter name if used

    # Request
    messages: list[dict[str, str]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 1024
    tools: list[dict] = field(default_factory=list)
    tools_requested: bool = False        # True if `tools` was non-empty in request
                                         # CRITICAL: controls scoring path in DataCleaner/AutoTagger

    # Response
    response_content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Pipeline metadata (populated during processing)
    latency_ms: float = 0.0
    fitness_score: float | None = None   # Set by DataCleaner
    is_valid: bool | None = None         # Set by DataCleaner
    tag: str | None = None               # Set by AutoTagger: "sft" | "kto" | "grpo" | "discard"
    dataset_version: str | None = None   # Set by DatasetStager
    errors: list[str] = field(default_factory=list)

    # Tenant (reserved for cloud multi-tenancy)
    tenant_id: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> InferenceLogRecord:
        """Deserialize from dict, ignoring unknown fields."""
        ...
```

#### LogFilter

```python
@dataclass
class LogFilter:
    """Filter criteria for querying inference logs."""

    since: str | None = None             # ISO 8601 -- logs at or after
    until: str | None = None             # ISO 8601 -- logs at or before
    model_id: str | None = None
    tag: str | list[str] | None = None   # "sft", "kto", "grpo", "discard", None (untagged)
    min_score: float | None = None
    max_score: float | None = None
    is_valid: bool | None = None
    dataset_version: str | None = None   # None = unused logs only
    has_tool_calls: bool | None = None
    limit: int | None = None
```

#### DatasetVersion

```python
@dataclass
class DatasetVersion:
    """Metadata for a staged dataset version."""

    version_id: str                      # e.g. "v003"
    created_at: str                      # ISO 8601 UTC
    source_model_id: str
    record_counts: dict[str, int]        # {"sft": 612, "kto_pos": 180, "kto_neg": 55, "grpo": 23}
    file_paths: dict[str, str]           # {"sft": "Datasets/flywheel/v003/sft_training.jsonl", ...}
    content_hash: str                    # SHA-256 of concatenated dataset files
    parent_version: str | None = None
    filter_criteria: dict = field(default_factory=dict)
    training_run_id: str | None = None   # Set after retrain completes

    def to_dict(self) -> dict:
        ...

    @classmethod
    def from_dict(cls, data: dict) -> DatasetVersion:
        ...
```

#### TaggedExample

```python
@dataclass
class TaggedExample:
    """An inference log with its assigned training tag and formatted for output."""

    log_id: str
    tag: str                             # "sft" | "kto" | "grpo"
    conversations: list[dict[str, str]]  # ChatML format
    label: bool | None = None            # True for SFT/KTO-pos, False for KTO-neg, None for GRPO
    reward: float | None = None          # For GRPO: fitness score as reward signal
    fitness_score: float = 0.0
    tag_source: str = "rule"             # "rule" | "judge"
```

### 4.2 Storage Schema

#### SQLite Tables (local -- `.tracking/flywheel.db`)

```sql
-- Inference log index (actual content stays in JSONL files)
CREATE TABLE inference_logs (
    log_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,        -- ISO 8601
    model_id        TEXT NOT NULL,
    adapter_name    TEXT,
    has_tool_calls  INTEGER NOT NULL DEFAULT 0,  -- 0/1
    tools_requested INTEGER NOT NULL DEFAULT 0,  -- 0/1: were tools in the request?
    fitness_score   REAL,                 -- NULL = not yet scored
    is_valid        INTEGER,              -- NULL = not yet scored, 0/1
    tag             TEXT,                 -- NULL = untagged, "sft"/"kto"/"grpo"/"discard"
    tag_source      TEXT,                 -- "rule"/"judge"
    dataset_version TEXT,                 -- NULL = unused
    source_file     TEXT NOT NULL,        -- JSONL filename for log retrieval
    line_number     INTEGER NOT NULL,     -- Line offset in source_file
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_logs_tag ON inference_logs(tag);
CREATE INDEX idx_logs_score ON inference_logs(fitness_score);
CREATE INDEX idx_logs_unused ON inference_logs(dataset_version) WHERE dataset_version IS NULL;
CREATE INDEX idx_logs_timestamp ON inference_logs(timestamp);

-- Dataset version manifests
CREATE TABLE dataset_versions (
    version_id      TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    source_model_id TEXT NOT NULL,
    record_counts   TEXT NOT NULL,        -- JSON: {"sft": 612, "kto_pos": 180, ...}
    file_paths      TEXT NOT NULL,        -- JSON: {"sft": "path", "kto": "path", ...}
    content_hash    TEXT NOT NULL,
    parent_version  TEXT,
    filter_criteria TEXT,                 -- JSON
    training_run_id TEXT,                 -- Set after retrain
    FOREIGN KEY (parent_version) REFERENCES dataset_versions(version_id)
);
```

#### Postgres Tables (cloud -- schema per tenant)

```sql
-- Created per tenant: CREATE SCHEMA tenant_{id};

CREATE TABLE tenant_{id}.inference_logs (
    -- Same columns as SQLite, plus:
    tenant_id       TEXT NOT NULL DEFAULT '{id}',
    -- Postgres-specific: use JSONB for tools/messages if needed for querying
    request_tools   JSONB,
    -- Partitioned by month for large-scale deployments
    PRIMARY KEY (log_id, timestamp)
) PARTITION BY RANGE (timestamp);

CREATE TABLE tenant_{id}.dataset_versions (
    -- Same as SQLite schema
);
```

### 4.3 Data Flow

```
inference_logs/2026-03-20.jsonl    (raw JSONL, written by InferenceLogger)
         |
         v  [DataCleaner reads JSONL, scores via FitnessEvaluator]
         |
         v  [AutoTagger reads scored logs, assigns tags]
         |
         v  [DatasetStager reads tagged logs, writes versioned datasets]
         |
Datasets/flywheel/v003/
    sft_training.jsonl             (label: true, score >= 0.8)
    kto_training.jsonl             (mixed: true for pos, false for neg)
    grpo_training.jsonl            (reward field set from fitness score)
    manifest.json                  (DatasetVersion metadata)
```

### 4.4 Lineage Chain (RunRegistry)

```
inference_log_batch (ingestion run)
    |
    +--[link_runs]--> flywheel_cycle (tagging + staging run)
                          |
                          +--[link_runs]--> sft / kto / grpo training run
                                                |
                                                +--[link_runs]--> evaluation run
```

All lineage is tracked via existing `RunRegistry.link_runs()`. The `dataset_source` field on training RunRecords stores the version identifier (e.g., `"flywheel/v003"`).

---

## 5. Interface Specifications

### 5.1 InferenceLogger (`shared/flywheel/inference_logger.py`)

```python
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class InferenceLogger:
    """Async JSONL writer for inference request/response pairs.

    Captures OpenAI-format chat completion requests and responses,
    writes them as InferenceLogRecord entries to date-partitioned JSONL files,
    and indexes them in the LogCatalog for downstream pipeline queries.

    Thread-safe via asyncio.Queue. Non-blocking: callers enqueue log entries
    and return immediately. A background task drains the queue and writes.

    Args:
        log_dir: Directory for JSONL files (e.g., "inference_logs/")
        catalog: LogCatalog instance for indexing
        config: FlywheelConfig for rotation settings
        enabled: When False, all log calls are no-ops (zero overhead)
    """

    def __init__(
        self,
        log_dir: Path,
        catalog: LogCatalog,
        config: FlywheelConfig,
        enabled: bool = True,
    ) -> None: ...

    async def start(self) -> None:
        """Start the background writer task. Call once at proxy startup."""
        ...

    async def stop(self) -> None:
        """Flush pending writes and stop the background task. Call at shutdown."""
        ...

    async def log_inference(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
        model_id: str,
        adapter_name: str | None = None,
    ) -> str:
        """Enqueue an inference log entry. Returns the log_id immediately.

        Args:
            request: OpenAI chat completion request body
            response: OpenAI chat completion response body
            latency_ms: End-to-end latency in milliseconds
            model_id: Model identifier from the request
            adapter_name: LoRA adapter name if applicable

        Returns:
            The generated log_id (UUID4 string)
        """
        ...

    def _build_record(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
        model_id: str,
        adapter_name: str | None,
    ) -> InferenceLogRecord:
        """Extract fields from OpenAI-format request/response into InferenceLogRecord."""
        ...

    def _get_log_file(self) -> Path:
        """Return today's JSONL file path (date-partitioned)."""
        ...

    async def _writer_loop(self) -> None:
        """Background loop: drain queue, batch-write to JSONL, index in catalog."""
        ...
```

**Key design decisions**:
- Uses `asyncio.Queue` (not `threading.Lock`) since the proxy is async (FastAPI/uvicorn)
- Date-partitioned JSONL files (`inference_logs/2026-03-20.jsonl`) for natural rotation
- Background writer batches writes (configurable `flush_interval_seconds`, default 1.0)
- Catalog indexing happens in the writer loop after JSONL write succeeds

### 5.2 LoggingProxy (`services/proxy/app.py`)

```python
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response

from shared.flywheel.catalog import create_catalog
from shared.flywheel.config import FlywheelConfig, load_flywheel_config
from shared.flywheel.inference_logger import InferenceLogger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the proxy."""
    config = load_flywheel_config()
    catalog = await create_catalog(
        backend=config.catalog_backend,
        path=config.catalog_path,
        url=config.catalog_url,
        tenant_id=config.tenant_id,
    )
    http_client = httpx.AsyncClient(
        base_url=f"http://{config.vllm_host}:{config.vllm_port}",
        timeout=httpx.Timeout(config.proxy_timeout_seconds),
    )
    inference_logger = InferenceLogger(
        log_dir=config.log_dir,
        catalog=catalog,
        config=config,
    )
    await inference_logger.start()

    app.state.config = config
    app.state.catalog = catalog
    app.state.http_client = http_client
    app.state.logger = inference_logger

    yield

    await inference_logger.stop()
    await http_client.aclose()
    await catalog.close()


app = FastAPI(title="Flywheel Logging Proxy", lifespan=lifespan)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str) -> Response:
    """Forward all requests to vLLM, logging chat completions.

    Only POST /v1/chat/completions requests are logged.
    All other requests are forwarded transparently.

    Auth headers are passed through without inspection.
    """
    ...


@app.get("/flywheel/health")
async def health() -> dict:
    """Health check endpoint for the proxy itself."""
    ...


@app.get("/flywheel/stats")
async def stats() -> dict:
    """Return logging statistics (total logged, today's count, queue depth)."""
    ...
```

**Key design decisions**:
- Catch-all route forwards everything to vLLM transparently
- Only `POST /v1/chat/completions` triggers logging (detected by path match)
- Auth headers passed through without inspection or storage
- `/flywheel/health` and `/flywheel/stats` are proxy-specific endpoints (prefixed to avoid collision with vLLM routes)
- `httpx.AsyncClient` for non-blocking forwarding
- Error handling: if logging fails, the response is still returned to the caller (logging must never block inference)

### 5.3 LogCatalog (`shared/flywheel/catalog.py`)

```python
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LogCatalog(Protocol):
    """Async interface for the inference log catalog.

    Two implementations:
    - SQLiteLogCatalog: local, zero-config, aiosqlite
    - PostgresLogCatalog: cloud, multi-tenant, asyncpg
    """

    async def initialize(self) -> None:
        """Create tables if they don't exist. Called once at startup."""
        ...

    async def close(self) -> None:
        """Close database connections. Called at shutdown."""
        ...

    async def insert_log(self, record: InferenceLogRecord) -> str:
        """Index an inference log record. Returns the log_id.

        The record's content is stored in JSONL files; the catalog stores
        only the index (log_id, timestamp, model_id, score, tag, etc.)
        plus a pointer to the source file and line number.
        """
        ...

    async def insert_logs_batch(self, records: list[InferenceLogRecord]) -> int:
        """Batch-insert inference log indexes. Returns count inserted."""
        ...

    async def find_logs(self, filters: LogFilter) -> list[InferenceLogRecord]:
        """Query logs matching filter criteria.

        Returns InferenceLogRecord instances with index fields populated.
        Full content (messages, response) must be read from the JSONL source file.
        """
        ...

    async def count_logs(self, filters: LogFilter) -> int:
        """Count logs matching filter criteria without loading them."""
        ...

    async def update_score(
        self, log_id: str, fitness_score: float, is_valid: bool, errors: list[str]
    ) -> None:
        """Update fitness score and validation status for a log entry."""
        ...

    async def update_tag(
        self, log_id: str, tag: str, tag_source: str
    ) -> None:
        """Update the training tag for a log entry."""
        ...

    async def mark_used(
        self, log_ids: list[str], dataset_version: str
    ) -> None:
        """Mark logs as consumed by a dataset version (prevents reuse)."""
        ...

    async def create_dataset_version(
        self, version: DatasetVersion
    ) -> str:
        """Store a dataset version manifest. Returns version_id."""
        ...

    async def get_dataset_version(
        self, version_id: str
    ) -> DatasetVersion | None:
        """Retrieve a dataset version manifest by ID."""
        ...

    async def get_latest_dataset_version(self) -> DatasetVersion | None:
        """Retrieve the most recent dataset version."""
        ...


class SQLiteLogCatalog:
    """Local SQLite implementation using aiosqlite.

    Uses WAL mode for concurrent read access from dashboard/CLI
    while the pipeline writes. Single writer is fine for local use.

    Args:
        db_path: Path to SQLite database file (e.g., ".tracking/flywheel.db")
    """

    def __init__(self, db_path: Path) -> None: ...
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def insert_log(self, record: InferenceLogRecord) -> str: ...
    async def insert_logs_batch(self, records: list[InferenceLogRecord]) -> int: ...
    async def find_logs(self, filters: LogFilter) -> list[InferenceLogRecord]: ...
    async def count_logs(self, filters: LogFilter) -> int: ...
    async def update_score(self, log_id: str, fitness_score: float, is_valid: bool, errors: list[str]) -> None: ...
    async def update_tag(self, log_id: str, tag: str, tag_source: str) -> None: ...
    async def mark_used(self, log_ids: list[str], dataset_version: str) -> None: ...
    async def create_dataset_version(self, version: DatasetVersion) -> str: ...
    async def get_dataset_version(self, version_id: str) -> DatasetVersion | None: ...
    async def get_latest_dataset_version(self) -> DatasetVersion | None: ...


class PostgresLogCatalog:
    """Cloud Postgres implementation using asyncpg.

    Schema-per-tenant isolation: all tables live under tenant_{id} schema.
    Connection pooling via asyncpg.Pool for concurrent proxy requests.

    Args:
        dsn: PostgreSQL connection string
        tenant_id: Tenant identifier for schema isolation
        pool_size: Connection pool size (default 10)
    """

    def __init__(self, dsn: str, tenant_id: str, pool_size: int = 10) -> None: ...
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    # ... same methods as SQLiteLogCatalog ...


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
    ...
```

### 5.4 DataCleaner (`shared/flywheel/cleaner.py`)

```python
import logging
from typing import Callable

from shared.validation.fitness import FitnessEvaluator, FitnessResult, create_fitness_evaluator

logger = logging.getLogger(__name__)


class PIIDetector(Protocol):
    """Interface for PII detection (future feature).

    v1 implementation is a no-op stub. Designed for extension
    with regex-based or ML-based detectors.
    """

    def detect(self, text: str) -> list[PIIMatch]:
        """Detect PII in text. Returns list of matches with type and span."""
        ...

    def scrub(self, text: str) -> str:
        """Replace detected PII with placeholders."""
        ...


@dataclass
class PIIMatch:
    """A detected PII instance."""
    pii_type: str       # "email", "phone", "ssn", "name", etc.
    start: int
    end: int
    text: str


class NoOpPIIDetector:
    """v1 stub: passes all text through unchanged."""

    def detect(self, text: str) -> list[PIIMatch]:
        return []

    def scrub(self, text: str) -> str:
        return text


class DataCleaner:
    """Scores inference logs using FitnessEvaluator and applies PII scrubbing.

    Reads unscored logs from the catalog, evaluates each with FitnessEvaluator,
    and updates the catalog with fitness_score, is_valid, and errors.

    Args:
        catalog: LogCatalog instance
        config: FlywheelConfig (controls scoring method, max_errors, etc.)
        pii_detector: PII detection implementation (default: NoOpPIIDetector)
    """

    def __init__(
        self,
        catalog: LogCatalog,
        config: FlywheelConfig,
        pii_detector: PIIDetector | None = None,
    ) -> None: ...

    async def clean_logs(
        self,
        filters: LogFilter | None = None,
        batch_size: int = 100,
    ) -> CleaningResult:
        """Score and clean all unscored inference logs matching filters.

        Reads logs in batches, evaluates fitness, scrubs PII, and updates
        the catalog with results.

        Args:
            filters: Optional additional filters (default: unscored logs only)
            batch_size: Number of logs to process per batch

        Returns:
            CleaningResult with counts and score distribution
        """
        ...

    def _evaluate_log(self, record: InferenceLogRecord) -> FitnessResult:
        """Evaluate a single inference log using FitnessEvaluator.

        Reconstructs model_output from the log's response_content and tool_calls,
        then passes it through the evaluator.
        """
        ...

    def _read_log_content(self, record: InferenceLogRecord) -> dict:
        """Read full log content from the source JSONL file.

        The catalog stores only index fields. Full content (messages,
        response) is read from source_file at line_number.
        """
        ...


@dataclass
class CleaningResult:
    """Summary of a cleaning run."""
    total_processed: int = 0
    scored: int = 0
    pii_scrubbed: int = 0
    errors: int = 0
    score_distribution: dict[str, int] = field(default_factory=dict)
    # e.g. {"0.0-0.3": 45, "0.3-0.8": 120, "0.8-1.0": 300}
```

**Key design decisions**:
- FitnessEvaluator is created with `max_errors_before_zero=5` and `scoring_method="error_count"` (pinned in config, not left to default)
- PII detection is a Protocol with a no-op stub for v1 -- interface defined now for future implementation
- Catalog updates are batched (update_score in a transaction per batch)
- Full log content is read on-demand from JSONL source files, not stored in SQLite

### 5.5 AutoTagger (`shared/flywheel/tagger.py`)

```python
import logging
from typing import Any

from shared.judge.judge_service import JudgeService

logger = logging.getLogger(__name__)


class AutoTagger:
    """Tags inference logs for SFT/KTO/GRPO training based on quality scores.

    Rule-based classification handles ~80% of logs (clear pass/fail).
    LLM judge fallback handles the ambiguous middle tier (0.4-0.7 band).

    Tag rules (configurable via FlywheelConfig):
        score >= sft_threshold (0.8)      -> "sft"  (positive example)
        kto_min <= score < sft_threshold  -> "kto"  (negative example for KTO)
        score < kto_min (0.3)             -> "discard"
        has_tool_calls AND is_valid       -> also eligible for "grpo"

    For scores in the ambiguous band (0.4-0.7), the LLM judge is consulted
    to decide whether the example is actually good enough for SFT or should
    stay as KTO negative.

    Args:
        catalog: LogCatalog instance
        config: FlywheelConfig (tag thresholds)
        judge: Optional JudgeService for ambiguous-band fallback
    """

    def __init__(
        self,
        catalog: LogCatalog,
        config: FlywheelConfig,
        judge: JudgeService | None = None,
    ) -> None: ...

    async def tag_logs(
        self,
        filters: LogFilter | None = None,
        batch_size: int = 100,
        use_judge: bool = True,
    ) -> TaggingResult:
        """Tag all scored-but-untagged logs matching filters.

        Args:
            filters: Optional additional filters (default: scored + untagged)
            batch_size: Number of logs to process per batch
            use_judge: Whether to invoke LLM judge for ambiguous examples

        Returns:
            TaggingResult with counts per tag
        """
        ...

    def _classify_by_rules(self, record: InferenceLogRecord) -> str:
        """Apply rule-based classification. Returns tag string.

        IMPORTANT: The scoring path depends on whether tools were present in
        the original request. FitnessEvaluator returns 0.0 for non-tool-call
        responses (via no_tool_call_score), but that does NOT mean the response
        is bad -- it means tools weren't requested. The tagger MUST check
        tools_requested before applying score thresholds.

        Decision logic:
            if score is None: return "unscored"

            # Non-tool-call path: tools were NOT in the request
            if not record.tools_requested:
                # Score is meaningless (FitnessEvaluator gives 0.0 by default)
                # Use text_quality_threshold from config instead
                if config.text_response_policy == "sft":
                    return "sft"     # Include as text-only SFT positive
                elif config.text_response_policy == "skip":
                    return "discard" # Not useful for tool-calling training
                else:
                    return "kto"     # Default: KTO negative (model should have used tools?)

            # Tool-call path: tools WERE in the request
            if score >= config.sft_threshold: return "sft"
            if score < config.kto_min_threshold: return "discard"
            if has_tool_calls and is_valid: return "grpo"
            if config.ambiguous_min <= score <= config.ambiguous_max: return "ambiguous"
            return "kto"
        """
        ...

    async def _judge_ambiguous(
        self, records: list[InferenceLogRecord]
    ) -> list[tuple[str, str]]:
        """Invoke LLM judge on ambiguous examples.

        Returns list of (log_id, tag) pairs where tag is "sft" or "kto".
        Uses shared/judge/JudgeService with a flywheel-specific rubric.
        """
        ...

    def _build_grpo_examples(
        self, records: list[InferenceLogRecord]
    ) -> list[InferenceLogRecord]:
        """Identify logs eligible for GRPO training.

        A log is GRPO-eligible if:
        - has_tool_calls is True
        - is_valid is True (tool call passes schema validation)
        - fitness_score can serve as reward signal

        GRPO examples get tagged "grpo" IN ADDITION to their sft/kto tag.
        The stager handles dual-tagged examples.
        """
        ...


@dataclass
class TaggingResult:
    """Summary of a tagging run."""
    total_processed: int = 0
    sft_count: int = 0
    kto_count: int = 0
    grpo_count: int = 0
    discard_count: int = 0
    judge_invocations: int = 0
    errors: int = 0
```

**Key design decisions**:
- **Non-tool requests get a separate scoring path**: FitnessEvaluator returns `no_tool_call_score` (default 0.0) for responses without tool calls, but this score is meaningless when tools were never requested. The tagger checks `record.tools_requested` BEFORE applying score thresholds. Non-tool requests are handled by `text_response_policy` config ("sft" to include, "skip" to discard, "kto" for KTO negative).
- GRPO eligibility is orthogonal to SFT/KTO tagging: a log can be tagged both "sft" and "grpo". The stager decides which dataset to include it in.
- The ambiguous band (0.4-0.7) is configurable. The judge is optional -- if no JudgeService is provided, ambiguous examples default to "kto".
- The LLM judge uses a flywheel-specific rubric (not the general-purpose SynthChat rubrics). This rubric evaluates whether the tool call is semantically appropriate for the user request, not just structurally valid.

### 5.6 DatasetStager (`shared/flywheel/stager.py`)

```python
import hashlib
import json
import logging
from pathlib import Path

from shared.experiment_tracking.registry import RunRegistry
from shared.experiment_tracking.schema import RunRecord

logger = logging.getLogger(__name__)


class DatasetStager:
    """Assembles tagged examples into versioned JSONL training datasets.

    Reads tagged logs from the catalog, formats them as ChatML training data
    (matching existing Datasets/ conventions), writes versioned JSONL files,
    and registers the flywheel cycle in RunRegistry.

    Output structure:
        Datasets/flywheel/v003/
            sft_training.jsonl      (label: true)
            kto_training.jsonl      (mixed label: true/false)
            grpo_training.jsonl     (reward field)
            manifest.json           (DatasetVersion metadata)

    Args:
        catalog: LogCatalog instance
        config: FlywheelConfig
        datasets_dir: Base directory for staged datasets (default: Datasets/flywheel/)
    """

    def __init__(
        self,
        catalog: LogCatalog,
        config: FlywheelConfig,
        datasets_dir: Path | None = None,
    ) -> None: ...

    async def stage_dataset(
        self,
        filters: LogFilter | None = None,
    ) -> StagingResult:
        """Stage all tagged-but-unused logs into a new dataset version.

        1. Query catalog for tagged, unused logs
        2. Format each log into ChatML training format
        3. Write versioned JSONL files
        4. Compute content hash
        5. Create DatasetVersion in catalog
        6. Register flywheel_cycle RunRecord in RunRegistry
        7. Mark all used logs in catalog

        Args:
            filters: Optional additional filters

        Returns:
            StagingResult with version info and counts
        """
        ...

    def _next_version_id(self) -> str:
        """Determine next version ID (e.g., "v004") by scanning Datasets/flywheel/."""
        ...

    def _format_sft_example(self, record: InferenceLogRecord, content: dict) -> dict:
        """Format a log as an SFT training example.

        Returns: {"conversations": [...], "label": true}
        """
        ...

    def _format_kto_example(self, record: InferenceLogRecord, content: dict) -> dict:
        """Format a log as a KTO training example.

        SFT-threshold logs: {"conversations": [...], "label": true}
        KTO-band logs: {"conversations": [...], "label": false}
        """
        ...

    def _format_grpo_example(self, record: InferenceLogRecord, content: dict) -> dict:
        """Format a log as a GRPO training example.

        Returns: {"conversations": [...], "reward": <fitness_score>}
        """
        ...

    def _compute_content_hash(self, file_paths: dict[str, Path]) -> str:
        """SHA-256 hash of all staged dataset files."""
        ...

    def _register_flywheel_cycle(
        self, version: DatasetVersion
    ) -> str:
        """Register this staging cycle in RunRegistry and return run_id.

        Uses flywheel_cycle_to_run_record() adapter from
        shared/experiment_tracking/adapters.py.
        """
        ...


@dataclass
class StagingResult:
    """Summary of a staging run."""
    version_id: str = ""
    sft_count: int = 0
    kto_pos_count: int = 0
    kto_neg_count: int = 0
    grpo_count: int = 0
    total_records: int = 0
    file_paths: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""
    run_id: str = ""
```

**Adapter function** (to be added to `shared/experiment_tracking/adapters.py`):

```python
def flywheel_cycle_to_run_record(
    cycle_data: dict[str, Any],
    output_dir: str,
    *,
    run_id: str | None = None,
    parent_run_id: str | None = None,
) -> RunRecord:
    """Convert flywheel cycle metadata to a RunRecord.

    Args:
        cycle_data: Dict with keys:
            - version_id: Dataset version string (e.g., "v003")
            - record_counts: {"sft": N, "kto_pos": N, "kto_neg": N, "grpo": N}
            - filter_criteria: Tag thresholds used
            - source_model_id: Model that generated the inference logs
            - content_hash: SHA-256 of staged datasets
        output_dir: Path to Datasets/flywheel/vN/
        run_id: Optional pre-generated run ID
        parent_run_id: inference_log_batch run_id

    Returns:
        RunRecord with run_type="flywheel_cycle"
    """
    ...
```

### 5.7 FlywheelOrchestrator (`shared/flywheel/orchestrator.py`)

```python
import asyncio
import logging
import subprocess
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class RetrainMode(Enum):
    """How to handle GPU resources during retraining."""
    GPU_MUTEX = "gpu_mutex"      # Stop vLLM -> train -> restart (single GPU)
    HOT_SWAP = "hot_swap"        # Train elsewhere, hot-swap adapter (multi-GPU or cloud)
    CLOUD = "cloud"              # Offload to cloud backend (HF Jobs, RunPod, Modal)


class FlywheelOrchestrator:
    """Coordinates the full flywheel pipeline and retrain lifecycle.

    Pipeline stages: ingest -> clean -> tag -> stage -> (optional) retrain

    Retrain modes:
    - GPU_MUTEX: Single GPU (RTX 3090). Stops vLLM, trains, restarts with new adapter.
    - HOT_SWAP: Training on separate GPU. Hot-swaps adapter via vLLM API.
    - CLOUD: Offloads training to cloud backend. Downloads adapter, hot-swaps.

    Args:
        catalog: LogCatalog instance
        config: FlywheelConfig
        cleaner: DataCleaner instance
        tagger: AutoTagger instance
        stager: DatasetStager instance
    """

    def __init__(
        self,
        catalog: LogCatalog,
        config: FlywheelConfig,
        cleaner: DataCleaner,
        tagger: AutoTagger,
        stager: DatasetStager,
    ) -> None: ...

    async def run_cycle(
        self,
        skip_retrain: bool = False,
        retrain_mode: RetrainMode | None = None,
        dry_run: bool = False,
    ) -> CycleResult:
        """Execute a full flywheel cycle: clean -> tag -> stage -> retrain.

        Args:
            skip_retrain: If True, stop after staging (prepare data only)
            retrain_mode: Override config's default retrain mode
            dry_run: If True, show what would happen without executing

        Returns:
            CycleResult with per-stage summaries
        """
        ...

    async def check_readiness(self) -> ReadinessReport:
        """Check if enough data has accumulated to justify a retrain cycle.

        Thresholds from config:
        - min_new_examples: Minimum untagged logs since last cycle
        - min_sft_examples: Minimum expected SFT examples
        - min_quality_score: Average fitness score threshold

        Returns:
            ReadinessReport with ready flag and breakdown
        """
        ...

    async def _stop_vllm(self) -> bool:
        """Stop the vLLM server process. Returns True if stopped successfully."""
        ...

    async def _start_vllm(self, adapter_path: str | None = None) -> bool:
        """Start vLLM with optional LoRA adapter. Returns True if started."""
        ...

    async def _hot_swap_adapter(self, adapter_path: str, adapter_name: str) -> bool:
        """Hot-swap LoRA adapter via vLLM API (POST /v1/load_lora_adapter).

        Uses load_inplace=true for zero-downtime update.
        """
        ...

    async def _run_training(
        self,
        dataset_version: DatasetVersion,
        retrain_mode: RetrainMode,
    ) -> TrainingResult:
        """Execute training using the staged dataset.

        For GPU_MUTEX and HOT_SWAP modes, invokes the appropriate trainer
        script via subprocess (e.g., train_sft.py --dataset-file <path>).

        For CLOUD mode, uses existing Trainers/cloud/ infrastructure.
        """
        ...

    async def _select_trainer(
        self, dataset_version: DatasetVersion
    ) -> tuple[str, list[str]]:
        """Select trainer script and args based on dataset composition.

        Priority: SFT if sft_count > 0, KTO if kto_count > 0 and no SFT.
        GRPO runs separately if grpo_count > 0.

        Returns: (script_path, args_list)
        """
        ...

    def status(self) -> FlywheelStatus:
        """Return current flywheel status (sync method for CLI use).

        Checks: vLLM running?, logs since last cycle, last dataset version,
        readiness estimate.
        """
        ...


@dataclass
class ReadinessReport:
    """Assessment of whether a retrain cycle should be triggered."""
    ready: bool = False
    new_log_count: int = 0
    estimated_sft: int = 0
    estimated_kto: int = 0
    estimated_grpo: int = 0
    avg_quality_score: float = 0.0
    reasons: list[str] = field(default_factory=list)  # Why ready or not ready


@dataclass
class TrainingResult:
    """Outcome of a training run."""
    success: bool = False
    run_id: str = ""
    adapter_path: str = ""
    training_type: str = ""   # "sft" | "kto" | "grpo"
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class CycleResult:
    """Summary of a full flywheel cycle."""
    cleaning: CleaningResult | None = None
    tagging: TaggingResult | None = None
    staging: StagingResult | None = None
    training: TrainingResult | None = None
    hot_swap_success: bool | None = None
    total_duration_seconds: float = 0.0


@dataclass
class FlywheelStatus:
    """Current state of the flywheel system."""
    vllm_running: bool = False
    vllm_model: str = ""
    active_adapter: str | None = None
    total_logs: int = 0
    unprocessed_logs: int = 0
    last_cycle_at: str | None = None
    last_dataset_version: str | None = None
    readiness: ReadinessReport | None = None
```

### 5.8 FlywheelConfig (`shared/flywheel/config.py`)

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.utilities import load_yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("configs/flywheel/default.yaml")


@dataclass
class FlywheelConfig:
    """Configuration for the flywheel pipeline.

    Loaded from YAML config file. All thresholds have sensible defaults
    that can be overridden per-deployment.
    """

    # -- Scoring thresholds --
    sft_threshold: float = 0.8           # score >= this -> SFT positive
    kto_min_threshold: float = 0.3       # score < this -> discard
    ambiguous_min: float = 0.4           # scores in [ambiguous_min, ambiguous_max] -> judge
    ambiguous_max: float = 0.7
    max_errors_before_zero: int = 5      # FitnessEvaluator config: controls score granularity
                                         # With 5: scores are 1.0, 0.8, 0.6, 0.4, 0.2, 0.0
                                         # With 10: scores are 1.0, 0.9, 0.8, ..., 0.0 (finer)
    scoring_method: str = "error_count"  # "binary" | "error_count" | "error_penalty"
    error_penalty_per_error: float = 0.1 # For "error_penalty" method only
    no_tool_call_score: float = 0.0      # Score for non-tool-call responses

    # -- Non-tool-call handling --
    text_response_policy: str = "skip"   # How to handle responses where tools were NOT requested
                                         # "sft"  = include as text-only SFT positive
                                         # "skip" = discard (default -- not useful for tool training)
                                         # "kto"  = include as KTO negative

    # -- FitnessEvaluator config --
    fitness_config_path: str | None = None  # Path to fitness YAML config file
                                            # If set, overrides validation_rules and scoring settings
                                            # IMPORTANT: Without config, FitnessEvaluator scores
                                            # everything with tool calls as 1.0 (no validation runs)

    # -- GRPO --
    grpo_enabled: bool = True
    grpo_reward_scale: float = 1.0       # Multiplier on fitness_score for GRPO reward

    # -- Storage --
    catalog_backend: str = "sqlite"      # "sqlite" | "postgres"
    catalog_path: str = ".tracking/flywheel.db"  # SQLite path
    catalog_url: str | None = None       # Postgres connection URL
    tenant_id: str | None = None         # Postgres tenant ID
    log_dir: str = "inference_logs"      # JSONL log directory
    datasets_dir: str = "Datasets/flywheel"  # Staged dataset directory

    # -- Logging proxy --
    proxy_port: int = 8080
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    proxy_timeout_seconds: float = 120.0
    flush_interval_seconds: float = 1.0  # InferenceLogger write batch interval

    # -- vLLM --
    vllm_adapter_name: str = "current-adapter"
    vllm_adapter_path: str | None = None
    vllm_base_model: str | None = None
    vllm_max_lora_rank: int = 64
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enable_runtime_lora: bool = True

    # -- Retrain --
    retrain_mode: str = "gpu_mutex"      # "gpu_mutex" | "hot_swap" | "cloud"
    retrain_trainer: str = "sft"         # "sft" | "kto" | "auto"
    min_new_examples: int = 500          # Readiness: min unprocessed logs
    min_sft_examples: int = 100          # Readiness: min estimated SFT examples
    min_quality_score: float = 0.6       # Readiness: min average quality

    # -- Cloud retrain (when retrain_mode="cloud") --
    cloud_provider: str | None = None    # "hf_jobs" | "runpod" | "modal"
    cloud_config_path: str | None = None # Path to cloud backend config

    # -- Log rotation --
    log_retention_days: int = 30
    compress_after_days: int = 7

    # -- Validation rules for FitnessEvaluator --
    validation_rules: list[dict] = field(default_factory=list)

    def to_fitness_config(self) -> dict[str, Any]:
        """Build FitnessEvaluator config dict from flywheel settings.

        If fitness_config_path is set, loads and returns the external YAML
        config directly. Otherwise, builds config from inline fields.

        IMPORTANT: An empty validations list means FitnessEvaluator runs
        zero validators and scores everything 1.0. The flywheel MUST
        either set fitness_config_path or provide validation_rules.

        Returns config suitable for create_fitness_evaluator().
        """
        if self.fitness_config_path is not None:
            return load_yaml(self.fitness_config_path)
        if not self.validation_rules:
            logger.warning(
                "No fitness_config_path or validation_rules set; "
                "FitnessEvaluator will score all responses 1.0"
            )
        return {
            "validations": self.validation_rules,
            "scoring": {
                "method": self.scoring_method,
                "no_tool_call_score": self.no_tool_call_score,
                "params": {
                    "max_errors_before_zero": self.max_errors_before_zero,
                    "penalty_per_error": self.error_penalty_per_error,
                },
            },
        }


def load_flywheel_config(
    config_path: str | Path | None = None,
) -> FlywheelConfig:
    """Load flywheel config from YAML file.

    Falls back to configs/flywheel/default.yaml if no path specified.
    Missing fields use dataclass defaults.

    Args:
        config_path: Path to YAML config file

    Returns:
        Populated FlywheelConfig instance
    """
    ...
```

**Default config file** (`configs/flywheel/default.yaml`):

```yaml
# Flywheel pipeline configuration

# Scoring thresholds
sft_threshold: 0.8
kto_min_threshold: 0.3
ambiguous_min: 0.4
ambiguous_max: 0.7
max_errors_before_zero: 5       # Controls score granularity (5 -> 0.2 steps, 10 -> 0.1 steps)
scoring_method: error_count      # "binary" | "error_count" | "error_penalty"
# error_penalty_per_error: 0.1  # Uncomment if using "error_penalty" method

# Non-tool-call handling
# FitnessEvaluator scores non-tool-call responses as 0.0 (no schema to validate).
# This policy controls how the tagger treats those responses:
text_response_policy: skip       # "sft" | "kto" | "skip"

# Fitness validation config
# CRITICAL: Without a fitness config, FitnessEvaluator scores everything 1.0.
# Point this at a YAML file with validation rules (same format as SynthChat fitness.yaml).
fitness_config_path: configs/flywheel/fitness.yaml

# GRPO
grpo_enabled: true
grpo_reward_scale: 1.0

# Storage
catalog_backend: sqlite
catalog_path: .tracking/flywheel.db
log_dir: inference_logs
datasets_dir: Datasets/flywheel

# Proxy
proxy_port: 8080
vllm_host: localhost
vllm_port: 8000
proxy_timeout_seconds: 120

# vLLM
vllm_adapter_name: current-adapter
vllm_max_lora_rank: 64
vllm_gpu_memory_utilization: 0.9
vllm_enable_runtime_lora: true

# Retrain
retrain_mode: gpu_mutex
retrain_trainer: auto
min_new_examples: 500
min_sft_examples: 100
min_quality_score: 0.6

# Log rotation
log_retention_days: 30
compress_after_days: 7

# Validation rules for FitnessEvaluator
# Minimum rules to prevent trivial 1.0 scores
validation_rules:
  - type: json
    path: "tool_calls[0].function.name"
    required: true
  - type: json
    path: "tool_calls[0].function.arguments"
    required: true
```

**Flywheel fitness config** (`configs/flywheel/fitness.yaml`):

```yaml
# Fitness validation rules for flywheel inference logs.
# This file is loaded by FitnessEvaluator via FlywheelConfig.fitness_config_path.
# Without this file (or validation_rules in default.yaml), ALL responses score 1.0
# and the tagger cannot meaningfully classify examples.
#
# Format follows the same schema as SynthChat/Evaluator fitness configs.
# Each rule defines a validation check. Failures increment the error count,
# which maps to fitness_score via max_errors_before_zero.

validations:
  # Tool call structure validation
  - type: json
    name: tool_call_name_present
    path: "tool_calls[0].function.name"
    required: true

  - type: json
    name: tool_call_args_valid
    path: "tool_calls[0].function.arguments"
    required: true

  # Add domain-specific rules below as your model evolves.
  # Example: validate that tool arguments parse as valid JSON
  # - type: json_parse
  #   name: args_parseable
  #   path: "tool_calls[0].function.arguments"

scoring:
  method: error_count
  no_tool_call_score: 0.0
  params:
    max_errors_before_zero: 5
```

> **Calibration note**: With `max_errors_before_zero=5` and `error_count` method, scores
> increment in 0.2 steps (0.0, 0.2, 0.4, 0.6, 0.8, 1.0). For finer granularity, increase
> `max_errors_before_zero` (e.g., 10 gives 0.1 steps). When changing this value, review
> `sft_threshold` and `kto_min_threshold` to ensure thresholds still land on achievable scores.

### 5.9 CLI Integration (`tuner/handlers/flywheel_handler.py`)

```python
import asyncio
import logging
from pathlib import Path
from typing import Any

from tuner.handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class FlywheelHandler(BaseHandler):
    """Handler for `tuner flywheel` CLI subcommands.

    Subcommands:
        status      Show flywheel system status
        run-cycle   Execute a full flywheel cycle (clean -> tag -> stage -> retrain)
        configure   Interactive configuration wizard
        ingest      Manually trigger log ingestion
        stage       Stage a new dataset version (no retrain)
        readiness   Check retrain readiness
        logs        Show inference log statistics
        versions    List staged dataset versions
    """

    def handle(self, args: Any) -> None:
        """Route to appropriate subcommand handler."""
        ...

    def _handle_status(self, args: Any) -> None:
        """Display flywheel status: vLLM state, log counts, last cycle, readiness."""
        ...

    def _handle_run_cycle(self, args: Any) -> None:
        """Execute a full flywheel cycle.

        Options:
            --skip-retrain    Stop after staging
            --retrain-mode    Override: gpu_mutex | hot_swap | cloud
            --dry-run         Show what would happen
            --config          Path to flywheel config
        """
        ...

    def _handle_configure(self, args: Any) -> None:
        """Interactive configuration wizard for flywheel settings."""
        ...

    def _handle_readiness(self, args: Any) -> None:
        """Check and display retrain readiness report."""
        ...

    def _handle_stage(self, args: Any) -> None:
        """Stage a new dataset version without retraining."""
        ...

    def _handle_logs(self, args: Any) -> None:
        """Show inference log statistics and recent activity."""
        ...

    def _handle_versions(self, args: Any) -> None:
        """List all staged dataset versions with record counts."""
        ...
```

**CLI registration** (modifications to existing files):

In `tuner/cli/parser.py`, add:
```python
flywheel_parser = subparsers.add_parser("flywheel", help="Data flywheel pipeline")
flywheel_sub = flywheel_parser.add_subparsers(dest="flywheel_action")

flywheel_sub.add_parser("status", help="Show flywheel system status")

run_cycle = flywheel_sub.add_parser("run-cycle", help="Execute flywheel cycle")
run_cycle.add_argument("--skip-retrain", action="store_true")
run_cycle.add_argument("--retrain-mode", choices=["gpu_mutex", "hot_swap", "cloud"])
run_cycle.add_argument("--dry-run", action="store_true")
run_cycle.add_argument("--config", type=str, help="Path to flywheel config")

flywheel_sub.add_parser("configure", help="Configure flywheel settings")
flywheel_sub.add_parser("readiness", help="Check retrain readiness")

stage = flywheel_sub.add_parser("stage", help="Stage dataset version")
stage.add_argument("--config", type=str)

flywheel_sub.add_parser("logs", help="Show log statistics")
flywheel_sub.add_parser("versions", help="List dataset versions")
```

In `tuner/cli/router.py`, add:
```python
"flywheel": FlywheelHandler,
```

In `run.sh` TUI menu, add:
```
[9] Flywheel (self-improving pipeline)
```

---

## 6. Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Async interface | `aiosqlite` / `asyncpg` | Proxy is async (FastAPI); sync->async retrofit is painful; consistent interface |
| Log format | JSONL (date-partitioned) | Matches existing patterns (`InteractionLogger`); human-readable; trainer-compatible |
| Log catalog | SQLite WAL (local) / Postgres (cloud) | Zero-config local; schema-per-tenant cloud; same SQL semantics |
| Config format | YAML + dataclass | Matches existing `configs/` pattern; type-safe with defaults |
| HTTP proxy | FastAPI + httpx | Async, OpenAPI docs, httpx for non-blocking forwarding |
| Data structures | dataclasses | Matches existing codebase (`RunRecord`, `FitnessResult`); no new Pydantic dep |
| Queue pattern | `asyncio.Queue` | Proxy is async; threading.Lock not needed |
| Content hash | SHA-256 | Standard; deterministic; sufficient for dataset integrity |
| Version naming | Sequential `vNNN` | Simple, human-readable, sorted lexicographically |

---

## 7. Security Architecture

### 7.1 Inference Data

- Logging is **opt-in**: proxy only logs when explicitly started
- No credentials are stored in logs (auth headers are not captured)
- PII scrubbing interface defined (v1 is no-op; v2 adds regex/ML detectors)
- JSONL files are local to the deployment; no external transmission

### 7.2 Tenant Isolation (Cloud)

- Postgres schema-per-tenant: `tenant_{id}.inference_logs`
- No row-level filtering needed -- complete schema isolation
- Tenant ID validated on catalog creation; cannot cross-query

### 7.3 vLLM Runtime LoRA

- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` required
- Per vLLM docs: "intended for isolated, trusted environments"
- Acceptable for single-user local flywheel; document in deployment guide

### 7.4 Proxy Security

- Auth headers passed through to vLLM without inspection or storage
- No new auth mechanism introduced by the proxy itself
- `/flywheel/health` and `/flywheel/stats` endpoints are unauthenticated (local use)

---

## 8. Deployment Architecture

### 8.1 Local Development (Single GPU)

```
Terminal 1: vllm serve <model> --enable-lora --port 8000
Terminal 2: uvicorn services.proxy.app:app --port 8080
Terminal 3: tuner flywheel run-cycle  (when ready)
```

GPU mutex lifecycle:
1. Proxy captures inference logs continuously
2. User runs `tuner flywheel readiness` to check
3. User runs `tuner flywheel run-cycle`
4. Orchestrator: stop vLLM -> train -> restart vLLM with new adapter
5. Loop closes

### 8.2 Docker Compose (Production)

```yaml
# docker-compose.flywheel.yml
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: --model <base> --enable-lora --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - ./models:/models
    ports:
      - "8000:8000"

  proxy:
    build: ./services/proxy
    depends_on: [vllm]
    environment:
      - FLYWHEEL_CONFIG=/app/configs/flywheel/default.yaml
    volumes:
      - ./inference_logs:/app/inference_logs
      - ./.tracking:/app/.tracking
    ports:
      - "8080:8080"
```

---

## 9. Implementation Guidelines

### 9.1 For Coders

- **Follow existing patterns**: Look at `shared/judge/interaction_logger.py` for JSONL writing, `shared/experiment_tracking/adapters.py` for RunRecord conversion, `shared/validation/fitness.py` for FitnessEvaluator usage
- **Dataclasses, not Pydantic**: All data structures use `@dataclass` with `field(default_factory=...)` for mutable defaults
- **JSONL format**: Match existing ChatML format in `Datasets/` -- `{"conversations": [...], "label": true/false}`
- **Error handling**: Log and continue. Pipeline errors must never crash the proxy or lose inference data. Wrap all pipeline operations in try/except with logging.
- **Pre-commit hook**: Avoid variable names containing "token" (use `api_key`, `auth_key`, `hf_credential`). The hook pattern-matches `print\s*\(.*token` case-insensitively.
- **File size**: Keep each module under 500 lines. `catalog.py` may exceed this with two implementations; consider splitting `sqlite_catalog.py` and `postgres_catalog.py` if needed.

### 9.2 Module Dependency Graph

```
config.py                       (no internal deps)
    ^
    |
catalog.py                      (imports config)
    ^
    |
inference_logger.py             (imports catalog, config)
    ^
    |
cleaner.py                      (imports catalog, config; uses shared/validation/fitness.py)
    ^
    |
tagger.py                       (imports catalog, config; optionally uses shared/judge/)
    ^
    |
stager.py                       (imports catalog, config; uses shared/experiment_tracking/)
    ^
    |
orchestrator.py                 (imports all above; coordinates pipeline)
```

Dependencies flow strictly downward. No circular imports.

### 9.3 Testing Hooks

- `LogCatalog` is a Protocol: tests use an in-memory implementation
- `PIIDetector` is a Protocol: tests inject `NoOpPIIDetector` or a mock
- `JudgeService` is optional: tagger tests run without LLM judge
- `FitnessEvaluator` accepts inline config: tests provide known validation rules
- `FlywheelOrchestrator` subprocess calls are mockable via dependency injection

---

## 10. Implementation Roadmap

### Phase 1 -- Capture (can ship independently)

| # | File | Dependencies |
|---|------|-------------|
| 1 | `shared/flywheel/config.py` | None |
| 2 | `shared/flywheel/catalog.py` | config.py |
| 3 | `shared/flywheel/inference_logger.py` | catalog.py, config.py |
| 4 | `services/proxy/app.py` | inference_logger.py, catalog.py, config.py |
| 5 | `configs/flywheel/default.yaml` | None |
| 5b | `configs/flywheel/fitness.yaml` | None |

**Milestone**: Proxy captures inference logs to JSONL and indexes in SQLite.

### Phase 2 -- Process

| # | File | Dependencies |
|---|------|-------------|
| 6 | `shared/flywheel/cleaner.py` | catalog.py, config.py, `shared/validation/fitness.py` |
| 7 | `shared/flywheel/tagger.py` | catalog.py, config.py, `shared/judge/` (optional) |

**Milestone**: Logs are scored, tagged, and queryable by tag.

### Phase 3 -- Version

| # | File | Dependencies |
|---|------|-------------|
| 8 | `shared/flywheel/stager.py` | catalog.py, config.py, `shared/experiment_tracking/` |
| 9 | `shared/experiment_tracking/adapters.py` (extend) | schema.py |

**Milestone**: Versioned datasets staged in `Datasets/flywheel/vN/`.

### Phase 4 -- Retrain + CLI

| # | File | Dependencies |
|---|------|-------------|
| 10 | `shared/flywheel/orchestrator.py` | All flywheel modules |
| 11 | `tuner/handlers/flywheel_handler.py` | orchestrator.py |
| 12 | `tuner/cli/parser.py` (modify) | None |
| 13 | `tuner/cli/router.py` (modify) | flywheel_handler.py |
| 14 | `run.sh` (modify) | None |

**Milestone**: `tuner flywheel run-cycle` executes full pipeline.

### Phase 5 -- Docker

| # | File | Dependencies |
|---|------|-------------|
| 15 | `docker-compose.flywheel.yml` | services/proxy/ |
| 16 | `requirements-flywheel.txt` | None |

**Milestone**: Containerized deployment.

### Parallel Opportunities

- Phase 1 items 1-3 (flywheel package) and item 4 (proxy) can be developed concurrently
- Phase 2 items 6 and 7 are independent (cleaner and tagger don't depend on each other)
- Phase 4 items 12-14 (CLI registration) can be developed while item 10 (orchestrator) is in progress

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Flywheel amplifies model errors | Medium | High | Holdout eval set; eval score tracking per version; human review gate for first N cycles |
| GPU mutex downtime during training | High | Medium | Clearly documented; cloud offload option eliminates downtime |
| Proxy adds latency | Low | Medium | Async forwarding; benchmark target <10ms overhead |
| SQLite contention under load | Low | Low | WAL mode; single writer pattern; only CLI reads are concurrent |
| Tagger misclassification | Medium | Medium | Conservative thresholds (0.8/0.3); manual review of first staged dataset |
| JSONL file growth | Low | Low | Daily rotation; configurable retention; archive/compress after N days |
| Pre-commit hook blocks commits | Medium | Low | Avoid "token" in variable names; documented workaround |

---

## 12. Architectural Decisions Record

### ADR-001: Async LogCatalog from Day One

**Status**: Accepted

**Context**: The logging proxy runs in an async context (FastAPI/uvicorn). The catalog could use sync SQLite for local and switch to async only for Postgres.

**Decision**: Use async interface (`aiosqlite` + `asyncpg`) for both backends.

**Rationale**: Retrofitting sync to async later requires changing every caller. The overhead of `aiosqlite` over sync `sqlite3` is negligible. One interface, two backends, zero migration cost.

### ADR-002: JSONL + SQLite Index (Not SQLite-Only Storage)

**Status**: Accepted

**Context**: We could store full inference log content in SQLite BLOB columns.

**Decision**: Store content in JSONL files; use SQLite only as an index.

**Rationale**: JSONL files are directly readable by trainers, debuggable with standard tools (`jq`, `head`), and match the existing `InteractionLogger` pattern. SQLite stays lightweight as an index.

### ADR-003: Schema-Per-Tenant for Postgres

**Status**: Accepted

**Context**: Multi-tenant data isolation can be achieved via row-level filtering or schema-level separation.

**Decision**: Schema-per-tenant (`tenant_{id}.inference_logs`).

**Rationale**: Eliminates the risk of a missing WHERE clause leaking cross-tenant data. No RLS policies to maintain. Each tenant's data is physically separated.

### ADR-004: Dual Tagging for GRPO

**Status**: Accepted

**Context**: A log that scores >= 0.8 (SFT quality) may also have valid tool calls (GRPO eligible). Should it be tagged once or twice?

**Decision**: GRPO eligibility is orthogonal. A log tagged "sft" can also be used for GRPO training.

**Rationale**: GRPO uses fitness_score as a continuous reward signal, not a binary label. High-quality SFT examples are also high-reward GRPO examples. The stager handles dual inclusion by writing the example to both datasets with appropriate formatting.

### ADR-005: No Ingestion Module

**Status**: Accepted

**Context**: The plan included `shared/flywheel/ingestion.py` for normalizing raw logs to canonical format.

**Decision**: Remove the ingestion module. The proxy writes InferenceLogRecord directly in canonical format.

**Rationale**: Since we control the proxy (the only log source), there's no format mismatch to normalize. The `InferenceLogger._build_record()` method handles the OpenAI-to-InferenceLogRecord conversion at write time. Adding an ingestion layer adds complexity with no value. If external log sources are added later, an ingestion module can be introduced then.

**Deviation from plan**: The plan listed `shared/flywheel/ingestion.py` as a Phase 2 file. This is removed because the proxy already produces canonical records. The cleaner reads directly from the catalog.

---

## Quality Checklist

- [x] All PREPARE phase requirements addressed
- [x] Components have single, clear responsibilities
- [x] Interfaces are well-defined (Protocol + concrete implementations)
- [x] Non-functional requirements covered (scalability via Postgres, performance via async, security via tenant isolation)
- [x] Security considerations embedded (no credential logging, PII interface, tenant isolation)
- [x] Architecture is testable (Protocol-based DI, optional dependencies)
- [x] Implementation path is clear with parallel opportunities
- [x] Existing codebase patterns followed (dataclasses, JSONL, ChatML format)
- [x] All deviations from plan documented with rationale (ADR-005)
