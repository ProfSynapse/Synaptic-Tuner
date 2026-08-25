"""Durable SQLite journal for one-shot provider effects."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

from ._log_policy import (
    EventCode, LogLevel, MAX_QUERY_LIMIT, MAX_RETAINED_BYTES,
    MAX_RETAINED_RECORDS, MessageCode, checked_log_fields,
)
from ._provider import EffectIdentity, EffectKind, ExecutionScope, ProviderJobRef
from .execution import (
    AccessContext, ArtifactVerificationState, ExecutionGrant, LogCursor, LogEntry,
    LogPage, RunRef, RunState, RunStatus,
)


SCHEMA_VERSION = 2
APPLICATION_ID = 0x53594E4A
SCHEMA_DESCRIPTOR = "synaptic.jobstore/sqlite/v2:grants,runs,effects,observations,state_events,logs,log_meta,cursors"
MAX_CURSOR_ROWS_PER_RUN = 1024
CURSOR_TTL_SECONDS = 86_400
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\\-]{0,255}")


class StoreError(RuntimeError):
    pass


class CorruptStore(StoreError):
    pass


class RecoveryRequired(CorruptStore):
    code = "job_store_recovery_required"

    def __init__(self, message: str = "job store requires explicit recovery") -> None:
        super().__init__(message)


class AccessDenied(StoreError):
    pass


class OperationConflict(StoreError):
    pass


class GrantRejected(StoreError):
    pass


class InvalidTransition(StoreError):
    pass


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_reparse(path: Path) -> bool:
    info = path.lstat()
    attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


@dataclass(frozen=True, slots=True)
class JobStoreLocation:
    database_path: Path
    control_plane_state_root: Path
    forbidden_code_roots: tuple[Path, ...]

    def __post_init__(self) -> None:
        database = Path(self.database_path)
        root = Path(self.control_plane_state_root)
        forbidden = tuple(Path(item) for item in self.forbidden_code_roots)
        if not forbidden:
            raise ValueError("at least one forbidden engine code root is required")
        if not database.is_absolute() or not root.is_absolute() or any(
            not item.is_absolute() for item in forbidden
        ):
            raise ValueError("job-store paths must be absolute")
        object.__setattr__(self, "database_path", database)
        object.__setattr__(self, "control_plane_state_root", root)
        object.__setattr__(self, "forbidden_code_roots", forbidden)


@dataclass(frozen=True, slots=True)
class GrantBinding:
    operation_key: str
    scope: ExecutionScope
    plan_fingerprint: str
    source_digest: str
    workload_digest: str
    artifact_slot_ref: str
    quote_digest: str
    resource_digest: str
    allowed_secret_refs_digest: str
    issued_at: str
    expires_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ExecutionScope):
            raise TypeError("scope must be ExecutionScope")
        object.__setattr__(self, "operation_key", _safe_ref(self.operation_key, "operation_key"))
        object.__setattr__(self, "artifact_slot_ref", _safe_ref(self.artifact_slot_ref, "artifact_slot_ref"))
        for name in ("plan_fingerprint", "source_digest", "workload_digest", "quote_digest", "resource_digest", "allowed_secret_refs_digest"):
            object.__setattr__(self, name, _fixed_digest(getattr(self, name), name))
        issued = _utc_timestamp(self.issued_at, "issued_at")
        expires = _utc_timestamp(self.expires_at, "expires_at")
        if expires <= issued:
            raise ValueError("grant expiry must be after issue time")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)


@dataclass(frozen=True, slots=True)
class SubmissionClaim:
    run: RunRef
    identity: EffectIdentity
    new_claim: bool
    state: RunState


@dataclass(frozen=True, slots=True)
class StoredEffect:
    run: RunRef
    identity: EffectIdentity
    status: str
    provider_job: ProviderJobRef | None


@dataclass(frozen=True, slots=True)
class ReconciliationClaim:
    effect: StoredEffect
    claim_token: str


def _digest(domain: str, value: str) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + value.encode("utf-8")).hexdigest()


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _safe_ref(value: str, name: str) -> str:
    value = _require_text(value, name)
    if _SAFE_REF_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded safe reference")
    return value


def _fixed_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical SHA-256 digest")
    return value


def _utc_timestamp(value: str, name: str) -> str:
    value = _require_text(value, name)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class JobStore:
    """One-connection-per-operation, fail-closed durable lifecycle store."""

    def __init__(
        self,
        location: JobStoreLocation,
        *,
        clock: Callable[[], str],
        id_factory: Callable[[str], str],
        token_factory: Callable[[], bytes] = secrets.token_bytes,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        if not isinstance(location, JobStoreLocation):
            raise TypeError("location must be JobStoreLocation")
        self.location = location
        self.path = location.database_path
        self._clock = clock
        self._id_factory = id_factory
        self._token_factory = token_factory
        if not callable(token_factory):
            raise TypeError("token_factory must be callable")
        if not isinstance(busy_timeout_ms, int) or busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self._busy_timeout_ms = busy_timeout_ms
        self._trusted_file_identity: tuple[int, int] | None = None
        self._open_or_create()

    def _now(self) -> str:
        return _utc_timestamp(self._clock(), "clock")

    def _new_token(self) -> str:
        raw = self._token_factory()
        if not isinstance(raw, bytes) or len(raw) != 32:
            raise ValueError("token_factory must return exactly 32 random bytes")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _validate_location(self) -> None:
        # Residual assumption: the host ACL prevents hostile replacement between
        # this last validation and SQLite's open; Python exposes no portable
        # openat/no-follow API for sqlite3 database plus sidecar creation.
        root = self.location.control_plane_state_root
        paths = (self.path, root, *self.location.forbidden_code_roots)
        if os.name == "nt" and any(str(item).startswith("\\\\") for item in paths):
            raise CorruptStore("UNC job-store paths are prohibited")
        if os.name == "nt" and ":" in self.path.name:
            raise CorruptStore("database alternate streams are prohibited")
        if not root.exists() or not root.is_dir():
            raise CorruptStore("host-owned state root must be an existing directory")
        if not self.path.parent.exists() or not self.path.parent.is_dir():
            raise CorruptStore("database parent directory must be host-created")

        def reject_component_hazards(path: Path) -> None:
            current = path
            while True:
                if os.path.lexists(current) and _is_reparse(current):
                    raise CorruptStore("path components must not be links or reparse points")
                if current.parent == current:
                    break
                current = current.parent

        reject_component_hazards(root)
        reject_component_hazards(self.path.parent)
        lexical_root = Path(os.path.abspath(root))
        lexical_db = Path(os.path.abspath(self.path))
        resolved_root = root.resolve(strict=True)
        resolved_parent = self.path.parent.resolve(strict=True)
        resolved_db = resolved_parent / self.path.name
        if not _is_relative_to(lexical_db, lexical_root) or not _is_relative_to(resolved_db, resolved_root):
            raise CorruptStore("database must remain beneath the host state root")
        root_device = root.stat().st_dev
        if self.path.parent.stat().st_dev != root_device:
            raise CorruptStore("database path must remain on the host-root filesystem")
        for forbidden in self.location.forbidden_code_roots:
            forbidden_lexical = Path(os.path.abspath(forbidden))
            forbidden_resolved = forbidden.resolve(strict=False)
            if _is_relative_to(lexical_db, forbidden_lexical) or _is_relative_to(resolved_db, forbidden_resolved):
                raise CorruptStore("database must remain outside engine code roots")
        sidecars = (
            self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm"),
            Path(str(self.path) + "-journal"),
        )
        for candidate in sidecars:
            if candidate.parent != self.path.parent:
                raise CorruptStore("SQLite sidecars must share the database directory")
            if not os.path.lexists(candidate):
                continue
            if _is_reparse(candidate) or not candidate.is_file():
                raise CorruptStore("database files must be regular non-reparse files")
            details = candidate.stat()
            if details.st_nlink != 1:
                raise CorruptStore("database files must not have hard links")
            if details.st_dev != root_device:
                raise CorruptStore("database files must remain on the host-root filesystem")
            resolved = candidate.resolve(strict=True)
            if not _is_relative_to(resolved, resolved_root):
                raise CorruptStore("database sidecar escaped the host state root")
            if any(_is_relative_to(resolved, item.resolve(strict=False)) for item in self.location.forbidden_code_roots):
                raise CorruptStore("database sidecar entered an engine code root")
    def _sidecar_paths(self) -> tuple[Path, Path, Path]:
        return (
            Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm"),
            Path(str(self.path) + "-journal"),
        )

    def _raw_header_identity(self) -> tuple[int, int]:
        self._validate_location()
        try:
            before = self.path.stat()
            before_stable = (
                before.st_dev, before.st_ino, before.st_size,
                getattr(before, "st_mtime_ns", int(before.st_mtime * 1_000_000_000)),
            )
            with self.path.open("rb", buffering=0) as stream:
                header = stream.read(100)
            after = self.path.stat()
            after_stable = (
                after.st_dev, after.st_ino, after.st_size,
                getattr(after, "st_mtime_ns", int(after.st_mtime * 1_000_000_000)),
            )
        except (OSError, ValueError, TypeError) as exc:
            raise CorruptStore("database raw-header read failed") from exc
        if before_stable != after_stable:
            raise CorruptStore("database identity changed during raw-header read")
        if len(header) != 100 or header[:16] != b"SQLite format 3\x00":
            raise CorruptStore("database raw header is not SQLite format 3")
        raw_user_version = int.from_bytes(header[60:64], "big")
        raw_application_id = int.from_bytes(header[68:72], "big")
        if raw_user_version != SCHEMA_VERSION or raw_application_id != APPLICATION_ID:
            raise CorruptStore("database raw application or user-version identity mismatch")
        return (before.st_dev, before.st_ino)

    def _validate_trusted_identity(self) -> None:
        self._validate_location()
        if self._trusted_file_identity is None:
            raise CorruptStore("database file identity has not been trusted")
        try:
            current = self.path.stat()
        except OSError as exc:
            raise CorruptStore("trusted database file is unavailable") from exc
        if (current.st_dev, current.st_ino) != self._trusted_file_identity:
            raise CorruptStore("trusted database file identity changed")

    def _immutable_validate_cold_existing(self) -> None:
        raw_identity = self._raw_header_identity()
        sidecars = tuple(path for path in self._sidecar_paths() if os.path.lexists(path))
        if sidecars:
            raise RecoveryRequired("SQLite sidecars require explicit recovery")
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro&immutable=1", uri=True, isolation_level=None
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if application_id != APPLICATION_ID or version != SCHEMA_VERSION:
                raise CorruptStore("immutable SQLite identity disagrees with raw header")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise CorruptStore("database integrity check failed")
            self._validate_schema(connection)
        except CorruptStore:
            raise
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            raise CorruptStore("immutable database validation failed") from exc
        finally:
            if connection is not None:
                connection.close()
        if self._raw_header_identity() != raw_identity:
            raise CorruptStore("database identity changed during immutable validation")
        self._trusted_file_identity = raw_identity

    def _connect(self) -> sqlite3.Connection:
        self._validate_trusted_identity()
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                str(self.path), timeout=self._busy_timeout_ms / 1000, isolation_level=None
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA trusted_schema = OFF")
            journal = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            connection.execute("PRAGMA synchronous = FULL")
            readings = (
                int(connection.execute("PRAGMA foreign_keys").fetchone()[0]),
                int(connection.execute("PRAGMA busy_timeout").fetchone()[0]),
                int(connection.execute("PRAGMA trusted_schema").fetchone()[0]),
                int(connection.execute("PRAGMA synchronous").fetchone()[0]),
                int(connection.execute("PRAGMA application_id").fetchone()[0]),
                int(connection.execute("PRAGMA user_version").fetchone()[0]),
            )
            expected = (1, self._busy_timeout_ms, 0, 2, APPLICATION_ID, SCHEMA_VERSION)
            if journal != "wal" or readings != expected:
                raise CorruptStore("required SQLite safety pragmas were not retained")
            self._validate_trusted_identity()
            return connection
        except CorruptStore:
            if connection is not None:
                connection.close()
            raise
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            if connection is not None:
                connection.close()
            raise CorruptStore("database writable connection failed closed") from exc

    def _enable_wal_after_trust(self) -> None:
        connection = self._connect()
        connection.close()
        self._validate_trusted_identity()
        if any(os.path.lexists(path) for path in self._sidecar_paths()):
            raise RecoveryRequired("WAL enablement did not return to a cold sidecar-free state")

    def _open_or_create(self) -> None:
        self._validate_location()
        if os.path.lexists(self.path):
            self._immutable_validate_cold_existing()
            self._enable_wal_after_trust()
            return
        connection: sqlite3.Connection | None = None
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(descriptor)
            connection = sqlite3.connect(
                str(self.path), timeout=self._busy_timeout_ms / 1000, isolation_level=None
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA trusted_schema = OFF")
            if str(connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]).lower() != "delete":
                raise CorruptStore("new database refused rollback-journal initialization")
            connection.execute("PRAGMA synchronous = FULL")
            self._create_schema(connection)
        except CorruptStore:
            raise
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            raise CorruptStore("exclusive database initialization failed") from exc
        finally:
            if connection is not None:
                connection.close()
        self._immutable_validate_cold_existing()
        self._enable_wal_after_trust()

    @classmethod
    def _recover_sidecars(
        cls, location: JobStoreLocation, *, busy_timeout_ms: int = 5_000
    ) -> None:
        """Refuse in-process recovery without inspecting or mutating evidence.

        Functional recovery is deferred to a separately designed offline
        snapshot/replay workflow.
        """
        del cls, location, busy_timeout_ms
        raise RecoveryRequired(
            "in-process sidecar recovery is disabled; offline snapshot/replay is required"
        )
    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE store_metadata (
              singleton INTEGER PRIMARY KEY CHECK(singleton=1),
              schema_descriptor TEXT NOT NULL, schema_fingerprint TEXT NOT NULL
            );
            CREATE TABLE grants (
              grant_digest TEXT PRIMARY KEY, principal_ref TEXT NOT NULL,
              project_ref TEXT NOT NULL, provider TEXT NOT NULL,
              account_ref TEXT NOT NULL, namespace_ref TEXT NOT NULL,
              operation_key TEXT NOT NULL, plan_fingerprint TEXT NOT NULL,
              source_digest TEXT NOT NULL, workload_digest TEXT NOT NULL,
              artifact_slot_ref TEXT NOT NULL, quote_digest TEXT NOT NULL,
              resource_digest TEXT NOT NULL, allowed_secret_refs_digest TEXT NOT NULL,
              issued_at TEXT NOT NULL, expires_at TEXT NOT NULL,
              consumed_run_id TEXT UNIQUE, consumed_at TEXT
            );
            CREATE TABLE runs (
              run_id TEXT PRIMARY KEY, project_ref TEXT NOT NULL,
              principal_ref TEXT NOT NULL, provider TEXT NOT NULL,
              account_ref TEXT NOT NULL, namespace_ref TEXT NOT NULL,
              operation_key TEXT NOT NULL, plan_fingerprint TEXT NOT NULL,
              canonical_plan TEXT NOT NULL, source_digest TEXT NOT NULL,
              workload_digest TEXT NOT NULL, artifact_slot_ref TEXT NOT NULL,
              grant_digest TEXT NOT NULL UNIQUE REFERENCES grants(grant_digest),
              state TEXT NOT NULL, artifact_state TEXT NOT NULL,
              revision INTEGER NOT NULL, message_code TEXT,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              UNIQUE(project_ref, operation_key)
            );
            CREATE TABLE effects (
              effect_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              kind TEXT NOT NULL CHECK(kind IN ('submit','cancel')),
              provider TEXT NOT NULL, account_ref TEXT NOT NULL,
              namespace_ref TEXT NOT NULL, effect_key TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('claimed','started','confirmed','absent','ambiguous')),
              claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
              provider_job_id TEXT, receipt_digest TEXT, result_code TEXT,
              reconcile_claim_digest TEXT, reconcile_claimed_at TEXT,
              UNIQUE(run_id, kind),
              UNIQUE(provider, account_ref, namespace_ref, effect_key)
            );
            CREATE TABLE observations (
              observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL REFERENCES runs(run_id), kind TEXT NOT NULL,
              result_code TEXT NOT NULL, digest TEXT, observed_at TEXT NOT NULL
            );
            CREATE TABLE state_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL REFERENCES runs(run_id), revision INTEGER NOT NULL,
              prior_state TEXT, state TEXT NOT NULL, event_code TEXT NOT NULL,
              message_code TEXT NOT NULL, created_at TEXT NOT NULL,
              UNIQUE(run_id, revision)
            );
            CREATE TABLE logs (
              run_id TEXT NOT NULL REFERENCES runs(run_id), sequence INTEGER NOT NULL,
              timestamp TEXT NOT NULL, level TEXT NOT NULL, event TEXT NOT NULL,
              message TEXT NOT NULL, byte_count INTEGER NOT NULL,
              PRIMARY KEY(run_id, sequence)
            );
            CREATE TABLE log_meta (
              run_id TEXT PRIMARY KEY REFERENCES runs(run_id), next_sequence INTEGER NOT NULL,
              floor_sequence INTEGER NOT NULL, retained_records INTEGER NOT NULL,
              retained_bytes INTEGER NOT NULL, dropped_records INTEGER NOT NULL
            );
            CREATE TABLE cursors (
              token_digest TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              after_sequence INTEGER NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX idx_runs_project_created ON runs(project_ref, created_at, run_id);
            CREATE INDEX idx_effects_run_kind ON effects(run_id, kind);
            CREATE UNIQUE INDEX idx_submit_provider_job ON effects(
              provider, account_ref, namespace_ref, provider_job_id
            ) WHERE kind='submit' AND provider_job_id IS NOT NULL;
            CREATE INDEX idx_observations_run ON observations(run_id, observation_id);
            CREATE INDEX idx_events_run ON state_events(run_id, event_id);
            CREATE INDEX idx_logs_run_sequence ON logs(run_id, sequence);
            CREATE INDEX idx_cursors_run ON cursors(run_id, created_at);
            """.replace("+", "")
        )
        actual_fingerprint = JobStore._schema_fingerprint(connection)
        connection.execute(
            "INSERT INTO store_metadata VALUES (1,?,?)",
            (SCHEMA_DESCRIPTOR, actual_fingerprint),
        )
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.execute("COMMIT")
    @staticmethod
    def _schema_fingerprint(connection: sqlite3.Connection) -> str:
        objects = [tuple(row) for row in connection.execute(
            """SELECT type,name,tbl_name,sql FROM sqlite_master
               WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name""".replace("+", "")
        )]
        if any(row[0] not in {"table", "index"} for row in objects):
            raise CorruptStore("views and triggers are prohibited in the JobStore schema")
        tables = sorted(row[1] for row in objects if row[0] == "table")
        table_details: dict[str, object] = {}
        for table in tables:
            table_details[table] = {
                "xinfo": [tuple(row) for row in connection.execute(
                    f"PRAGMA table_xinfo({json.dumps(table)})"
                )],
                "foreign_keys": [tuple(row) for row in connection.execute(
                    f"PRAGMA foreign_key_list({json.dumps(table)})"
                )],
                "indexes": [],
            }
            indexes = [tuple(row) for row in connection.execute(
                f"PRAGMA index_list({json.dumps(table)})"
            )]
            index_details = []
            for index in indexes:
                index_name = index[1]
                index_details.append({
                    "list": index,
                    "xinfo": [tuple(row) for row in connection.execute(
                        f"PRAGMA index_xinfo({json.dumps(index_name)})"
                    )],
                })
            table_details[table]["indexes"] = sorted(
                index_details, key=lambda item: item["list"][1]
            )
        descriptor = canonical_json({"objects": objects, "tables": table_details})
        return hashlib.sha256(descriptor.encode("utf-8")).hexdigest()

    @staticmethod
    def _expected_schema_fingerprint() -> str:
        pristine = sqlite3.connect(":memory:", isolation_level=None)
        pristine.row_factory = sqlite3.Row
        try:
            JobStore._create_schema(pristine)
            return JobStore._schema_fingerprint(pristine)
        finally:
            pristine.close()

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        metadata = connection.execute(
            "SELECT schema_descriptor,schema_fingerprint FROM store_metadata WHERE singleton=1"
        ).fetchone()
        if metadata is None or connection.execute(
            "SELECT COUNT(*) FROM store_metadata"
        ).fetchone()[0] != 1:
            raise CorruptStore("database schema metadata is missing or non-singular")
        actual = JobStore._schema_fingerprint(connection)
        expected = JobStore._expected_schema_fingerprint()
        if metadata["schema_descriptor"] != SCHEMA_DESCRIPTOR:
            raise CorruptStore("database schema descriptor mismatch")
        if metadata["schema_fingerprint"] != actual or actual != expected:
            raise CorruptStore("database actual-schema fingerprint mismatch")
    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _register_grant_from_authority(
        self, capability: object, access: AccessContext, grant: ExecutionGrant, binding: GrantBinding
    ) -> None:
        from ._authority import _is_registrar_capability
        if not _is_registrar_capability(capability):
            raise AccessDenied("grant registration requires authority capability")
        now = self._now()
        grant_digest = _digest("synaptic.execution-grant/v1", grant.grant_ref)
        with self._transaction() as connection:
            try:
                connection.execute(
                    """INSERT INTO grants VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)""",
                    (grant_digest, access.principal_ref, access.project_ref,
                     binding.scope.provider, binding.scope.account_ref, binding.scope.namespace_ref,
                     binding.operation_key, binding.plan_fingerprint, binding.source_digest,
                     binding.workload_digest, binding.artifact_slot_ref, binding.quote_digest,
                     binding.resource_digest, binding.allowed_secret_refs_digest,
                     binding.issued_at, binding.expires_at),
                )
            except sqlite3.IntegrityError as exc:
                raise OperationConflict("grant already exists") from exc
        del now

    def claim_submission(
        self,
        access: AccessContext,
        grant: ExecutionGrant,
        binding: GrantBinding,
        *,
        canonical_plan: str,
    ) -> SubmissionClaim:
        now = self._now()
        grant_digest = _digest("synaptic.execution-grant/v1", grant.grant_ref)
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM grants WHERE grant_digest=?", (grant_digest,)).fetchone()
            if row is None:
                raise GrantRejected("grant is not registered")
            expected = (access.principal_ref, access.project_ref, binding.scope.provider,
                        binding.scope.account_ref, binding.scope.namespace_ref, binding.operation_key,
                        binding.plan_fingerprint, binding.source_digest, binding.workload_digest,
                        binding.artifact_slot_ref, binding.quote_digest, binding.resource_digest,
                        binding.allowed_secret_refs_digest, binding.issued_at, binding.expires_at)
            actual = tuple(row[name] for name in (
                "principal_ref", "project_ref", "provider", "account_ref", "namespace_ref",
                "operation_key", "plan_fingerprint", "source_digest", "workload_digest",
                "artifact_slot_ref", "quote_digest", "resource_digest",
                "allowed_secret_refs_digest", "issued_at", "expires_at"))
            if actual != expected or binding.issued_at > now or now >= binding.expires_at:
                raise GrantRejected("grant binding or lifetime is invalid")
            existing = connection.execute(
                "SELECT * FROM runs WHERE project_ref=? AND operation_key=?",
                (access.project_ref, binding.operation_key),
            ).fetchone()
            if existing is not None:
                if existing["principal_ref"] != access.principal_ref:
                    raise AccessDenied("run ownership mismatch")
                if existing["plan_fingerprint"] != binding.plan_fingerprint or existing["grant_digest"] != grant_digest:
                    raise OperationConflict("operation key is already bound differently")
                effect = self._effect_row(connection, existing["run_id"], EffectKind.SUBMIT)
                return SubmissionClaim(
                    RunRef(existing["run_id"], existing["project_ref"]),
                    self._identity_from_row(effect), False, RunState(existing["state"]),
                )
            if row["consumed_run_id"] is not None:
                raise GrantRejected("grant has already been consumed")
            run_id = _require_text(self._id_factory("run"), "run_id")
            effect_id = _require_text(self._id_factory("effect"), "effect_id")
            effect_key = _digest(
                "synaptic.provider-effect/v1",
                "|".join((binding.scope.provider, binding.scope.account_ref,
                          binding.scope.namespace_ref, binding.operation_key, EffectKind.SUBMIT.value)),
            )
            connection.execute(
                """INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, access.project_ref, access.principal_ref, binding.scope.provider,
                 binding.scope.account_ref, binding.scope.namespace_ref, binding.operation_key,
                 binding.plan_fingerprint, canonical_plan, binding.source_digest,
                 binding.workload_digest, binding.artifact_slot_ref, grant_digest,
                 RunState.SUBMITTING.value, ArtifactVerificationState.PENDING.value,
                 1, MessageCode.AUTHORITY_CONSUMED.value, now, now),
            )
            connection.execute(
                """INSERT INTO effects(effect_id,run_id,kind,provider,account_ref,namespace_ref,
                   effect_key,status,claimed_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (effect_id, run_id, EffectKind.SUBMIT.value, binding.scope.provider,
                 binding.scope.account_ref, binding.scope.namespace_ref, effect_key, "claimed", now),
            )
            connection.execute(
                "UPDATE grants SET consumed_run_id=?, consumed_at=? WHERE grant_digest=? AND consumed_run_id IS NULL",
                (run_id, now, grant_digest),
            )
            connection.execute(
                "INSERT INTO log_meta VALUES (?,0,0,0,0,0)", (run_id,)
            )
            self._event(connection, run_id, 1, None, RunState.SUBMITTING,
                        EventCode.AUTHORIZED, MessageCode.AUTHORITY_CONSUMED, now)
            return SubmissionClaim(
                RunRef(run_id, access.project_ref),
                EffectIdentity(effect_id, effect_key, EffectKind.SUBMIT, binding.scope),
                True, RunState.SUBMITTING,
            )

    @staticmethod
    def _effect_row(connection: sqlite3.Connection, run_id: str, kind: EffectKind) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM effects WHERE run_id=? AND kind=?", (run_id, kind.value)
        ).fetchone()
        if row is None:
            raise CorruptStore("required effect record is missing")
        return row

    @staticmethod
    def _identity_from_row(row: sqlite3.Row) -> EffectIdentity:
        return EffectIdentity(
            row["effect_id"], row["effect_key"], EffectKind(row["kind"]),
            ExecutionScope(row["provider"], row["account_ref"], row["namespace_ref"]),
        )

    def effect(self, access: AccessContext, run: RunRef, kind: EffectKind) -> StoredEffect:
        with self._read() as connection:
            row = self._owned_run(connection, access, run)
            effect = self._effect_row(connection, row["run_id"], kind)
            job = ProviderJobRef(effect["provider_job_id"]) if effect["provider_job_id"] else None
            return StoredEffect(run, self._identity_from_row(effect), effect["status"], job)

    def mark_effect_started(self, access: AccessContext, run: RunRef, kind: EffectKind) -> StoredEffect:
        now = self._now()
        with self._transaction() as connection:
            row = self._owned_run(connection, access, run)
            effect = self._effect_row(connection, row["run_id"], kind)
            if effect["status"] != "claimed":
                raise InvalidTransition("effect is not claimable")
            state = RunState.SUBMITTING if kind is EffectKind.SUBMIT else RunState.CANCELLING
            revision = row["revision"] + 1
            connection.execute(
                "UPDATE effects SET status='started', started_at=? WHERE effect_id=? AND status='claimed'",
                (now, effect["effect_id"]),
            )
            connection.execute(
                "UPDATE runs SET state=?, revision=?, message_code=?, updated_at=? WHERE run_id=? AND revision=?",
                (state.value, revision, MessageCode.EFFECT_STARTED.value, now, run.run_id, row["revision"]),
            )
            event = EventCode.SUBMISSION_STARTED if kind is EffectKind.SUBMIT else EventCode.CANCELLATION_STARTED
            self._event(connection, run.run_id, revision, RunState(row["state"]), state,
                        event, MessageCode.EFFECT_STARTED, now)
            job = ProviderJobRef(effect["provider_job_id"]) if effect["provider_job_id"] else None
            return StoredEffect(run, self._identity_from_row(effect), "started", job)

    def finish_effect(
        self, access: AccessContext, run: RunRef, kind: EffectKind, *,
        effect_status: str, state: RunState, message: MessageCode,
        provider_job: ProviderJobRef | None = None, receipt_digest: str | None = None,
        observation_result: str | None = None,
        reconciliation_token: str | None = None,
    ) -> RunStatus:
        if effect_status not in {"confirmed", "absent", "ambiguous"}:
            raise ValueError("invalid final effect status")
        if effect_status == "confirmed" and (provider_job is None or receipt_digest is None):
            raise ValueError("confirmed effects require provider job and receipt digest")
        if provider_job is not None:
            _safe_ref(provider_job.provider_job_id, "provider_job_id")
        if receipt_digest is not None:
            _fixed_digest(receipt_digest, "receipt_digest")
        now = self._now()
        with self._transaction() as connection:
            row = self._owned_run(connection, access, run)
            effect = self._effect_row(connection, run.run_id, kind)
            if effect["status"] not in {"started", "claimed", "ambiguous"}:
                raise InvalidTransition("effect is already final")
            claim_digest = effect["reconcile_claim_digest"]
            supplied_digest = (_digest("synaptic.reconcile-claim/v1", reconciliation_token)
                               if reconciliation_token is not None else None)
            if claim_digest != supplied_digest:
                raise InvalidTransition("reconciliation claim token mismatch")
            persisted_provider_job_id = (
                provider_job.provider_job_id if provider_job is not None else None
            )
            if kind is EffectKind.CANCEL:
                original_target = effect["provider_job_id"]
                if original_target is None:
                    raise CorruptStore("cancellation effect is missing its target job")
                _safe_ref(original_target, "provider_job_id")
                if persisted_provider_job_id is not None and persisted_provider_job_id != original_target:
                    raise InvalidTransition("cancellation target cannot change")
                persisted_provider_job_id = original_target
            revision = row["revision"] + 1
            connection.execute(
                """UPDATE effects SET status=?, finished_at=?, provider_job_id=?,
                   receipt_digest=?, result_code=?, reconcile_claim_digest=NULL,
                   reconcile_claimed_at=NULL WHERE effect_id=?""",
                (effect_status, now, persisted_provider_job_id,
                 receipt_digest, observation_result or message.value, effect["effect_id"]),
            )
            connection.execute(
                "UPDATE runs SET state=?, revision=?, message_code=?, updated_at=? WHERE run_id=? AND revision=?",
                (state.value, revision, message.value, now, run.run_id, row["revision"]),
            )
            if observation_result is not None:
                connection.execute(
                    "INSERT INTO observations(run_id,kind,result_code,digest,observed_at) VALUES (?,?,?,?,?)",
                    (run.run_id, kind.value, observation_result, receipt_digest, now),
                )
            event = {
                (EffectKind.SUBMIT, "confirmed"): EventCode.SUBMITTED,
                (EffectKind.SUBMIT, "absent"): EventCode.NOT_SUBMITTED,
                (EffectKind.SUBMIT, "ambiguous"): EventCode.SUBMISSION_AMBIGUOUS,
                (EffectKind.CANCEL, "confirmed"): EventCode.CANCELLATION_ACCEPTED,
                (EffectKind.CANCEL, "absent"): EventCode.CANCELLATION_FAILED,
                (EffectKind.CANCEL, "ambiguous"): EventCode.CANCELLATION_AMBIGUOUS,
            }[(kind, effect_status)]
            self._event(connection, run.run_id, revision, RunState(row["state"]), state, event, message, now)
            return RunStatus(run, state, ArtifactVerificationState(row["artifact_state"]), now, message.value)

    def claim_reconcile(
        self, access: AccessContext, run: RunRef, kind: EffectKind
    ) -> ReconciliationClaim:
        now = self._now()
        token = self._new_token()
        token_digest = _digest("synaptic.reconcile-claim/v1", token)
        with self._transaction() as connection:
            row = self._owned_run(connection, access, run)
            effect = self._effect_row(connection, run.run_id, kind)
            if effect["status"] not in {"started", "ambiguous"}:
                raise InvalidTransition("only an uncertain started effect can be reconciled")
            if effect["reconcile_claim_digest"] is not None:
                raise InvalidTransition("effect reconciliation is already claimed")
            revision = row["revision"] + 1
            changed = connection.execute(
                "UPDATE effects SET reconcile_claim_digest=?,reconcile_claimed_at=? "
                "WHERE effect_id=? AND reconcile_claim_digest IS NULL",
                (token_digest, now, effect["effect_id"]),
            ).rowcount
            if changed != 1:
                raise InvalidTransition("effect reconciliation claim was lost")
            connection.execute(
                "UPDATE runs SET state=?,revision=?,message_code=?,updated_at=? WHERE run_id=? AND revision=?",
                (RunState.RECONCILING.value, revision, MessageCode.EFFECT_OUTCOME_UNKNOWN.value,
                 now, run.run_id, row["revision"]),
            )
            self._event(connection, run.run_id, revision, RunState(row["state"]), RunState.RECONCILING,
                        EventCode.RECONCILIATION_STARTED, MessageCode.EFFECT_OUTCOME_UNKNOWN, now)
            job = ProviderJobRef(effect["provider_job_id"]) if effect["provider_job_id"] else None
            stored = StoredEffect(run, self._identity_from_row(effect), effect["status"], job)
            return ReconciliationClaim(stored, token)
    def reconciliation_kind(self, access: AccessContext, run: RunRef) -> EffectKind:
        with self._read() as connection:
            self._owned_run(connection, access, run)
            rows = connection.execute(
                """SELECT kind FROM effects WHERE run_id=?
                   AND (status IN ('started','ambiguous') OR reconcile_claim_digest IS NOT NULL)
                   ORDER BY CASE kind WHEN 'cancel' THEN 0 ELSE 1 END""".replace("+", ""),
                (run.run_id,),
            ).fetchall()
            if len(rows) != 1:
                raise CorruptStore("run must have exactly one uncertain effect")
            return EffectKind(rows[0]["kind"])

    def recover(self) -> None:
        now = self._now()
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT r.*,e.status AS effect_status,e.kind AS effect_kind
                   FROM runs r JOIN effects e ON e.run_id=r.run_id
                   WHERE e.status IN ('claimed','started')
                      OR e.reconcile_claim_digest IS NOT NULL""".replace("+", "")
            ).fetchall()
            for row in rows:
                kind = EffectKind(row["effect_kind"])
                if row["effect_status"] == "claimed":
                    state = RunState.NOT_SUBMITTED if kind is EffectKind.SUBMIT else RunState.CANCEL_FAILED
                    effect_status = "absent"
                    message = MessageCode.EFFECT_DEFINITIVELY_ABSENT
                    event = EventCode.NOT_SUBMITTED if kind is EffectKind.SUBMIT else EventCode.CANCELLATION_FAILED
                else:
                    state = RunState.RECONCILE_REQUIRED
                    effect_status = row["effect_status"]
                    message = MessageCode.EFFECT_OUTCOME_UNKNOWN
                    event = (EventCode.SUBMISSION_AMBIGUOUS if kind is EffectKind.SUBMIT
                             else EventCode.CANCELLATION_AMBIGUOUS)
                revision = row["revision"] + 1
                connection.execute(
                    """UPDATE effects SET status=?,
                       finished_at=CASE WHEN ?='absent' THEN ? ELSE finished_at END,
                       reconcile_claim_digest=NULL,reconcile_claimed_at=NULL
                       WHERE run_id=? AND kind=?""".replace("+", ""),
                    (effect_status, effect_status, now, row["run_id"], kind.value),
                )
                connection.execute(
                    "UPDATE runs SET state=?,revision=?,message_code=?,updated_at=? WHERE run_id=?",
                    (state.value, revision, message.value, now, row["run_id"]),
                )
                self._event(connection, row["run_id"], revision, RunState(row["state"]),
                            state, event, message, now)
    def create_cancel_effect(self, access: AccessContext, run: RunRef) -> StoredEffect | None:
        now = self._now()
        terminal = {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED, RunState.NOT_SUBMITTED}
        with self._transaction() as connection:
            row = self._owned_run(connection, access, run)
            state = RunState(row["state"])
            if state in terminal:
                return None
            existing = connection.execute(
                "SELECT * FROM effects WHERE run_id=? AND kind=?", (run.run_id, EffectKind.CANCEL.value)
            ).fetchone()
            if existing is not None:
                job = ProviderJobRef(existing["provider_job_id"]) if existing["provider_job_id"] else None
                return StoredEffect(run, self._identity_from_row(existing), existing["status"], job)
            submit = self._effect_row(connection, run.run_id, EffectKind.SUBMIT)
            if not submit["provider_job_id"]:
                raise InvalidTransition("submission must be reconciled before cancellation")
            effect_id = _require_text(self._id_factory("effect"), "effect_id")
            effect_key = _digest("synaptic.provider-effect/v1", submit["effect_key"] + "|cancel")
            connection.execute(
                """INSERT INTO effects(effect_id,run_id,kind,provider,account_ref,namespace_ref,
                   effect_key,status,claimed_at,provider_job_id) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (effect_id, run.run_id, EffectKind.CANCEL.value, row["provider"], row["account_ref"],
                 row["namespace_ref"], effect_key, "claimed", now, submit["provider_job_id"]),
            )
            revision = row["revision"] + 1
            connection.execute(
                "UPDATE runs SET state=?,revision=?,message_code=?,updated_at=? WHERE run_id=?",
                (RunState.CANCEL_REQUESTED.value, revision, MessageCode.AUTHORITY_CONSUMED.value, now, run.run_id),
            )
            self._event(connection, run.run_id, revision, state, RunState.CANCEL_REQUESTED,
                        EventCode.CANCEL_REQUESTED, MessageCode.AUTHORITY_CONSUMED, now)
            return StoredEffect(
                run, EffectIdentity(effect_id, effect_key, EffectKind.CANCEL,
                                    ExecutionScope(row["provider"], row["account_ref"], row["namespace_ref"])),
                "claimed", ProviderJobRef(submit["provider_job_id"]),
            )

    def observe_state(
        self, access: AccessContext, run: RunRef, state: RunState,
        observation_digest: str,
    ) -> RunStatus:
        _fixed_digest(observation_digest, "observation_digest")
        now = self._now()
        with self._transaction() as connection:
            row = self._owned_run(connection, access, run)
            revision = row["revision"] + 1
            message = MessageCode.PROVIDER_STATE_OBSERVED
            connection.execute(
                "INSERT INTO observations(run_id,kind,result_code,digest,observed_at) VALUES (?,?,?,?,?)",
                (run.run_id, "status", state.value, observation_digest, now),
            )
            connection.execute(
                "UPDATE runs SET state=?,revision=?,message_code=?,updated_at=? WHERE run_id=?",
                (state.value, revision, message.value, now, run.run_id),
            )
            self._event(connection, run.run_id, revision, RunState(row["state"]), state,
                        EventCode.STATE_OBSERVED, message, now)
            return RunStatus(run, state, ArtifactVerificationState(row["artifact_state"]), now, message.value)

    def status(self, access: AccessContext, run: RunRef) -> RunStatus:
        with self._read() as connection:
            row = self._owned_run(connection, access, run)
            return RunStatus(run, RunState(row["state"]), ArtifactVerificationState(row["artifact_state"]),
                             row["updated_at"], row["message_code"])

    def list_runs(self, access: AccessContext, limit: int = 50) -> tuple[RunRef, ...]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._read() as connection:
            rows = connection.execute(
                "SELECT run_id,project_ref FROM runs WHERE project_ref=? AND principal_ref=? ORDER BY created_at,run_id LIMIT ?",
                (access.project_ref, access.principal_ref, limit),
            ).fetchall()
            return tuple(RunRef(row["run_id"], row["project_ref"]) for row in rows)

    def logs(self, access: AccessContext, run: RunRef, cursor: LogCursor | None, limit: int) -> LogPage:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_QUERY_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_QUERY_LIMIT}")
        with self._transaction() as connection:
            self._owned_run(connection, access, run)
            after = self._decode_cursor(connection, run, cursor) if cursor else -1
            meta = connection.execute("SELECT * FROM log_meta WHERE run_id=?", (run.run_id,)).fetchone()
            if meta is None:
                raise CorruptStore("log metadata is missing")
            truncated = after < meta["floor_sequence"] - 1
            effective_after = max(after, meta["floor_sequence"] - 1)
            rows = connection.execute(
                "SELECT * FROM logs WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                (run.run_id, effective_after, limit + 1),
            ).fetchall()
            more = len(rows) > limit
            rows = rows[:limit]
            entries = tuple(LogEntry(row["sequence"], row["timestamp"], row["level"],
                                     row["event"], row["message"]) for row in rows)
            has_more = bool(rows) and (more or rows[-1]["sequence"] < meta["next_sequence"] - 1)
            next_cursor = self._encode_cursor(connection, run, rows[-1]["sequence"]) if has_more else None
            return LogPage(run, entries, next_cursor, truncated)
    def _event(
        self, connection: sqlite3.Connection, run_id: str, revision: int,
        prior: RunState | None, state: RunState, event: EventCode,
        message: MessageCode, now: str,
    ) -> None:
        level = LogLevel.ERROR if "ambiguous" in state.value or state in {RunState.FAILED, RunState.CANCEL_FAILED} else LogLevel.INFO
        level_value, event_value, message_value = checked_log_fields(level, event, message)
        connection.execute(
            "INSERT INTO state_events(run_id,revision,prior_state,state,event_code,message_code,created_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, revision, prior.value if prior else None, state.value, event_value, message_value, now),
        )
        self._append_log(connection, run_id, now, level_value, event_value, message_value)

    def _append_log(self, connection: sqlite3.Connection, run_id: str, now: str,
                    level: str, event: str, message: str) -> None:
        meta = connection.execute("SELECT * FROM log_meta WHERE run_id=?", (run_id,)).fetchone()
        if meta is None:
            raise CorruptStore("log metadata is missing")
        byte_count = len((now + level + event + message).encode("utf-8"))
        sequence = meta["next_sequence"]
        connection.execute("INSERT INTO logs VALUES (?,?,?,?,?,?,?)",
                           (run_id, sequence, now, level, event, message, byte_count))
        records = meta["retained_records"] + 1
        retained_bytes = meta["retained_bytes"] + byte_count
        floor = meta["floor_sequence"]
        dropped = meta["dropped_records"]
        while records > MAX_RETAINED_RECORDS or retained_bytes > MAX_RETAINED_BYTES:
            oldest = connection.execute(
                "SELECT sequence,byte_count FROM logs WHERE run_id=? ORDER BY sequence LIMIT 1", (run_id,)
            ).fetchone()
            if oldest is None:
                raise CorruptStore("log accounting is inconsistent")
            connection.execute("DELETE FROM logs WHERE run_id=? AND sequence=?", (run_id, oldest["sequence"]))
            records -= 1
            retained_bytes -= oldest["byte_count"]
            floor = oldest["sequence"] + 1
            dropped += 1
        connection.execute(
            "UPDATE log_meta SET next_sequence=?,floor_sequence=?,retained_records=?,retained_bytes=?,dropped_records=? WHERE run_id=?",
            (sequence + 1, floor, records, retained_bytes, dropped, run_id),
        )

    @staticmethod
    def _owned_run(connection: sqlite3.Connection, access: AccessContext, run: RunRef) -> sqlite3.Row:
        if run.project_ref != access.project_ref:
            raise AccessDenied("project ownership mismatch")
        row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run.run_id,)).fetchone()
        if row is None or row["project_ref"] != access.project_ref or row["principal_ref"] != access.principal_ref:
            raise AccessDenied("run is unavailable")
        return row

    def _encode_cursor(
        self, connection: sqlite3.Connection, run: RunRef, sequence: int
    ) -> LogCursor:
        token = self._new_token()
        token_digest = _digest("synaptic.log-cursor/v2", token)
        now = self._now()
        cutoff = (datetime.fromisoformat(now.replace("Z", "+00:00"))
                  - timedelta(seconds=CURSOR_TTL_SECONDS)).isoformat().replace("+00:00", "Z")
        connection.execute("DELETE FROM cursors WHERE created_at<=?", (cutoff,))
        connection.execute(
            """DELETE FROM cursors WHERE token_digest IN (
                 SELECT token_digest FROM cursors WHERE run_id=?
                 ORDER BY created_at DESC,token_digest DESC LIMIT -1 OFFSET ?
               )""",
            (run.run_id, MAX_CURSOR_ROWS_PER_RUN - 1),
        )
        try:
            connection.execute(
                "INSERT INTO cursors(token_digest,run_id,after_sequence,created_at) VALUES (?,?,?,?)",
                (token_digest, run.run_id, sequence, now),
            )
        except sqlite3.IntegrityError as exc:
            raise CorruptStore("opaque cursor token collision") from exc
        return LogCursor(token)

    def _decode_cursor(
        self, connection: sqlite3.Connection, run: RunRef, cursor: LogCursor
    ) -> int:
        if len(cursor.value) != 43 or re.fullmatch(r"[A-Za-z0-9_-]{43}", cursor.value) is None:
            raise ValueError("invalid log cursor")
        now = self._now()
        cutoff = (datetime.fromisoformat(now.replace("Z", "+00:00"))
                  - timedelta(seconds=CURSOR_TTL_SECONDS)).isoformat().replace("+00:00", "Z")
        connection.execute("DELETE FROM cursors WHERE created_at<=?", (cutoff,))
        token_digest = _digest("synaptic.log-cursor/v2", cursor.value)
        row = connection.execute(
            "SELECT run_id,after_sequence FROM cursors WHERE token_digest=?",
            (token_digest,),
        ).fetchone()
        if row is None or row["run_id"] != run.run_id:
            raise ValueError("invalid log cursor")
        return int(row["after_sequence"])

def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


__all__ = [
    "AccessDenied", "CorruptStore", "GrantBinding", "GrantRejected",
    "InvalidTransition", "JobStore", "JobStoreLocation", "OperationConflict", "SCHEMA_VERSION",
    "StoredEffect", "SubmissionClaim", "canonical_json",
]
