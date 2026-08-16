"""
shared/experiment_tracking/registry.py

Append-only JSONL registry for unified run tracking. Stores RunRecord entries
and supports query/filter operations. Handles run linkage (e.g. eval -> training).

Used by: local_tracker.py (auto-registration), adapters.py (manual registration),
         CLI list-runs (query), Evaluator (parent linkage).
"""
from __future__ import annotations

import errno
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, BinaryIO

from .schema import RunFilter, RunRecord

logger = logging.getLogger(__name__)

# Link records are stored as separate JSONL lines alongside RunRecords.
# They have a "__link__" marker to distinguish them from run entries.
_LINK_MARKER = "__link__"

_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_POLL_SECONDS = 0.01
_REPLACE_TIMEOUT_SECONDS = 10.0
_REPLACE_POLL_SECONDS = 0.01
_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _is_lock_contention(error: OSError) -> bool:
    """Return whether an open/publish error is transient lock contention."""

    return (
        isinstance(error, PermissionError)
        or error.errno in {errno.EACCES, errno.EAGAIN, errno.EBUSY}
        or getattr(error, "winerror", None) in {32, 33}
    )


def _is_windows_replace_contention(error: OSError) -> bool:
    """Classify only Windows sharing/access violations as retryable."""

    return os.name == "nt" and getattr(error, "winerror", None) in {5, 32, 33}


def _replace_with_retry(source: Path | str, target: Path | str) -> None:
    """Replace a registry while retaining its complete temp across contention."""

    deadline = time.monotonic() + _REPLACE_TIMEOUT_SECONDS
    while True:
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if not _is_windows_replace_contention(exc):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out replacing registry after Windows sharing violation: {target}"
                ) from exc
            time.sleep(min(_REPLACE_POLL_SECONDS, remaining))


def _thread_lock_for(path: Path) -> threading.RLock:
    key = path.resolve()
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


class _PathLock(AbstractContextManager["_PathLock"]):
    """Bounded thread/process lock backed by a kernel-owned file lock.

    The adjacent file persists as diagnostic owner metadata. Recovery never
    relies on age or PID probing: the operating system releases the lock when
    its owning process exits, including abnormal termination.
    """

    def __init__(self, target: Path) -> None:
        self._target = target.resolve()
        self._lock_path = self._target.with_name(f"{self._target.name}.lock")
        self._thread_lock = _thread_lock_for(self._target)
        self._token = uuid.uuid4().hex
        self._handle: BinaryIO | None = None

    def __enter__(self) -> "_PathLock":
        self._thread_lock.acquire()
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                handle = self._open_published_lock_file()
                if handle is not None and self._try_native_lock(handle):
                    self._handle = handle
                    self._write_owner_metadata()
                    return self
                if handle is not None:
                    handle.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring registry lock: {self._lock_path}"
                    )
                time.sleep(_LOCK_POLL_SECONDS)
        except Exception:
            self._thread_lock.release()
            raise

    def _open_published_lock_file(self) -> BinaryIO | None:
        """Open only a complete sentinel, publishing it atomically if absent."""

        if not self._lock_path.exists() and not self._publish_lock_sentinel():
            return None
        try:
            handle = self._lock_path.open("r+b")
        except FileNotFoundError:
            return None
        except OSError as exc:
            # Windows sharing violations during another process's open/close
            # are contention, not terminal failures.
            if _is_lock_contention(exc):
                return None
            raise
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() < 1:
                handle.close()
                return None
            handle.seek(0)
            return handle
        except OSError:
            handle.close()
            return None

    def _publish_lock_sentinel(self) -> bool:
        """Publish a fully flushed sentinel without exposing an empty file."""

        fd = -1
        tmp = ""
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(self._lock_path.parent),
                prefix=f".{self._lock_path.name}.",
                suffix=".init",
            )
            if os.write(fd, b"\0") != 1:
                raise OSError("Could not initialize registry lock sentinel")
            os.fsync(fd)
            os.close(fd)
            fd = -1
            try:
                os.link(tmp, self._lock_path)
            except FileExistsError:
                pass
            except OSError as exc:
                if _is_lock_contention(exc):
                    return False
                raise
            return self._lock_path.exists()
        except OSError as exc:
            if _is_lock_contention(exc):
                return False
            raise
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp:
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _try_native_lock(handle: BinaryIO) -> bool:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            elif sys.platform.startswith(("linux", "darwin", "freebsd")):
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                raise RuntimeError("No supported native registry lock is available")
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    def _write_owner_metadata(self) -> None:
        assert self._handle is not None
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "token": self._token,
                "acquired_at_ns": time.time_ns(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(b"\0" + payload)
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def _release_owner_metadata(self) -> None:
        assert self._handle is not None
        self._handle.seek(1)
        try:
            payload = json.loads(self._handle.read().decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if payload.get("token") != self._token:
            logger.error("Registry lock owner token changed before release: %s", self._lock_path)
            return
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(b"\0")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    @staticmethod
    def _native_unlock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._handle is not None:
                try:
                    self._release_owner_metadata()
                finally:
                    try:
                        self._native_unlock(self._handle)
                    finally:
                        self._handle.close()
                        self._handle = None
        finally:
            self._thread_lock.release()


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
        # In-memory cache invalidated by file mtime change
        self._cache_records: list[RunRecord] | None = None
        self._cache_links: list[dict[str, Any]] | None = None
        self._cache_mtime: float = 0.0

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

        with _PathLock(self._path):
            # Re-read under the path lock so idempotency covers all writers.
            existing = self._scan_records()
            for existing_record in existing:
                if existing_record.output_dir == record.output_dir:
                    logger.warning(
                        "Skipping duplicate registration for output_dir %s (existing run: %s)",
                        record.output_dir, existing_record.run_id,
                    )
                    return existing_record.run_id

            previous = self._path.read_bytes() if self._path.exists() else b""
            separator = b"" if not previous or previous.endswith((b"\n", b"\r")) else b"\n"
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(previous)
                    handle.write(separator)
                    handle.write(line.encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                _replace_with_retry(tmp, self._path)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
                raise

        self._invalidate_cache()
        logger.info("Registered run %s (%s)", record.run_id, record.run_type)
        return record.run_id

    @property
    def _links_path(self) -> Path:
        """Path to the separate links JSONL file alongside the registry."""
        return self._path.parent / "links.jsonl"

    def link_runs(
        self, child_run_id: str, parent_run_id: str, relationship: str = "parent"
    ) -> None:
        """Record a link between two runs (e.g. evaluation -> training).

        Links are stored in a separate links.jsonl file alongside the registry.
        For backward compatibility, link records in the main registry file are
        still read (but new links are always written to links.jsonl).

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
        with open(self._links_path, "a", encoding="utf-8") as f:
            f.write(line)
        self._invalidate_cache()

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

    # -- Cache management ---------------------------------------------------

    def _current_mtime(self) -> float:
        """Return the combined mtime of registry + links files (0.0 if absent)."""
        mtime = 0.0
        try:
            mtime += self._path.stat().st_mtime
        except OSError:
            pass
        try:
            mtime += self._links_path.stat().st_mtime
        except OSError:
            pass
        return mtime

    def _invalidate_cache(self) -> None:
        """Force a cache refresh on the next read."""
        self._cache_records = None
        self._cache_links = None
        self._cache_mtime = 0.0

    def _ensure_cache(self) -> None:
        """Reload cache if the file has been modified since last read."""
        current_mtime = self._current_mtime()
        if self._cache_records is not None and current_mtime == self._cache_mtime:
            return
        self._cache_records = self._scan_records()
        self._cache_links = self._scan_links()
        self._cache_mtime = current_mtime

    # -- Low-level I/O (no caching) ----------------------------------------

    def _scan_records(self) -> list[RunRecord]:
        """Scan the registry file for RunRecord entries, skipping malformed lines."""
        if not self._path.exists():
            return []

        records: list[RunRecord] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    if _LINK_MARKER in data:
                        continue  # Skip link records
                    records.append(RunRecord.from_dict(data))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    logger.warning(
                        "Skipping malformed line %d in %s: %s",
                        line_num, self._path, exc,
                    )
        return records

    def _scan_links(self) -> list[dict[str, Any]]:
        """Scan both the separate links file and the registry for link records.

        New links are written to links.jsonl. For backward compatibility, link
        records embedded in registry.jsonl are also loaded.
        """
        links: list[dict[str, Any]] = []

        # Read from dedicated links file first (preferred location)
        if self._links_path.exists():
            with open(self._links_path, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                        if _LINK_MARKER in data:
                            links.append(data)
                    except (json.JSONDecodeError, TypeError):
                        continue

        # Also check the main registry for legacy link records
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                        if _LINK_MARKER in data:
                            links.append(data)
                    except (json.JSONDecodeError, TypeError):
                        continue

        return links

    # -- Cached read methods -----------------------------------------------

    def _load_records(self) -> list[RunRecord]:
        """Return cached RunRecord entries, refreshing if file changed."""
        self._ensure_cache()
        return list(self._cache_records or [])

    def _load_links(self) -> list[dict[str, Any]]:
        """Return cached link records, refreshing if file changed."""
        self._ensure_cache()
        return list(self._cache_links or [])
