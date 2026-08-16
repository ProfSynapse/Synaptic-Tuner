"""Crash-safe incremental persistence and resume for the batch verbs.

Location: tuner/batch/persistence.py
Purpose: The durability contract shared by ``batch-generate`` and
    ``batch-capture``. This module is the actual point of the batch feature:
    a preempted or killed job must never lose completed work and must resume
    exactly where it left off.
Used by: tuner.batch.runner, the two batch handlers.

Contract
--------
* Every output row is either FULLY persisted or ABSENT — never a truncated
  JSON line and never a half-written tensor file. JSONL rows are written as a
  single ``line + "\n"`` write followed by ``flush()`` + ``os.fsync()``. Tensor
  files are written to a temp name and atomically ``os.replace``-d into place.
* A ``checkpoint.json`` in the out-dir tracks the set of done ids plus a config
  hash of the invocation (model, engine, decode params, seed, ...). Resuming
  with a different config is refused with a clear error.
* Processing order is deterministic; the done-set skip is BY ID, not by index,
  so a completed-then-resumed run produces the identical artifact set as an
  uninterrupted run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


CHECKPOINT_FILENAME = "checkpoint.json"
_CHECKPOINT_VERSION = 1


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Return a stable hash of the invocation config.

    The config dict is serialized with sorted keys so key ordering never
    affects the hash. Any value that changes *what* is produced (model, engine,
    decode params, seed, layers, persist dtype, ...) must be included by the
    caller; incidental values (out-dir, sync command, resume flag) must not.
    """
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sanitize_id(row_id: str) -> str:
    """Map an arbitrary row id to a filesystem-safe stem.

    Collisions across distinct ids are avoided by appending a short hash of the
    original id whenever sanitization actually changed the string.
    """
    text = str(row_id)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    # Guard against empty / dotfile stems and against pathological lengths.
    if not safe or safe in {".", ".."}:
        safe = "row"
    if safe != text or len(safe) > 128:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        safe = f"{safe[:100]}_{digest}"
    return safe


def _fsync_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/create is durable. Best-effort.

    Directory fsync is not supported on every platform/filesystem (Windows in
    particular); failures here must never abort a run.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        pass


class JsonlAppender:
    """Append complete JSON rows to a JSONL file, one durable write per row.

    Each row is serialized to a single line and written with a trailing
    newline in one ``write`` call, then flushed and fsynced. A crash therefore
    leaves the file ending on a complete line: readers never see a partial row.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            _fsync_file(handle)

    def append_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        """Append several rows, fsyncing once after all are written.

        Used for a batch flush: the whole batch is written then fsynced, so a
        crash mid-batch still leaves only complete lines (Python buffers the
        writes and each ``json.dumps`` line is newline-terminated).
        """
        rows = list(rows)
        if not rows:
            return
        with open(self.path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            _fsync_file(handle)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` atomically via a temp file + ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        _fsync_file(handle)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def read_jsonl_ids(path: Path, id_field: str = "id") -> List[str]:
    """Return the ids already present in a JSONL artifact.

    Tolerant of a trailing partial line (which the atomic append contract makes
    impossible, but resume must be robust to a torn file from any source):
    a line that fails to parse or lacks the id field is skipped.
    """
    path = Path(path)
    ids: List[str] = []
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if id_field in row:
                ids.append(str(row[id_field]))
    return ids


class ConfigMismatchError(RuntimeError):
    """Raised when ``--resume`` targets an out-dir written by a different config."""


class RunCheckpoint:
    """Tracks done ids + the invocation config hash in ``checkpoint.json``.

    The done set is the source of truth for skip-on-resume. It is reconciled
    against the actual JSONL index at load time so a row that made it into the
    index but not the checkpoint (crash between the two writes) is still counted
    as done — the artifacts, not the bookkeeping, define completion.
    """

    def __init__(
        self,
        out_dir: Path,
        config_hash: str,
        *,
        state_dir: Path | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.state_dir = Path(state_dir) if state_dir is not None else self.out_dir
        self.config_hash = config_hash
        self.done_ids: Set[str] = set()
        self.path = self.state_dir / CHECKPOINT_FILENAME

    @classmethod
    def load_or_create(
        cls,
        out_dir: Path,
        config: Dict[str, Any],
        *,
        resume: bool,
        index_ids: Optional[Iterable[str]] = None,
        state_dir: Path | None = None,
    ) -> "RunCheckpoint":
        """Load an existing checkpoint (verifying config) or create a fresh one.

        Args:
            out_dir: The run output directory.
            config: The hashable invocation config (see ``compute_config_hash``).
            resume: If True, an existing checkpoint is loaded and its config hash
                must match. If False and a checkpoint already exists, that is an
                error (refuse to silently overwrite / mix a prior run).
            index_ids: Ids observed in the JSONL artifact, used to reconcile the
                done set with what actually landed on disk.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_state_dir = Path(state_dir) if state_dir is not None else out_dir
        selected_state_dir.mkdir(parents=True, exist_ok=True)
        config_hash = compute_config_hash(config)
        cp = cls(out_dir, config_hash, state_dir=selected_state_dir)
        cp_path = selected_state_dir / CHECKPOINT_FILENAME

        if cp_path.exists():
            existing = json.loads(cp_path.read_text(encoding="utf-8"))
            existing_hash = existing.get("config_hash")
            if not resume:
                raise ConfigMismatchError(
                    f"Output directory {out_dir} already contains a run "
                    f"({CHECKPOINT_FILENAME}). Pass --resume to continue it, or "
                    f"choose a fresh --out-dir. Refusing to overwrite."
                )
            if existing_hash != config_hash:
                raise ConfigMismatchError(
                    "Cannot --resume: the checkpoint in "
                    f"{out_dir} was written with a different configuration.\n"
                    f"  existing config_hash: {existing_hash}\n"
                    f"  current  config_hash: {config_hash}\n"
                    "Model, engine, decode params, or seed changed. Resume only "
                    "works across identical configs; use a fresh --out-dir."
                )
            cp.done_ids = set(str(x) for x in existing.get("done_ids", []))

        if index_ids is not None:
            cp.done_ids.update(str(x) for x in index_ids)
        # Persist the (possibly reconciled) checkpoint so a subsequent crash
        # before the first batch still has a valid, hash-stamped checkpoint.
        cp._write()
        return cp

    def is_done(self, row_id: str) -> bool:
        return str(row_id) in self.done_ids

    def mark_done(self, row_ids: Iterable[str]) -> None:
        """Record ids as done and durably rewrite the checkpoint."""
        for rid in row_ids:
            self.done_ids.add(str(rid))
        self._write()

    def _write(self) -> None:
        payload = {
            "version": _CHECKPOINT_VERSION,
            "config_hash": self.config_hash,
            "done_ids": sorted(self.done_ids),
            "count": len(self.done_ids),
        }
        atomic_write_bytes(
            self.path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
