"""Append-only, fsynced, hash-chained JSONL log.

Each line is exactly ``canonical_json(record)``. A record carries ``run_id``,
``seq``, ``prev`` (digest of the previous record), ``type``, ``time``, ``data``,
``linked`` and its own ``digest`` over everything else. Verification fails
closed: a bad line, gap, altered byte or partial tail raises. The only repair
is an explicit audited quarantine of an incomplete final line.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from .canon import canonical_bytes, fsync_dir, sha256_hex, strict_loads
from .errors import JournalCorrupt, PartialTail


def genesis_prev(binding_digest: str, log_name: str) -> str:
    return sha256_hex(f"{binding_digest}:{log_name}".encode())


def _digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "digest"}
    return sha256_hex(canonical_bytes(body))


def verify_file(path: Path, run_id: str, genesis: str) -> list[dict]:
    """Return all records or raise ``JournalCorrupt`` / ``PartialTail``."""
    try:
        blob = Path(path).read_bytes()
    except FileNotFoundError:
        return []
    if not blob:
        return []
    lines = blob.split(b"\n")
    partial = lines.pop()  # '' if the file ends with a newline
    records: list[dict] = []
    prev = genesis
    for i, line in enumerate(lines):
        where = f"{path.name} line {i + 1}"
        try:
            rec = strict_loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise JournalCorrupt(f"{where}: not valid JSON", line=i + 1) from exc
        if not isinstance(rec, dict) or canonical_bytes(rec) != line:
            raise JournalCorrupt(f"{where}: not canonical record bytes", line=i + 1)
        if rec.get("run_id") != run_id:
            raise JournalCorrupt(f"{where}: run_id mismatch", line=i + 1)
        if rec.get("seq") != i + 1:
            raise JournalCorrupt(f"{where}: sequence gap/reorder", line=i + 1)
        if rec.get("prev") != prev:
            raise JournalCorrupt(f"{where}: previous-digest link broken", line=i + 1)
        if rec.get("digest") != _digest(rec):
            raise JournalCorrupt(f"{where}: digest mismatch (record altered)", line=i + 1)
        prev = rec["digest"]
        records.append(rec)
    if partial:
        raise PartialTail(
            f"{path.name}: incomplete final line ({len(partial)} bytes) after {len(records)} valid records",
            valid_records=len(records),
            valid_bytes=len(blob) - len(partial),
            partial_bytes=len(partial),
        )
    return records


class HashChainLog:
    def __init__(
        self,
        path: Path,
        run_id: str,
        genesis: str,
        clock: Callable[[], float] = time.time,
        _allow_missing: bool = True,
    ):
        self.path = Path(path)
        self.run_id = run_id
        self.genesis = genesis
        self._clock = clock
        self.records: list[dict] = verify_file(self.path, run_id, genesis)
        new = not self.path.exists()
        self._fd = os.open(str(self.path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        if new:
            fsync_dir(self.path.parent)

    @property
    def head(self) -> str:
        return self.records[-1]["digest"] if self.records else self.genesis

    @property
    def next_seq(self) -> int:
        return len(self.records) + 1

    def append(self, rtype: str, data: dict, linked: dict | None = None) -> dict:
        rec: dict[str, Any] = {
            "run_id": self.run_id,
            "seq": self.next_seq,
            "prev": self.head,
            "type": rtype,
            "time": round(self._clock(), 6),
            "data": data,
            "linked": linked or {},
        }
        rec["digest"] = _digest(rec)
        line = canonical_bytes(rec) + b"\n"
        view = memoryview(line)
        while view:
            n = os.write(self._fd, view)
            view = view[n:]
        os.fsync(self._fd)
        self.records.append(rec)
        return rec

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def quarantine_partial_tail(path: Path, quarantine_dir: Path, run_id: str, genesis: str) -> dict:
    """Move an incomplete final line aside (never deleted) and return its audit facts.

    Preconditions are re-checked: every complete line must verify and the
    remainder must lack a trailing newline.
    """
    try:
        verify_file(path, run_id, genesis)
        raise JournalCorrupt(f"{path.name} has no partial tail to recover")
    except PartialTail as exc:
        keep = exc.details["valid_bytes"]
    blob = path.read_bytes()
    tail = blob[keep:]
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(quarantine_dir.iterdir())) + 1
    qpath = quarantine_dir / f"{path.name}.{n:03d}.partial"
    fd = os.open(str(qpath), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, tail)
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(quarantine_dir)
    with open(path, "r+b") as fh:
        fh.truncate(keep)
        fh.flush()
        os.fsync(fh.fileno())
    return {
        "log": path.name,
        "quarantined_file": str(qpath.relative_to(quarantine_dir.parent)),
        "quarantined_sha256": sha256_hex(tail),
        "quarantined_bytes": len(tail),
        "valid_prefix_bytes": keep,
    }
