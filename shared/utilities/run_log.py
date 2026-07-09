"""Resumable per-item run log for long evaluation loops.

A multi-hour per-item evaluation loop (generate, then grade, one row at a
time) that buffers every result in memory and writes output only at the end
loses the entire run to a crash in the final stretch. ``RunLog`` gives such
a loop a durable, append-only per-item record plus an atomic summary write,
so a killed process can be restarted and pick up exactly where it left off,
and a separate process can peek at progress cheaply without disturbing the
writer.

Layout for a log opened at ``path``:

- ``<path>``              append-only JSON-lines log, one record per item.
- ``<path>.meta.json``    run identity: schema_version + a fingerprint over
                          the run config, plus a completion flag.
- ``<path>.summary.json`` final aggregate, written atomically by finalize().

Keys are opaque strings. Composite keys (arm + row + dose, say) should be
joined into a single string by the caller before calling record() /
iter_pending() -- a JSON round trip cannot distinguish a tuple key from a
list, so treating keys as anything but plain strings is a latent bug.

Each record is appended with a flush + fsync. At the multi-second-per-item
cadence typical of a generate-then-grade loop, the fsync cost (single-digit
milliseconds) is immaterial next to the item cost, and it is what makes the
log survive a hard kill: once record() returns, that item is durable on
disk even if the process dies on the very next line.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

SCHEMA_VERSION = 1

T = TypeVar("T")


class RunLogError(RuntimeError):
    """Raised when a run log cannot be safely opened, resumed, or written."""


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(run_config: dict, schema_version: int) -> str:
    payload = _canonical_json({"schema_version": schema_version, "run_config": run_config})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scan(path: Path, key_field: str) -> tuple[dict[str, dict], int]:
    """Scan a JSONL log, returning ``{key: record}`` for well-formed lines
    and the byte offset immediately following the last well-formed line.

    A malformed FINAL line is the one shape of corruption a durable append
    log admits: every prior line already reached fsync before the next
    write began, so only the tail can be torn by a kill mid-append. That
    line is dropped silently and the returned offset excludes it, so the
    caller can truncate the file back to a clean state. A malformed line
    that is NOT the final one means something other than a crash mid-append
    corrupted the file, and this raises rather than silently discarding
    history.
    """
    records: dict[str, dict] = {}
    if not path.exists() or path.stat().st_size == 0:
        return records, 0

    data = path.read_bytes()
    n = len(data)
    good_end = 0
    pos = 0
    while pos < n:
        newline_at = data.find(b"\n", pos)
        is_final_chunk = newline_at == -1
        end = n if is_final_chunk else newline_at
        raw_line = data[pos:end]
        line = raw_line.strip()
        next_pos = n if is_final_chunk else newline_at + 1

        if not line:
            good_end = next_pos
            pos = next_pos
            continue

        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if is_final_chunk:
                break
            raise RunLogError(
                f"malformed non-final line in run log {path} at byte offset "
                f"{pos}; this is corruption beyond the tolerated "
                "torn-final-line case and will not be silently discarded"
            )

        key = rec.get(key_field)
        if key is not None:
            records[str(key)] = rec
        good_end = next_pos
        pos = next_pos

    return records, good_end


class RunLog:
    """Append-only, resumable per-item log for one evaluation run."""

    def __init__(
        self,
        path: str | Path,
        run_config: dict,
        *,
        fresh: bool = False,
        key_field: str = "key",
    ) -> None:
        self.path = Path(path)
        self.meta_path = self.path.with_name(self.path.name + ".meta.json")
        self.summary_path = self.path.with_name(self.path.name + ".summary.json")
        self.key_field = key_field
        self._fingerprint = _fingerprint(run_config, SCHEMA_VERSION)

        if fresh:
            for candidate in (self.path, self.meta_path, self.summary_path):
                candidate.unlink(missing_ok=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != self._fingerprint:
                raise RunLogError(
                    f"run_config fingerprint mismatch for {self.path}: this "
                    "log was opened with a different run_config. Resume "
                    "must be the same run -- use a new path, or pass "
                    "fresh=True to intentionally start over."
                )
        else:
            self._write_meta(complete=False)

        self._records, good_end = _scan(self.path, self.key_field)
        if self.path.exists() and self.path.stat().st_size > good_end:
            # Torn final line from a kill mid-append: drop it before the
            # next append lands.
            with self.path.open("r+b") as fh:
                fh.truncate(good_end)

        self._fh = self.path.open("ab")

    # -- lifecycle ---------------------------------------------------

    def _write_meta(self, *, complete: bool) -> None:
        meta = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self._fingerprint,
            "complete": complete,
        }
        tmp_path = self.meta_path.with_name(self.meta_path.name + ".tmp")
        tmp_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.meta_path)

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- reading -------------------------------------------------------

    def done_keys(self) -> set[str]:
        """Completed item keys: the initial scan at open, kept current by
        every record() call made through this instance."""
        return set(self._records.keys())

    def iter_pending(self, items: Iterable[T], key_fn: Callable[[T], Any]) -> Iterator[T]:
        """Yield items from ``items`` whose key is not yet done, in order."""
        done = self.done_keys()
        for item in items:
            if str(key_fn(item)) in done:
                continue
            yield item

    @classmethod
    def peek_done_keys(cls, path: str | Path, *, key_field: str = "key") -> set[str]:
        """Read-only snapshot of completed keys.

        Safe to call from another process while a RunLog writer is live on
        the same path: this never opens the file for writing and never
        truncates it, so it cannot race or corrupt the writer. A torn final
        line from a writer mid-append is simply excluded from the snapshot.
        """
        records, _ = _scan(Path(path), key_field)
        return set(records.keys())

    # -- writing ---------------------------------------------------------

    def record(self, key: str, payload: dict) -> None:
        """Append one item's result. Durable on disk when this returns."""
        rec = dict(payload)
        rec[self.key_field] = key
        line = json.dumps(rec, default=str) + "\n"
        self._fh.write(line.encode("utf-8"))
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._records[str(key)] = rec

    def finalize(self, summary: dict) -> None:
        """Atomically write the run's summary and mark the run complete."""
        tmp_path = self.summary_path.with_name(self.summary_path.name + ".tmp")
        data = json.dumps(summary, indent=2, default=str).encode("utf-8")
        with tmp_path.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, self.summary_path)
        self._write_meta(complete=True)
