"""Consumer-owned, best-effort public progress retention.

The journal is evidence, not a coordinator restart or submission authority.
Cursors and duplicate detection last only for this process. One bounded page
per poll and no final drain mean even terminal runs may have missing tails.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunLogsRequest
from tuner.execution.observation_logs import decode_training_progress_log
from synaptic_tuner.api.v1._contract import canonical_bytes


class RunProgressConsumer:
    MAX_RECORDS = 10_001
    MAX_BYTES = 64 * 1024 * 1024

    def __init__(self, runs, run: TrainingRunRef, root: Path):
        self.runs, self.run, self.root = runs, run, root
        self.cursor = None
        self.status = "progress_waiting"
        self.disabled = False
        self._seen: dict[int, bytes] = {}
        self._bytes = 0
        self.path = root / ("progress-" + hashlib.sha256(
            canonical_bytes(run.to_dict())).hexdigest() + ".jsonl")
        self._fd = None
        self._created = False
        self._file_identity = None

    def _open(self):
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(directory)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise ValueError
            self._fd = os.open(self.path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                               | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            self._created = True
            info = os.fstat(self._fd)
            self._file_identity = (info.st_dev, info.st_ino)
            os.fsync(directory)
        finally:
            os.close(directory)

    def close(self):
        if self._fd is not None:
            descriptor, self._fd = self._fd, None
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _disable(self, status):
        self.status, self.disabled = status, True
        self.close()

    def result(self):
        return {"status": self.status, "records_path": str(self.path) if self._created else None,
                "retained_records": len(self._seen)}

    def poll(self):
        """Failures affect telemetry only; temporary unreadable pages retry later."""
        if self.disabled:
            return
        try:
            page = self.runs.logs(RunLogsRequest(self.run, self.cursor, 100, 65536))
        except Exception:
            self.status = "progress_read_unavailable"
            return
        try:
            pending = []
            for entry in page.entries:
                record = decode_training_progress_log(entry, expected_run=self.run)
                raw = canonical_bytes(record.to_dict())
                previous = self._seen.get(record.sequence)
                if previous is not None:
                    if previous != raw:
                        self._disable("progress_sequence_conflict")
                        return
                    continue
                pending.append((record.sequence, raw))
            if (len(self._seen) + len(pending) > self.MAX_RECORDS
                    or self._bytes + sum(len(raw) + 1 for _, raw in pending) > self.MAX_BYTES):
                self._disable("progress_retention_exhausted")
                return
        except Exception:
            self._disable("progress_record_invalid")
            return
        try:
            if pending and self._fd is None:
                self._open()
            for sequence, raw in pending:
                remaining = memoryview(raw + b"\n")
                while remaining:
                    written = os.write(self._fd, remaining)
                    if written <= 0:
                        raise OSError
                    remaining = remaining[written:]
                self._seen[sequence] = raw
                self._bytes += len(raw) + 1
            if pending:
                os.fsync(self._fd)
            if page.next_cursor is not None:
                self.cursor = page.next_cursor
            self.status = "progress_retained" if self._seen else "progress_waiting"
        except Exception:
            self._disable("progress_storage_unavailable")


def start_progress_consumer(runs, run, root, *, capture_enabled, logs_enabled):
    """No reads or filesystem effects when capture or advertised logs are absent."""
    if capture_enabled is not True or logs_enabled is not True:
        return None
    return RunProgressConsumer(runs, run, root)
