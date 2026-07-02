"""Generic partial-artifact sync hook for the batch verbs.

Location: tuner/batch/sync_hook.py
Purpose: After every N newly persisted rows (and once at the end), run a
    user-supplied shell command so cloud wrappers can push partial artifacts to
    durable storage. This is what makes preemption non-lossy end to end: the
    persistence layer flushes each row to local disk, and this hook periodically
    ships that disk to somewhere that survives the job.
Used by: tuner.batch.runner.

Design notes
------------
* The command is arbitrary shell supplied by the operator; it is generic (no
  provider assumptions). Two env vars are exported to it:
    - ``TUNER_SYNC_DIR``:    the run out-dir to push.
    - ``TUNER_SYNC_REASON``: ``periodic`` or ``final``.
* Sync failures WARN and continue. A sync problem (network blip, expired
  credentials) must never kill a compute job that is making progress.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


class SyncHook:
    """Fires a shell command every ``sync_every`` newly persisted rows.

    A ``sync_every`` of 0 (or a missing command) disables periodic syncing;
    ``final()`` still fires once at the end when a command is configured.
    """

    def __init__(
        self,
        out_dir: Path,
        sync_cmd: Optional[str],
        sync_every: int = 0,
        *,
        warn=None,
    ):
        self.out_dir = Path(out_dir)
        self.sync_cmd = sync_cmd or None
        self.sync_every = max(0, int(sync_every or 0))
        self._since_last = 0
        self._warn = warn or (lambda msg: print(msg, file=sys.stderr))

    @property
    def enabled(self) -> bool:
        return self.sync_cmd is not None

    def note_rows(self, n_new: int) -> None:
        """Record ``n_new`` newly persisted rows; sync if the threshold is hit."""
        if not self.enabled or self.sync_every <= 0 or n_new <= 0:
            return
        self._since_last += n_new
        if self._since_last >= self.sync_every:
            self._run("periodic")
            self._since_last = 0

    def final(self) -> None:
        """Fire one final sync at the end of the run (if a command is set)."""
        if not self.enabled:
            return
        self._run("final")
        self._since_last = 0

    def _run(self, reason: str) -> None:
        env = dict(os.environ)
        env["TUNER_SYNC_DIR"] = str(self.out_dir)
        env["TUNER_SYNC_REASON"] = reason
        try:
            result = subprocess.run(
                self.sync_cmd,
                shell=True,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self._warn(
                    f"[batch] sync-cmd ({reason}) exited {result.returncode}; "
                    f"continuing. stderr: {result.stderr.strip()[:500]}"
                )
        except Exception as exc:  # noqa: BLE001 - sync must never kill the run
            self._warn(f"[batch] sync-cmd ({reason}) failed to launch: {exc}; continuing.")
