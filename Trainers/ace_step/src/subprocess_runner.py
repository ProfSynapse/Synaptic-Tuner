"""
ACE-STEP subprocess runner — the load-bearing error-handling boundary.

Location: Trainers/ace_step/src/subprocess_runner.py
Purpose:  Run an ACE-STEP `train.py` invocation (stage-1 `fixed --preprocess` or
          stage-2 `fixed`) as a child process with the NON-NEGOTIABLE error contract
          (contract §1.2.5, plan risk row): capture the returncode, tee stderr to
          BOTH the console and the run log, and RAISE on a nonzero exit so a failed
          `train.py` can NEVER look like success. No silent swallow.
Used by:  Trainers/ace_step/train_ace_step.py (stage-2 `fixed`) and
          Trainers/ace_step/src/data_loader.py (stage-1 `fixed --preprocess`).

Contract: docs/architecture/ace-step-pipeline-contract.md §1.2 item 5.

Design note: stderr is drained line-by-line in a DEDICATED thread so the operator
sees failures live AND they are persisted to the run log, while the pipe can never
fill and deadlock the main thread that is waiting on the child (F-4 hardening — a
single-threaded reader coupled to process.wait() could stall under a stderr flood).
stdout is inherited by the console as-is (ACE-STEP's own progress output), so it is
never piped — there is no second pipe to drain.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

# Read-only import: the ACE_STEP_HOME seam resolver (config_translation owns it; this
# module does not edit it). Used to run train.py with the ACE-STEP repo as its cwd.
from config_translation import resolve_ace_step_home


class AceStepSubprocessError(RuntimeError):
    """Raised when an ACE-STEP `train.py` subcommand exits with a nonzero code.

    Carries the stage name, the return code, and the argv so the failure is fully
    attributable in logs and re-raisable context.
    """

    def __init__(self, stage: str, returncode: int, argv: list[str]):
        self.stage = stage
        self.returncode = returncode
        self.argv = argv
        super().__init__(
            f"ACE-STEP '{stage}' subprocess failed with exit code {returncode}. "
            f"argv: {' '.join(argv)}"
        )


def _tee_stream(stream: IO[str], sinks: tuple[IO[str], ...]) -> None:
    """Drain a text stream line-by-line into every sink, flushing each line.

    Runs in its own thread (see run_ace_step_subprocess) so the source pipe is
    drained independently of the main thread — preventing a full-pipe deadlock if
    the child floods the stream while the main thread blocks on process.wait().
    Each line is flushed immediately so failures surface live on the console and are
    persisted to the log without buffering.
    """
    for line in stream:
        for sink in sinks:
            sink.write(line)
            sink.flush()


def run_ace_step_subprocess(
    argv: list[str],
    *,
    repo_root: Path,
    stage: str,
    log_dir: Path,
) -> None:
    """Run an ACE-STEP `train.py` subcommand, teeing stderr and raising on failure.

    Args:
        argv:      The fully translated argv (e.g. ["python", "train.py", "fixed", ...]).
                   Built by the §1.3 translation table; PROVISIONAL until the
                   build-time `--help` byte-confirm passes.
        repo_root: Synthetic-Conversations repo root. Used to resolve the ACE-STEP
                   home dir (the ACE_STEP_HOME seam) which is passed as the child's
                   cwd, so train.py runs with its own repo as the working directory
                   (defensive: closes a latent footgun if train.py ever resolves a
                   path relative to its cwd rather than to argv[0]).
        stage:     "preprocess" | "fixed" — names the stage in logs + exceptions.
        log_dir:   Directory to write the per-stage stderr log into (created if absent).

    Raises:
        AceStepSubprocessError: the subprocess exited nonzero (the contract's
            raise-on-nonzero guarantee — a failed train.py must not look like success).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stderr_log = log_dir / f"ace_step_{stage}_{timestamp}.stderr.log"

    print(f"[ace_step:{stage}] running: {' '.join(argv)}")
    print(f"[ace_step:{stage}] stderr tee -> {stderr_log}")

    # Stream stderr line-by-line so failures surface live AND are persisted; stdout
    # inherits the console so ACE-STEP's progress output is visible unbuffered.
    with open(stderr_log, "w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            argv,
            cwd=str(resolve_ace_step_home(repo_root)),  # M-f: run in the ACE-STEP repo dir
            stdout=None,                 # inherit console stdout (live progress)
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,                   # line-buffered
        )
        assert process.stderr is not None  # PIPE guarantees a stream
        # F-4: drain stderr in a DEDICATED thread so the pipe is cleared independently
        # of the main thread. A single-threaded reader coupled to process.wait() could
        # deadlock if the child floods stderr faster than it is consumed; the drainer
        # reads to EOF (child exit) regardless of main-thread state. raise-on-nonzero
        # below is unchanged.
        tee_thread = threading.Thread(
            target=_tee_stream,
            args=(process.stderr, (sys.stderr, stderr_handle)),
            daemon=True,
        )
        tee_thread.start()
        returncode = process.wait()
        tee_thread.join()  # flush all stderr before evaluating the returncode

    if returncode != 0:
        # RAISE — do NOT swallow. A nonzero train.py must propagate as a failure.
        raise AceStepSubprocessError(stage=stage, returncode=returncode, argv=argv)

    print(f"[ace_step:{stage}] completed (exit 0)")
