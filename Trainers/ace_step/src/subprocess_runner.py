"""
ACE-STEP subprocess runner — the load-bearing error-handling boundary.

Location: Trainers/ace_step/src/subprocess_runner.py
Purpose:  Run an ACE-STEP `train.py` subcommand (preprocess or fixed) as a child
          process with the NON-NEGOTIABLE error contract (contract §1.2.5, plan
          risk row): capture the returncode, tee stderr to BOTH the console and the
          run log, and RAISE on a nonzero exit so a failed `train.py` can NEVER look
          like success. No silent swallow.
Used by:  Trainers/ace_step/train_ace_step.py (stage-2 `fixed`) and
          Trainers/ace_step/src/data_loader.py (stage-1 `preprocess`).

Contract: docs/architecture/ace-step-pipeline-contract.md §1.2 item 5.

Design note: stderr is streamed line-by-line so the operator sees failures live AND
they are persisted to the run log, rather than buffered until the process exits.
stdout is streamed to the console as-is (ACE-STEP's own progress output).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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
        repo_root: Synthetic-Conversations repo root (informational; the ACE-STEP
                   repo cwd is encoded in argv[1] by the caller's path resolution).
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
            stdout=None,                 # inherit console stdout (live progress)
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,                   # line-buffered
        )
        assert process.stderr is not None  # PIPE guarantees a stream
        for line in process.stderr:
            sys.stderr.write(line)       # tee to console
            stderr_handle.write(line)    # tee to log
        returncode = process.wait()

    if returncode != 0:
        # RAISE — do NOT swallow. A nonzero train.py must propagate as a failure.
        raise AceStepSubprocessError(stage=stage, returncode=returncode, argv=argv)

    print(f"[ace_step:{stage}] completed (exit 0)")
