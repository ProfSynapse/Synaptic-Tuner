"""P0 (CPU/CI) regression tests for the ACE-STEP subprocess runner (F-4).

``subprocess_runner.run_ace_step_subprocess`` is the load-bearing error-handling
boundary: it must (a) tee the child's stderr to BOTH the console and the run log
with ZERO loss, (b) RAISE ``AceStepSubprocessError`` (returncode preserved) on a
nonzero exit so a failed ``train.py`` can never look like success, and (c) NOT
deadlock when the child floods stderr faster than it is consumed.

(c) is the F-4 regression target: the runner drains stderr in a DEDICATED daemon
thread (``_tee_stream``) while the main thread does ``wait()`` then ``join()``. A
single-threaded reader coupled to ``process.wait()`` deadlocks once the child
writes more than the OS pipe buffer (~64 KB) — the child blocks on write, the
parent blocks on wait, forever. The FLOOD test writes ~20 000 stderr lines (well
past the buffer) and asserts the call COMPLETES (no hang), drains zero-loss, AND
still raises on the nonzero exit.

CPU-only, no GPU, no real ACE-STEP model: the "child" is a tiny inline
``python -c`` script. This file imports + exercises subprocess_runner only — it
does NOT edit it (bc-2's lane).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "Trainers" / "ace_step" / "src"
if str(_SRC) not in sys.path:
    # subprocess_runner + config_translation are unique module names in the repo
    # (verified — no bare-import shadow hazard), so a direct import is safe. The
    # runner's own `from config_translation import ...` resolves from this same dir.
    sys.path.insert(0, str(_SRC))

import subprocess_runner  # noqa: E402
from subprocess_runner import AceStepSubprocessError, run_ace_step_subprocess  # noqa: E402


@pytest.fixture(autouse=True)
def _ace_step_home(tmp_path, monkeypatch):
    """Point the ACE_STEP_HOME seam at a REAL dir so Popen's cwd exists.

    run_ace_step_subprocess runs the child with cwd=resolve_ace_step_home(repo_root),
    which defaults to repo_root/vendor/ACE-Step-1.5 (absent in CI). Setting the env
    var makes resolve_ace_step_home return this existing tmp dir — so the child can
    actually start without provisioning the vendor clone.
    """
    home = tmp_path / "ace_step_home"
    home.mkdir()
    monkeypatch.setenv("ACE_STEP_HOME", str(home))


def _py_child(script: str) -> list[str]:
    """argv that runs an inline python child (the stand-in for `python train.py ...`)."""
    return [sys.executable, "-c", script]


def _stderr_log(log_dir: Path) -> Path:
    """Return the single ace_step_*.stderr.log the runner wrote (timestamped name)."""
    logs = list(log_dir.glob("*.stderr.log"))
    assert len(logs) == 1, f"expected exactly one stderr log, got {logs}"
    return logs[0]


def _run_in_thread(fn, timeout: float = 30.0):
    """Run ``fn`` in a worker thread; return (thread, holder).

    holder gets ``exc`` (the raised exception) or ``ok=True``. Used by the FLOOD
    test so a DEADLOCK manifests as ``thread.is_alive()`` after the join timeout
    rather than hanging the whole suite.
    """
    holder: dict = {}

    def target():
        try:
            fn()
            holder["ok"] = True
        except BaseException as exc:  # noqa: BLE001 - capture to assert in caller
            holder["exc"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout)
    return t, holder


# ---------------------------------------------------------------------------
# SUCCESS: stdout + stderr written, exit 0 -> teed to the log, NO raise
# ---------------------------------------------------------------------------

def test_success_tees_stderr_and_does_not_raise(tmp_path):
    log_dir = tmp_path / "logs"
    script = (
        "import sys\n"
        "sys.stdout.write('progress line\\n')\n"
        "sys.stderr.write('warn one\\n')\n"
        "sys.stderr.write('warn two\\n')\n"
        "sys.exit(0)\n"
    )
    # Returns None (no raise) on exit 0.
    result = run_ace_step_subprocess(
        _py_child(script), repo_root=REPO_ROOT, stage="fixed", log_dir=log_dir
    )
    assert result is None

    log_text = _stderr_log(log_dir).read_text(encoding="utf-8")
    # Both stderr lines were teed to the persistent log sink (zero-loss).
    assert "warn one" in log_text
    assert "warn two" in log_text


# ---------------------------------------------------------------------------
# FAILURE: nonzero exit -> raises AceStepSubprocessError with rc/stage/argv preserved
# ---------------------------------------------------------------------------

def test_nonzero_exit_raises_with_returncode_preserved(tmp_path):
    log_dir = tmp_path / "logs"
    argv = _py_child(
        "import sys\n"
        "sys.stderr.write('fatal: boom\\n')\n"
        "sys.exit(3)\n"
    )
    with pytest.raises(AceStepSubprocessError) as excinfo:
        run_ace_step_subprocess(argv, repo_root=REPO_ROOT, stage="preprocess", log_dir=log_dir)

    err = excinfo.value
    assert err.returncode == 3, "the contract requires the child's exit code be preserved"
    assert err.stage == "preprocess"
    assert err.argv == argv
    # The stderr written before exit was still teed to the log (tee ran before raise).
    assert "fatal: boom" in _stderr_log(log_dir).read_text(encoding="utf-8")


def test_success_message_path_is_distinct_from_failure(tmp_path):
    """A child that writes NOTHING to stderr and exits 0 still must not raise."""
    log_dir = tmp_path / "logs"
    run_ace_step_subprocess(
        _py_child("import sys; sys.exit(0)"),
        repo_root=REPO_ROOT,
        stage="fixed",
        log_dir=log_dir,
    )
    # Log exists (opened for write) but is empty — no false stderr content.
    assert _stderr_log(log_dir).read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# FLOOD: ~20000 stderr lines -> no deadlock, zero-loss drain, still raises (F-4)
# ---------------------------------------------------------------------------

def test_stderr_flood_does_not_deadlock_and_drains_zero_loss(tmp_path):
    """The F-4 regression: a stderr flood past the OS pipe buffer must NOT deadlock.

    Without the dedicated drain thread, ~20000 lines (~260 KB) overflow the ~64 KB
    pipe buffer: the child blocks on write, the parent blocks on wait() — forever.
    Running the call in a worker thread with a join timeout turns that hang into an
    assertable ``thread.is_alive()`` instead of stalling the whole suite.
    """
    log_dir = tmp_path / "logs"
    n_lines = 20000
    argv = _py_child(
        "import sys\n"
        f"for i in range({n_lines}):\n"
        "    sys.stderr.write('err line %d\\n' % i)\n"
        "sys.exit(3)\n"  # nonzero so we ALSO confirm raise-after-flood
    )

    def call():
        run_ace_step_subprocess(argv, repo_root=REPO_ROOT, stage="fixed", log_dir=log_dir)

    thread, holder = _run_in_thread(call, timeout=30.0)

    # (1) No deadlock: the call returned within the timeout.
    assert not thread.is_alive(), (
        "run_ace_step_subprocess DEADLOCKED under a stderr flood — the F-4 dedicated "
        "drain thread regressed (single-threaded read coupled to wait())."
    )
    # (2) Still raises on the nonzero exit even after a flood.
    assert isinstance(holder.get("exc"), AceStepSubprocessError), (
        f"expected AceStepSubprocessError after the flood, got {holder!r}"
    )
    assert holder["exc"].returncode == 3
    # (3) Zero-loss: every one of the flooded lines reached the persistent log sink.
    log_text = _stderr_log(log_dir).read_text(encoding="utf-8")
    drained = sum(1 for line in log_text.splitlines() if line.startswith("err line "))
    assert drained == n_lines, f"stderr drain lost lines: {drained}/{n_lines} reached the log"
