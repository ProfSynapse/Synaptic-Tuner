from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import math
import io
from pathlib import Path

import pytest

from Evaluator.owned_process import OwnedProcessError, OwnedProcessLease


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
_REAL_KILLPG = os.killpg
_ACTIVE_LEASES: list[OwnedProcessLease] = []


def _force_cleanup(lease: OwnedProcessLease) -> None:
    if lease in _ACTIVE_LEASES:
        _ACTIVE_LEASES.remove(lease)
    try:
        os.waitid(os.P_PID, lease.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return
    try:
        _REAL_KILLPG(lease.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        lease._process.wait(timeout=2)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass


@pytest.fixture(autouse=True)
def _cleanup_process_families():
    del _ACTIVE_LEASES[:]
    try:
        yield
    finally:
        for lease in tuple(_ACTIVE_LEASES):
            if lease.cleanup_pending:
                _force_cleanup(lease)
        del _ACTIVE_LEASES[:]


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return False
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, FileNotFoundError):
        return False


def _wait_file(path: Path) -> int:
    for _ in range(300):
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.01)
    raise AssertionError("grandchild pid was not published")


def _family(tmp_path: Path, *, ignore_term: bool = False, parent_exits: bool = False):
    pid_file = tmp_path / "grandchild.pid"
    grandchild = (
        "import os,pathlib,signal,time;"
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_term else "")
        + f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
        + "time.sleep(60)"
    )
    parent = (
        "import pathlib,subprocess,sys,time,signal;"
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_term else "")
        + f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]);"
        + ("sys.exit(0)" if parent_exits else "time.sleep(60)")
    )
    lease = OwnedProcessLease.spawn(
        (sys.executable, "-c", parent), cwd=tmp_path, environment=dict(os.environ)
    )
    _ACTIVE_LEASES.append(lease)
    try:
        return lease, _wait_file(pid_file)
    except BaseException:
        _force_cleanup(lease)
        raise


def test_close_stops_child_and_grandchild(tmp_path: Path) -> None:
    lease, grandchild = _family(tmp_path)
    assert lease.close(term_timeout=1, kill_timeout=1)
    assert not lease.cleanup_pending
    assert not _alive(grandchild)
    assert lease.close()


def test_kill_follows_term_timeout(tmp_path: Path) -> None:
    lease, grandchild = _family(tmp_path, ignore_term=True)
    assert lease.close(term_timeout=0.05, kill_timeout=1)
    assert not _alive(grandchild)


def test_parent_early_exit_still_stops_group(tmp_path: Path) -> None:
    lease, grandchild = _family(tmp_path, parent_exits=True)
    time.sleep(0.05)
    assert lease.close(term_timeout=1, kill_timeout=1)
    assert not _alive(grandchild)


def test_keyboard_interrupt_cleans_family(tmp_path: Path) -> None:
    lease, grandchild = _family(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with lease:
            raise KeyboardInterrupt
    assert not _alive(grandchild)


def test_denied_signal_is_unresolved_and_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lease, grandchild = _family(tmp_path)
    real_killpg = os.killpg
    calls: list[tuple[int, int]] = []

    def denied(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))
        raise PermissionError

    monkeypatch.setattr(os, "killpg", denied)
    assert not lease.close(term_timeout=0, kill_timeout=0)
    assert lease.cleanup_pending
    assert not lease.close()
    assert len(calls) == 1
    monkeypatch.setattr(os, "killpg", real_killpg)
    _force_cleanup(lease)
    for _ in range(100):
        if not _alive(grandchild):
            break
        time.sleep(0.01)


def test_start_failure_does_not_create_a_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        raise OSError("closed test failure")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError, match="closed test failure"):
        OwnedProcessLease.spawn((sys.executable, "-c", "pass"), cwd=tmp_path, environment={})


def test_guard_never_signals_an_already_reaped_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys as _sys

    lease = OwnedProcessLease.spawn(
        (sys.executable, "-c", "pass"), cwd=tmp_path, environment={}
    )
    _ACTIVE_LEASES.append(lease)
    lease._process.wait(timeout=2)
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        _sys.modules[__name__],
        "_REAL_KILLPG",
        lambda pgid, sig: calls.append((pgid, sig)),
    )
    _force_cleanup(lease)
    assert calls == []


def test_invalid_inputs_fail_before_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def unexpected(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    with pytest.raises(TypeError):
        OwnedProcessLease.spawn("echo", cwd=tmp_path, environment={})  # type: ignore[arg-type]
    assert not called


@pytest.mark.parametrize("value", [True, -1, math.nan, math.inf, -math.inf, 301])
def test_nonfinite_or_unbounded_timeout_is_denied(tmp_path: Path, value: float) -> None:
    lease, _ = _family(tmp_path)
    try:
        with pytest.raises(TypeError):
            lease.close(term_timeout=value)
    finally:
        _force_cleanup(lease)


def test_direct_construction_cannot_adopt_a_group() -> None:
    with pytest.raises(TypeError, match="created by spawn"):
        OwnedProcessLease(object(), 1, 1)  # type: ignore[arg-type]


def test_changed_leader_identity_is_never_signaled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import Evaluator.owned_process as owned

    lease, _ = _family(tmp_path)
    monkeypatch.setattr(owned, "_identity", lambda pid: ("S", lease.pid, -1))
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(AssertionError("signaled")))
    assert not lease.close()
    assert lease.cleanup_pending
    monkeypatch.undo()
    _force_cleanup(lease)


def test_cleanup_exception_does_not_mask_keyboard_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lease, _ = _family(tmp_path)
    real_killpg = os.killpg
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(KeyboardInterrupt):
        with lease:
            raise KeyboardInterrupt
    assert lease.cleanup_pending
    monkeypatch.setattr(os, "killpg", real_killpg)
    _force_cleanup(lease)


def test_unknown_member_identity_retains_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import Evaluator.owned_process as owned

    lease, grandchild = _family(tmp_path, ignore_term=True)
    real_identity = owned._identity

    def obscured(pid: int):
        if pid == grandchild:
            return owned._UNKNOWN
        return real_identity(pid)

    monkeypatch.setattr(owned, "_identity", obscured)
    assert not lease.close(term_timeout=0.1, kill_timeout=0.1)
    assert lease.cleanup_pending
    monkeypatch.undo()
    assert lease.close(term_timeout=0.1, kill_timeout=1)


@pytest.mark.parametrize("mode", ["malformed", "oserror"])
def test_identity_read_failures_are_unknown(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    import builtins
    import Evaluator.owned_process as owned

    if mode == "malformed":
        monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: io.BytesIO(b"bad"))
    else:
        monkeypatch.setattr(
            builtins, "open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("denied"))
        )
    assert owned._identity(os.getpid()) is owned._UNKNOWN


def test_missing_identity_is_absent() -> None:
    import Evaluator.owned_process as owned

    assert owned._identity(2**30) is None


def test_spawn_identity_failure_retains_family_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import Evaluator.owned_process as owned

    real_identity = owned._identity
    calls = 0

    def first_unknown(pid: int):
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(0.1)
            return owned._UNKNOWN
        return real_identity(pid)

    monkeypatch.setattr(owned, "_identity", first_unknown)
    with pytest.raises(OwnedProcessError) as caught:
        OwnedProcessLease.spawn(
            (
                sys.executable,
                "-c",
                "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); time.sleep(60)",
            ),
            cwd=tmp_path,
            environment=dict(os.environ),
        )
    lease = caught.value.cleanup_lease
    _ACTIVE_LEASES.append(lease)
    monkeypatch.setattr(owned, "_identity", real_identity)
    assert lease.close(term_timeout=1, kill_timeout=1)


def test_census_entry_budget_is_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import Evaluator.owned_process as owned

    lease, _ = _family(tmp_path)
    monkeypatch.setattr(owned, "_MAX_PROC_ENTRIES", 0)
    assert not lease.close(term_timeout=0.01, kill_timeout=0.01)
    assert lease.cleanup_pending
    monkeypatch.setattr(owned, "_MAX_PROC_ENTRIES", 131_072)
    assert lease.close(term_timeout=1, kill_timeout=1)


def test_census_expired_deadline_is_unknown(tmp_path: Path) -> None:
    lease, _ = _family(tmp_path)
    try:
        assert lease._members(0.0) is None
        assert lease.cleanup_pending
    finally:
        assert lease.close(term_timeout=1, kill_timeout=1)


def test_spawn_identity_keyboard_interrupt_keeps_cleanup_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import Evaluator.owned_process as owned

    real_identity = owned._identity
    monkeypatch.setattr(
        owned,
        "_identity",
        lambda pid: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        OwnedProcessLease.spawn(
            (sys.executable, "-c", "import time;time.sleep(60)"),
            cwd=tmp_path,
            environment=dict(os.environ),
        )
    lease = caught.value.cleanup_lease
    _ACTIVE_LEASES.append(lease)
    monkeypatch.setattr(owned, "_identity", real_identity)
    assert lease.close(term_timeout=1, kill_timeout=1)


def test_unsupported_platform_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import Evaluator.owned_process as owned

    monkeypatch.setattr(owned.os, "name", "nt")
    with pytest.raises(OwnedProcessError, match="POSIX"):
        OwnedProcessLease.spawn(("ignored",), cwd=tmp_path, environment={})
