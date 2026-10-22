from dataclasses import replace
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1
from tuner.runtime.native_process import NativeInvocation, NativeOwnedProcess, NativeProcessLimits
from tuner.runtime import native_process_guardian as guardian


@pytest.fixture
def invocation():
    previous = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-process-test-", dir=Path.home()) as name:
            root = Path(name)
            # A fixture shim keeps the test interpreter's real stdlib location.
            # Production admission additionally requires the captured Mach-O executable.
            executable = root / "python"
            executable.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
            executable.chmod(0o700)
            guardian_path = root / "guardian.py"
            guardian_path.write_bytes(Path(guardian.__file__).read_bytes())
            worker = root / "worker.py"
            worker.write_text("print('bounded worker output')\n")
            request = root / "request.json"
            request.write_text("{}")
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            yield NativeInvocation(
                "attempt-1", "a" * 64, str(executable), digest(executable),
                str(guardian_path), digest(guardian_path), str(worker), digest(worker),
                str(request), digest(request), str(root), NativeProcessLimits(2, 1, 1024, 1024**2),
            )
    finally:
        os.umask(previous)


def worker(invocation, text):
    path = Path(invocation.worker)
    path.write_text(text)
    return replace(invocation, worker_digest=hashlib.sha256(path.read_bytes()).hexdigest())


def guardian_script(invocation, text):
    path = Path(invocation.guardian)
    path.write_text(text)
    return replace(invocation, guardian_digest=hashlib.sha256(path.read_bytes()).hexdigest())


def test_real_bounded_process_retains_claim_spawn_outcome_and_never_replays(invocation):
    records = InMemoryDurableRecordStoreV1()
    process = NativeOwnedProcess(records)
    outcome = process.run_once(invocation)
    assert outcome["reason"] == "worker_exited" and outcome["exit_code"] == 0
    assert outcome["termination_confirmed"] is True and outcome["stdout_bytes"] > 0
    assert process.observe(invocation)["state"] == "terminated"
    assert NativeOwnedProcess(records).cancel(invocation) is False
    with pytest.raises(ValueError, match="already exists"):
        process.run_once(invocation)


@pytest.mark.parametrize("script,reason", [
    ("import time; time.sleep(60)", "deadline"),
    ("import os\nwhile True: os.write(1, b'x' * 65536)", "output_limit"),
    ("import os\nwhile True: os.write(2, b'x' * 65536)", "output_limit"),
])
def test_deadline_and_output_are_actual_process_bounds(invocation, script, reason):
    invocation = worker(invocation, script)
    outcome = NativeOwnedProcess(InMemoryDurableRecordStoreV1()).run_once(invocation)
    assert outcome["reason"] == reason
    assert outcome["termination_confirmed"] is True and outcome["exit_code"] != 0


def test_current_owner_can_cancel_without_replaying_numeric_pid(invocation):
    invocation = worker(invocation, "import time; time.sleep(60)")
    process = NativeOwnedProcess(InMemoryDurableRecordStoreV1())
    outputs = []
    thread = threading.Thread(target=lambda: outputs.append(process.run_once(invocation)))
    thread.start()
    deadline = time.monotonic() + 2
    accepted = False
    while time.monotonic() < deadline:
        if process.cancel(invocation):
            accepted = True
            break
        time.sleep(.01)
    thread.join(timeout=6)
    assert accepted and not thread.is_alive()
    assert outputs[0]["reason"] == "cancel_requested"
    assert process.cancel(invocation) is False


def test_spawn_retention_failure_preserves_uncertain_claim(invocation):
    class FailingStore(InMemoryDurableRecordStoreV1):
        def create(self, *, partition, key, canonical):
            if key.endswith("/spawn"):
                raise RuntimeError("simulated interrupted storage")
            return super().create(partition=partition, key=key, canonical=canonical)
    records = FailingStore()
    process = NativeOwnedProcess(records)
    with pytest.raises(RuntimeError):
        process.run_once(invocation)
    assert process.observe(invocation) == {"state": "uncertain", "automatic_replay": False}
    with pytest.raises(ValueError, match="already exists"):
        NativeOwnedProcess(records).run_once(invocation)


@pytest.mark.parametrize("script,expected", [
    ("raise SystemExit(123)\n", {"category": "guardian_exit", "exit_code": 123}),
    ("import sys\nsys.stdout.write('sensitive worker output')\n",
     {"category": "guardian_response_invalid"}),
    ("import sys\nsys.stdout.buffer.write(b'\\xff')\n",
     {"category": "guardian_response_invalid"}),
])
def test_guardian_protocol_failure_retains_only_closed_diagnostic(invocation, script, expected):
    invocation = guardian_script(invocation, script)
    records = InMemoryDurableRecordStoreV1()
    process = NativeOwnedProcess(records)
    with pytest.raises(ValueError, match="native guardian outcome unavailable"):
        process.run_once(invocation)
    assert process.observe(invocation) == {"state": "uncertain", "automatic_replay": False}
    raw = records.read(partition="effects", key=f"native-mlx/{invocation.attempt}/diagnostic").canonical
    diagnostic = json.loads(raw)
    assert diagnostic["value"] == expected
    assert "sensitive" not in raw.decode()
    assert records.read(partition="effects", key=f"native-mlx/{invocation.attempt}/outcome") is None
    with pytest.raises(ValueError, match="already exists"):
        process.run_once(invocation)


def test_real_guardian_classifies_spawn_failure_without_exception_text(invocation):
    read_fd, write_fd = os.pipe()
    spec = {
        "argv": ["/this-native-worker-does-not-exist"],
        "cwd": invocation.private_root, "environment": {},
        "timeout_seconds": 1, "termination_grace_seconds": 1,
        "maximum_output_bytes": 1024, "maximum_file_bytes": 1024**2,
    }
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", invocation.guardian, str(read_fd)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=(read_fd,),
        )
        os.close(read_fd)
        read_fd = None
        out, err = proc.communicate(json.dumps(spec).encode() + b"\n", timeout=5)
        assert proc.returncode == 122 and out == b"" and err == b""
    finally:
        if read_fd is not None:
            os.close(read_fd)
        os.close(write_fd)


def test_byte_drift_rejects_before_claim(invocation):
    Path(invocation.worker).write_text("changed")
    process = NativeOwnedProcess(InMemoryDurableRecordStoreV1())
    with pytest.raises(ValueError, match="bytes differ"):
        process.run_once(invocation)
    assert process.observe(invocation) == {"state": "unclaimed"}


def test_guardian_stops_worker_after_owner_pipe_eof(invocation):
    invocation = worker(invocation, "import time; time.sleep(60)")
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", invocation.guardian, str(read_fd)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(read_fd,),
    )
    os.close(read_fd)
    spec = {
        "argv": [sys.executable, "-I", "-S", "-B", invocation.worker],
        "cwd": invocation.private_root, "environment": {},
        "timeout_seconds": 10, "termination_grace_seconds": 1,
        "maximum_output_bytes": 1024, "maximum_file_bytes": 1024**2,
    }
    os.close(write_fd)
    out, err = proc.communicate(json.dumps(spec).encode() + b"\n", timeout=5)
    assert proc.returncode == 0, err
    outcome = json.loads(out)
    assert outcome["reason"] == "owner_lost" and outcome["termination_confirmed"] is True


def test_darwin_exit_observer_uses_kqueue_without_reaping(monkeypatch):
    calls = []

    class Queue:
        def control(self, changes, maximum, timeout):
            calls.append(changes)
            if len(calls) == 1:
                return []
            return [type("Event", (), {"flags": 0, "ident": 42, "fflags": 4})()]

        def close(self):
            calls.append("closed")
    monkeypatch.setattr(guardian.sys, "platform", "darwin")
    monkeypatch.setattr(guardian.select, "kqueue", Queue, raising=False)
    monkeypatch.setattr(guardian.select, "kevent", lambda *a, **k: (a, k), raising=False)
    for name, value in (("KQ_FILTER_PROC", 1), ("KQ_EV_ADD", 1), ("KQ_EV_ONESHOT", 2),
                        ("KQ_NOTE_EXIT", 4), ("KQ_EV_ERROR", 8)):
        monkeypatch.setattr(guardian.select, name, value, raising=False)
    monkeypatch.setattr(guardian.os, "waitid", lambda *a: pytest.fail("Darwin must not call waitid"))
    observer = guardian._ExitObserver(type("Child", (), {"pid": 42})())
    assert observer.exited() is True
    observer.close()
    assert calls[-1] == "closed"


def test_empty_owned_group_still_reaps_child_after_both_signals_miss(monkeypatch):
    calls = []

    class Child:
        pid = 42

        def wait(self, *, timeout):
            calls.append(("wait", timeout))
            return 0

    def no_group(pid, sig):
        calls.append((pid, sig))
        raise ProcessLookupError("group already empty")

    monkeypatch.setattr(guardian.os, "killpg", no_group)
    phase = {"name": "termination"}
    observer = type("Observer", (), {"exited": lambda self: True})()
    assert guardian._terminate(Child(), 1, observer, phase) == 0
    assert calls == [(42, signal.SIGTERM), (42, signal.SIGKILL), ("wait", 1)]
    assert phase["name"] == "termination_wait"


def test_exited_leader_does_not_skip_owned_descendant_cleanup(monkeypatch):
    calls = []
    descendant_live = True

    class Child:
        pid = 42

        def wait(self, *, timeout):
            assert descendant_live is False
            calls.append("reap")
            return 0

    def signal_group(pid, sig):
        nonlocal descendant_live
        assert pid == 42
        calls.append(sig)
        if sig == signal.SIGKILL:
            descendant_live = False

    monkeypatch.setattr(guardian.os, "killpg", signal_group)
    observer = type("Observer", (), {"exited": lambda self: True})()
    assert guardian._terminate(Child(), 1, observer) == 0
    assert calls == [signal.SIGTERM, signal.SIGKILL, "reap"]


@pytest.mark.parametrize("failure,expected,code", [
    ("term", "termination_term_other_os", 133),
    ("kill", "termination_kill_other_os", 137),
    ("wait", "termination_wait", 131),
])
def test_termination_failure_subphase_is_closed(monkeypatch, failure, expected, code):
    class Child:
        pid = 42

        def wait(self, *, timeout):
            if failure == "wait":
                raise OSError("private wait detail")
            return 0

    def signal_group(pid, sig):
        if (failure == "term" and sig == signal.SIGTERM
                or failure == "kill" and sig == signal.SIGKILL):
            raise OSError("private signal detail")

    monkeypatch.setattr(guardian.os, "killpg", signal_group)
    phase = {"name": "termination"}
    observer = type("Observer", (), {"exited": lambda self: True})()
    with pytest.raises(OSError):
        guardian._terminate(Child(), 1, observer, phase)
    assert phase["name"] == expected
    assert guardian._FAILURE_CODES[expected] == code


@pytest.mark.parametrize("sig,prefix", [
    (signal.SIGTERM, "termination_term"),
    (signal.SIGKILL, "termination_kill"),
])
@pytest.mark.parametrize("failure,suffix", [
    (PermissionError(errno.EPERM, "private detail"), "eperm"),
    (PermissionError(errno.EACCES, "private detail"), "other_os"),
    (OSError(errno.EIO, "private detail"), "other_os"),
    (RuntimeError("private detail"), "non_os"),
])
def test_owned_group_signal_failure_class_is_closed(monkeypatch, sig, prefix, failure, suffix):
    calls = []

    def fail(pid, actual_sig):
        calls.append((pid, actual_sig))
        raise failure

    monkeypatch.setattr(guardian.os, "killpg", fail)
    phase = {"name": prefix}
    with pytest.raises(type(failure)):
        guardian._signal_owned_group(42, sig, phase)
    assert calls == [(42, sig)]
    assert phase["name"] == prefix + "_" + suffix
    assert guardian._FAILURE_CODES[phase["name"]] in range(132, 140)


@pytest.mark.parametrize("sig,prefix,code", [
    (signal.SIGTERM, "termination_term", 134),
    (signal.SIGKILL, "termination_kill", 138),
])
def test_missing_group_signal_api_is_closed(monkeypatch, sig, prefix, code):
    monkeypatch.delattr(guardian.os, "killpg")
    phase = {"name": prefix}
    with pytest.raises(RuntimeError):
        guardian._signal_owned_group(42, sig, phase)
    assert guardian._FAILURE_CODES[phase["name"]] == code


def test_guardian_emits_only_closed_signal_code(monkeypatch, capsys):
    class Input:
        buffer = io.BytesIO(b"{}\n")

    def refused(spec, control_fd, phase):
        phase["name"] = "termination_term_eperm"
        raise PermissionError(errno.EPERM, "private path and argument")

    monkeypatch.setattr(guardian.sys, "argv", ["guardian.py", "3"])
    monkeypatch.setattr(guardian.sys, "stdin", Input())
    monkeypatch.setattr(guardian, "supervise", refused)
    assert guardian.main() == 132
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("failed_signal,expected_phase", [
    (signal.SIGTERM, "termination_term_eperm"),
    (signal.SIGKILL, "termination_kill_other_os"),
])
def test_cleanup_preserves_first_termination_failure_and_reaps_owned_child(
    monkeypatch, invocation, failed_signal, expected_phase,
):
    read_fd, write_fd = os.pipe()
    original_killpg = os.killpg
    original_popen = subprocess.Popen
    children = []
    failed = False
    signals = []

    def own_child(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    def fail_first_kill(pid, sig):
        nonlocal failed
        signals.append(sig)
        if sig == failed_signal and not failed:
            failed = True
            if sig == signal.SIGTERM:
                raise PermissionError(errno.EPERM, "private signal detail")
            raise OSError(errno.EIO, "private signal detail")
        return original_killpg(pid, sig)

    monkeypatch.setattr(guardian.os, "killpg", fail_first_kill)
    monkeypatch.setattr(guardian.subprocess, "Popen", own_child)
    spec = {
        "argv": [sys.executable, "-I", "-S", "-B", invocation.worker],
        "cwd": invocation.private_root, "environment": {},
        "timeout_seconds": 2, "termination_grace_seconds": 1,
        "maximum_output_bytes": 1024, "maximum_file_bytes": 1024**2,
    }
    phase = {"name": "protocol"}
    try:
        with pytest.raises(OSError):
            guardian.supervise(spec, read_fd, phase)
        assert failed and phase["name"] == expected_phase
        assert len(children) == 1 and children[0].returncode is not None
        assert signals.count(signal.SIGTERM) >= 1
        assert signal.SIGKILL in signals
    finally:
        os.close(read_fd)
        os.close(write_fd)
