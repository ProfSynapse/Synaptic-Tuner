from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
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
