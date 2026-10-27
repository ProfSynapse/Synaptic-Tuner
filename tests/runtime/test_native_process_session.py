"""Owned transport fixtures only; no ML imports or public capability proof."""
from dataclasses import replace
import os
from pathlib import Path
import resource
import threading
import time

import pytest

from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1
from tests.runtime import test_native_process as fixtures
from tuner.runtime.native_process import NativeOwnedProcess
from tuner.runtime.native_process_session import NativeProcessSession

invocation = fixtures.invocation
worker = fixtures.worker
guardian_script = fixtures.guardian_script


def terminal(session, invocation):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        observed = session.poll(invocation)
        if observed["state"] == "terminated":
            return observed
        time.sleep(.01)
    pytest.fail("fixture guardian did not finish within bound")


def test_nonblocking_start_and_poll_use_real_thread_affine_sqlite(invocation):
    invocation = worker(invocation, "from pathlib import Path\nimport time\n"
        "while not Path('release').exists(): time.sleep(.01)\nprint('complete')\n")
    with SQLiteDurableStoreV1(Path(invocation.private_root) / "session.sqlite3", create=True) as records:
        session = NativeProcessSession(records)
        try:
            assert session.start_once(invocation)["state"] == "running"
            assert session.observe(invocation)["state"] == "running"
            (Path(invocation.private_root) / "release").write_text("release")
            observed = terminal(session, invocation)
            assert observed["outcome"]["exit_code"] == 0
            assert observed["outcome"]["termination_confirmed"] is True
            assert NativeOwnedProcess(records).observe(invocation) == observed
            with pytest.raises(ValueError, match="already exists"):
                session.start_once(invocation)
        finally:
            session.close()


def test_start_with_high_numbered_descriptors_uses_real_guardian(invocation, monkeypatch):
    # Do not change pytest's resource limits. Exercise legal host descriptors
    # above select()'s fixed FD_SETSIZE while retaining actual process cleanup.
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < 1100:
        pytest.skip("host soft descriptor limit below high-FD fixture bound")
    occupied, stdin_fds = [], []
    from tuner.runtime import native_process_session as transport
    original_spawn = transport.subprocess.Popen

    def spawn(*args, **kwargs):
        child = original_spawn(*args, **kwargs)
        stdin_fds.append(child.stdin.fileno())
        return child

    monkeypatch.setattr(transport.subprocess, "Popen", spawn)
    records = InMemoryDurableRecordStoreV1()
    session = NativeProcessSession(records)
    try:
        for _ in range(1052):
            occupied.append(os.open(os.devnull, os.O_RDONLY))
            if occupied[-1] >= 1050:
                break
        assert occupied[-1] >= 1050
        assert session.start_once(invocation)["state"] == "running"
        assert stdin_fds[0] > 1023
        outcome = terminal(session, invocation)
        assert outcome["outcome"]["exit_code"] == 0
        assert outcome["outcome"]["termination_confirmed"] is True
        assert NativeOwnedProcess(records).observe(invocation) == outcome
        with pytest.raises(ValueError, match="already exists"):
            session.start_once(invocation)
    finally:
        session.close()
        for descriptor in occupied:
            os.close(descriptor)


def test_cross_thread_cancel_does_not_touch_thread_affine_store(invocation):
    invocation = worker(invocation, "import time\nwhile True: time.sleep(.01)\n")
    with SQLiteDurableStoreV1(Path(invocation.private_root) / "session.sqlite3", create=True) as records:
        session = NativeProcessSession(records)
        try:
            assert session.cancel(invocation) is False
            session.start_once(invocation)
            results = []
            thread = threading.Thread(target=lambda: results.append(session.cancel(invocation)))
            thread.start();thread.join(timeout=2)
            assert not thread.is_alive() and results == [True]
            assert session.cancel(invocation) is True  # One request, no pipe flooding.
            assert terminal(session, invocation)["outcome"]["reason"] == "cancel_requested"
            assert session.cancel(invocation) is False
        finally:
            session.close()


def test_cold_session_cannot_adopt_cancel_or_replay_live_attempt(invocation):
    invocation = worker(invocation, "import time\nwhile True: time.sleep(.01)\n")
    records = InMemoryDurableRecordStoreV1()
    owner, cold = NativeProcessSession(records), NativeProcessSession(records)
    try:
        owner.start_once(invocation)
        assert cold.observe(invocation) == {"state": "uncertain", "automatic_replay": False}
        assert cold.cancel(invocation) is False
        with pytest.raises(ValueError, match="already exists"):
            cold.start_once(invocation)
        assert owner.cancel(invocation)
        terminal(owner, invocation)
    finally:
        owner.close();cold.close()


def test_owner_pipe_close_preserves_uncertainty_and_no_replay(invocation):
    invocation = worker(invocation, "import time\nwhile True: time.sleep(.01)\n")
    records = InMemoryDurableRecordStoreV1()
    owner = NativeProcessSession(records)
    owner.start_once(invocation)
    owner.close()
    cold = NativeProcessSession(records)
    assert cold.observe(invocation) == {"state": "uncertain", "automatic_replay": False}
    assert cold.cancel(invocation) is False
    with pytest.raises(ValueError, match="already exists"):
        cold.start_once(invocation)


def test_store_failure_after_spawn_relinquishes_pipe_without_replaying(invocation):
    class FailingStore(InMemoryDurableRecordStoreV1):
        def create(self, *, partition, key, canonical):
            if key.endswith("/spawn"):
                raise RuntimeError("fixture interrupted persistence")
            return super().create(partition=partition, key=key, canonical=canonical)
    records = FailingStore()
    owner = NativeProcessSession(records)
    with pytest.raises(RuntimeError): owner.start_once(invocation)
    assert owner.observe(invocation) == {"state": "uncertain", "automatic_replay": False}
    assert owner.cancel(invocation) is False
    with pytest.raises(ValueError, match="already exists"):
        NativeProcessSession(records).start_once(invocation)


def test_validation_failure_before_claim_does_not_dispatch(invocation):
    records = InMemoryDurableRecordStoreV1()
    Path(invocation.worker).write_text("changed")
    session = NativeProcessSession(records)
    with pytest.raises(ValueError, match="input differs"): session.start_once(invocation)
    assert session.observe(invocation) == {"state": "unclaimed"}


def test_wrong_thread_poll_refuses_before_store_access(invocation):
    session = NativeProcessSession(InMemoryDurableRecordStoreV1())
    failures = []
    def other():
        try: session.observe(invocation)
        except ValueError: failures.append(True)
    thread = threading.Thread(target=other);thread.start();thread.join(timeout=2)
    assert not thread.is_alive() and failures == [True]


def test_live_invocation_binding_is_not_just_attempt_name(invocation):
    invocation = worker(invocation, "import time\nwhile True: time.sleep(.01)\n")
    owner = NativeProcessSession(InMemoryDurableRecordStoreV1())
    try:
        owner.start_once(invocation)
        wrong = replace(invocation, binding_digest="b" * 64)
        assert owner.cancel(wrong) is False
        with pytest.raises(ValueError, match="live invocation differs"): owner.poll(wrong)
        assert owner.cancel(invocation)
        terminal(owner, invocation)
    finally:
        owner.close()


def test_forged_python_equality_cannot_select_live_stop_authority(invocation):
    invocation = worker(invocation, "import time\nwhile True: time.sleep(.01)\n")
    class Forged:
        attempt = invocation.attempt
        def __eq__(self, other): return True
    owner = NativeProcessSession(InMemoryDurableRecordStoreV1())
    try:
        owner.start_once(invocation)
        assert owner.cancel(Forged()) is False
        with pytest.raises(TypeError): owner.poll(Forged())
        assert owner.cancel(invocation)
        terminal(owner, invocation)
    finally:
        owner.close()


@pytest.mark.parametrize("output", ["x" * 4097, "{}", '{"termination_confirmed":false}'])
def test_bad_guardian_receipt_stays_uncertain(invocation, output):
    invocation = guardian_script(invocation, "import sys\nsys.stdin.buffer.readline(65537)\nprint(" + repr(output) + ")\n")
    records = InMemoryDurableRecordStoreV1();owner = NativeProcessSession(records)
    try:
        owner.start_once(invocation)
        with pytest.raises(ValueError): terminal(owner, invocation)
        assert NativeOwnedProcess(records).observe(invocation) == {"state": "uncertain", "automatic_replay": False}
    finally:
        owner.close()
