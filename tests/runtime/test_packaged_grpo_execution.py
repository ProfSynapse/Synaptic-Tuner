"""CPU process controls for the separately bounded packaged GRPO child."""
import os
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace

import pytest

from tuner.runtime.packaged_sft_execution import _run_child_process


def _invoke(tmp_path, code):
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    invocation = SimpleNamespace(
        argv=(sys.executable, "-I", "-B", "-c", code),
        cwd=tmp_path,
        environment={"PATH": "/usr/bin:/bin"},
        retained_fds=(),
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
    )
    return invocation, read_fd, write_fd


def _run(invocation, read_fd, write_fd, deadline):
    return _run_child_process(
        invocation, read_fd, write_fd, maximum=2, every_n_steps=1,
        on_metric_frame=None, on_metric_exhausted=None, deadline=deadline,
    )


def _expire_when_ready(monkeypatch, marker):
    from tuner.runtime import packaged_sft_execution as seam
    original = time.monotonic
    deadline = original() + 30
    # Startup gets the same finite original budget; only after the real child
    # arms its marker do we advance this module-local observation clock.
    monkeypatch.setattr(seam, "time", SimpleNamespace(
        monotonic=lambda: deadline + 1 if marker.exists() else original()))
    return deadline


def test_expired_budget_has_no_child_or_output(tmp_path):
    invocation, read_fd, write_fd = _invoke(tmp_path, "raise AssertionError")
    with pytest.raises(TimeoutError):
        _run(invocation, read_fd, write_fd, time.monotonic() - 1)
    assert not list(tmp_path.iterdir())
    for descriptor in (read_fd, write_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_stalled_owned_child_is_killed_and_reaped(tmp_path, monkeypatch):
    pid_file = tmp_path / "pid"
    code = ("import os,time\nwith open('pid.tmp','w') as stream:\n"
            " stream.write(str(os.getpid()))\nos.rename('pid.tmp','pid')\ntime.sleep(60)")
    invocation, read_fd, write_fd = _invoke(tmp_path, code)
    deadline = _expire_when_ready(monkeypatch, pid_file)
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        _run(invocation, read_fd, write_fd, deadline)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_success_preserves_raw_exit_without_sft_evidence(tmp_path):
    invocation, read_fd, write_fd = _invoke(tmp_path, "raise SystemExit(7)")
    assert _run(invocation, read_fd, write_fd, time.monotonic() + 5) == 7


def test_descendant_holding_pipe_is_terminated_at_budget(tmp_path, monkeypatch):
    code = ("import os,time\npid=os.fork()\nif pid:\n"
            " with open('descendant.tmp','w') as stream:\n  stream.write(str(pid))\n"
            " os.rename('descendant.tmp','descendant')\nelse:\n time.sleep(60)")
    invocation, read_fd, write_fd = _invoke(tmp_path, code)
    deadline = _expire_when_ready(monkeypatch, tmp_path / "descendant")
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        _run(invocation, read_fd, write_fd, deadline)
    pid = int((tmp_path / "descendant").read_text())
    # A grandchild is not ours to waitpid; verified dead/zombie is termination,
    # not a promise to reap processes owned by the machine's init process.
    until = time.monotonic() + 3
    while time.monotonic() < until:
        try:
            status = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1]
        except FileNotFoundError:
            return
        if status.startswith("Z "):
            return
        time.sleep(0.01)
    pytest.fail("owned process-group descendant remained live")


@pytest.mark.parametrize("confirmation_close_failure", (False, True))
def test_retained_custody_rejects_model_mutation_and_closes(
        tmp_path, monkeypatch, confirmation_close_failure):
    from Trainers.sft import runtime_v1 as core
    from tuner.runtime import packaged_sft_execution as seam
    import hashlib

    raw = b'{"prompt":"bounded"}\n'
    dataset = tmp_path / "input.jsonl"
    dataset.write_bytes(raw)
    roots = {name: tmp_path / name for name in ("artifacts", "state", "tracking", "cache", "tmp")}
    for root in roots.values():
        root.mkdir(mode=0o700)
    paths = seam.PackagedSFTPaths(prepared_input=dataset, **roots)
    model = {"ref": "org/model", "revision": "a" * 40, "tokenizer_revision": "a" * 40}
    execution = SimpleNamespace(prepared_input_size_bytes=len(raw),
        prepared_input_content_digest=hashlib.sha256(raw).hexdigest())
    snapshot = core._model_snapshot_path(model, paths.cache)
    snapshot.mkdir(parents=True)
    member = snapshot / "weights.bin"
    member.write_bytes(b"original")
    monkeypatch.setattr(seam, "_inspect_parent_release", lambda release: None)
    directories = seam._hold_paths(paths)
    retained = None
    try:
        if confirmation_close_failure:
            original = seam._close_resources
            calls = []
            def reject_first_close(resources):
                result = original(resources)
                calls.append(tuple(resources))
                return True if len(calls) == 1 else result
            monkeypatch.setattr(seam, "_close_resources", reject_first_close)
            with pytest.raises(seam.PackagedSFTExecutionError) as caught:
                seam._prepared_execution_inputs(paths=paths, execution=execution,
                    directories=directories, release=None, model=model,
                    model_preparer=lambda *_: snapshot)
            assert caught.value.stage == "REVALIDATION"
            assert calls[0] and all(directory.fd is None for directory in calls[0])
            return
        with seam._prepared_execution_inputs(paths=paths, execution=execution,
                directories=directories, release=None, model=model,
                model_preparer=lambda *_: snapshot) as retained:
            retained.check()
            assert retained.dataset_bytes == raw
            assert (retained.private_copy.snapshot / "weights.bin").read_bytes() == b"original"
            member.write_bytes(b"mutation")
            with pytest.raises(ValueError):
                retained.check()
        assert all(directory.fd is None for directory in retained.snapshots)
        assert all(file.fd is None for file in retained.private_copy.files)
    finally:
        seam._close_resources(directories)



def test_output_parent_failure_closes_pipe_descriptors(tmp_path):
    invocation, read_fd, write_fd = _invoke(tmp_path, "raise AssertionError")
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"file")
    invocation.stdout_path = blocked / "stdout"
    with pytest.raises(FileExistsError):
        _run(invocation, read_fd, write_fd, time.monotonic() + 10)
    for descriptor in (read_fd, write_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_unconfirmed_reap_is_finite_closed_uncertainty(tmp_path, monkeypatch):
    from tuner.runtime import packaged_sft_execution as seam
    invocation, read_fd, write_fd = _invoke(tmp_path, "unused")
    waits = []
    class Process:
        pid = 123
        def poll(self):
            return None
        def wait(self, *, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("private", timeout)
    monkeypatch.setattr(seam.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(seam.os, "killpg", lambda *args: None)
    ticks = iter([0, 0, 2])
    monkeypatch.setattr(seam.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="^PACKAGED_CHILD_CLEANUP_UNCERTAIN$"):
        _run(invocation, read_fd, write_fd, 1)
    assert waits == [5.0]


def test_retained_check_refuses_inventory_close_failure(monkeypatch):
    from tuner.runtime import packaged_sft_execution as seam
    class Held:
        def close(self):
            raise OSError("private")
    retained = seam._PreparedExecutionInputs(
        SimpleNamespace(cache=Path("/cache")), object(), (), None, "sft",
        b"original", Path("/snapshot"), (), {}, SimpleNamespace(check=lambda: None),
    )
    monkeypatch.setattr(seam, "_input", lambda *args: b"original")
    monkeypatch.setattr(seam, "_snapshot_inventory", lambda *args: ((Held(),), {}))
    with pytest.raises(seam.PackagedSFTExecutionError) as caught:
        retained.check()
    assert caught.value.stage == "REVALIDATION"
