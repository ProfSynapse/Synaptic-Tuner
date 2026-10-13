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


def test_expired_budget_has_no_child_or_output(tmp_path):
    invocation, read_fd, write_fd = _invoke(tmp_path, "raise AssertionError")
    with pytest.raises(TimeoutError):
        _run(invocation, read_fd, write_fd, time.monotonic() - 1)
    assert not list(tmp_path.iterdir())
    for descriptor in (read_fd, write_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_stalled_owned_child_is_killed_and_reaped(tmp_path):
    pid_file = tmp_path / "pid"
    code = "import os,time; open('pid','w').write(str(os.getpid())); time.sleep(60)"
    invocation, read_fd, write_fd = _invoke(tmp_path, code)
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        _run(invocation, read_fd, write_fd, time.monotonic() + 1)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_success_preserves_raw_exit_without_sft_evidence(tmp_path):
    invocation, read_fd, write_fd = _invoke(tmp_path, "raise SystemExit(7)")
    assert _run(invocation, read_fd, write_fd, time.monotonic() + 5) == 7


def test_descendant_holding_pipe_is_terminated_at_budget(tmp_path):
    code = ("import os,time; pid=os.fork(); "
            "open('descendant','w').write(str(pid)) if pid else time.sleep(60)")
    invocation, read_fd, write_fd = _invoke(tmp_path, code)
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        _run(invocation, read_fd, write_fd, time.monotonic() + 0.5)
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


def test_retained_custody_rejects_model_mutation_and_closes(tmp_path, monkeypatch):
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

