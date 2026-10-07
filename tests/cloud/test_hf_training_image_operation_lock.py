from __future__ import annotations

import threading
import time
import subprocess
import sys
from pathlib import Path

import pytest

from tuner.cloud import hf_training_image_operation_lock as operation


REPOSITORY = "docker.io/unsloth/unsloth"
DIGEST = "sha256:" + "a" * 64


def test_operation_key_is_domain_separated_deterministic_and_identity_bound() -> None:
    first = operation.operation_lock_key(REPOSITORY, DIGEST)
    assert len(first) == 64
    assert first == operation.operation_lock_key(REPOSITORY, DIGEST)
    assert first != operation.operation_lock_key(REPOSITORY, "sha256:" + "b" * 64)
    for repository, digest in (("UPPER/repo", DIGEST), (REPOSITORY, "latest"), ("../escape", DIGEST)):
        with pytest.raises(operation.ImageOperationLockError, match="OPERATION_LOCK_INVALID"):
            operation.operation_lock_key(repository, digest)


def test_same_identity_threads_serialize_on_native_operation_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operation, "_lock_root", lambda: tmp_path / "locks")
    entered: list[str] = []
    first_inside = threading.Event()
    release_first = threading.Event()

    def first() -> None:
        with operation.image_operation_lock(REPOSITORY, DIGEST):
            entered.append("first")
            first_inside.set()
            assert release_first.wait(2)

    def second() -> None:
        assert first_inside.wait(2)
        with operation.image_operation_lock(REPOSITORY, DIGEST):
            entered.append("second")

    one = threading.Thread(target=first)
    two = threading.Thread(target=second)
    one.start()
    two.start()
    assert first_inside.wait(2)
    time.sleep(0.05)
    assert entered == ["first"]
    release_first.set()
    one.join(2)
    two.join(2)
    assert not one.is_alive() and not two.is_alive()
    assert entered == ["first", "second"]


def test_different_identity_locks_do_not_block_each_other(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operation, "_lock_root", lambda: tmp_path / "locks")
    with operation.image_operation_lock(REPOSITORY, DIGEST):
        with operation.image_operation_lock(REPOSITORY, "sha256:" + "b" * 64):
            pass


def test_lock_acquisition_timeout_is_closed(monkeypatch) -> None:
    class BusyLock:
        def acquire(self, timeout: float) -> bool:
            assert 0 <= timeout <= operation.LOCK_ACQUIRE_SECONDS
            return False
        def release(self) -> None:
            raise AssertionError("not acquired")

    monkeypatch.setattr(operation, "_thread_lock", lambda _key: BusyLock())
    with pytest.raises(operation.ImageOperationLockError, match="OPERATION_LOCK_TIMEOUT"):
        with operation.image_operation_lock(REPOSITORY, DIGEST):
            pass


def test_native_contention_timeout_is_closed_and_descriptor_cleaned(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operation, "_lock_root", lambda: tmp_path / "locks")
    readings = iter((0.0, 0.0, 31.0, 31.0, 31.0))
    monkeypatch.setattr(operation.time, "monotonic", lambda: next(readings, 31.0))
    monkeypatch.setattr(operation, "_native_try_lock", lambda _descriptor: False)
    with pytest.raises(operation.ImageOperationLockError, match="OPERATION_LOCK_TIMEOUT"):
        with operation.image_operation_lock(REPOSITORY, DIGEST):
            pass


def test_cleanup_failure_has_closed_reason(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operation, "_lock_root", lambda: tmp_path / "locks")
    monkeypatch.setattr(
        operation, "_native_unlock",
        lambda _descriptor: (_ for _ in ()).throw(OSError("secret")),
    )
    with pytest.raises(operation.ImageOperationLockError, match="OPERATION_LOCK_CLEANUP_FAILED"):
        with operation.image_operation_lock(REPOSITORY, DIGEST):
            pass


def test_cleanup_deadline_overrun_has_closed_reason(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operation, "_lock_root", lambda: tmp_path / "locks")
    readings = iter((0.0, 0.0, 0.0, 0.0, 61.0))
    monkeypatch.setattr(operation.time, "monotonic", lambda: next(readings, 61.0))
    with pytest.raises(operation.ImageOperationLockError, match="OPERATION_LOCK_CLEANUP_FAILED"):
        with operation.image_operation_lock(REPOSITORY, DIGEST):
            pass


def test_noncanonical_deadline_overrides_are_rejected() -> None:
    with pytest.raises(operation.ImageOperationLockError, match="OPERATION_LOCK_INVALID"):
        with operation.image_operation_lock(REPOSITORY, DIGEST, acquisition_seconds=31):
            pass


def test_malformed_persistent_lock_file_is_rejected(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "locks"
    root.mkdir()
    key = operation.operation_lock_key(REPOSITORY, DIGEST)
    (root / f"{key}.lock").write_bytes(b"hostile")
    monkeypatch.setattr(operation, "_lock_root", lambda: root)
    with pytest.raises(operation.ImageOperationLockError, match="OPERATION_LOCK_INVALID"):
        with operation.image_operation_lock(REPOSITORY, DIGEST):
            pass


def test_same_identity_processes_share_kernel_lock(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import sys,time;from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);"
        "from tuner.cloud import hf_training_image_operation_lock as o;"
        "o._lock_root=lambda:Path(sys.argv[2]);"
        "c=o.image_operation_lock('docker.io/unsloth/unsloth','sha256:'+'a'*64);"
        "c.__enter__();print(sys.argv[3],flush=True);time.sleep(float(sys.argv[4]));c.__exit__(None,None,None)"
    )
    root = tmp_path / "process-locks"
    first = subprocess.Popen(
        [sys.executable, "-I", "-c", code, str(repo), str(root), "FIRST", "0.5"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert first.stdout is not None
    assert first.stdout.readline().strip() == "FIRST"
    started = time.monotonic()
    second = subprocess.run(
        [sys.executable, "-I", "-c", code, str(repo), str(root), "SECOND", "0"],
        capture_output=True, text=True, timeout=3, check=False,
    )
    elapsed = time.monotonic() - started
    first.wait(timeout=3)
    assert first.returncode == 0 and second.returncode == 0
    assert second.stdout.strip() == "SECOND" and elapsed >= 0.3
