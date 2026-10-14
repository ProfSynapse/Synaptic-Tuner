"""Finite parent diagnostics with real local descriptors and seals, no provider."""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from Trainers.sft import runtime_v1 as core
from tuner.runtime import packaged_training_worker as worker
from tuner.runtime import packaged_sft_execution as execution
from tests.runtime.test_packaged_training_worker import _cpu_release

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="actual Linux memfd and kernel seals required",
)


def _fixture(monkeypatch):
    release = _cpu_release()
    digest = hashlib.sha256(b"reviewed trainer").hexdigest()
    # Isolate diagnostic boundary tests from separately tested release admission.
    # Filesystem operations, descriptor custody and sealing remain real.
    monkeypatch.setattr(worker, "parse_packaged_runtime_release", lambda _: release)
    monkeypatch.setattr(
        worker, "admit_packaged_training_release", lambda *a, **k: release
    )
    monkeypatch.setattr(worker, "_parent_trainer_reference", lambda _: digest)
    monkeypatch.setattr(worker, "_local_cpu_core", lambda _: core)
    spawned = []

    def child(command, **kwargs):
        spawned.append(command)
        assert (
            os.pread(kwargs["pass_fds"][0], len(worker.LOCAL_CPU_DATA), 0)
            == worker.LOCAL_CPU_DATA
        )
        kwargs["stdout"].write(
            execution._canonical(worker.local_cpu_result(release, digest))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", child)
    return release, spawned


def _fail(*args, **kwargs):
    raise RuntimeError("PRIVATE_EXCEPTION_SENTINEL")


@pytest.mark.parametrize(
    "stage",
    [
        "RELEASE_ADMISSION",
        "PARENT_RELEASE",
        "INPUT_PREPARATION",
        "FIXTURE_SETUP",
        "FIXTURE_WRITE",
        "TRANSPORT",
        "CHILD_RESULT",
        "POST_CHILD_CUSTODY",
        "CLEANUP",
    ],
)
def test_each_parent_boundary_has_closed_stage_and_no_raw_text(monkeypatch, stage):
    _, spawned = _fixture(monkeypatch)
    if stage == "RELEASE_ADMISSION":
        monkeypatch.setattr(worker, "parse_packaged_runtime_release", _fail)
    elif stage == "PARENT_RELEASE":
        monkeypatch.setattr(worker, "_parent_trainer_reference", _fail)
    elif stage == "INPUT_PREPARATION":
        monkeypatch.setattr(core, "_sealed_prepared_dataset", _fail)
    elif stage == "FIXTURE_SETUP":
        monkeypatch.setattr(tempfile, "TemporaryDirectory", _fail)
    elif stage == "CHILD_RESULT":
        monkeypatch.setattr(subprocess, "run", _fail)
    elif stage == "POST_CHILD_CUSTODY":
        original = execution._HeldModelFile.check
        checks = []

        def check(self):
            checks.append(1)
            if len(checks) > 1:
                _fail()
            return original(self)

        monkeypatch.setattr(execution._HeldModelFile, "check", check)
    elif stage == "CLEANUP":
        original = execution._HeldModelFile.close

        def close(self):
            original(self)
            _fail()

        monkeypatch.setattr(execution._HeldModelFile, "close", close)
    else:
        original = os.write
        fail_at = {"FIXTURE_WRITE": 2, "TRANSPORT": 3}[stage]
        writes = []

        def write(fd, data):
            writes.append(1)
            if len(writes) == fail_at:
                _fail()
            return original(fd, data)

        monkeypatch.setattr(os, "write", write)
    with pytest.raises(worker.PackagedLocalCPUStageError) as rejected:
        worker.qualify_installed_child({})
    assert rejected.value.stage == stage
    assert str(rejected.value) == "PACKAGED_LOCAL_CPU_REJECTED"
    assert rejected.value.__suppress_context__
    assert len(spawned) == (1 if stage in {"POST_CHILD_CUSTODY", "CLEANUP"} else 0)


def test_primary_failure_survives_cleanup_failure_and_all_closes_attempted(monkeypatch):
    _fixture(monkeypatch)
    original_memfd = core._linux_memfd_create
    original_close = os.close
    input_fds = []
    closed = []

    def memfd(*args):
        fd = original_memfd(*args)
        input_fds.append(fd)
        return fd

    def close(fd):
        closed.append(fd)
        return original_close(fd)

    monkeypatch.setattr(core, "_linux_memfd_create", memfd)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(subprocess, "run", _fail)
    leaf_close = execution._HeldModelFile.close

    def failing_leaf_close(self):
        leaf_close(self)
        _fail()

    monkeypatch.setattr(execution._HeldModelFile, "close", failing_leaf_close)
    with pytest.raises(worker.PackagedLocalCPUStageError) as rejected:
        worker.qualify_installed_child({})
    assert rejected.value.stage == "CHILD_RESULT"
    assert input_fds[0] in closed
    with pytest.raises(OSError):
        os.fstat(input_fds[0])


def test_primary_write_failure_survives_writer_close_failure(monkeypatch):
    _fixture(monkeypatch)
    original_write, original_close = os.write, os.close
    bad = []

    def write(fd, data):
        if data == worker.LOCAL_CPU_MODEL:
            bad.append(fd)
            _fail()
        return original_write(fd, data)

    def close(fd):
        original_close(fd)
        if fd in bad:
            _fail()

    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(os, "close", close)
    with pytest.raises(worker.PackagedLocalCPUStageError) as rejected:
        worker.qualify_installed_child({})
    assert rejected.value.stage == "FIXTURE_WRITE"


def test_success_result_keeps_canonical_bytes(monkeypatch):
    release, spawned = _fixture(monkeypatch)
    expected = worker.local_cpu_result(
        release, hashlib.sha256(b"reviewed trainer").hexdigest()
    )
    assert execution._canonical(
        worker.qualify_installed_child({})
    ) == execution._canonical(expected)
    assert len(spawned) == 1


@pytest.mark.parametrize(
    "module",
    [
        "subprocess",
        "tempfile",
        "contextlib",
        "tuner.runtime.packaged_sft_execution",
    ],
)
def test_import_failure_is_release_admission_before_resources(monkeypatch, module):
    import builtins

    _fixture(monkeypatch)
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == module:
            raise ImportError("PRIVATE_IMPORT_DETAILS")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(
        core,
        "_linux_memfd_create",
        lambda *_a: pytest.fail("created input after import failure"),
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: pytest.fail("child after import failure")
    )
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(worker.PackagedLocalCPUStageError) as rejected:
        worker.qualify_installed_child({})
    assert rejected.value.stage == "RELEASE_ADMISSION"
    assert rejected.value.__suppress_context__
    assert str(rejected.value) == "PACKAGED_LOCAL_CPU_REJECTED"


@pytest.mark.parametrize("fault", ["write", "seal", "readback"])
def test_preparation_fault_closes_real_descriptor_before_child(monkeypatch, fault):
    _, spawned = _fixture(monkeypatch)
    created = []
    original = core._linux_memfd_create

    def create(name):
        descriptor = original(name)
        created.append(descriptor)
        return descriptor

    monkeypatch.setattr(core, "_linux_memfd_create", create)

    def os_failure(*args, **kwargs):
        raise OSError("PRIVATE_OS_DETAILS")

    if fault == "write":
        monkeypatch.setattr(os, "write", os_failure)
    elif fault == "seal":
        monkeypatch.setattr(fcntl, "fcntl", os_failure)
    else:
        monkeypatch.setattr(os, "pread", lambda *args: b"")
    with pytest.raises(worker.PackagedLocalCPUStageError) as rejected:
        worker.qualify_installed_child({})
    assert rejected.value.stage == "INPUT_PREPARATION" and not spawned
    assert len(created) == 1
    with pytest.raises(OSError):
        os.fstat(created[0])


def test_missing_python_memfd_and_fcntl_names_use_real_sealed_source(monkeypatch):
    release, spawned = _fixture(monkeypatch)
    for name in ("memfd_create", "MFD_ALLOW_SEALING", "MFD_CLOEXEC"):
        monkeypatch.delattr(os, name, raising=False)
    for name in (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_WRITE",
    ):
        monkeypatch.delattr(fcntl, name, raising=False)
    original_run = subprocess.run

    def check(command, **kwargs):
        descriptor = kwargs["pass_fds"][0]
        required = (
            core._LINUX_F_SEAL_SEAL
            | core._LINUX_F_SEAL_SHRINK
            | core._LINUX_F_SEAL_GROW
            | core._LINUX_F_SEAL_WRITE
        )
        assert fcntl.fcntl(descriptor, core._LINUX_F_GET_SEALS) & required == required
        with pytest.raises(PermissionError):
            os.write(descriptor, b"x")
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", check)
    assert worker.qualify_installed_child({}) == worker.local_cpu_result(
        release, hashlib.sha256(b"reviewed trainer").hexdigest()
    )
    assert len(spawned) == 1


def test_missing_sealing_dependency_fails_before_fd_or_child(monkeypatch):
    import builtins
    _, spawned = _fixture(monkeypatch)
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("PRIVATE_IMPORT_DETAILS")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setattr(core, "_linux_memfd_create", lambda *_a: pytest.fail("input after dependency failure"))
    with pytest.raises(worker.PackagedLocalCPUStageError) as rejected:
        worker.qualify_installed_child({})
    assert rejected.value.stage == "INPUT_PREPARATION" and not spawned
