from __future__ import annotations

import sys
import hashlib
import json
import warnings
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

import tuner.runtime.packaged_training_worker as worker
from tuner.runtime.packaged_worker_closure import load_packaged_worker_closure
from tuner.runtime.packaged_training_worker import (
    PACKAGED_TRAINING_WORKER_ENTRYPOINT,
    PackagedLocalCPUStageError,
    PackagedTrainingWorkerError,
    admit_packaged_training_release,
)
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1


def _release() -> PackagedTrainingRuntimeReleaseV1:
    closure = load_packaged_worker_closure().digest
    return PackagedTrainingRuntimeReleaseV1.build(
        release_ref="runtime:test", package_name="synaptic-tuner", package_version="1.2.3",
        package_digest="a" * 64, source_provenance_digest="b" * 64,
        worker_entrypoint=PACKAGED_TRAINING_WORKER_ENTRYPOINT, worker_closure_digest=closure,
        image_ref="registry.example/synaptic/tuner@sha256:" + "c" * 64,
        image_digest="c" * 64, python_implementation="cpython", python_version="3.12.3",
        python_executable="/usr/local/bin/python", python_executable_digest="d" * 64,
        installed_distributions_digest="e" * 64, installed_distribution_count=1,
        platform_system="linux", platform_machine="x86_64", cuda_version=None,
        runtime_facts={"gpu": False}, compatible_methods=("sft",),
        compatible_models=(("Qwen/Qwen3.5-4B", "1" * 40),),
        compatible_dataset_formats=("syntunia-sft-row/v2",),
        workload_schema="synaptic-sft-workload/v1",
        prepared_input_schema="synaptic-prepared-training-input/v1",
        artifact_contract_schema="synaptic-sft-artifacts/v1",
    )


def test_admission_accepts_exact_embedded_release() -> None:
    release = _release()
    assert admit_packaged_training_release(
        release.canonical_bytes(), expected_release_digest=release.manifest_digest
    ) == release


@pytest.mark.parametrize(
    "payload, digest",
    [(b"{}", "a" * 64), (b"x" * (128 * 1024 + 1), "a" * 64)],
    ids=("noncanonical", "oversize"),
)
def test_admission_rejects_noncanonical_or_oversize_release(payload: bytes, digest: str) -> None:
    with pytest.raises(PackagedTrainingWorkerError, match="PACKAGED_RELEASE_REJECTED"):
        admit_packaged_training_release(payload, expected_release_digest=digest)


def test_admission_rejects_wrong_release_digest() -> None:
    release = _release()
    with pytest.raises(PackagedTrainingWorkerError, match="PACKAGED_RELEASE_REJECTED"):
        admit_packaged_training_release(release.canonical_bytes(), expected_release_digest="0" * 64)


@pytest.mark.parametrize("fault", [RuntimeError, SystemExit, KeyboardInterrupt, GeneratorExit])
def test_admission_collapses_manifest_loader_fault(monkeypatch: pytest.MonkeyPatch, fault: type[BaseException]) -> None:
    release = _release()
    monkeypatch.setattr(worker, "load_packaged_worker_closure", lambda: (_ for _ in ()).throw(fault("PRIVATE_SENTINEL")))
    with pytest.raises(PackagedTrainingWorkerError, match="PACKAGED_RELEASE_REJECTED") as rejected:
        worker.admit_packaged_training_release(
            release.canonical_bytes(), expected_release_digest=release.manifest_digest
        )
    assert rejected.value.__suppress_context__
    assert "PRIVATE_SENTINEL" not in str(rejected.value)


def _wheel_bytes(members: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as archive, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for name, data in members:
            archive.writestr(name, data)
    return output.getvalue()


def _reference_fixture(
    monkeypatch: pytest.MonkeyPatch, wheel_raw: bytes, *, filename: str = "synaptic_tuner-1.2.3-py3-none-any.whl",
    build_digest: str | None = None, release_digest: str | None = None,
) -> SimpleNamespace:
    wheel_digest = hashlib.sha256(wheel_raw).hexdigest()
    inputs = {"wheel": {"filename": filename, "distribution": "synaptic-tuner",
                        "version": "1.2.3", "sha256": build_digest or wheel_digest}}
    inputs_raw = (json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")

    def read(path: Path, maximum: int = 1024 * 1024) -> bytes:
        if path == Path("/opt/synaptic-runtime/build-inputs.json"):
            assert maximum == worker._MAX_BUILD_INPUTS_BYTES
            return inputs_raw
        assert path == Path("/opt/synaptic-runtime") / filename
        assert maximum == worker._MAX_WHEEL_BYTES
        return wheel_raw

    monkeypatch.setattr(worker, "stable_read", read)
    return SimpleNamespace(package_digest=release_digest or wheel_digest)


def test_parent_reference_uses_pinned_wheel_not_parent_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = b"print('installed child compiles this')\n"
    release = _reference_fixture(monkeypatch, _wheel_bytes([("Trainers/sft/train_sft.py", trainer)]))
    monkeypatch.setattr(worker.sys, "executable", "/unrelated/host/python")

    assert worker._parent_trainer_reference(release) == hashlib.sha256(trainer).hexdigest()


@pytest.mark.parametrize("wrong", ["build", "release", "wheel"])
def test_parent_reference_rejects_wrong_wheel_hash(monkeypatch: pytest.MonkeyPatch, wrong: str) -> None:
    wheel_raw = _wheel_bytes([("Trainers/sft/train_sft.py", b"pass\n")])
    release = _reference_fixture(monkeypatch, wheel_raw,
                                 build_digest="0" * 64 if wrong in {"build", "wheel"} else None,
                                 release_digest="0" * 64 if wrong in {"release", "wheel"} else None)
    with pytest.raises(ValueError):
        worker._parent_trainer_reference(release)


@pytest.mark.parametrize("filename", ["../escape.whl", "subdir/other.whl", "..\\other.whl", "/tmp/other.whl", "name.zip"])
def test_parent_reference_rejects_unsafe_wheel_path(monkeypatch: pytest.MonkeyPatch, filename: str) -> None:
    release = _reference_fixture(monkeypatch, _wheel_bytes([("Trainers/sft/train_sft.py", b"pass\n")]), filename=filename)
    with pytest.raises(ValueError):
        worker._parent_trainer_reference(release)


@pytest.mark.parametrize("members", [
    [],
    [("Trainers/sft/not_train_sft.py", b"pass\n")],
    [("Trainers/sft/train_sft.py", b"a"), ("Trainers/sft/train_sft.py", b"b")],
    [("Trainers/sft/train_sft.py", b"a" * (4 * 1024 * 1024 + 1))],
])
def test_parent_reference_rejects_missing_duplicate_or_oversized_trainer(
    monkeypatch: pytest.MonkeyPatch, members: list[tuple[str, bytes]]
) -> None:
    release = _reference_fixture(monkeypatch, _wheel_bytes(members))
    with pytest.raises(ValueError):
        worker._parent_trainer_reference(release)


@pytest.mark.skipif(sys.platform != "linux", reason="sealed local CPU qualification requires Linux")
@pytest.mark.parametrize("extra_output", [b"", b"\n"])
def test_qualification_from_nonisolated_parent_keeps_child_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_output: bytes
) -> None:
    import fcntl
    import os
    import subprocess

    from tuner.runtime import packaged_sft_execution as execution

    release = SimpleNamespace(
        manifest_digest="a" * 64, python_executable="/opt/python/bin/python",
        canonical_bytes=lambda: b"release",
    )
    trainer_digest = hashlib.sha256(b"pass\n").hexdigest()
    monkeypatch.setattr(worker, "parse_packaged_runtime_release", lambda _: release)
    monkeypatch.setattr(worker, "admit_packaged_training_release", lambda *args, **kwargs: release)
    monkeypatch.setattr(worker, "_parent_trainer_reference", lambda _: trainer_digest)
    original_flags = sys.flags
    class NonIsolatedFlags:
        isolated = 0

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(worker.sys, "flags", NonIsolatedFlags())
    assert sys.executable != release.python_executable
    monkeypatch.setenv("HF_TOKEN", "PRIVATE_CREDENTIAL_SENTINEL")

    def inspect_child(command, **kwargs):
        assert command[:4] == [release.python_executable, "-I", "-m", "tuner.runtime.packaged_sft_child"]
        assert kwargs["env"] == worker.local_cpu_environment(release)
        assert "PRIVATE_CREDENTIAL_SENTINEL" not in repr(kwargs["env"])
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert len(kwargs["pass_fds"]) == 1
        fd = kwargs["pass_fds"][0]
        seals = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
        assert seals & fcntl.F_SEAL_WRITE
        assert os.pread(fd, len(worker.LOCAL_CPU_DATA), 0) == worker.LOCAL_CPU_DATA
        expected = worker.local_cpu_result(release, trainer_digest)
        kwargs["stdout"].write(execution._canonical(expected) + extra_output)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", inspect_child)
    if extra_output:
        with pytest.raises(PackagedLocalCPUStageError) as rejected:
            worker.qualify_installed_child({})
        assert rejected.value.stage == "CHILD_RESULT"
        assert str(rejected.value) == "PACKAGED_LOCAL_CPU_REJECTED"
    else:
        assert worker.qualify_installed_child({}) == worker.local_cpu_result(
            release, trainer_digest
        )


@pytest.mark.skipif(sys.platform != "linux", reason="local CPU qualification requires Linux")
def test_qualification_distinguishes_parent_release_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = SimpleNamespace(manifest_digest="a" * 64, canonical_bytes=lambda: b"release")
    monkeypatch.setattr(worker, "parse_packaged_runtime_release", lambda _: release)
    monkeypatch.setattr(worker, "admit_packaged_training_release", lambda *args, **kwargs: release)
    monkeypatch.setattr(worker, "_parent_trainer_reference", lambda _: (_ for _ in ()).throw(RuntimeError("PRIVATE_SENTINEL")))

    with pytest.raises(PackagedLocalCPUStageError) as rejected:
        worker.qualify_installed_child({})
    assert rejected.value.stage == "PARENT_RELEASE"
    assert str(rejected.value) == "PACKAGED_LOCAL_CPU_REJECTED"
    assert rejected.value.__suppress_context__


@pytest.mark.skipif(sys.platform != "linux", reason="local CPU qualification requires Linux")
@pytest.mark.parametrize("returncode, stderr_bytes", [(2, b""), (0, b"PRIVATE_SENTINEL")])
def test_qualification_collapses_child_exit_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int, stderr_bytes: bytes
) -> None:
    import subprocess

    from tuner.runtime import packaged_sft_execution as execution

    release = SimpleNamespace(
        manifest_digest="a" * 64, python_executable="/opt/python/bin/python",
        canonical_bytes=lambda: b"release",
    )
    trainer_digest = hashlib.sha256(b"pass\n").hexdigest()
    monkeypatch.setattr(worker, "parse_packaged_runtime_release", lambda _: release)
    monkeypatch.setattr(worker, "admit_packaged_training_release", lambda *args, **kwargs: release)
    monkeypatch.setattr(worker, "_parent_trainer_reference", lambda _: trainer_digest)

    def fail_child(command, **kwargs):
        kwargs["stdout"].write(execution._canonical(worker.local_cpu_result(release, trainer_digest)))
        kwargs["stderr"].write(stderr_bytes)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(subprocess, "run", fail_child)
    with pytest.raises(PackagedLocalCPUStageError) as rejected:
        worker.qualify_installed_child({})
    assert rejected.value.stage == "CHILD_RESULT"
    assert str(rejected.value) == "PACKAGED_LOCAL_CPU_REJECTED"


@pytest.mark.skipif(sys.platform != "linux", reason="local CPU child requires Linux")
def test_local_cpu_child_rejects_nonisolated_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    from tuner.runtime import packaged_sft_child

    original_flags = sys.flags
    class NonIsolatedFlags:
        isolated = 0

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(packaged_sft_child.sys, "flags", NonIsolatedFlags())
    with pytest.raises(ValueError):
        packaged_sft_child.run_local_cpu_child(["--qualify-local", "transport", "0" * 64])
