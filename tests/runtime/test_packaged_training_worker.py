from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tuner.runtime.packaged_training_worker as worker
from tuner.runtime.packaged_worker_closure import load_packaged_worker_closure
from tuner.runtime.packaged_training_worker import (
    PACKAGED_TRAINING_WORKER_ENTRYPOINT,
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
    trainer = tmp_path / "trainer.py"
    trainer.write_bytes(b"pass\n")
    monkeypatch.setattr(worker, "parse_packaged_runtime_release", lambda _: release)
    monkeypatch.setattr(worker, "admit_packaged_training_release", lambda *args, **kwargs: release)
    monkeypatch.setattr(execution, "_inspect_release", lambda _: trainer)
    original_flags = sys.flags
    class NonIsolatedFlags:
        isolated = 0

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(worker.sys, "flags", NonIsolatedFlags())
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
        expected = worker.local_cpu_result(release, execution._digest(trainer.read_bytes()))
        kwargs["stdout"].write(execution._canonical(expected) + extra_output)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", inspect_child)
    if extra_output:
        with pytest.raises(ValueError):
            worker.qualify_installed_child({})
    else:
        assert worker.qualify_installed_child({}) == worker.local_cpu_result(
            release, execution._digest(trainer.read_bytes())
        )


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
