from __future__ import annotations

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
