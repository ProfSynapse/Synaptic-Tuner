"""Real isolated CPU protocol; package/image observations explicitly substituted."""

from pathlib import Path
import subprocess
import sys

import pytest

from tuner.runtime import packaged_training_worker as worker
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1
from tests.runtime.test_packaged_runtime_releases import _release_values
from tests.runtime import cpu_qualification_image as image

SOURCE = Path(worker.__file__).absolute().parents[2]
HELPERS = Path(__file__).absolute().parent
BOOTSTRAP = """
import sys,json
from pathlib import Path
sys.path[:0]=[sys.argv[2],sys.argv[1]]
import cpu_qualification_image
cpu_qualification_image.install(Path(sys.argv[1]),json.loads(Path(sys.argv[3]).read_bytes()))
from tuner.runtime import packaged_sft_child as child
assert Path(child.__file__).absolute().parents[2] == Path(sys.argv[1])
sys.argv=['packaged_sft_child',*sys.argv[4:]]
try:
    raise SystemExit(child.run_local_cpu_child(sys.argv[1:]))
except Exception as error:
    locations=[]; trace=error.__traceback__
    while trace is not None:
        locations.append([Path(trace.tb_frame.f_code.co_filename).name,trace.tb_lineno]);trace=trace.tb_next
    print(json.dumps({'class':type(error).__name__,'locations':locations}),file=sys.stderr)
    raise SystemExit(2)
"""


def _release(method, config, **changes):
    values = _release_values(
        worker_entrypoint=worker.PACKAGED_TRAINING_WORKER_ENTRYPOINT,
        worker_closure_digest=config[method]["closure_digest"],
        package_digest=config["wheel_sha256"],
        python_executable=sys.executable,
        compatible_methods=(method,),
        compatible_dataset_formats=("synaptic-grpo-prompts/v1",)
        if method == "grpo"
        else ("syntunia-sft-row/v2",),
        workload_schema=f"synaptic-packaged-{method}-workload/v1",
        artifact_contract_schema=f"synaptic-{method}-artifacts/v1",
    )
    values.update(changes)
    return PackagedTrainingRuntimeReleaseV1.build(**values)


def _image(tmp_path, monkeypatch, mutation=None):
    config_path, config = image.capture(SOURCE, tmp_path, mutation=mutation)
    image.install(SOURCE, config, monkeypatch.setattr)
    return config_path, config


def _child(monkeypatch, config_path):
    original = subprocess.run
    calls = []

    def run(command, **kwargs):
        assert command[:4] == [
            sys.executable,
            "-I",
            "-m",
            "tuner.runtime.packaged_sft_child",
        ]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "HF_TOKEN" not in kwargs["env"]
        assert len(kwargs["pass_fds"]) == 1
        calls.append(tuple(command))
        actual = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            BOOTSTRAP,
            str(SOURCE),
            str(HELPERS),
            str(config_path),
            *command[4:],
        ]
        result = original(actual, **kwargs)
        if result.returncode:
            kwargs["stderr"].seek(0)
            print(kwargs["stderr"].read().decode())
        return result

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.parametrize("method", ["sft", "grpo"])
def test_actual_isolated_cpu_protocol_selects_exact_release(
    tmp_path, monkeypatch, method
):
    config_path, config = _image(tmp_path, monkeypatch)
    release = _release(method, config)
    calls = _child(monkeypatch, config_path)
    if method == "grpo":
        with pytest.raises(worker.PackagedTrainingWorkerError):
            worker.admit_packaged_training_release(
                release.canonical_bytes(),
                expected_release_digest=release.manifest_digest,
            )
    result = worker.qualify_installed_child(release.to_dict())
    trainer = (
        "Trainers/sft/train_sft.py"
        if method == "sft"
        else "Trainers/grpo/runtime_v1.py"
    )
    import hashlib

    assert result == worker.local_cpu_result(
        release, hashlib.sha256((SOURCE / trainer).read_bytes()).hexdigest()
    )
    assert len(calls) == 1
    assert result["training_executed"] is False and result["gpu_qualified"] is False
    assert result["provider_qualified"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"workload_schema": "unknown-workload/v1"},
        {"compatible_methods": ("grpo", "sft")},
        {"artifact_contract_schema": "synaptic-sft-artifacts/v1"},
        {"prepared_input_schema": "other-input/v1"},
        {"compatible_dataset_formats": ("syntunia-sft-row/v2",)},
    ],
)
def test_mixed_release_contracts_reject_before_process(tmp_path, monkeypatch, changes):
    _, config = _image(tmp_path, monkeypatch)
    release = _release("grpo", config, **changes)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("spawned invalid release"),
    )
    with pytest.raises(ValueError, match="PACKAGED_CPU_METHOD_REJECTED"):
        worker.qualify_installed_child(release.to_dict())


def test_wrong_outer_closure_rejects_before_image_read(tmp_path, monkeypatch):
    _, config = _image(tmp_path, monkeypatch)
    release = _release(
        "grpo", config, worker_closure_digest=config["sft"]["closure_digest"]
    )
    monkeypatch.setattr(
        worker, "_parent_trainer_reference", lambda *args: pytest.fail("read trainer")
    )
    with pytest.raises(worker.PackagedTrainingWorkerError):
        worker.qualify_installed_child(release.to_dict())


@pytest.mark.parametrize("mutation", ["missing_grpo_trainer", "wrong_wheel_python"])
def test_wrong_trainer_or_owned_wheel_import_never_qualifies(
    tmp_path, monkeypatch, mutation
):
    path, config = _image(tmp_path, monkeypatch, mutation)
    release = _release("grpo", config)
    calls = _child(monkeypatch, path)
    with pytest.raises(worker.PackagedLocalCPUStageError):
        worker.qualify_installed_child(release.to_dict())
    assert len(calls) == (0 if mutation == "missing_grpo_trainer" else 1)


def test_legacy_sft_schema_keeps_existing_result_bytes(tmp_path, monkeypatch):
    path, config = _image(tmp_path, monkeypatch)
    release = _release("sft", config, workload_schema="synaptic-sft-workload/v1")
    _child(monkeypatch, path)
    result = worker.qualify_installed_child(release.to_dict())
    assert image.canonical(result) == image.canonical(
        worker.local_cpu_result(release, result["trainer_sha256"])
    )
