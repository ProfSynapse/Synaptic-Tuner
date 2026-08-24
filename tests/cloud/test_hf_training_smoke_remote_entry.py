from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tuner.cloud import hf_training_smoke_remote_entry as remote
from tuner.cloud.hf_training_smoke_contract import (
    RUNTIME_PYTHON_IMPLEMENTATION,
    RUNTIME_PYTHON_VERSION,
)


def _safe_open(filename, framework, device="cpu"):
    return None


def _torch_load(f, *, weights_only=None):
    return None


class _TrainerCallback:
    def on_optimizer_step(self, args, state, control, **kwargs):
        return None


def _install_cpu_modules(
    monkeypatch, events: list[str], *, available=True, device_name="NVIDIA A10G",
    count=1, capability=(8, 6), total_memory=24 * 1024**3,
):
    class _Properties:
        pass

    properties = _Properties()
    properties.total_memory = total_memory

    class _Cuda:
        @staticmethod
        def is_available():
            events.append("cuda.available")
            return available

        @staticmethod
        def device_count():
            events.append("cuda.count")
            return count

        @staticmethod
        def get_device_name(index):
            events.append("cuda.name")
            return device_name

        @staticmethod
        def get_device_capability(index):
            events.append("cuda.capability")
            return capability

        @staticmethod
        def get_device_properties(index):
            events.append("cuda.properties")
            return properties

    torch = types.ModuleType("torch")
    torch.cuda = _Cuda()
    torch.load = _torch_load
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = _safe_open
    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = _TrainerCallback
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _install_unsloth(monkeypatch, events: list[str], *, loader=..., import_error=None):
    class _FastLanguageModel:
        @staticmethod
        def from_pretrained(model_name, *, revision=None):
            return model_name, revision

    module = types.ModuleType("unsloth")

    def resolve_attr(name):
        if name == "FastLanguageModel":
            events.append("unsloth.import")
            if import_error is not None:
                raise import_error
            if loader is None:
                return types.SimpleNamespace(from_pretrained=None)
            return _FastLanguageModel
        raise AttributeError(name)

    module.__dict__["__getattr__"] = resolve_attr
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return _FastLanguageModel


def _lock(path: Path, *, signatures=None) -> Path:
    cpu_signatures = {
        "TrainerCallback.on_optimizer_step": str(inspect.signature(_TrainerCallback.on_optimizer_step)),
        "safetensors.safe_open": str(inspect.signature(_safe_open)),
        "torch.load": str(inspect.signature(_torch_load)),
        "unsloth.import": "GPU_RUNTIME_REQUIRED",
    }
    payload = {
        "runtime": {
            "python_implementation": RUNTIME_PYTHON_IMPLEMENTATION,
            "python": RUNTIME_PYTHON_VERSION,
            "packages": {"torch": "2.9.0"},
            "signatures": cpu_signatures if signatures is None else signatures,
        }
    }
    path.write_text(json.dumps(payload), encoding="ascii")
    return path


@pytest.fixture(autouse=True)
def _package_version(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.9.0")
    monkeypatch.setattr(remote.platform, "python_implementation", lambda: RUNTIME_PYTHON_IMPLEMENTATION)
    monkeypatch.setattr(remote.platform, "python_version", lambda: RUNTIME_PYTHON_VERSION)


def test_runtime_attests_gpu_before_unsloth_import(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events)

    evidence = remote._runtime_evidence(_lock(tmp_path / "lock.json"))

    assert set(evidence) == {
        "python_implementation", "python", "packages", "signatures",
        "gpu_observed_signatures", "cuda",
    }
    assert events == [
        "cuda.available", "cuda.count", "cuda.name", "cuda.capability",
        "cuda.properties", "unsloth.import",
    ]
    assert evidence["cuda"]["device_name"] == "NVIDIA A10G"
    assert evidence["python_implementation"] == "CPython"
    assert evidence["python"] == "3.11.14"
    assert evidence["signatures"]["unsloth.import"] == "GPU_RUNTIME_REQUIRED"
    assert "FastLanguageModel.from_pretrained" in evidence["gpu_observed_signatures"]


@pytest.mark.parametrize(
    ("implementation", "version", "message", "expected_events"),
    [
        ("PyPy", RUNTIME_PYTHON_VERSION, "implementation drifted", ["python.implementation"]),
        (True, RUNTIME_PYTHON_VERSION, "implementation drifted", ["python.implementation"]),
        (RUNTIME_PYTHON_IMPLEMENTATION, "3.12.7", "runtime drifted", ["python.implementation", "python.version"]),
        (RUNTIME_PYTHON_IMPLEMENTATION, 31114, "runtime drifted", ["python.implementation", "python.version"]),
    ],
)
def test_runtime_identity_fails_before_packages_imports_and_gpu(
    monkeypatch, tmp_path: Path, implementation: object, version: object,
    message: str, expected_events: list[str],
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        remote.platform, "python_implementation",
        lambda: events.append("python.implementation") or implementation,
    )
    monkeypatch.setattr(
        remote.platform, "python_version",
        lambda: events.append("python.version") or version,
    )
    monkeypatch.setattr(
        remote.importlib.metadata, "version",
        lambda name: events.append("package.version") or "2.9.0",
    )
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events)

    with pytest.raises(remote.RemoteTrainingSmokeError, match=message):
        remote._runtime_evidence(_lock(tmp_path / "lock.json"))

    assert events == expected_events


def test_runtime_identity_is_checked_in_order_before_packages_and_gpu(
    monkeypatch, tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        remote.platform, "python_implementation",
        lambda: events.append("python.implementation") or RUNTIME_PYTHON_IMPLEMENTATION,
    )
    monkeypatch.setattr(
        remote.platform, "python_version",
        lambda: events.append("python.version") or RUNTIME_PYTHON_VERSION,
    )
    monkeypatch.setattr(
        remote.importlib.metadata, "version",
        lambda name: events.append("package.version") or "2.9.0",
    )
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events)

    remote._runtime_evidence(_lock(tmp_path / "lock.json"))

    assert events[:3] == ["python.implementation", "python.version", "package.version"]
    assert events.index("package.version") < events.index("cuda.available")


@pytest.mark.parametrize(
    ("field", "value", "message", "expected_events"),
    [
        ("python_implementation", "PyPy", "implementation drifted", ["python.implementation"]),
        ("python_implementation", True, "implementation drifted", ["python.implementation"]),
        (
            "python", "3.12.7", "runtime drifted",
            ["python.implementation", "python.version"],
        ),
        (
            "python", 31114, "runtime drifted",
            ["python.implementation", "python.version"],
        ),
    ],
)
def test_locked_runtime_identity_drift_fails_before_packages_imports_and_gpu(
    monkeypatch, tmp_path: Path, field: str, value: object,
    message: str, expected_events: list[str],
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        remote.platform, "python_implementation",
        lambda: events.append("python.implementation") or RUNTIME_PYTHON_IMPLEMENTATION,
    )
    monkeypatch.setattr(
        remote.platform, "python_version",
        lambda: events.append("python.version") or RUNTIME_PYTHON_VERSION,
    )
    monkeypatch.setattr(
        remote.importlib.metadata, "version",
        lambda name: events.append("package.version") or "2.9.0",
    )
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events)
    lock = _lock(tmp_path / "lock.json")
    payload = json.loads(lock.read_text(encoding="ascii"))
    payload["runtime"][field] = value
    lock.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(remote.RemoteTrainingSmokeError, match=message):
        remote._runtime_evidence(lock)

    assert events == expected_events


@pytest.mark.parametrize("total_memory", [23_068_672_000, 25_769_803_776])
def test_a10g_memory_bounds_are_inclusive(monkeypatch, tmp_path: Path, total_memory: int) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events, total_memory=total_memory)
    _install_unsloth(monkeypatch, events)
    evidence = remote._runtime_evidence(_lock(tmp_path / "lock.json"))
    assert evidence["cuda"]["total_memory"] == total_memory


@pytest.mark.parametrize(
    ("device_name", "count", "message"),
    [("NVIDIA L4", 1, "approved A10G"), ("NVIDIA A10G", 2, "exactly one")],
)
def test_wrong_gpu_fails_before_unsloth_import(
    monkeypatch, tmp_path: Path, device_name: str, count: int, message: str
) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events, device_name=device_name, count=count)
    _install_unsloth(monkeypatch, events)

    with pytest.raises(remote.RemoteTrainingSmokeError, match=message):
        remote._runtime_evidence(_lock(tmp_path / "lock.json"))

    assert "unsloth.import" not in events


@pytest.mark.parametrize(
    ("overrides", "expected_events"),
    [
        ({"available": 1}, ["cuda.available"]),
        ({"count": True}, ["cuda.available", "cuda.count"]),
        ({"device_name": "NVIDIA A10G "}, ["cuda.available", "cuda.count", "cuda.name"]),
        (
            {"capability": [8, 6]},
            ["cuda.available", "cuda.count", "cuda.name", "cuda.capability"],
        ),
        (
            {"total_memory": 25_769_803_777},
            ["cuda.available", "cuda.count", "cuda.name", "cuda.capability", "cuda.properties"],
        ),
    ],
)
def test_hostile_gpu_types_and_identity_fail_in_exact_query_order(
    monkeypatch, tmp_path: Path, overrides: dict[str, object], expected_events: list[str]
) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events, **overrides)
    _install_unsloth(monkeypatch, events)

    with pytest.raises(remote.RemoteTrainingSmokeError):
        remote._runtime_evidence(_lock(tmp_path / "lock.json"))

    assert events == expected_events


def test_requires_exact_gpu_runtime_sentinel_before_gpu_probe(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events)
    signatures = {
        "TrainerCallback.on_optimizer_step": str(inspect.signature(_TrainerCallback.on_optimizer_step)),
        "safetensors.safe_open": str(inspect.signature(_safe_open)),
        "torch.load": str(inspect.signature(_torch_load)),
        "unsloth.import": "captured-on-cpu",
    }

    with pytest.raises(remote.RemoteTrainingSmokeError, match="GPU-only Unsloth sentinel"):
        remote._runtime_evidence(_lock(tmp_path / "lock.json", signatures=signatures))

    assert events == []


def test_missing_gpu_side_loader_fails_closed(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events, loader=None)

    with pytest.raises(remote.RemoteTrainingSmokeError, match="not callable"):
        remote._runtime_evidence(_lock(tmp_path / "lock.json"))


def test_gpu_side_import_exception_is_normalized_fail_closed(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events, import_error=RuntimeError("hostile import"))

    with pytest.raises(remote.RemoteTrainingSmokeError, match="GPU-side Unsloth import failed"):
        remote._runtime_evidence(_lock(tmp_path / "lock.json"))


def test_gpu_side_signature_is_bounded(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events)
    fast_language_model = _install_unsloth(monkeypatch, events)
    lock = _lock(tmp_path / "lock.json")
    real_signature = inspect.signature

    def hostile_signature(target):
        if target is fast_language_model.from_pretrained:
            return "x" * (remote._MAX_GPU_SIGNATURE_BYTES + 1)
        return real_signature(target)

    monkeypatch.setattr(remote.inspect, "signature", hostile_signature)
    with pytest.raises(remote.RemoteTrainingSmokeError, match="signature is unbounded"):
        remote._runtime_evidence(lock)


def test_gpu_side_signature_must_be_printable_ascii(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events)
    fast_language_model = _install_unsloth(monkeypatch, events)
    lock = _lock(tmp_path / "lock.json")
    real_signature = inspect.signature

    def hostile_signature(target):
        if target is fast_language_model.from_pretrained:
            return "bad\nline"
        return real_signature(target)

    monkeypatch.setattr(remote.inspect, "signature", hostile_signature)
    with pytest.raises(remote.RemoteTrainingSmokeError, match="signature is unbounded"):
        remote._runtime_evidence(lock)


def test_cpu_safe_signature_drift_fails_before_gpu_probe(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events)
    lock = _lock(tmp_path / "lock.json")
    payload = json.loads(lock.read_text(encoding="ascii"))
    payload["runtime"]["signatures"]["torch.load"] = "(drifted)"
    lock.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(remote.RemoteTrainingSmokeError, match="callable signature drifted"):
        remote._runtime_evidence(lock)

    assert events == []


def test_package_version_drift_fails_before_runtime_imports(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    _install_cpu_modules(monkeypatch, events)
    _install_unsloth(monkeypatch, events)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.9.1")

    with pytest.raises(remote.RemoteTrainingSmokeError, match="package version drifted"):
        remote._runtime_evidence(_lock(tmp_path / "lock.json"))

    assert events == []


def test_runtime_attestation_precedes_dataset_and_training_actions() -> None:
    source = inspect.getsource(remote.run)
    assert source.index("_runtime_evidence(runtime_lock)") < source.index("_validate_dataset(dataset)")
    assert source.index("_runtime_evidence(runtime_lock)") < source.index("subprocess.run(")


def test_main_sanitizes_all_failure_details(monkeypatch, capfd) -> None:
    secret = "hf_secret_that_must_not_escape"

    def fail(argv):
        os.write(1, secret.encode("ascii"))
        os.write(2, secret.encode("ascii"))
        raise RuntimeError(secret)

    monkeypatch.setattr(remote, "run", fail)
    assert remote.main(["--hostile", secret]) == 125
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == remote._PRIVATE_FAILURE_BYTES.decode("ascii")
    assert json.loads(captured.err) == {
        "reason_code": "REMOTE_TRAINING_SMOKE_REJECTED",
        "schema_version": "synaptic-hf-training-private-error/v1",
    }
    assert captured.err.count("\n") == 1
    assert secret not in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("stage", "reason_code", "exit_code"),
    [
        ("credential", "REMOTE_CREDENTIAL_REJECTED", 120),
        ("runtime", "REMOTE_RUNTIME_REJECTED", 121),
        ("artifact", "REMOTE_ARTIFACT_REJECTED", 122),
        ("trainer", "REMOTE_TRAINER_REJECTED", 123),
    ],
)
def test_main_exposes_only_closed_failure_stage(
    monkeypatch, capfd, stage: str, reason_code: str, exit_code: int
) -> None:
    secret = "hf_private_failure_detail"

    def fail(argv):
        os.write(1, secret.encode("ascii"))
        os.write(2, secret.encode("ascii"))
        raise remote.RemoteTrainingSmokeError(secret, stage=stage)

    monkeypatch.setattr(remote, "run", fail)
    assert remote.main([]) == exit_code
    captured = capfd.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "reason_code": reason_code,
        "schema_version": "synaptic-hf-training-private-error/v1",
    }
    assert captured.err.count("\n") == 1
    assert secret not in captured.err
    assert "Traceback" not in captured.err


def test_remote_failure_stage_must_be_from_closed_set() -> None:
    with pytest.raises(ValueError, match="failure stage is invalid"):
        remote.RemoteTrainingSmokeError("private", stage="unexpected")


def test_main_sanitizes_argparse_boundary(capfd) -> None:
    assert remote.main(["--unknown", "hf_secret"]) == 125
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == remote._PRIVATE_FAILURE_BYTES.decode("ascii")
    assert json.loads(captured.err)["reason_code"] == "REMOTE_TRAINING_SMOKE_REJECTED"


def test_main_discards_incidental_success_output(monkeypatch, capfd) -> None:
    def succeed(argv):
        os.write(1, b"private stdout")
        os.write(2, b"private stderr")
        return {}

    monkeypatch.setattr(remote, "run", succeed)
    assert remote.main([]) == 0
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("failure", [False, True])
def test_main_contains_real_child_process_output(monkeypatch, capfd, failure: bool) -> None:
    secret = "native-child-secret"

    def child(argv):
        subprocess.run(
            [
                sys.executable, "-c",
                "import os;os.write(1,b'native-child-secret');os.write(2,b'native-child-secret')",
            ],
            check=True,
        )
        if failure:
            raise RuntimeError(secret)
        return {}

    monkeypatch.setattr(remote, "run", child)
    assert remote.main([]) == (125 if failure else 0)
    captured = capfd.readouterr()
    assert captured.out == ""
    if failure:
        assert captured.err == remote._PRIVATE_FAILURE_BYTES.decode("ascii")
        assert json.loads(captured.err)["reason_code"] == "REMOTE_TRAINING_SMOKE_REJECTED"
    else:
        assert captured.err == ""
    assert secret not in captured.err


def test_main_contains_high_volume_native_output_without_buffering(monkeypatch, capfd) -> None:
    def noisy(argv):
        chunk = b"private-native-output" * 4096
        for _ in range(64):
            os.write(1, chunk)
            os.write(2, chunk)
        return {}

    monkeypatch.setattr(remote, "run", noisy)
    assert remote.main([]) == 0
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_restores_process_descriptors(monkeypatch, capfd) -> None:
    monkeypatch.setattr(remote, "run", lambda argv: {})
    assert remote.main([]) == 0
    os.write(1, b"restored-out")
    os.write(2, b"restored-err")
    captured = capfd.readouterr()
    assert captured.out == "restored-out"
    assert captured.err == "restored-err"


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit, BaseException])
def test_main_sanitizes_base_exception(monkeypatch, capfd, failure_type) -> None:
    secret = "private-base-exception-detail"

    def fail(argv):
        os.write(1, secret.encode("ascii"))
        os.write(2, secret.encode("ascii"))
        raise failure_type(secret)

    monkeypatch.setattr(remote, "run", fail)
    assert remote.main([]) == 125
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == remote._PRIVATE_FAILURE_BYTES.decode("ascii")
    assert secret not in captured.err
    assert "Traceback" not in captured.err


def test_main_sanitizes_descriptor_setup_failure(monkeypatch, capfd) -> None:
    def fail_enter(self):
        raise OSError("private descriptor failure")

    monkeypatch.setattr(remote._SilenceProcessDescriptors, "__enter__", fail_enter)
    assert remote.main([]) == 125
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == remote._PRIVATE_FAILURE_BYTES.decode("ascii")


def test_trainer_subprocess_uses_fixed_null_standard_streams() -> None:
    source = inspect.getsource(remote.run)
    assert "stdin=subprocess.DEVNULL" in source
    assert "stdout=subprocess.DEVNULL" in source
    assert "stderr=subprocess.DEVNULL" in source
