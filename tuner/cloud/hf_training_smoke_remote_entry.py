"""Constant, no-shell remote entrypoint for the protected HF training smoke."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path

from tuner.cloud.hf_training_smoke_artifacts import build_inventory, safetensors_identity
from tuner.cloud.hf_training_smoke_contract import (
    RUNTIME_PYTHON_IMPLEMENTATION,
    RUNTIME_PYTHON_VERSION,
)
from tuner.cloud.hf_training_smoke_workload import (
    DATASET, DATASET_SHA256, MODEL_REVISION, RECIPE_SHA256,
    TrainingSmokeWorkloadError, build_workload, validate_remote_argv,
)


_FAILURE_STAGES = {
    "credential": ("REMOTE_CREDENTIAL_REJECTED", 120),
    "runtime": ("REMOTE_RUNTIME_REJECTED", 121),
    "artifact": ("REMOTE_ARTIFACT_REJECTED", 122),
    "trainer": ("REMOTE_TRAINER_REJECTED", 123),
    "input": ("REMOTE_INPUT_REJECTED", 124),
}


class RemoteTrainingSmokeError(RuntimeError):
    def __init__(self, message: str, *, stage: str | None = None) -> None:
        if stage is not None and stage not in _FAILURE_STAGES:
            raise ValueError("Protected failure stage is invalid")
        super().__init__(message)
        self.stage = stage


@contextmanager
def _failure_stage(stage: str):
    """Map every in-phase failure to one closed non-secret diagnostic."""

    if stage not in _FAILURE_STAGES:
        raise ValueError("Protected failure stage is invalid")
    try:
        yield
    except RemoteTrainingSmokeError as exc:
        if exc.stage is not None:
            raise
        raise RemoteTrainingSmokeError("Protected remote phase failed", stage=stage) from exc
    except BaseException as exc:
        raise RemoteTrainingSmokeError("Protected remote phase failed", stage=stage) from exc


_GPU_RUNTIME_SENTINEL = "GPU_RUNTIME_REQUIRED"
_MAX_GPU_SIGNATURE_BYTES = 4096
_APPROVED_GPU_NAME = "NVIDIA A10G"
_APPROVED_COMPUTE_CAPABILITY = (8, 6)
_MIN_A10G_MEMORY_BYTES = 23_068_672_000
_MAX_A10G_MEMORY_BYTES = 25_769_803_776
_PRIVATE_FAILURE = {
    "reason_code": "REMOTE_TRAINING_SMOKE_REJECTED",
    "schema_version": "synaptic-hf-training-private-error/v1",
}


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


_PRIVATE_FAILURE_BYTES = _canonical_json(_PRIVATE_FAILURE)


def _flush_private_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


class _SilenceProcessDescriptors:
    """Redirect process fd 0/1/2 without retaining private output in memory."""

    def __init__(self) -> None:
        self._null: int | None = None
        self._saved: dict[int, int] = {}

    def __enter__(self) -> None:
        _flush_private_streams()
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
        self._null = os.open(os.devnull, flags)
        try:
            for descriptor in (1, 2):
                self._saved[descriptor] = os.dup(descriptor)
            for descriptor in (1, 2):
                os.dup2(self._null, descriptor)
        except Exception:
            self._restore()
            raise

    def _restore(self) -> None:
        _flush_private_streams()
        for descriptor in (1, 2):
            saved = self._saved.get(descriptor)
            if saved is not None:
                try:
                    os.dup2(saved, descriptor)
                except OSError:
                    pass
        for saved in self._saved.values():
            try:
                os.close(saved)
            except OSError:
                pass
        self._saved.clear()
        if self._null is not None:
            try:
                os.close(self._null)
            except OSError:
                pass
            self._null = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._restore()
        return False


def _private_failure(stage: str | None) -> tuple[bytes, int]:
    classified = _FAILURE_STAGES.get(stage or "")
    if classified is None:
        return _PRIVATE_FAILURE_BYTES, 125
    reason_code, exit_code = classified
    return _canonical_json({
        "reason_code": reason_code,
        "schema_version": "synaptic-hf-training-private-error/v1",
    }), exit_code


def _write_private_failure(stage: str | None = None) -> int:
    payload, exit_code = _private_failure(stage)
    try:
        os.write(2, payload)
    except OSError:
        pass
    return exit_code


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


def _sha256_file(path: Path, maximum: int = 512 * 1024 * 1024) -> str:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise RemoteTrainingSmokeError("Protected input is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            digest.update(chunk)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if remaining <= 0 or not stat.S_ISREG(opened.st_mode):
        raise RemoteTrainingSmokeError("Protected input exceeds its bound")
    return digest.hexdigest()


def _copy_regular(source: Path, destination: Path, *, maximum: int = 512 * 1024 * 1024) -> None:
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise RemoteTrainingSmokeError("Protected copy source is not a bounded regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RemoteTrainingSmokeError("Protected artifact destination already exists")
    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    source_fd = os.open(source, source_flags)
    destination_fd = None
    source_digest = hashlib.sha256()
    try:
        opened = os.fstat(source_fd)
        destination_fd = os.open(destination, destination_flags, 0o600)
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > maximum:
                raise RemoteTrainingSmokeError("Protected copy source exceeds its bound")
            source_digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                offset += os.write(destination_fd, chunk[offset:])
        after = source.lstat()
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
    signatures = {
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        for value in (before, opened, after)
    }
    if len(signatures) != 1:
        raise RemoteTrainingSmokeError("Protected copy source changed during read")
    if source_digest.hexdigest() != _sha256_file(destination, maximum):
        raise RemoteTrainingSmokeError("Protected artifact copy changed bytes")


def _validate_dataset(path: Path) -> None:
    if _sha256_file(path, 4 * 1024 * 1024) != DATASET_SHA256:
        raise RemoteTrainingSmokeError("Protected dataset digest mismatch")
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteTrainingSmokeError("Protected dataset is invalid JSONL") from exc
    if len(rows) != 1 or not isinstance(rows[0], dict) or not isinstance(rows[0].get("conversations"), list):
        raise RemoteTrainingSmokeError("Protected dataset must contain exactly one conversation row")


def _reject_remote_credentials() -> None:
    forbidden = {"HF_TOKEN", "HF_API_KEY", "WANDB_API_KEY", "HUGGING_FACE_HUB_TOKEN"}
    present = sorted(key for key in forbidden if os.environ.get(key))
    if present:
        raise RemoteTrainingSmokeError("Protected job environment contains forbidden credentials", stage="credential")


def _runtime_evidence(runtime_lock: Path) -> dict[str, object]:
    """Recheck captured dependency/signature/CUDA evidence inside the real job."""

    try:
        lock = json.loads(runtime_lock.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteTrainingSmokeError("Protected runtime lock could not be read", stage="runtime") from exc
    runtime = lock.get("runtime") if isinstance(lock, dict) else None
    if not isinstance(runtime, dict):
        raise RemoteTrainingSmokeError("Protected Python runtime drifted", stage="runtime")
    python_implementation = platform.python_implementation()
    if (
        type(python_implementation) is not str
        or python_implementation != RUNTIME_PYTHON_IMPLEMENTATION
        or runtime.get("python_implementation") != RUNTIME_PYTHON_IMPLEMENTATION
    ):
        raise RemoteTrainingSmokeError("Protected Python implementation drifted", stage="runtime")
    python_version = platform.python_version()
    if (
        type(python_version) is not str
        or python_version != RUNTIME_PYTHON_VERSION
        or runtime.get("python") != RUNTIME_PYTHON_VERSION
    ):
        raise RemoteTrainingSmokeError("Protected Python runtime drifted", stage="runtime")
    packages = runtime.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise RemoteTrainingSmokeError("Protected runtime package lock is empty", stage="runtime")
    observed_packages: dict[str, str] = {}
    for name, expected in sorted(packages.items()):
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RemoteTrainingSmokeError("Protected runtime package is missing", stage="runtime") from exc
        if observed != expected:
            raise RemoteTrainingSmokeError("Protected runtime package version drifted", stage="runtime")
        observed_packages[name] = observed

    try:
        import torch
        from safetensors import safe_open
        from transformers import TrainerCallback
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected CPU-safe runtime imports failed", stage="runtime") from exc

    resolvers = {
        "TrainerCallback.on_optimizer_step": TrainerCallback.on_optimizer_step,
        "safetensors.safe_open": safe_open,
        "torch.load": torch.load,
    }
    signatures = runtime.get("signatures")
    if not isinstance(signatures, dict):
        raise RemoteTrainingSmokeError("Protected runtime signature set drifted", stage="runtime")
    expected_signatures = dict(signatures)
    if expected_signatures.pop("unsloth.import", None) != _GPU_RUNTIME_SENTINEL:
        raise RemoteTrainingSmokeError("Protected runtime lacks the GPU-only Unsloth sentinel", stage="runtime")
    if set(expected_signatures) != set(resolvers):
        raise RemoteTrainingSmokeError("Protected runtime signature set drifted", stage="runtime")
    try:
        observed_signatures = {
            name: str(inspect.signature(target)) for name, target in resolvers.items()
        }
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected CPU-safe callable inspection failed", stage="runtime") from exc
    if observed_signatures != expected_signatures:
        raise RemoteTrainingSmokeError("Protected runtime callable signature drifted", stage="runtime")

    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected CUDA identity check failed", stage="runtime") from exc
    if type(cuda_available) is not bool or cuda_available is not True:
        raise RemoteTrainingSmokeError("Protected smoke requires an available CUDA GPU", stage="runtime")
    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected CUDA identity check failed", stage="runtime") from exc
    if type(device_count) is not int or device_count != 1:
        raise RemoteTrainingSmokeError("Protected smoke requires exactly one visible CUDA GPU", stage="runtime")
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected CUDA identity check failed", stage="runtime") from exc
    if type(device_name) is not str or device_name != _APPROVED_GPU_NAME:
        raise RemoteTrainingSmokeError("Protected smoke did not run on the approved A10G hardware", stage="runtime")
    try:
        compute_capability = torch.cuda.get_device_capability(0)
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected CUDA identity check failed", stage="runtime") from exc
    if (
        type(compute_capability) is not tuple
        or len(compute_capability) != 2
        or any(type(part) is not int for part in compute_capability)
        or compute_capability != _APPROVED_COMPUTE_CAPABILITY
    ):
        raise RemoteTrainingSmokeError("Protected A10G compute capability is invalid", stage="runtime")
    try:
        properties = torch.cuda.get_device_properties(0)
        total_memory = properties.total_memory
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected CUDA identity check failed", stage="runtime") from exc
    if (
        type(total_memory) is not int
        or not _MIN_A10G_MEMORY_BYTES <= total_memory <= _MAX_A10G_MEMORY_BYTES
    ):
        raise RemoteTrainingSmokeError("Protected A10G memory identity is invalid", stage="runtime")

    try:
        from unsloth import FastLanguageModel

        from_pretrained = FastLanguageModel.from_pretrained
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected GPU-side Unsloth import failed", stage="runtime") from exc
    if not callable(from_pretrained):
        raise RemoteTrainingSmokeError("Protected GPU-side model loader is not callable", stage="runtime")
    try:
        gpu_signature = str(inspect.signature(from_pretrained))
        signature_bytes = gpu_signature.encode("utf-8")
    except Exception as exc:
        raise RemoteTrainingSmokeError("Protected GPU-side callable inspection failed", stage="runtime") from exc
    if (
        not signature_bytes
        or len(signature_bytes) > _MAX_GPU_SIGNATURE_BYTES
        or any(byte < 0x20 or byte > 0x7E for byte in signature_bytes)
    ):
        raise RemoteTrainingSmokeError("Protected GPU-side callable signature is unbounded", stage="runtime")

    return {
        "python_implementation": python_implementation,
        "python": python_version, "packages": observed_packages,
        "signatures": {**observed_signatures, "unsloth.import": _GPU_RUNTIME_SENTINEL},
        "gpu_observed_signatures": {"FastLanguageModel.from_pretrained": gpu_signature},
        "cuda": {
            "available": True, "device_count": 1, "device_name": device_name,
            "compute_capability": list(compute_capability),
            "total_memory": total_memory,
        },
    }


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise RemoteTrainingSmokeError("Protected remote arguments are invalid")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        raise RemoteTrainingSmokeError("Protected remote arguments are invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _PrivateArgumentParser(add_help=False)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--source-lock-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-slot", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--engine-root", required=True)
    return parser


def run(argv: list[str]) -> dict[str, object]:
    with _failure_stage("input"):
        try:
            validate_remote_argv(argv)
        except TrainingSmokeWorkloadError as exc:
            raise RemoteTrainingSmokeError("Protected remote arguments are invalid") from exc
        args = _parser().parse_args(argv)
    _reject_remote_credentials()

    with _failure_stage("input"):
        project = Path(args.project_root)
        engine = Path(args.engine_root)
        if project.as_posix() != "/workspace/project" or engine.as_posix() != "/workspace/engine":
            raise RemoteTrainingSmokeError("Protected logical source roots drifted")
        recipe = Path(args.recipe)
        runtime_lock = Path(args.runtime_lock)
        source_lock = Path(args.source_lock)
        if args.recipe_sha256 != RECIPE_SHA256 or _sha256_file(recipe, 64 * 1024) != RECIPE_SHA256:
            raise RemoteTrainingSmokeError("Protected recipe identity mismatch")
        if _sha256_file(source_lock, 4 * 1024 * 1024) != args.source_lock_sha256:
            raise RemoteTrainingSmokeError("Protected source-lock identity mismatch")

    with _failure_stage("runtime"):
        if _sha256_file(runtime_lock, 128 * 1024) != args.runtime_lock_sha256:
            raise RemoteTrainingSmokeError("Protected runtime-lock identity mismatch")
        runtime_evidence = _runtime_evidence(runtime_lock)

    with _failure_stage("input"):
        dataset = project / DATASET
        _validate_dataset(dataset)
        workload = build_workload(
            project, source_lock_sha256=args.source_lock_sha256,
            artifact_slot=args.artifact_slot, runtime_lock_path=runtime_lock,
        )
        if tuple(argv) != workload.argv:
            raise RemoteTrainingSmokeError("Executed argv does not match the protected workload")

    with _failure_stage("artifact"):
        artifact_root = Path(args.artifact_root)
        if artifact_root.as_posix() != "/workspace/artifacts" or not artifact_root.is_dir():
            raise RemoteTrainingSmokeError("Protected artifact mount is unavailable")
        if any(artifact_root.iterdir()):
            raise RemoteTrainingSmokeError("Protected artifact prefix must be empty")
        _copy_regular(source_lock, artifact_root / "source-lock.json", maximum=4 * 1024 * 1024)
        _write_json(
            artifact_root / "exclusive-sentinel.json",
            {"schema_version": "synaptic-hf-training-exclusive-sentinel/v1", "artifact_slot": args.artifact_slot},
        )

    with _failure_stage("trainer"):
        private_root = Path("/workspace/private-training")
        run_root = private_root / args.artifact_slot
        evidence_path = run_root / "step-evidence.json"
        environment = {
            key: value for key, value in os.environ.items()
            if key not in {"HF_TOKEN", "HF_API_KEY", "WANDB_API_KEY", "HUGGING_FACE_HUB_TOKEN"}
        }
        environment.update(
            {
                "HOME": "/workspace/empty-home", "HF_HOME": "/workspace/cache/huggingface",
                "HF_TOKEN_PATH": "/workspace/empty-home/no-token",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "WANDB_DISABLED": "true",
                "PYTHONNOUSERSITE": "1",
            }
        )
        command = [
            sys.executable, str(project / "Trainers/sft/train_sft.py"),
            "--protected-smoke-config", str(recipe),
            "--protected-smoke-evidence", str(evidence_path),
            "--output-root", str(private_root), "--run-timestamp", args.artifact_slot,
            "--no-dashboard", "--quiet",
        ]
        completed = subprocess.run(
            command, cwd=project, env=environment, check=False, timeout=1500,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RemoteTrainingSmokeError("Protected trainer exited unsuccessfully")

    with _failure_stage("artifact"):
        checkpoint = run_root / "checkpoints" / "checkpoint-1"
        final_model = run_root / "final_model"
        copies = {
            checkpoint / "adapter_model.safetensors": artifact_root / "checkpoint-1/adapter_model.safetensors",
            checkpoint / "adapter_config.json": artifact_root / "checkpoint-1/adapter_config.json",
            checkpoint / "trainer_state.json": artifact_root / "checkpoint-1/trainer_state.json",
            checkpoint / "optimizer.pt": artifact_root / "checkpoint-1/optimizer.pt",
            checkpoint / "scheduler.pt": artifact_root / "checkpoint-1/scheduler.pt",
            final_model / "adapter_model.safetensors": artifact_root / "final_model/adapter_model.safetensors",
            final_model / "adapter_config.json": artifact_root / "final_model/adapter_config.json",
            final_model / "tokenizer_config.json": artifact_root / "final_model/tokenizer_config.json",
            run_root / "training_lineage.json": artifact_root / "training_lineage.json",
            evidence_path: artifact_root / "step-evidence.json",
        }
        for source, destination in copies.items():
            _copy_regular(source, destination)
        checkpoint_identity = safetensors_identity(artifact_root / "checkpoint-1/adapter_model.safetensors")
        final_identity = safetensors_identity(artifact_root / "final_model/adapter_model.safetensors")
        if checkpoint_identity != final_identity:
            raise RemoteTrainingSmokeError("Protected checkpoint and final adapter differ")
        evidence = json.loads((artifact_root / "step-evidence.json").read_text(encoding="utf-8"))
        evidence["serialized_adapter_identity"] = final_identity
        _write_json(artifact_root / "step-evidence.json", evidence)
        runtime_lock_document = json.loads(runtime_lock.read_text(encoding="ascii"))
        result = {
            "schema_version": "synaptic-hf-training-job-result/v1", "status": "COMPLETED",
            "publication": False, "model_revision": MODEL_REVISION,
            "dataset_sha256": DATASET_SHA256, "workload_sha256": workload.workload_sha256,
            "adapter_identity": final_identity, "artifact_slot": args.artifact_slot,
            "runtime_lock_id": runtime_lock_document["lock_id"],
            "runtime": runtime_evidence,
        }
        manifest = {
            "schema_version": "synaptic-hf-training-manifest/v1", "status": "COMPLETED",
            "publication": False, "workload_sha256": workload.workload_sha256,
            "artifact_slot": args.artifact_slot,
        }
        _write_json(artifact_root / "result.json", result)
        _write_json(artifact_root / "manifest.json", manifest)
        _write_json(artifact_root / "inventory.json", build_inventory(artifact_root))
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        with _SilenceProcessDescriptors():
            run(list(sys.argv[1:] if argv is None else argv))
    except RemoteTrainingSmokeError as exc:
        return _write_private_failure(exc.stage)
    except BaseException:
        return _write_private_failure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
