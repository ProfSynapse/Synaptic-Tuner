"""Provider-independent definition of the single protected HF training smoke."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from tuner.cloud.hf_training_smoke_contract import validate_runtime_lock as validate_runtime_lock_contract
from tuner.cloud.hf_volume_transport import (
    HFVerifiedVolumeSpec,
    build_training_provider_command,
)
from tuner.core.exceptions import CloudProviderError


RECIPE_PATH = "Trainers/recipes/protected/hf_smollm2_135m_training_smoke.yaml"
RUNTIME_LOCK_PATH = "Trainers/cloud/runtime-locks/hf_training_smoke_unsloth_2026_1_2.json"
RECIPE_SHA256 = "1d1b898731d8d9cb874c50d6c9770be9f4f02e368c214679e26e5f91bfef2e65"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "a91318be21aeaf0879874faa161dcb40c68847e9"
DATASET = "Datasets/tools_datasets/non_thinking/contentManager/smoke_tools_v2.5.jsonl"
DATASET_SHA256 = "1e0d08073ca5f8400899b2cb61c8459177600a10fd30a2c2c53eaa3f4a38d854"
DATASET_GIT_BLOB = "1623a48f1980a0e3d39aa5b59fb3856f6d3b2408"
HARDWARE = "a10g-small"
SOURCE_MOUNT = "/workspace/synaptic-bootstrap-input"
PROJECT_ROOT = "/workspace/project"
ENGINE_ROOT = "/workspace/engine"
ARTIFACT_MOUNT = "/workspace/artifacts"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TrainingSmokeWorkloadError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _regular_bytes(path: Path, *, maximum: int) -> bytes:
    import os
    import stat

    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise TrainingSmokeWorkloadError("Protected workload input is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if len(content) > maximum or not stat.S_ISREG(opened.st_mode):
        raise TrainingSmokeWorkloadError("Protected workload input exceeds its bound")
    return content


def validate_recipe(path: Path) -> bytes:
    content = _regular_bytes(path, maximum=64 * 1024)
    if hashlib.sha256(content).hexdigest() != RECIPE_SHA256:
        raise TrainingSmokeWorkloadError("Protected recipe bytes do not match the reviewed identity")
    try:
        recipe = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise TrainingSmokeWorkloadError("Protected recipe is invalid") from exc
    if not isinstance(recipe, Mapping):
        raise TrainingSmokeWorkloadError("Protected recipe must be an object")
    protected = recipe.get("protected")
    expected = {
        "capability": "hf.training-smoke", "dataset_sha256": DATASET_SHA256,
        "dataset_git_blob": DATASET_GIT_BLOB, "dataset_rows": 1,
        "hardware": HARDWARE, "publication": False, "retries": 0,
        "job_secrets": [],
    }
    if protected != expected:
        raise TrainingSmokeWorkloadError("Protected recipe authority fields drifted")
    model = recipe.get("model", {})
    training = recipe.get("training", {})
    lora = recipe.get("lora", {})
    dataset = recipe.get("dataset", {})
    if (
        model.get("model_name") != MODEL
        or model.get("model_revision") != MODEL_REVISION
        or model.get("anonymous") is not True
        or model.get("trust_remote_code") is not False
        or model.get("use_safetensors") is not True
        or model.get("load_in_4bit") is not False
        or dataset.get("local_file") != f"{PROJECT_ROOT}/{DATASET}"
        or training.get("max_steps") != 1
        or training.get("per_device_train_batch_size") != 1
        or training.get("gradient_accumulation_steps") != 1
        or training.get("save_steps") != 1
        or training.get("save_total_limit") != 1
        or training.get("logging_steps") != 1
        or lora.get("r") != 8
        or lora.get("lora_alpha") != 16
        or lora.get("random_state") != 3407
    ):
        raise TrainingSmokeWorkloadError("Protected recipe runtime fields drifted")
    return content


def validate_runtime_lock(path: Path) -> tuple[dict[str, object], bytes]:
    content = _regular_bytes(path, maximum=128 * 1024)
    try:
        lock = json.loads(content.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingSmokeWorkloadError("Runtime lock is invalid") from exc
    if not isinstance(lock, dict):
        raise TrainingSmokeWorkloadError("Runtime lock has a noncanonical shape")
    try:
        lock = validate_runtime_lock_contract(lock)
    except CloudProviderError as exc:
        raise TrainingSmokeWorkloadError("Runtime lock is not a reviewed canonical contract") from exc
    image = lock["image"]
    child = image.get("child_digest")
    expected_provider_reference = f"unsloth/unsloth@{child}"
    if (
        image.get("registry_repository") != "docker.io/unsloth/unsloth"
        or image.get("provider_repository") != "unsloth/unsloth"
        or image.get("provider_reference") != expected_provider_reference
    ):
        raise TrainingSmokeWorkloadError("Runtime lock provider image identity drifted")
    if _canonical_json(lock) != content:
        raise TrainingSmokeWorkloadError("Runtime lock is not canonically encoded")
    return lock, content


@dataclass(frozen=True)
class ProtectedWorkload:
    argv: tuple[str, ...]
    remote_argv: tuple[str, ...]
    provider_command: tuple[str, ...]
    remote_argv_sha256: str
    provider_command_sha256: str
    command: str
    recipe_sha256: str
    runtime_lock_sha256: str
    workload_sha256: str
    image: str


_ARGUMENT_ORDER = (
    "--recipe", "--recipe-sha256", "--runtime-lock", "--runtime-lock-sha256",
    "--source-lock", "--source-lock-sha256", "--artifact-root", "--artifact-slot",
    "--project-root", "--engine-root",
)


def validate_remote_argv(argv: Sequence[str]) -> None:
    if len(argv) != len(_ARGUMENT_ORDER) * 2:
        raise TrainingSmokeWorkloadError("Protected remote argv has the wrong length")
    if tuple(argv[::2]) != _ARGUMENT_ORDER:
        raise TrainingSmokeWorkloadError("Protected remote argv order or option set drifted")
    if any("=" in option or not option.startswith("--") for option in argv[::2]):
        raise TrainingSmokeWorkloadError("Protected remote argv uses a forbidden option form")
    if len(set(argv[::2])) != len(_ARGUMENT_ORDER):
        raise TrainingSmokeWorkloadError("Protected remote argv contains duplicate options")


def build_workload(
    repository: Path, *, source_lock_sha256: str, artifact_slot: str,
    runtime_lock_path: Path | None = None,
    source_volume_spec: HFVerifiedVolumeSpec | None = None,
    expected_project_root: str | None = None,
    expected_engine_root: str | None = None,
    expected_project_commit: str | None = None,
    expected_engine_commit: str | None = None,
    expected_mode: str | None = None,
) -> ProtectedWorkload:
    if not _HEX64.fullmatch(source_lock_sha256) or not _HEX64.fullmatch(artifact_slot):
        raise TrainingSmokeWorkloadError("Protected workload identities must be lowercase SHA-256")
    recipe_path = repository / RECIPE_PATH
    validate_recipe(recipe_path)
    lock_path = runtime_lock_path or repository / RUNTIME_LOCK_PATH
    lock, lock_bytes = validate_runtime_lock(lock_path)
    lock_digest = hashlib.sha256(lock_bytes).hexdigest()
    image = lock["image"]["provider_reference"]
    argv = (
        "--recipe", f"{PROJECT_ROOT}/{RECIPE_PATH}",
        "--recipe-sha256", RECIPE_SHA256,
        "--runtime-lock", f"{PROJECT_ROOT}/{RUNTIME_LOCK_PATH}",
        "--runtime-lock-sha256", lock_digest,
        "--source-lock", f"{SOURCE_MOUNT}/source-lock.json",
        "--source-lock-sha256", source_lock_sha256,
        "--artifact-root", ARTIFACT_MOUNT,
        "--artifact-slot", artifact_slot,
        "--project-root", PROJECT_ROOT,
        "--engine-root", ENGINE_ROOT,
    )
    validate_remote_argv(argv)
    provider_inputs = (
        source_volume_spec, expected_project_root, expected_engine_root,
        expected_project_commit, expected_engine_commit, expected_mode,
    )
    if any(value is not None for value in provider_inputs):
        if any(value is None for value in provider_inputs):
            raise TrainingSmokeWorkloadError(
                "Protected provider command requires every authenticated source binding"
            )
        assert source_volume_spec is not None
        provider_command = build_training_provider_command(
            source_volume_spec, remote_argv=argv,
            expected_project_root=str(expected_project_root),
            expected_engine_root=str(expected_engine_root),
            expected_project_commit=str(expected_project_commit),
            expected_engine_commit=str(expected_engine_commit),
            expected_mode=str(expected_mode),
        )
    else:
        provider_command = ()
    remote_argv_sha256 = hashlib.sha256(_canonical_json(list(argv))).hexdigest()
    provider_command_sha256 = hashlib.sha256(
        _canonical_json(list(provider_command))
    ).hexdigest()
    command = "python -m tuner.cloud.hf_training_smoke_remote_entry " + " ".join(argv)
    identity = {
        "schema_version": "synaptic-hf-training-workload/v1",
        "module": "tuner.cloud.hf_training_smoke_remote_entry",
        "recipe_sha256": RECIPE_SHA256, "runtime_lock_sha256": lock_digest,
        "model": MODEL, "model_revision": MODEL_REVISION,
        "dataset_sha256": DATASET_SHA256, "dataset_git_blob": DATASET_GIT_BLOB,
        "hardware": HARDWARE, "image": image,
    }
    return ProtectedWorkload(
        argv=argv, remote_argv=argv, provider_command=provider_command,
        remote_argv_sha256=remote_argv_sha256,
        provider_command_sha256=provider_command_sha256,
        command=command, recipe_sha256=RECIPE_SHA256,
        runtime_lock_sha256=lock_digest,
        workload_sha256=hashlib.sha256(_canonical_json(identity)).hexdigest(), image=image,
    )


__all__ = [
    "ARTIFACT_MOUNT", "DATASET", "DATASET_GIT_BLOB", "DATASET_SHA256",
    "ENGINE_ROOT", "HARDWARE", "MODEL", "MODEL_REVISION", "PROJECT_ROOT",
    "ProtectedWorkload", "RECIPE_PATH", "RECIPE_SHA256", "RUNTIME_LOCK_PATH",
    "TrainingSmokeWorkloadError", "build_workload", "validate_recipe",
    "validate_remote_argv", "validate_runtime_lock",
]
