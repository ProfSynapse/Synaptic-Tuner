"""Closed, no-follow verification for protected training-smoke artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping

from tuner.cloud.hf_training_smoke_contract import (
    RUNTIME_PYTHON_IMPLEMENTATION,
    RUNTIME_PYTHON_VERSION,
)


MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NUMBER_CHARACTERS = 128
EXPECTED_PATHS = (
    "source-lock.json",
    "exclusive-sentinel.json",
    "checkpoint-1/adapter_model.safetensors",
    "checkpoint-1/adapter_config.json",
    "checkpoint-1/trainer_state.json",
    "checkpoint-1/optimizer.pt",
    "checkpoint-1/scheduler.pt",
    "final_model/adapter_model.safetensors",
    "final_model/adapter_config.json",
    "final_model/tokenizer_config.json",
    "training_lineage.json",
    "step-evidence.json",
    "result.json",
    "manifest.json",
    "inventory.json",
)
JSON_PATHS = tuple(path for path in EXPECTED_PATHS if path.endswith(".json"))


class TrainingSmokeArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactExpectation:
    source_lock_sha256: str
    workload_sha256: str
    model_revision: str
    dataset_sha256: str
    artifact_slot: str
    runtime_lock_id: str
    runtime_python_implementation: str
    runtime_python: str
    runtime_packages: tuple[tuple[str, str], ...]
    runtime_signatures: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for value in (
            self.source_lock_sha256, self.workload_sha256, self.dataset_sha256,
            self.artifact_slot, self.runtime_lock_id,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise TrainingSmokeArtifactError("Artifact expectation contains an invalid SHA-256 identity")
        if re.fullmatch(r"[0-9a-f]{40}", self.model_revision) is None:
            raise TrainingSmokeArtifactError("Artifact expectation contains an invalid model revision")
        if self.runtime_python_implementation != RUNTIME_PYTHON_IMPLEMENTATION:
            raise TrainingSmokeArtifactError(
                "Artifact expectation contains an invalid runtime Python implementation"
            )
        if self.runtime_python != RUNTIME_PYTHON_VERSION:
            raise TrainingSmokeArtifactError("Artifact expectation contains an invalid runtime Python")
        for label, entries in (
            ("packages", self.runtime_packages), ("signatures", self.runtime_signatures),
        ):
            if (
                type(entries) is not tuple
                or not entries
                or any(
                    type(entry) is not tuple
                    or len(entry) != 2
                    or type(entry[0]) is not str
                    or type(entry[1]) is not str
                    or not entry[0]
                    or not entry[1]
                    for entry in entries
                )
                or tuple(sorted(entries)) != entries
                or len({entry[0] for entry in entries}) != len(entries)
            ):
                raise TrainingSmokeArtifactError(
                    f"Artifact expectation contains invalid runtime {label}"
                )
        if dict(self.runtime_signatures).get("unsloth.import") != "GPU_RUNTIME_REQUIRED":
            raise TrainingSmokeArtifactError("Artifact expectation lacks the canonical Unsloth sentinel")
        if {entry[0] for entry in self.runtime_signatures} != {
            "TrainerCallback.on_optimizer_step",
            "safetensors.safe_open",
            "torch.load",
            "unsloth.import",
        }:
            raise TrainingSmokeArtifactError("Artifact expectation has a noncanonical signature set")


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _validate_root(root: Path) -> Path:
    root = Path(os.path.abspath(root))
    cursor = root
    chain = []
    while cursor != cursor.parent:
        chain.append(cursor)
        cursor = cursor.parent
    for component in reversed(chain):
        info = component.lstat()
        if _is_link_or_reparse(info):
            raise TrainingSmokeArtifactError("Artifact root traverses a link or reparse point")
    if not root.is_dir():
        raise TrainingSmokeArtifactError("Artifact root is not a directory")
    return root


def _inventory_paths(root: Path) -> tuple[str, ...]:
    found: list[str] = []
    stack = [(root, PurePosixPath())]
    while stack:
        directory, relative = stack.pop()
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name.casefold(), reverse=True)
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if _is_link_or_reparse(info):
                raise TrainingSmokeArtifactError("Artifact tree contains a link or reparse point")
            child = relative / entry.name
            value = child.as_posix()
            if any(part in {"", ".", ".."} for part in child.parts) or "\\" in value:
                raise TrainingSmokeArtifactError("Artifact path is not canonical")
            if stat.S_ISDIR(info.st_mode):
                stack.append((Path(entry.path), child))
            elif stat.S_ISREG(info.st_mode):
                found.append(value)
            else:
                raise TrainingSmokeArtifactError("Artifact tree contains a special file")
    if len({path.casefold() for path in found}) != len(found):
        raise TrainingSmokeArtifactError("Artifact paths collide under case normalization")
    return tuple(sorted(found))


def read_regular(root: Path, relative: str, *, maximum: int = MAX_FILE_BYTES) -> bytes:
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TrainingSmokeArtifactError("Artifact path escapes its root") from exc
    info = path.lstat()
    if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise TrainingSmokeArtifactError("Artifact is not a bounded regular file")
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
        raise TrainingSmokeArtifactError("Artifact exceeds its bound")
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
    ):
        raise TrainingSmokeArtifactError("Artifact changed while being read")
    return content


def _json(root: Path, relative: str, *, maximum: int = 2 * 1024 * 1024) -> object:
    content = read_regular(root, relative, maximum=maximum)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TrainingSmokeArtifactError("Artifact JSON contains a duplicate object key")
            result[key] = value
        return result

    def bounded_int(literal: str) -> int:
        if len(literal) > MAX_JSON_NUMBER_CHARACTERS:
            raise TrainingSmokeArtifactError("Artifact JSON contains an oversized numeric literal")
        return int(literal)

    def bounded_float(literal: str) -> float:
        if len(literal) > MAX_JSON_NUMBER_CHARACTERS:
            raise TrainingSmokeArtifactError("Artifact JSON contains an oversized numeric literal")
        value = float(literal)
        if not math.isfinite(value):
            raise TrainingSmokeArtifactError("Artifact JSON contains a non-finite number")
        return value

    def reject_constant(literal: str) -> object:
        raise TrainingSmokeArtifactError("Artifact JSON contains a non-finite number")

    try:
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=reject_duplicates,
            parse_int=bounded_int, parse_float=bounded_float, parse_constant=reject_constant,
        )
    except TrainingSmokeArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError, OverflowError) as exc:
        raise TrainingSmokeArtifactError(f"Artifact JSON is invalid: {relative}") from exc
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise TrainingSmokeArtifactError("Artifact JSON exceeds its nesting bound")
        if isinstance(current, dict):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
    return value


def safetensors_identity(path: Path) -> str:
    """Hash canonical tensor names, shapes, dtypes and raw values."""

    from safetensors import safe_open

    digest = hashlib.sha256()
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            if not keys:
                raise TrainingSmokeArtifactError("Adapter safetensors is empty")
            for key in keys:
                tensor = handle.get_tensor(key).detach().cpu().contiguous()
                raw = tensor.view(-1).view(__import__("torch").uint8).numpy().tobytes()
                digest.update(key.encode("utf-8") + b"\0")
                digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
                digest.update(str(tensor.dtype).encode("ascii") + b"\0")
                digest.update(hashlib.sha256(raw).digest())
    except TrainingSmokeArtifactError:
        raise
    except Exception as exc:
        raise TrainingSmokeArtifactError("Adapter safetensors could not be inspected safely") from exc
    return digest.hexdigest()


def build_inventory(root: Path) -> dict[str, object]:
    root = _validate_root(root)
    paths = _inventory_paths(root)
    if "inventory.json" in paths:
        paths = tuple(path for path in paths if path != "inventory.json")
    files = []
    total = 0
    for relative in paths:
        content = read_regular(root, relative)
        total += len(content)
        if total > MAX_TOTAL_BYTES:
            raise TrainingSmokeArtifactError("Artifact tree exceeds its aggregate byte bound")
        files.append({"path": relative, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return {"schema_version": "synaptic-hf-training-artifact-inventory/v1", "files": files, "total_bytes": total}


def verify_artifact_tree(
    root: Path, expectation: ArtifactExpectation, *,
    tensor_identity: Callable[[Path], str] = safetensors_identity,
) -> dict[str, object]:
    root = _validate_root(root)
    actual = _inventory_paths(root)
    if actual != tuple(sorted(EXPECTED_PATHS)):
        raise TrainingSmokeArtifactError("Artifact tree is not the exact closed inventory")
    parsed_json = {relative: _json(root, relative) for relative in JSON_PATHS}
    inventory = parsed_json["inventory.json"]
    if not isinstance(inventory, dict) or inventory != build_inventory(root):
        raise TrainingSmokeArtifactError("Artifact inventory does not match point-in-time contents")
    manifest = parsed_json["manifest.json"]
    result = parsed_json["result.json"]
    sentinel = parsed_json["exclusive-sentinel.json"]
    evidence = parsed_json["step-evidence.json"]
    lineage = parsed_json["training_lineage.json"]
    trainer_state = parsed_json["checkpoint-1/trainer_state.json"]
    source_lock = read_regular(root, "source-lock.json", maximum=4 * 1024 * 1024)
    if hashlib.sha256(source_lock).hexdigest() != expectation.source_lock_sha256:
        raise TrainingSmokeArtifactError("Artifact source-lock identity mismatch")
    manifest_keys = {"schema_version", "status", "publication", "artifact_slot", "workload_sha256"}
    if not isinstance(manifest, Mapping) or set(manifest) != manifest_keys or (
        manifest.get("schema_version") != "synaptic-hf-training-manifest/v1"
        or manifest.get("status") != "COMPLETED"
        or manifest.get("publication") is not False
        or manifest.get("artifact_slot") != expectation.artifact_slot
        or manifest.get("workload_sha256") != expectation.workload_sha256
    ):
        raise TrainingSmokeArtifactError("Completed manifest semantics are invalid")
    result_keys = {
        "schema_version", "status", "publication", "model_revision", "dataset_sha256",
        "workload_sha256", "adapter_identity", "artifact_slot", "runtime_lock_id", "runtime",
    }
    if not isinstance(result, Mapping) or set(result) != result_keys or (
        result.get("schema_version") != "synaptic-hf-training-job-result/v1"
        or result.get("status") != "COMPLETED"
        or result.get("publication") is not False
        or result.get("model_revision") != expectation.model_revision
        or result.get("dataset_sha256") != expectation.dataset_sha256
        or result.get("workload_sha256") != expectation.workload_sha256
        or result.get("artifact_slot") != expectation.artifact_slot
        or result.get("runtime_lock_id") != expectation.runtime_lock_id
    ):
        raise TrainingSmokeArtifactError("Training result semantics are invalid")
    if not isinstance(sentinel, Mapping) or set(sentinel) != {"schema_version", "artifact_slot"} or (
        sentinel.get("schema_version") != "synaptic-hf-training-exclusive-sentinel/v1"
        or sentinel.get("artifact_slot") != expectation.artifact_slot
    ):
        raise TrainingSmokeArtifactError("Exclusive sentinel semantics are invalid")
    runtime = result.get("runtime")
    expected_runtime_keys = {
        "python_implementation", "python", "packages", "signatures",
        "gpu_observed_signatures", "cuda",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != expected_runtime_keys:
        raise TrainingSmokeArtifactError("Training result runtime evidence is not closed")
    if (
        runtime.get("python_implementation") != RUNTIME_PYTHON_IMPLEMENTATION
        or runtime.get("python_implementation") != expectation.runtime_python_implementation
    ):
        raise TrainingSmokeArtifactError("Training result runtime Python implementation drifted")
    if (
        runtime.get("python") != RUNTIME_PYTHON_VERSION
        or runtime.get("python") != expectation.runtime_python
    ):
        raise TrainingSmokeArtifactError("Training result runtime Python drifted")
    packages = runtime.get("packages")
    expected_packages = dict(expectation.runtime_packages)
    if not isinstance(packages, Mapping) or set(packages) != set(expected_packages) or dict(packages) != expected_packages:
        raise TrainingSmokeArtifactError("Training result runtime packages drifted")
    signatures = runtime.get("signatures")
    expected_signatures = dict(expectation.runtime_signatures)
    if (
        not isinstance(signatures, Mapping)
        or set(signatures) != set(expected_signatures)
        or dict(signatures) != expected_signatures
        or signatures.get("unsloth.import") != "GPU_RUNTIME_REQUIRED"
    ):
        raise TrainingSmokeArtifactError("Training result runtime signatures drifted")
    cuda = runtime.get("cuda") if isinstance(runtime, Mapping) else None
    expected_cuda_keys = {
        "available", "device_count", "device_name", "compute_capability", "total_memory",
    }
    if not isinstance(cuda, Mapping) or set(cuda) != expected_cuda_keys or (
        type(cuda.get("available")) is not bool
        or cuda.get("available") is not True
        or type(cuda.get("device_count")) is not int
        or cuda.get("device_count") != 1
        or type(cuda.get("device_name")) is not str
        or cuda.get("device_name") != "NVIDIA A10G"
        or type(cuda.get("compute_capability")) is not list
        or len(cuda.get("compute_capability", [])) != 2
        or any(type(part) is not int for part in cuda.get("compute_capability", []))
        or cuda.get("compute_capability") != [8, 6]
        or type(cuda.get("total_memory")) is not int
        or not 23_068_672_000 <= cuda.get("total_memory", 0) <= 25_769_803_776
    ):
        raise TrainingSmokeArtifactError("Training result lacks approved A10G CUDA evidence")
    gpu_signatures = runtime.get("gpu_observed_signatures") if isinstance(runtime, Mapping) else None
    if not isinstance(gpu_signatures, Mapping) or set(gpu_signatures) != {
        "FastLanguageModel.from_pretrained"
    }:
        raise TrainingSmokeArtifactError("Training result lacks closed GPU signature evidence")
    gpu_signature = gpu_signatures.get("FastLanguageModel.from_pretrained")
    if (
        type(gpu_signature) is not str
        or not 1 <= len(gpu_signature) <= 4096
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in gpu_signature)
    ):
        raise TrainingSmokeArtifactError("Training result GPU signature evidence is invalid")
    if not isinstance(evidence, Mapping) or (
        evidence.get("schema_version") != "synaptic-protected-sft-evidence/v1"
        or type(evidence.get("global_step")) is not int
        or evidence.get("global_step") != 1
        or type(evidence.get("optimizer_boundaries")) is not int
        or evidence.get("optimizer_boundaries") != 1
        or type(evidence.get("step_one_loss")) not in {int, float}
        or not math.isfinite(float(evidence["step_one_loss"]))
        or type(evidence.get("delta_l2")) not in {int, float}
        or not math.isfinite(float(evidence["delta_l2"]))
        or float(evidence["delta_l2"]) <= 0
        or type(evidence.get("scheduler_last_epoch")) is not int
        or evidence.get("scheduler_last_epoch") != 1
        or not isinstance(evidence.get("optimizer_steps"), list)
        or not evidence.get("optimizer_steps")
        or any(type(step) is not int or step != 1 for step in evidence["optimizer_steps"])
        or not isinstance(evidence.get("pre_step_identity"), str)
        or not isinstance(evidence.get("post_step_identity"), str)
        or evidence.get("pre_step_identity") == evidence.get("post_step_identity")
        or type(evidence.get("changed_tensor_count")) is not int
        or evidence.get("changed_tensor_count", 0) <= 0
    ):
        raise TrainingSmokeArtifactError("Optimizer evidence does not prove one finite update")
    if (
        not isinstance(trainer_state, Mapping)
        or type(trainer_state.get("global_step")) is not int
        or trainer_state.get("global_step") != 1
    ):
        raise TrainingSmokeArtifactError("Checkpoint global step is not one")
    if not isinstance(lineage, Mapping) or (
        lineage.get("model_revision") != expectation.model_revision
        or lineage.get("dataset_sha256") != expectation.dataset_sha256
        or type(lineage.get("max_steps")) is not int
        or lineage.get("max_steps") != 1
        or type(lineage.get("gradient_accumulation_steps")) is not int
        or lineage.get("gradient_accumulation_steps") != 1
    ):
        raise TrainingSmokeArtifactError("Training lineage does not bind the protected workload")
    checkpoint_identity = tensor_identity(root / "checkpoint-1" / "adapter_model.safetensors")
    final_identity = tensor_identity(root / "final_model" / "adapter_model.safetensors")
    if checkpoint_identity != final_identity:
        raise TrainingSmokeArtifactError("Checkpoint and final adapter identities differ")
    if result.get("adapter_identity") != final_identity or evidence.get("serialized_adapter_identity") != final_identity:
        raise TrainingSmokeArtifactError("Adapter identity evidence is inconsistent")
    optimizer_proof = {
        "optimizer_boundaries": 1,
        "global_step": 1,
        "optimizer_step": 1,
        "scheduler_step": 1,
        "loss": evidence["step_one_loss"],
        "max_steps": 1,
        "gradient_accumulation_steps": 1,
        "pre_adapter_sha256": evidence["pre_step_identity"],
        "post_adapter_sha256": final_identity,
        "checkpoint_adapter_sha256": final_identity,
        "final_adapter_sha256": final_identity,
        "trainable_weight_delta": evidence["delta_l2"],
    }
    return {
        "status": "VERIFIED", "inventory_sha256": hashlib.sha256(_canonical_json(inventory)).hexdigest(),
        "adapter_identity": final_identity, "global_step": 1,
        "optimizer_proof": optimizer_proof,
    }


__all__ = [
    "ArtifactExpectation", "EXPECTED_PATHS", "TrainingSmokeArtifactError",
    "build_inventory", "read_regular", "safetensors_identity", "verify_artifact_tree",
]
