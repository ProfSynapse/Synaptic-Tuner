"""Exact inventory and semantic verification for completed training runs."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tarfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from tuner.training.recipes import CompiledWorkload

from .artifacts import (
    ArtifactContract,
    ArtifactEntry,
    ArtifactIntegrity,
    ArtifactInventory,
    IntegrityVerification,
    InventoryVerification,
    verify_inventory,
)
from .dispatch import ProcessResult


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    INVALID = "invalid"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class SemanticCheck:
    code: str
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("semantic check code is required")
        if not isinstance(self.passed, bool):
            raise TypeError("semantic check result must be a boolean")


@runtime_checkable
class ArtifactReader(Protocol):
    """Bounded complete-read port.

    Implementations must return the complete artifact when its byte length is
    no greater than ``maximum``. They must raise ``ArtifactReadLimitExceeded``
    instead of truncating when the artifact is larger than that bound.
    """

    def read_bytes(self, artifact: ArtifactEntry, *, maximum: int) -> bytes: ...


class ArtifactReadError(RuntimeError):
    pass


class ArtifactReadLimitExceeded(ArtifactReadError):
    pass


MAX_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024
MAX_SEMANTIC_ARTIFACT_BYTES = 256 * 1024
MAX_ARCHIVE_MEMBERS = 1024
MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024 * 1024
MAX_ARCHIVE_EXPANDED_BYTES = 64 * 1024 * 1024 * 1024
_MODEL_CONFIGS = frozenset({"adapter_config.json", "config.json"})
_MODEL_PAYLOAD = re.compile(r"^(adapter_model|model)(?:-(\d{5})-of-(\d{5}))?\.safetensors$")
_TOKENIZER_CONFIGS = frozenset({"tokenizer_config.json"})
_TOKENIZER_PAYLOADS = frozenset({"tokenizer.json"})
_TOKENIZER_OPTIONAL = frozenset({
    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
    "merges.txt", "vocab.json",
})
_MODEL_OPTIONAL = frozenset({"generation_config.json", "README.md"})
_MAX_INDEX_BYTES = 16 * 1024 * 1024
_MAX_SHARDS = 1024
_MAX_TENSORS = 1_000_000
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_EXECUTION_EVIDENCE_SCHEMA = "synaptic-sft-execution-evidence/v1"
_SAFETENSORS_DTYPES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class _AuthenticatedArtifactReader:
    """Semantic reader backed only by bytes authenticated in this verification."""

    def __init__(self, authenticated: dict[str, bytes]) -> None:
        self._authenticated = dict(authenticated)

    def read_bytes(self, artifact: ArtifactEntry, *, maximum: int) -> bytes:
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
            raise ValueError("maximum must be a non-negative integer")
        try:
            value = self._authenticated[artifact.path]
        except KeyError as exc:
            raise ArtifactReadError("artifact bytes were not authenticated") from exc
        if len(value) > maximum:
            raise ArtifactReadLimitExceeded(
                "authenticated artifact exceeds semantic read bound"
            )
        return value


def authenticate_artifacts(
    inventory: ArtifactInventory,
    reader: ArtifactReader,
    *,
    maximum_artifact_bytes: int = MAX_ARTIFACT_BYTES,
) -> tuple[IntegrityVerification, ArtifactReader]:
    """Read and authenticate every inventory entry exactly once."""

    if not isinstance(inventory, ArtifactInventory):
        raise TypeError("inventory must be an ArtifactInventory")
    if not isinstance(reader, ArtifactReader):
        raise TypeError("reader must implement ArtifactReader")
    if (
        not isinstance(maximum_artifact_bytes, int)
        or isinstance(maximum_artifact_bytes, bool)
        or maximum_artifact_bytes < 1
    ):
        raise ValueError("maximum_artifact_bytes must be a positive integer")
    results: list[ArtifactIntegrity] = []
    authenticated: dict[str, bytes] = {}
    for artifact in inventory.entries:
        errors: list[str] = []
        if artifact.size > maximum_artifact_bytes:
            errors.append("declared_size_exceeds_bound")
        read_bound = min(artifact.size, maximum_artifact_bytes) + 1
        try:
            content = reader.read_bytes(artifact, maximum=read_bound)
            if not isinstance(content, bytes):
                raise TypeError("artifact reader must return bytes")
        except Exception:
            content = None
            errors.append("artifact_read_failed")
        actual_size = len(content) if content is not None else None
        actual_sha256 = (
            hashlib.sha256(content).hexdigest() if content is not None else None
        )
        if content is not None:
            if actual_size > read_bound:
                errors.append("artifact_read_bound_exceeded")
            if actual_size != artifact.size:
                errors.append("artifact_size_mismatch")
            if actual_sha256 != artifact.sha256:
                errors.append("artifact_digest_mismatch")
        valid = not errors
        if valid:
            authenticated[artifact.path] = content
        results.append(
            ArtifactIntegrity(
                artifact=artifact,
                valid=valid,
                actual_size=actual_size,
                actual_sha256=actual_sha256,
                errors=tuple(errors),
            )
        )
    verification = IntegrityVerification(
        valid=all(item.valid for item in results),
        artifacts=tuple(results),
    )
    return verification, _AuthenticatedArtifactReader(authenticated)


@runtime_checkable
class SemanticVerifier(Protocol):
    def verify(
        self,
        *,
        workload: CompiledWorkload,
        inventory: ArtifactInventory,
        reader: ArtifactReader,
    ) -> tuple[SemanticCheck, ...]: ...


class WorkloadBindingVerifier:
    """Base semantic check binding produced artifacts to the exact workload."""

    def __init__(self, *, closure_digest: str, closure_manifest_path: str) -> None:
        if not isinstance(closure_digest, str) or re.fullmatch(r"[0-9a-f]{64}", closure_digest) is None:
            raise ValueError("authoritative closure digest is invalid")
        normalized = _absolute_normalized_path(closure_manifest_path)
        if normalized is None:
            raise ValueError("authoritative closure manifest path is invalid")
        self._closure_digest = closure_digest
        self._closure_manifest_path = normalized

    def verify(
        self,
        *,
        workload: CompiledWorkload,
        inventory: ArtifactInventory,
        reader: ArtifactReader,
    ) -> tuple[SemanticCheck, ...]:
        records = inventory.for_role("workload_record")
        lineage = inventory.for_role("training_lineage")
        models = inventory.for_role("final_model")
        tokenizers = inventory.for_role("tokenizer")
        record_ok = False
        lineage_ok = False
        if len(records) == 1:
            try:
                record_ok = (
                    reader.read_bytes(
                        records[0], maximum=MAX_SEMANTIC_ARTIFACT_BYTES
                    )
                    == workload.canonical_bytes
                )
            except ArtifactReadError:
                record_ok = False
        if len(lineage) == 1:
            try:
                raw = reader.read_bytes(
                    lineage[0], maximum=MAX_SEMANTIC_ARTIFACT_BYTES
                )
                document = _strict_json(raw, require_canonical=True)
            except (
                ArtifactReadError,
                UnicodeError,
                json.JSONDecodeError,
                ValueError,
            ):
                document = None
            lineage_ok = _validate_lineage_document(
                document,
                workload,
                closure_digest=self._closure_digest,
                closure_manifest_path=self._closure_manifest_path,
            )
        model_members: frozenset[str] = frozenset()
        tokenizer_members: frozenset[str] = frozenset()
        model_ok = False
        tokenizer_ok = False
        distinct_ok = False
        try:
            if len(models) == 1:
                model_raw = reader.read_bytes(models[0], maximum=MAX_ARTIFACT_BYTES)
                model_members, model_ok = _validate_sft_archive(
                    model_raw, "model",
                    locked_model_ref=workload.document["configuration"]["document"]["model"]["ref"],
                )
            else:
                model_raw = b""
            if len(tokenizers) == 1:
                tokenizer_raw = reader.read_bytes(
                    tokenizers[0], maximum=MAX_ARTIFACT_BYTES
                )
                tokenizer_members, tokenizer_ok = _validate_sft_archive(
                    tokenizer_raw, "tokenizer"
                )
            else:
                tokenizer_raw = b""
            distinct_ok = (
                model_ok
                and tokenizer_ok
                and model_raw != tokenizer_raw
                and model_members.isdisjoint(tokenizer_members)
            )
        except (ArtifactReadError, ValueError, OSError, tarfile.TarError):
            model_ok = tokenizer_ok = distinct_ok = False
        return (
            SemanticCheck("workload_record_exact", record_ok),
            SemanticCheck("lineage_binds_workload", lineage_ok),
            SemanticCheck("final_model_semantic", model_ok),
            SemanticCheck("tokenizer_semantic", tokenizer_ok),
            SemanticCheck("model_tokenizer_disjoint", distinct_ok),
        )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is prohibited: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is prohibited")
    return parsed


def _unique_pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_type_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_type_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_type_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _strict_json(content: bytes, *, require_canonical: bool = False) -> object:
    if content.startswith(b"\xef\xbb\xbf"):
        raise ValueError("JSON BOM is prohibited")
    value = json.loads(
        content.decode("utf-8", errors="strict"),
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    if require_canonical and _canonical_json(value) != content:
        raise ValueError("JSON is not canonical")
    return value


def _validate_lineage_document(
    document: object,
    workload: CompiledWorkload,
    *,
    closure_digest: str,
    closure_manifest_path: str,
) -> bool:
    try:
        return _validate_lineage_document_unchecked(
            document, workload,
            closure_digest=closure_digest,
            closure_manifest_path=closure_manifest_path,
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
        return False


def _validate_lineage_document_unchecked(
    document: object, workload: CompiledWorkload, *,
    closure_digest: str, closure_manifest_path: str,
) -> bool:
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "workload_fingerprint",
        "execution_source",
        "configuration_revision",
        "identities",
        "trainer_exit_code",
        "execution_evidence",
        "execution_evidence_sha256",
        "trainer_lineage",
    }:
        return False
    evidence = document.get("execution_evidence")
    trainer = document.get("trainer_lineage")
    return (
        document.get("schema_version") == "synaptic-sft-training-lineage/v1"
        and document.get("workload_fingerprint") == workload.fingerprint
        and document.get("execution_source") == workload.document["execution_source"]
        and document.get("configuration_revision")
        == workload.document["configuration"]["revision"]
        and document.get("identities") == workload.document["identities"]
        and type(document.get("trainer_exit_code")) is int
        and document.get("trainer_exit_code") == 0
        and isinstance(evidence, dict)
        and document.get("execution_evidence_sha256")
        == hashlib.sha256(_canonical_json(evidence)).hexdigest()
        and _validate_execution_evidence(
            evidence, workload,
            closure_digest=closure_digest,
            closure_manifest_path=closure_manifest_path,
        )
        and _validate_embedded_trainer_lineage(trainer, workload, evidence)
    )


def _validate_execution_evidence(
    evidence: dict[str, object], workload: CompiledWorkload, *,
    closure_digest: str, closure_manifest_path: str,
) -> bool:
    if set(evidence) != {
        "schema_version", "workload_fingerprint", "configuration_revision",
        "model", "dataset", "sft", "argv", "environment", "cwd", "outputs", "result",
    }:
        return False
    config = workload.document["configuration"]["document"]
    model = config["model"]
    dataset = config["dataset"]
    expected_model = {
        "ref": model["ref"],
        "revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "load_in_4bit": model["load_in_4bit"],
    }
    reported_dataset = evidence.get("dataset")
    if not isinstance(reported_dataset, dict):
        return False
    expected_dataset = {
        "ref": dataset["ref"],
        "resolved_path": reported_dataset.get("resolved_path"),
        "revision": dataset["revision"],
        "content_digest": dataset["content_digest"],
    }
    argv = evidence.get("argv")
    environment = evidence.get("environment")
    outputs = evidence.get("outputs")
    result = evidence.get("result")
    if not (
        evidence.get("schema_version") == _EXECUTION_EVIDENCE_SCHEMA
        and evidence.get("workload_fingerprint") == workload.fingerprint
        and evidence.get("configuration_revision")
        == workload.document["configuration"]["revision"]
        and _json_type_equal(evidence.get("model"), expected_model)
        and _json_type_equal(reported_dataset, expected_dataset)
        and _json_type_equal(evidence.get("sft"), config["sft"])
        and isinstance(argv, list)
        and bool(argv)
        and all(isinstance(item, str) and item for item in argv)
        and isinstance(environment, dict)
        and isinstance(outputs, dict)
        and _json_type_equal(result, {"exit_code": 0, "status": "completed"})
    ):
        return False
    return _validate_evidence_paths_and_argv(
        evidence, workload,
        closure_digest=closure_digest,
        closure_manifest_path=closure_manifest_path,
    )


def _normalized_path(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.replace("\\", "/").rstrip("/")


_WRITABLE_RUNTIME_ROOTS = ("artifacts", "state", "tracking", "cache", "tmp")
_ROOT_ENVIRONMENT_KEYS = {
    "engine": "SYNAPTIC_ENGINE_ROOT",
    "project": "SYNAPTIC_PROJECT_ROOT",
    "artifacts": "SYNAPTIC_ARTIFACT_ROOT",
    "state": "SYNAPTIC_STATE_ROOT",
    "tracking": "SYNAPTIC_TRACKING_ROOT",
    "cache": "SYNAPTIC_CACHE_ROOT",
    "tmp": "SYNAPTIC_TMP_ROOT",
}


def _absolute_normalized_path(value: object) -> str | None:
    normalized = _normalized_path(value)
    if normalized is None or not (
        normalized.startswith("/") or re.fullmatch(r"[A-Za-z]:/.*", normalized)
    ):
        return None
    return normalized


def _join_path(root: str, *parts: str) -> str:
    return "/".join((root.rstrip("/"), *(part.strip("/") for part in parts)))


def _observed_runtime_roots(
    source_runtime: dict[str, object], environment: dict[str, object]
) -> dict[str, str] | None:
    """Bind one provider-observed writable relocation to the locked root layout."""

    locked_value = source_runtime.get("roots")
    capabilities = source_runtime.get("capability_roots")
    if (
        not isinstance(locked_value, dict)
        or set(locked_value) != set(_ROOT_ENVIRONMENT_KEYS)
        or not isinstance(capabilities, dict)
        or set(capabilities) != {"writable"}
    ):
        return None
    locked = {
        name: _absolute_normalized_path(value)
        for name, value in locked_value.items()
    }
    capability = _absolute_normalized_path(capabilities.get("writable"))
    if capability is None or any(value is None for value in locked.values()):
        return None
    locked_boundaries = {
        locked[name].rsplit("/", 1)[0]
        for name in _WRITABLE_RUNTIME_ROOTS
        if locked[name].endswith("/" + name)
    }
    if len(locked_boundaries) != 1:
        return None
    locked_boundary = next(iter(locked_boundaries))
    if not (
        locked_boundary == capability
        or locked_boundary.startswith(capability + "/")
    ) or any(
        locked[name] != _join_path(locked_boundary, name)
        for name in _WRITABLE_RUNTIME_ROOTS
    ):
        return None
    observed = {
        name: _absolute_normalized_path(environment.get(variable))
        for name, variable in _ROOT_ENVIRONMENT_KEYS.items()
    }
    if any(value is None for value in observed.values()):
        return None
    if observed["engine"] != locked["engine"] or observed["project"] != locked["project"]:
        return None
    observed_boundaries = {
        observed[name].rsplit("/", 1)[0]
        for name in _WRITABLE_RUNTIME_ROOTS
        if observed[name].endswith("/" + name)
    }
    if len(observed_boundaries) != 1:
        return None
    observed_boundary = next(iter(observed_boundaries))
    if not observed_boundary or any(
        observed[name] != _join_path(observed_boundary, name)
        for name in _WRITABLE_RUNTIME_ROOTS
    ):
        return None
    return {name: value for name, value in observed.items() if value is not None}


def _python_executable_matches(actual: str, interpreter: dict[str, object]) -> bool:
    planned = _absolute_normalized_path(interpreter.get("executable"))
    observed = _absolute_normalized_path(actual)
    version = interpreter.get("version")
    if planned is None or observed is None or not isinstance(version, str):
        return False
    if observed == planned:
        return True
    match = re.fullmatch(r"(\d+)\.(\d+)\.\d+", version)
    if match is None:
        return False
    parent, separator, filename = planned.rpartition("/")
    if not separator or filename not in {"python", "python3"}:
        return False
    return observed == _join_path(parent, f"python{match.group(1)}.{match.group(2)}")


def _validate_evidence_paths_and_argv(
    evidence: dict[str, object], workload: CompiledWorkload, *,
    closure_digest: str, closure_manifest_path: str,
) -> bool:
    source_runtime = workload.document["execution_source"].get("runtime")
    if not isinstance(source_runtime, dict) or not isinstance(source_runtime.get("roots"), dict):
        return False
    environment = evidence["environment"]
    if not isinstance(environment, dict):
        return False
    normalized = _observed_runtime_roots(source_runtime, environment)
    if normalized is None:
        return False
    config = workload.document["configuration"]["document"]
    relative_dataset = config["dataset"]["ref"].removeprefix("project://")
    dataset_path = f"{normalized['project']}/{relative_dataset}"
    expected_outputs = {
        "run_dir": f"{normalized['state']}/runtime-v1-trainer/output/runtime-v1",
        "final_model_dir": f"{normalized['state']}/runtime-v1-trainer/output/runtime-v1/final_model",
        "tokenizer_dir": f"{normalized['state']}/runtime-v1-trainer/output/runtime-v1/final_model",
        "lineage_path": f"{normalized['state']}/runtime-v1-trainer/output/runtime-v1/training_lineage.json",
    }
    outputs = {
        key: _normalized_path(value)
        for key, value in evidence["outputs"].items()
    }
    if (
        _normalized_path(evidence["dataset"]["resolved_path"]) != dataset_path
        or _normalized_path(evidence.get("cwd")) != normalized["tmp"]
        or outputs != expected_outputs
    ):
        return False
    interpreter = source_runtime.get("interpreter")
    environment_contract = source_runtime.get("environment")
    if not isinstance(interpreter, dict) or set(interpreter) != {
        "implementation", "version", "executable", "executable_digest"
    }:
        return False
    planned_environment = (
        environment_contract.get("variables")
        if isinstance(environment_contract, dict)
        and environment_contract.get("clear_inherited") is True
        else None
    )
    if not isinstance(planned_environment, dict):
        return False
    if (
        planned_environment.get("PYTHONPATH") != source_runtime["roots"]["engine"]
        or {"PYTHONHOME", "PYTHONUSERBASE", "HF_TOKEN"} & set(planned_environment)
    ):
        return False
    executable = _normalized_path(evidence["argv"][0])
    if executable is None or not _python_executable_matches(executable, interpreter):
        return False
    expected_argv = _expected_trainer_argv(
        executable, normalized, dataset_path, config, workload
    )
    if (
        [_normalized_path(item) for item in evidence["argv"]] != [
        _normalized_path(item) for item in expected_argv
        ]
    ):
        return False
    env = environment
    required_env = {
        "SYNAPTIC_ENGINE_ROOT": normalized["engine"],
        "SYNAPTIC_PROJECT_ROOT": normalized["project"],
        "SYNAPTIC_ARTIFACT_ROOT": normalized["artifacts"],
        "SYNAPTIC_STATE_ROOT": normalized["state"],
        "SYNAPTIC_TRACKING_ROOT": normalized["tracking"],
        "SYNAPTIC_CACHE_ROOT": normalized["cache"],
        "SYNAPTIC_TMP_ROOT": normalized["tmp"],
        "SYNAPTIC_WORKLOAD_FINGERPRINT": workload.fingerprint,
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "HF_HOME": f"{normalized['cache']}/huggingface",
        "TRANSFORMERS_CACHE": f"{normalized['cache']}/transformers",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SYNAPTIC_MODEL_SNAPSHOT": (
            f"{normalized['cache']}/model/"
            f"models--{str(config['model']['ref']).replace('/', '--')}/snapshots/"
            f"{config['model']['revision']}"
        ),
        "SYNAPTIC_WORKER_CLOSURE_MANIFEST": closure_manifest_path,
        "SYNAPTIC_WORKER_CLOSURE_DIGEST": closure_digest,
        "WANDB_DISABLED": "true",
    }
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in planned_environment.items()):
        return False
    expected_env = {
        **{key: value for key, value in planned_environment.items() if key != "PYTHONPATH"},
        **required_env,
    }
    return (
        env == expected_env
        and not ({"PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "HF_TOKEN"} & set(env))
    )


def _expected_trainer_argv(
    python: str,
    roots: dict[str, str],
    dataset_path: str,
    config: dict[str, object],
    workload: CompiledWorkload,
) -> list[str]:
    model = config["model"]
    sft = config["sft"]
    argv = [
        python,
        "-I",
        f"{roots['engine']}/tuner/runtime/offline_sft_worker.py",
        "--",
        "--model-name", str(model["ref"]), "--model-revision", str(model["revision"]),
        "--anonymous-model", "--model-cache-dir", f"{roots['cache']}/model",
        "--model-snapshot",
        f"{roots['cache']}/model/models--{str(model['ref']).replace('/', '--')}/snapshots/{model['revision']}",
        "--local-file", dataset_path, "--output-root",
        f"{roots['state']}/runtime-v1-trainer/output", "--run-timestamp", "runtime-v1",
        "--no-dashboard", "--quiet",
        "--runtime-v1-workload-fingerprint", workload.fingerprint,
        "--runtime-v1-configuration-revision", workload.document["configuration"]["revision"],
        "--runtime-v1-tokenizer-revision", str(model["tokenizer_revision"]),
        "--runtime-v1-dataset-revision", str(config["dataset"]["revision"]),
        "--runtime-v1-dataset-digest", str(config["dataset"]["content_digest"]),
    ]
    mappings = (
        ("batch_size", "--batch-size"),
        ("gradient_accumulation_steps", "--gradient-accumulation"),
        ("learning_rate", "--learning-rate"),
        ("max_steps", "--max-steps"), ("num_epochs", "--num-epochs"),
        ("max_seq_length", "--max-seq-length"), ("seed", "--seed"),
        ("save_steps", "--save-steps"), ("save_total_limit", "--save-total-limit"),
        ("lora_rank", "--lora-r"), ("lora_alpha", "--lora-alpha"),
        ("lora_dropout", "--lora-dropout"),
    )
    decimal_keys = {"learning_rate", "lora_dropout"}
    for key, flag in mappings:
        if key in sft:
            value = format(Decimal(str(sft[key])), "f") if key in decimal_keys else str(sft[key])
            argv.extend((flag, value))
    argv.extend(("--lora-target-modules", ",".join(sft["lora_target_modules"])))
    if sft["use_dora"]:
        argv.append("--use-dora")
    if sft["use_rslora"]:
        argv.append("--use-rslora")
    init_lora_weights = sft["init_lora_weights"]
    argv.extend((
        "--init-lora-weights",
        ("true" if init_lora_weights else "false")
        if isinstance(init_lora_weights, bool)
        else init_lora_weights,
    ))
    if sft["split_dataset"]:
        argv.append("--split-dataset")
    argv.append("--load-in-4bit" if model["load_in_4bit"] else "--no-load-in-4bit")
    return argv


def _validate_embedded_trainer_lineage(
    trainer: object, workload: CompiledWorkload, evidence: dict[str, object]
) -> bool:
    if not isinstance(trainer, dict):
        return False
    config = workload.document["configuration"]["document"]
    sft = config["sft"]
    projection = {
        "schema_version": "synaptic-sft-trainer-projection/v1",
        "workload_fingerprint": workload.fingerprint,
        "configuration_revision": workload.document["configuration"]["revision"],
        "model": {
            "ref": config["model"]["ref"], "revision": config["model"]["revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "load_in_4bit": config["model"]["load_in_4bit"],
        },
        "dataset": {
            "resolved_path": evidence["dataset"]["resolved_path"],
            "revision": config["dataset"]["revision"],
            "content_digest": config["dataset"]["content_digest"],
        },
        "training": {
            "batch_size": sft["batch_size"], "gradient_accumulation_steps": sft["gradient_accumulation_steps"],
            "learning_rate": float(sft["learning_rate"]), "max_steps": sft.get("max_steps", -1),
            "num_epochs": sft.get("num_epochs", 1), "max_seq_length": sft["max_seq_length"],
            "seed": sft["seed"], "save_steps": sft["save_steps"],
            "save_total_limit": sft["save_total_limit"], "split_dataset": sft["split_dataset"],
        },
        "lora": {
            "rank": sft["lora_rank"], "alpha": sft["lora_alpha"],
            "dropout": float(sft["lora_dropout"]), "target_modules": sft["lora_target_modules"],
            "use_dora": sft["use_dora"], "use_rslora": sft["use_rslora"],
            "init_lora_weights": sft["init_lora_weights"],
        },
        "outputs": {"run_dir": evidence["outputs"]["run_dir"], "final_model_dir": evidence["outputs"]["final_model_dir"]},
        "status": "completed",
    }
    model = trainer.get("model")
    dataset = trainer.get("dataset")
    training = trainer.get("training")
    lora = trainer.get("lora")
    runtime = trainer.get("runtime")
    legacy_overlaps = (
        _json_type_equal(trainer.get("training_type"), "SFT")
        and _json_type_equal(trainer.get("run_directory"), evidence["outputs"]["run_dir"])
        and isinstance(model, dict)
        and _json_type_equal(model.get("base_model"), config["model"]["ref"])
        and _json_type_equal(model.get("load_in_4bit"), config["model"]["load_in_4bit"])
        and isinstance(dataset, dict)
        and _json_type_equal(dataset.get("source"), evidence["dataset"]["resolved_path"])
        and isinstance(runtime, dict)
        and _json_type_equal(runtime.get("status"), "completed")
        and isinstance(training, dict)
        and _json_type_equal(training.get("batch_size"), sft["batch_size"])
        and _json_type_equal(
            training.get("gradient_accumulation_steps"),
            sft["gradient_accumulation_steps"],
        )
        and _json_type_equal(training.get("learning_rate"), float(sft["learning_rate"]))
        and _json_type_equal(training.get("max_steps"), sft.get("max_steps", -1))
        and _json_type_equal(training.get("max_seq_length"), sft["max_seq_length"])
        and _json_type_equal(training.get("seed"), sft["seed"])
        and isinstance(lora, dict)
        and _json_type_equal(lora.get("rank"), sft["lora_rank"])
        and _json_type_equal(lora.get("alpha"), sft["lora_alpha"])
        and _json_type_equal(lora.get("dropout"), float(sft["lora_dropout"]))
        and _json_type_equal(lora.get("target_modules"), sft["lora_target_modules"])
    )
    return legacy_overlaps and _json_type_equal(
        trainer.get("synaptic_runtime_projection"), projection
    )


def _validate_sft_archive(
    content: bytes, artifact_kind: str, *, locked_model_ref: str | None = None
) -> tuple[frozenset[str], bool]:
    if not content:
        return frozenset(), False
    names: set[str] = set()
    configs: set[str] = set()
    payloads: set[str] = set()
    indexes: dict[str, dict[str, object]] = {}
    tensor_info: dict[str, tuple[frozenset[str], int]] = {}
    expanded = 0
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_ARCHIVE_MEMBERS:
            return frozenset(), False
        for member in members:
            name = member.name
            path = PurePosixPath(name)
            if (
                not member.isfile()
                or not name
                or "\\" in name
                or path.is_absolute()
                or len(path.parts) != 1
                or any(part in {"", ".", ".."} for part in path.parts)
                or path.as_posix() != name
                or name in names
                or not 0 < member.size <= MAX_ARCHIVE_MEMBER_BYTES
            ):
                return frozenset(), False
            expanded += member.size
            if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
                return frozenset(), False
            extracted = archive.extractfile(member)
            if extracted is None:
                return frozenset(), False
            names.add(name)
            if artifact_kind == "model":
                is_index = name in {"adapter_model.safetensors.index.json", "model.safetensors.index.json"}
                if name not in _MODEL_CONFIGS and name not in _MODEL_OPTIONAL and not is_index and not _MODEL_PAYLOAD.fullmatch(name):
                    return frozenset(), False
                if is_index:
                    raw_index = extracted.read(min(member.size, _MAX_INDEX_BYTES) + 1)
                    if len(raw_index) != member.size:
                        return frozenset(), False
                    parsed_index = _strict_json(raw_index)
                    if not isinstance(parsed_index, dict):
                        return frozenset(), False
                    indexes[name] = parsed_index
                if name in _MODEL_CONFIGS:
                    content_value = extracted.read(min(member.size, 4 * 1024 * 1024) + 1)
                    if len(content_value) != member.size or not _valid_model_config(
                        name, content_value, locked_model_ref=locked_model_ref
                    ):
                        return frozenset(), False
                    configs.add(name)
                if _MODEL_PAYLOAD.fullmatch(name):
                    valid_payload, contained_tensors, tensor_bytes = _valid_safetensors_stream(extracted, member.size)
                    if not valid_payload:
                        return frozenset(), False
                    payloads.add(name)
                    tensor_info[name] = (contained_tensors, tensor_bytes)
                if name == "generation_config.json":
                    raw = extracted.read(min(member.size, 4 * 1024 * 1024) + 1)
                    if len(raw) != member.size or not _valid_bounded_json_object(raw):
                        return frozenset(), False
                if name == "README.md":
                    raw = extracted.read(min(member.size, 4 * 1024 * 1024) + 1)
                    if len(raw) != member.size or not _valid_nonempty_utf8(raw):
                        return frozenset(), False
            elif artifact_kind == "tokenizer":
                if name not in (
                    _TOKENIZER_CONFIGS | _TOKENIZER_PAYLOADS | _TOKENIZER_OPTIONAL
                ):
                    return frozenset(), False
                if name in _TOKENIZER_CONFIGS:
                    content_value = extracted.read(min(member.size, 4 * 1024 * 1024) + 1)
                    if len(content_value) != member.size or not _valid_tokenizer_config(
                        content_value
                    ):
                        return frozenset(), False
                    configs.add(name)
                if name in _TOKENIZER_PAYLOADS:
                    content_value = extracted.read(min(member.size, 512 * 1024 * 1024) + 1)
                    if len(content_value) != member.size or not _valid_tokenizer_json(
                        content_value
                    ):
                        return frozenset(), False
                    payloads.add(name)
                if name in _TOKENIZER_OPTIONAL:
                    content_value = extracted.read(min(member.size, 512 * 1024 * 1024) + 1)
                    if len(content_value) != member.size or not _valid_tokenizer_sidecar(name, content_value):
                        return frozenset(), False
            else:
                return frozenset(), False
    if artifact_kind == "model":
        adapter = "adapter_config.json" in configs
        full = "config.json" in configs
        if adapter == full:
            return frozenset(), False
        family = "adapter_model" if adapter else "model"
        matches = [_MODEL_PAYLOAD.fullmatch(name) for name in payloads]
        if any(match is None or match.group(1) != family for match in matches):
            return frozenset(), False
        sharded = [match for match in matches if match.group(2) is not None]
        index_name = f"{family}.safetensors.index.json"
        if sharded:
            totals = {int(match.group(3)) for match in sharded}
            numbers = {int(match.group(2)) for match in sharded}
            if len(totals) != 1:
                return frozenset(), False
            total = totals.pop()
            if (
                not 1 <= total <= _MAX_SHARDS
                or numbers != set(range(1, total + 1))
                or len(payloads) != total
                or set(indexes) != {index_name}
            ):
                return frozenset(), False
            index = indexes.get(index_name)
            if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
                return frozenset(), False
            weight_map = index.get("weight_map")
            metadata = index.get("metadata")
            if not isinstance(metadata, dict) or set(metadata) - {"total_size"} or (
                "total_size" in metadata and (
                    not isinstance(metadata["total_size"], int)
                    or isinstance(metadata["total_size"], bool)
                    or metadata["total_size"] != sum(info[1] for info in tensor_info.values())
                )
            ):
                return frozenset(), False
            if not isinstance(weight_map, dict) or not 0 < len(weight_map) <= _MAX_TENSORS or any(
                not isinstance(tensor, str) or not tensor or shard not in payloads
                for tensor, shard in weight_map.items()
            ) or set(weight_map.values()) != payloads:
                return frozenset(), False
            described = {shard: set() for shard in payloads}
            for tensor, shard in weight_map.items():
                described[shard].add(tensor)
            if any(described[shard] != set(tensor_info[shard][0]) for shard in payloads):
                return frozenset(), False
        elif payloads != {f"{family}.safetensors"} or indexes:
            return frozenset(), False
    return frozenset(names), bool(configs and payloads)


def _valid_model_config(name: str, content: bytes, *, locked_model_ref: str | None) -> bool:
    try:
        document = _strict_json(content)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    if name == "adapter_config.json":
        return (
            document.get("peft_type") == "LORA"
            and isinstance(document.get("base_model_name_or_path"), str)
            and document["base_model_name_or_path"] == locked_model_ref
        )
    return isinstance(document.get("model_type"), str) and bool(document["model_type"])


def _valid_tokenizer_config(content: bytes) -> bool:
    try:
        document = _strict_json(content)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(document, dict)
        and isinstance(document.get("tokenizer_class"), str)
        and bool(document["tokenizer_class"])
    )


def _valid_tokenizer_json(content: bytes) -> bool:
    try:
        document = _strict_json(content)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    model = document.get("model")
    return (
        isinstance(document.get("version"), str)
        and bool(document["version"])
        and isinstance(model, dict)
        and isinstance(model.get("type"), str)
        and bool(model["type"])
        and isinstance(model.get("vocab"), (dict, list))
        and bool(model["vocab"])
    )


def _valid_nonempty_utf8(content: bytes) -> bool:
    try:
        return bool(content.decode("utf-8").strip())
    except UnicodeDecodeError:
        return False


def _valid_json_tree(value: object, *, depth: int = 0) -> bool:
    if depth > 12:
        return False
    if value is None or isinstance(value, (str, bool, int)):
        return not isinstance(value, str) or len(value) <= 1_000_000
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return len(value) <= 100_000 and all(_valid_json_tree(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 100_000 and all(
            isinstance(key, str) and len(key) <= 4096 and _valid_json_tree(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def _valid_bounded_json_object(content: bytes) -> bool:
    try:
        document = _strict_json(content)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return False
    return isinstance(document, dict) and _valid_json_tree(document)


def _valid_tokenizer_sidecar(name: str, content: bytes) -> bool:
    if name in {"vocab.json", "added_tokens.json"}:
        try:
            document = _strict_json(content)
        except (UnicodeError, json.JSONDecodeError, ValueError):
            return False
        return (
            isinstance(document, dict)
            and bool(document)
            and len(document) <= 2_000_000
            and all(
                isinstance(token, str) and bool(token)
                and isinstance(index, int) and not isinstance(index, bool) and index >= 0
                for token, index in document.items()
            )
        )
    if name == "special_tokens_map.json":
        return _valid_bounded_json_object(content)
    if name == "merges.txt":
        if not _valid_nonempty_utf8(content):
            return False
        lines = [line for line in content.decode("utf-8").splitlines() if line and not line.startswith("#")]
        return bool(lines) and all(len(line.split()) == 2 for line in lines)
    if name == "chat_template.jinja":
        return _valid_nonempty_utf8(content)
    return False


def _valid_safetensors_stream(stream: object, total_size: int) -> tuple[bool, frozenset[str], int]:
    try:
        prefix = stream.read(8)
        if len(prefix) != 8:
            return False, frozenset(), 0
        header_size = int.from_bytes(prefix, "little", signed=False)
        if not 0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
            return False, frozenset(), 0
        header = stream.read(header_size)
        if len(header) != header_size:
            return False, frozenset(), 0
        document = _strict_json(header)
        data_read = 0
        has_nonzero_data = False
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            data_read += len(chunk)
            has_nonzero_data = has_nonzero_data or chunk.count(0) != len(chunk)
    except (UnicodeError, json.JSONDecodeError, ValueError, OSError):
        return False, frozenset(), 0
    data_size = total_size - 8 - header_size
    valid = (
        isinstance(document, dict)
        and data_read == data_size
        and has_nonzero_data
        and _valid_safetensors_index(document, data_size)
    )
    tensors = frozenset(name for name in document if name != "__metadata__") if isinstance(document, dict) else frozenset()
    return valid, tensors, data_size


def _valid_safetensors_index(document: dict[str, object], data_size: int) -> bool:
    metadata = document.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())
    ):
        return False
    intervals: list[tuple[int, int]] = []
    for name, descriptor in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            return False
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or dtype not in _SAFETENSORS_DTYPES
            or not isinstance(shape, list)
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in offsets)
        ):
            return False
        start, end = offsets
        if end <= start or end - start != math.prod(shape) * _SAFETENSORS_DTYPES[dtype]:
            return False
        intervals.append((start, end))
    if not intervals:
        return False
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start != cursor:
            return False
        cursor = end
    return cursor == data_size


@dataclass(frozen=True, slots=True)
class VerificationReport:
    status: VerificationStatus
    provider_completed: bool
    process_succeeded: bool
    inventory: InventoryVerification
    integrity: IntegrityVerification
    semantic_checks: tuple[SemanticCheck, ...]

    @property
    def success(self) -> bool:
        return self.status is VerificationStatus.VERIFIED


class VerificationService:
    def __init__(self, verifier: SemanticVerifier) -> None:
        if not isinstance(verifier, SemanticVerifier):
            raise TypeError("verifier must implement SemanticVerifier")
        self._verifier = verifier

    def verify(
        self,
        *,
        provider_completed: bool,
        process: ProcessResult,
        workload: CompiledWorkload,
        contract: ArtifactContract,
        inventory: ArtifactInventory,
        reader: ArtifactReader,
    ) -> VerificationReport:
        if not isinstance(provider_completed, bool):
            raise TypeError("provider_completed must be a boolean")
        if not isinstance(process, ProcessResult):
            raise TypeError("process must be a ProcessResult")
        inventory_result = verify_inventory(contract, inventory)
        integrity_result, authenticated_reader = authenticate_artifacts(
            inventory, reader
        )
        semantic = (
            self._verifier.verify(
                workload=workload,
                inventory=inventory,
                reader=authenticated_reader,
            )
            if integrity_result.valid
            else ()
        )
        if any(not isinstance(item, SemanticCheck) for item in semantic):
            raise TypeError("semantic verifier returned an invalid check")
        process_succeeded = process.exit_code == 0
        if not provider_completed:
            status = VerificationStatus.INCONCLUSIVE
        elif (
            not process_succeeded
            or not inventory_result.valid
            or not integrity_result.valid
            or any(not item.passed for item in semantic)
        ):
            status = VerificationStatus.INVALID
        elif not semantic:
            status = VerificationStatus.INCONCLUSIVE
        else:
            status = VerificationStatus.VERIFIED
        return VerificationReport(
            status=status,
            provider_completed=provider_completed,
            process_succeeded=process_succeeded,
            inventory=inventory_result,
            integrity=integrity_result,
            semantic_checks=tuple(semantic),
        )


__all__ = [
    "ArtifactReadError",
    "ArtifactReadLimitExceeded",
    "ArtifactReader",
    "MAX_ARTIFACT_BYTES",
    "MAX_SEMANTIC_ARTIFACT_BYTES",
    "SemanticCheck",
    "SemanticVerifier",
    "VerificationReport",
    "VerificationService",
    "VerificationStatus",
    "WorkloadBindingVerifier",
    "authenticate_artifacts",
]
