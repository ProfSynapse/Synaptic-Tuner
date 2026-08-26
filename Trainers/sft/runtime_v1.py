"""Concrete provider-neutral process entrypoint for canonical SFT workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Mapping, Protocol, runtime_checkable


MAX_WORKLOAD_BYTES = 256 * 1024
MAX_LINEAGE_BYTES = 4 * 1024 * 1024
EXECUTION_SOURCE_SCHEMA = "synaptic-execution-source/v1"
RUNTIME_SCHEMA = "synaptic-modal-runtime/v1"
_ENTRYPOINT = PurePosixPath("Trainers/sft/runtime_v1.py")
_ROOT_ENV = {
    "engine": "SYNAPTIC_ENGINE_ROOT",
    "project": "SYNAPTIC_PROJECT_ROOT",
    "artifacts": "SYNAPTIC_ARTIFACT_ROOT",
    "state": "SYNAPTIC_STATE_ROOT",
    "tracking": "SYNAPTIC_TRACKING_ROOT",
    "cache": "SYNAPTIC_CACHE_ROOT",
    "tmp": "SYNAPTIC_TMP_ROOT",
}
_WRITABLE_NAMES = ("artifacts", "state", "tracking", "cache", "tmp")
_INHERITED_ENV = (
    "PATH",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTHONIOENCODING",
    "LANG",
    "LC_ALL",
)
_MODEL_CONFIGS = frozenset({"adapter_config.json", "config.json"})
_MODEL_PAYLOADS = re.compile(r"^(adapter_model|model)(?:-(\d{5})-of-(\d{5}))?\.safetensors$")
_TOKENIZER_CONFIGS = frozenset({"tokenizer_config.json"})
_TOKENIZER_PAYLOADS = frozenset({"tokenizer.json"})
_TOKENIZER_OPTIONAL = frozenset({
    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
    "merges.txt", "vocab.json",
})
_MODEL_OPTIONAL = frozenset({"generation_config.json", "README.md"})
_KNOWN_IGNORED = frozenset({"training_args.bin"})
_MAX_INDEX_BYTES = 16 * 1024 * 1024
_MAX_SHARDS = 1024
_MAX_TENSORS = 1_000_000
_MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_SAFETENSORS_DTYPES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_EXECUTION_EVIDENCE_SCHEMA = "synaptic-sft-execution-evidence/v1"
_SFT_KEYS = {
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "max_steps",
    "num_epochs",
    "max_seq_length",
    "seed",
    "save_steps",
    "save_total_limit",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "use_dora",
    "use_rslora",
    "init_lora_weights",
    "split_dataset",
}
_REQUIRED_SFT_KEYS = _SFT_KEYS - {"max_steps", "num_epochs"}


class RuntimeV1Error(RuntimeError):
    """Closed runtime contract failure."""


class TrainerFailed(RuntimeV1Error):
    pass


def _json_type_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _json_type_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_type_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _version_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9]\d*(?:\.(?:0|[1-9]\d*)){1,3}", value) is None:
        raise RuntimeV1Error(f"{label} is not a strict runtime version")
    return tuple(int(part) for part in value.split("."))


def _validate_portable_runtime_requirements(requirements: object) -> None:
    if not isinstance(requirements, Mapping) or set(requirements) != {
        "schema_version", "python", "isolation", "allowed_environment",
        "trainer_projection_schema", "artifact_formats",
    }:
        raise RuntimeV1Error("portable runtime requirements are malformed")
    python = requirements["python"]
    if not isinstance(python, Mapping) or set(python) != {
        "implementation", "minimum_version", "maximum_version_exclusive"
    }:
        raise RuntimeV1Error("portable Python requirements are malformed")
    if python["implementation"] != sys.implementation.name:
        raise RuntimeV1Error("runtime Python implementation is unsupported")
    minimum = _version_tuple(python["minimum_version"], "minimum Python version")
    maximum = _version_tuple(python["maximum_version_exclusive"], "maximum Python version")
    current = tuple(sys.version_info[: max(len(minimum), len(maximum))])
    minimum_cmp = minimum + (0,) * (len(current) - len(minimum))
    maximum_cmp = maximum + (0,) * (len(current) - len(maximum))
    if not minimum_cmp <= current < maximum_cmp:
        raise RuntimeV1Error("runtime Python version is outside the portable requirement")
    if not _json_type_equal(
        requirements["isolation"], {"no_user_site": True, "safe_path": True}
    ):
        raise RuntimeV1Error("portable runtime isolation requirements are malformed")
    allowed = requirements["allowed_environment"]
    formats = requirements["artifact_formats"]
    if (
        not isinstance(allowed, list)
        or len(allowed) != len(set(allowed))
        or any(not isinstance(item, str) or not item for item in allowed)
        or requirements["trainer_projection_schema"] != "synaptic-sft-trainer-projection/v1"
        or not isinstance(formats, Mapping)
        or set(formats) != {"model", "tokenizer"}
        or formats["model"] != ["peft-safetensors", "full-safetensors"]
        or formats["tokenizer"] != "tokenizer-json"
    ):
        raise RuntimeV1Error("portable runtime requirements are unsupported")


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


def read_bounded_workload(stream: BinaryIO) -> bytes:
    payload = stream.read(MAX_WORKLOAD_BYTES + 1)
    if not isinstance(payload, bytes):
        raise RuntimeV1Error("workload stdin must be a binary stream")
    if not payload or len(payload) > MAX_WORKLOAD_BYTES:
        raise RuntimeV1Error("workload stdin is empty or exceeds its byte bound")
    if stream.read(1):
        raise RuntimeV1Error("workload stdin exceeds its byte bound")
    return payload


def _canonical_document(payload: bytes) -> dict[str, object]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise RuntimeV1Error("workload must not contain a BOM")
    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeV1Error("workload is not strict JSON") from exc
    if not isinstance(document, dict):
        raise RuntimeV1Error("workload root must be an object")
    try:
        canonical = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeV1Error("workload cannot be canonically encoded") from exc
    if canonical != payload:
        raise RuntimeV1Error("workload is not canonically encoded")
    return document


def _ensure_engine_import(engine_root: Path) -> None:
    value = str(engine_root)
    if value not in sys.path:
        sys.path.insert(0, value)


def _validate_schema(document: Mapping[str, object], engine_root: Path) -> None:
    try:
        from jsonschema.validators import validator_for
        from referencing import Registry, Resource
    except ImportError as exc:
        raise RuntimeV1Error("runtime schema dependencies are unavailable") from exc
    schema_path = engine_root / "schemas" / "synaptic-sft-workload-v1.schema.json"
    source_path = engine_root / "schemas" / "synaptic-execution-source-v1.schema.json"
    schema = _read_json_file(schema_path, maximum=128 * 1024)
    source_schema = _read_json_file(source_path, maximum=128 * 1024)
    schema_id = source_schema.get("$id")
    if not isinstance(schema_id, str):
        raise RuntimeV1Error("execution-source schema identity is invalid")
    validator_type = validator_for(schema)
    validator_type.check_schema(schema)
    registry = Registry().with_resource(schema_id, Resource.from_contents(source_schema))
    errors = tuple(validator_type(schema, registry=registry).iter_errors(document))
    if errors:
        raise RuntimeV1Error("workload failed the SFT v1 schema")


def _read_json_file(
    path: Path, *, maximum: int, require_canonical: bool = False
) -> dict[str, object]:
    content = _read_regular(path, maximum=maximum)
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeV1Error("runtime JSON artifact is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeV1Error("runtime JSON artifact must be an object")
    if require_canonical and _canonical_json(value) != content:
        raise RuntimeV1Error("engine JSON record is not canonical")
    return value


def _strict_json_bytes(content: bytes, *, label: str) -> object:
    if content.startswith(b"\xef\xbb\xbf"):
        raise RuntimeV1Error(f"{label} must not contain a BOM")
    try:
        return json.loads(
            content.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeV1Error(f"{label} is not strict JSON") from exc


def _read_regular(path: Path, *, maximum: int) -> bytes:
    _assert_no_redirected_components(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeV1Error("required runtime file is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size > maximum:
        raise RuntimeV1Error("runtime file must be bounded, regular, and link-free")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeV1Error("runtime file could not be read") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise RuntimeV1Error("runtime file must be bounded and regular")
            content = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        current = path.lstat()
    except OSError as exc:
        raise RuntimeV1Error("runtime file could not be read") from exc
    if len(content) > maximum:
        raise RuntimeV1Error("runtime file exceeds its byte bound")
    if (
        _file_identity(before) != _file_identity(after)
        or _stable_path_identity(after) != _stable_path_identity(current)
    ):
        raise RuntimeV1Error("runtime file changed while it was read")
    return content


@dataclass(frozen=True, slots=True)
class RuntimeRoots:
    engine: Path
    project: Path
    artifacts: Path
    state: Path
    tracking: Path
    cache: Path
    tmp: Path

    @property
    def writable(self) -> tuple[Path, ...]:
        return tuple(getattr(self, name) for name in _WRITABLE_NAMES)


def bind_runtime_roots(
    document: Mapping[str, object],
    environment: Mapping[str, str],
    *,
    engine_file: Path,
) -> RuntimeRoots:
    execution_source = document.get("execution_source")
    if not isinstance(execution_source, Mapping):
        raise RuntimeV1Error("workload execution source is missing")
    runtime = execution_source.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("schema_version") != RUNTIME_SCHEMA:
        raise RuntimeV1Error("execution source lacks the runtime contract")
    expected = runtime.get("roots")
    if not isinstance(expected, Mapping) or set(expected) != set(_ROOT_ENV):
        raise RuntimeV1Error("runtime-roots contract is incomplete")
    roots: dict[str, Path] = {}
    for name, variable in _ROOT_ENV.items():
        raw = environment.get(variable)
        locked = expected.get(name)
        if not isinstance(raw, str) or not raw or raw != locked:
            raise RuntimeV1Error("environment root does not match the locked runtime root")
        path = Path(raw)
        if not path.is_absolute():
            raise RuntimeV1Error("runtime roots must be absolute")
        _assert_no_redirected_components(path)
        if not path.exists() or not path.is_dir() or _is_redirect(path):
            raise RuntimeV1Error("runtime roots must be existing link-free directories")
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise RuntimeV1Error("runtime root contains an unresolved alias")
        roots[name] = resolved
    result = RuntimeRoots(**roots)
    _validate_root_topology(result, execution_source, engine_file=engine_file)
    return result


def _is_redirect(path: Path) -> bool:
    if path.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(path)):
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _assert_no_redirected_components(path: Path) -> None:
    """Reject symlink, junction, or reparse redirection in every existing component."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeV1Error("runtime path component is unavailable") from exc
        if _is_redirect(current):
            raise RuntimeV1Error("runtime path traverses a redirected component")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_path_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    # Windows reports a different ctime for an open handle and a path stat even
    # when both identify the same unchanged file.
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _require_contained_regular(path: Path, root: Path, *, label: str) -> Path:
    _assert_no_redirected_components(path)
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise RuntimeV1Error(f"{label} is unavailable") from exc
    if resolved != root and root not in resolved.parents:
        raise RuntimeV1Error(f"{label} escapes its locked root")
    if resolved != path or not stat.S_ISREG(info.st_mode) or _is_redirect(path):
        raise RuntimeV1Error(f"{label} must be a regular link-free file")
    return resolved


def _validate_root_topology(
    roots: RuntimeRoots,
    execution_source: Mapping[str, object],
    *,
    engine_file: Path,
) -> None:
    expected_entrypoint = roots.engine / Path(*_ENTRYPOINT.parts)
    actual_entrypoint = _require_contained_regular(
        engine_file, roots.engine, label="runtime entrypoint"
    )
    expected_entrypoint = _require_contained_regular(
        expected_entrypoint, roots.engine, label="runtime entrypoint"
    )
    if actual_entrypoint != expected_entrypoint:
        raise RuntimeV1Error("runtime entrypoint is outside the locked engine root")
    topology = execution_source.get("topology")
    sources = execution_source.get("sources")
    if not isinstance(sources, Mapping) or not isinstance(sources.get("engine"), Mapping):
        raise RuntimeV1Error("source topology is incomplete")
    if not isinstance(topology, Mapping) or topology.get("execution_mode") != "dual_clone":
        raise RuntimeV1Error("runtime v1 requires the finalized dual-clone topology")
    if roots.engine == roots.project:
        raise RuntimeV1Error("dual-clone roots must be distinct")
    # Engine/project relationships are governed by the locked source topology
    # above.  In particular, a superproject intentionally contains its engine
    # submodule.  Writable capabilities, however, must remain disjoint from
    # both source roots and from one another.
    for writable in roots.writable:
        for source in (roots.engine, roots.project):
            if (
                writable == source
                or writable in source.parents
                or source in writable.parents
            ):
                raise RuntimeV1Error("runtime roots overlap")
    for index, writable in enumerate(roots.writable):
        for other in roots.writable[index + 1 :]:
            if (
                writable == other
                or writable in other.parents
                or other in writable.parents
            ):
                raise RuntimeV1Error("runtime roots overlap")


def _resolve_relative(root: Path, value: str, *, require_file: bool) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        raise RuntimeV1Error("project path reference is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeV1Error("project path reference escapes its root")
    candidate = root.joinpath(*relative.parts)
    _assert_no_redirected_components(root)
    current = root
    for part in relative.parts:
        current = current / part
        if _is_redirect(current):
            raise RuntimeV1Error("project path reference traverses a redirect")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeV1Error("project path reference does not exist") from exc
    if root not in resolved.parents:
        raise RuntimeV1Error("project path reference escapes its root")
    if require_file:
        _require_contained_regular(candidate, root, label="project input")
    return resolved


def decode_and_validate_workload(
    payload: bytes,
    environment: Mapping[str, str],
    *,
    engine_file: Path,
) -> tuple[object, RuntimeRoots]:
    document = _canonical_document(payload)
    provisional_engine = environment.get("SYNAPTIC_ENGINE_ROOT", "")
    if not provisional_engine or not Path(provisional_engine).is_absolute():
        raise RuntimeV1Error("engine root is unavailable")
    provisional_path = Path(provisional_engine)
    _assert_no_redirected_components(provisional_path)
    try:
        engine_root = provisional_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeV1Error("engine root is unavailable") from exc
    expected_entrypoint = engine_root / Path(*_ENTRYPOINT.parts)
    if _require_contained_regular(
        engine_file, engine_root, label="runtime entrypoint"
    ) != _require_contained_regular(
        expected_entrypoint, engine_root, label="runtime entrypoint"
    ):
        raise RuntimeV1Error("engine root does not own this runtime entrypoint")
    _ensure_engine_import(engine_root)
    _validate_schema(document, engine_root)
    from synaptic_tuner.api.v1.training import CanonicalDocument
    from tuner.project.execution_source import ExecutionSourceV1
    from tuner.training.methods.sft import compile_sft_workload

    configuration = document.get("configuration")
    if not isinstance(configuration, Mapping) or not isinstance(
        configuration.get("document"), Mapping
    ):
        raise RuntimeV1Error("workload configuration is invalid")
    try:
        resolved_config = CanonicalDocument.from_mapping(configuration["document"])
        revision = hashlib.sha256(
            resolved_config.canonical_json.encode("utf-8")
        ).hexdigest()
        if configuration.get("revision") != revision:
            raise RuntimeV1Error("resolved configuration revision does not match")
        execution_source = ExecutionSourceV1.from_dict(document["execution_source"])
        workload = compile_sft_workload(
            resolved_config=resolved_config,
            execution_source=execution_source,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeV1Error("workload could not be reconstructed") from exc
    if workload.canonical_bytes != payload:
        raise RuntimeV1Error("workload bytes do not match deterministic compilation")
    expected_fingerprint = environment.get("SYNAPTIC_WORKLOAD_FINGERPRINT")
    if expected_fingerprint != workload.fingerprint:
        raise RuntimeV1Error("workload fingerprint does not match the dispatcher binding")
    roots = bind_runtime_roots(document, environment, engine_file=engine_file)
    return workload, roots


@dataclass(frozen=True, slots=True)
class TrainerInvocation:
    argv: tuple[str, ...]
    cwd: Path
    environment: tuple[tuple[str, str], ...]
    run_dir: Path
    final_model_dir: Path
    tokenizer_dir: Path
    lineage_path: Path
    projection_path: Path
    expected_projection: Mapping[str, object]
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True, slots=True)
class TrainerEvidence:
    exit_code: int
    final_model_dir: Path
    tokenizer_dir: Path
    lineage: Mapping[str, object]
    projection: Mapping[str, object]
    metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.exit_code) is not int:
            raise TypeError("trainer exit code must be an exact integer")


@runtime_checkable
class TrainerRunner(Protocol):
    def run(self, invocation: TrainerInvocation) -> TrainerEvidence: ...


def build_trainer_invocation(
    workload: object,
    roots: RuntimeRoots,
    environment: Mapping[str, str],
) -> TrainerInvocation:
    document = workload.document
    config = document["configuration"]["document"]
    model = config["model"]
    dataset = config["dataset"]
    sft = config["sft"]
    if not all(isinstance(value, Mapping) for value in (model, dataset, sft)):
        raise RuntimeV1Error("resolved SFT configuration is malformed")
    unknown = set(sft) - _SFT_KEYS
    if unknown:
        raise RuntimeV1Error("resolved SFT configuration contains unsupported keys")
    missing = _REQUIRED_SFT_KEYS - set(sft)
    if missing:
        raise RuntimeV1Error("resolved SFT configuration is not fully specified")
    model_revision = model.get("revision")
    if model.get("tokenizer_revision") != model_revision:
        raise RuntimeV1Error("runtime v1 requires one exact model/tokenizer snapshot")
    dataset_ref = dataset.get("ref")
    if not isinstance(dataset_ref, str) or not dataset_ref.startswith("project://"):
        raise RuntimeV1Error("runtime v1 requires a locked project-local dataset")
    dataset_path = _resolve_relative(
        roots.project, dataset_ref.removeprefix("project://"), require_file=True
    )
    sources = document["execution_source"]["sources"]
    project_revision = sources["project"]["commit"]
    if dataset.get("revision") != project_revision:
        raise RuntimeV1Error("project dataset revision must match the locked project commit")
    content_digest = dataset.get("content_digest")
    if not isinstance(content_digest, str) or hashlib.sha256(
        _read_regular(dataset_path, maximum=8 * 1024 * 1024 * 1024)
    ).hexdigest() != content_digest:
        raise RuntimeV1Error("project dataset content digest does not match")

    trainer_root = roots.state / "runtime-v1-trainer"
    if trainer_root.exists():
        raise RuntimeV1Error("trainer state path already exists")
    output_root = trainer_root / "output"
    run_dir = output_root / "runtime-v1"
    final_model_dir = run_dir / "final_model"
    trainer_path = _require_contained_regular(
        roots.engine / "Trainers" / "sft" / "train_sft.py",
        roots.engine,
        label="SFT trainer",
    )
    requirements = document.get("runtime_requirements")
    _validate_portable_runtime_requirements(requirements)
    runtime_lock = document["execution_source"].get("runtime")
    interpreter = runtime_lock.get("interpreter") if isinstance(runtime_lock, Mapping) else None
    execution_environment = runtime_lock.get("environment") if isinstance(runtime_lock, Mapping) else None
    if not isinstance(interpreter, Mapping) or set(interpreter) != {
        "implementation", "version", "executable", "executable_digest"
    }:
        raise RuntimeV1Error("execution interpreter is missing or malformed")
    if interpreter["implementation"] != sys.implementation.name or interpreter["version"] != ".".join(
        str(part) for part in sys.version_info[:3]
    ):
        raise RuntimeV1Error("execution interpreter identity does not match this runtime")
    python_executable = interpreter["executable"]
    if not isinstance(python_executable, str) or Path(python_executable).resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise RuntimeV1Error("resolved runtime interpreter does not match this runtime")
    planned_environment = (
        execution_environment.get("variables")
        if isinstance(execution_environment, Mapping)
        and execution_environment.get("clear_inherited") is True
        else None
    )
    allowed_environment = requirements.get("allowed_environment")
    if (
        not isinstance(planned_environment, Mapping)
        or not isinstance(allowed_environment, list)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in planned_environment.items())
        or not set(planned_environment).issubset(set(allowed_environment))
    ):
        raise RuntimeV1Error("resolved runtime environment violates portable requirements")
    argv = [
        str(Path(python_executable).resolve()),
        str(trainer_path),
        "--model-name",
        str(model["ref"]),
        "--model-revision",
        str(model_revision),
        "--anonymous-model",
        "--model-cache-dir",
        str(roots.cache / "model"),
        "--local-file",
        str(dataset_path),
        "--output-root",
        str(output_root),
        "--run-timestamp",
        "runtime-v1",
        "--no-dashboard",
        "--quiet",
        "--runtime-v1-workload-fingerprint",
        workload.fingerprint,
        "--runtime-v1-configuration-revision",
        str(document["configuration"]["revision"]),
        "--runtime-v1-tokenizer-revision",
        str(model["tokenizer_revision"]),
        "--runtime-v1-dataset-revision",
        str(dataset["revision"]),
        "--runtime-v1-dataset-digest",
        str(content_digest),
    ]
    _append_sft_arguments(argv, sft, model)
    child_env = dict(planned_environment)
    child_env.update(
        {
            _ROOT_ENV[name]: str(getattr(roots, name)) for name in _ROOT_ENV
        }
    )
    child_env.update(
        {
            "SYNAPTIC_WORKLOAD_FINGERPRINT": workload.fingerprint,
            "PYTHONPATH": str(roots.engine),
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "HF_HOME": str(roots.cache / "huggingface"),
            "TRANSFORMERS_CACHE": str(roots.cache / "transformers"),
            "WANDB_DISABLED": "true",
        }
    )
    expected_projection = _expected_trainer_projection(
        workload, dataset_path=dataset_path, run_dir=run_dir,
        final_model_dir=final_model_dir,
    )
    return TrainerInvocation(
        argv=tuple(argv),
        cwd=roots.tmp,
        environment=tuple(sorted(child_env.items())),
        run_dir=run_dir,
        final_model_dir=final_model_dir,
        tokenizer_dir=final_model_dir,
        lineage_path=run_dir / "training_lineage.json",
        projection_path=run_dir / "runtime_v1_projection.json",
        expected_projection=expected_projection,
        stdout_path=roots.tracking / "trainer.stdout.log",
        stderr_path=roots.tracking / "trainer.stderr.log",
    )


def _append_sft_arguments(
    argv: list[str], sft: Mapping[str, object], model: Mapping[str, object]
) -> None:
    mappings = (
        ("batch_size", "--batch-size", _positive_int),
        ("gradient_accumulation_steps", "--gradient-accumulation", _positive_int),
        ("learning_rate", "--learning-rate", _positive_decimal),
        ("max_steps", "--max-steps", _positive_int),
        ("num_epochs", "--num-epochs", _positive_int),
        ("max_seq_length", "--max-seq-length", _positive_int),
        ("seed", "--seed", _nonnegative_int),
        ("save_steps", "--save-steps", _positive_int),
        ("save_total_limit", "--save-total-limit", _positive_int),
        ("lora_rank", "--lora-r", _positive_int),
        ("lora_alpha", "--lora-alpha", _positive_int),
        ("lora_dropout", "--lora-dropout", _nonnegative_decimal),
    )
    if ("max_steps" in sft) == ("num_epochs" in sft):
        raise RuntimeV1Error("runtime v1 requires exactly one training duration")
    for key, flag, normalize in mappings:
        if key in sft:
            argv.extend((flag, normalize(sft[key], key)))
    targets = sft.get("lora_target_modules")
    if targets is not None:
        if not isinstance(targets, list) or not targets or any(
            not isinstance(item, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", item)
            for item in targets
        ):
            raise RuntimeV1Error("lora_target_modules is invalid")
        argv.extend(("--lora-target-modules", ",".join(targets)))
    for key, flag in (("use_dora", "--use-dora"), ("use_rslora", "--use-rslora")):
        if key in sft:
            if not isinstance(sft[key], bool):
                raise RuntimeV1Error(f"{key} must be a boolean")
            if sft[key]:
                argv.append(flag)
    if "init_lora_weights" in sft:
        value = sft["init_lora_weights"]
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise RuntimeV1Error("init_lora_weights is invalid")
        argv.extend(("--init-lora-weights", value))
    if "split_dataset" in sft:
        if not isinstance(sft["split_dataset"], bool):
            raise RuntimeV1Error("split_dataset must be a boolean")
        if sft["split_dataset"]:
            argv.append("--split-dataset")
    load_in_4bit = model.get("load_in_4bit")
    if load_in_4bit is not None:
        if not isinstance(load_in_4bit, bool):
            raise RuntimeV1Error("model.load_in_4bit must be a boolean")
        argv.append("--load-in-4bit" if load_in_4bit else "--no-load-in-4bit")


def _positive_int(value: object, name: str) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeV1Error(f"{name} must be a positive integer")
    return str(value)


def _nonnegative_int(value: object, name: str) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeV1Error(f"{name} must be a non-negative integer")
    return str(value)


def _decimal(value: object, name: str, *, positive: bool) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise RuntimeV1Error(f"{name} must be a decimal scalar")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeV1Error(f"{name} must be finite") from exc
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise RuntimeV1Error(f"{name} is outside its accepted range")
    return format(number, "f")


def _positive_decimal(value: object, name: str) -> str:
    return _decimal(value, name, positive=True)


def _nonnegative_decimal(value: object, name: str) -> str:
    return _decimal(value, name, positive=False)


class SubprocessTrainerRunner:
    def run(self, invocation: TrainerInvocation) -> TrainerEvidence:
        invocation.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with invocation.stdout_path.open("xb") as stdout, invocation.stderr_path.open("xb") as stderr:
            completed = subprocess.run(
                invocation.argv,
                cwd=invocation.cwd,
                env=dict(invocation.environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode != 0:
            return TrainerEvidence(completed.returncode, invocation.final_model_dir, invocation.tokenizer_dir, {}, {}, {})
        lineage = _read_json_file(invocation.lineage_path, maximum=MAX_LINEAGE_BYTES)
        projection = _read_json_file(
            invocation.projection_path,
            maximum=MAX_LINEAGE_BYTES,
            require_canonical=True,
        )
        metrics = lineage.get("results")
        return TrainerEvidence(
            completed.returncode,
            invocation.final_model_dir,
            invocation.tokenizer_dir,
            lineage,
            projection,
            metrics if isinstance(metrics, Mapping) else {},
        )


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    workload_fingerprint: str
    inventory_path: Path
    artifacts: tuple[Mapping[str, object], ...]


def execute_runtime(
    payload: bytes,
    *,
    environment: Mapping[str, str],
    runner: TrainerRunner,
    engine_file: Path = Path(__file__),
) -> RuntimeResult:
    workload, roots = decode_and_validate_workload(
        payload, environment, engine_file=engine_file
    )
    if not isinstance(runner, TrainerRunner):
        raise TypeError("runner must implement TrainerRunner")
    if any(roots.artifacts.iterdir()):
        raise RuntimeV1Error("artifact root must be empty")
    invocation = build_trainer_invocation(workload, roots, environment)
    evidence = runner.run(invocation)
    if not isinstance(evidence, TrainerEvidence):
        raise TypeError("trainer runner returned invalid evidence")
    if type(evidence.exit_code) is not int:
        raise TypeError("trainer exit code must be an exact integer")
    if evidence.exit_code != 0:
        raise TrainerFailed("trainer process failed")
    _validate_trainer_evidence(evidence, invocation, workload)
    execution_evidence = _build_execution_evidence(workload, invocation, evidence)
    execution_evidence_bytes = _canonical_json(execution_evidence)
    for directory in (evidence.final_model_dir, evidence.tokenizer_dir):
        _validate_artifact_directory(directory, roots.state)
    metrics = _normalize_metrics(evidence.metrics)
    artifacts: list[dict[str, object]] = []
    artifacts.append(_write_artifact(roots.artifacts, "workload_record", "workload.json", payload))
    lineage = {
        "schema_version": "synaptic-sft-training-lineage/v1",
        "workload_fingerprint": workload.fingerprint,
        "execution_source": workload.document["execution_source"],
        "configuration_revision": workload.document["configuration"]["revision"],
        "identities": workload.document["identities"],
        "trainer_exit_code": evidence.exit_code,
        "execution_evidence": execution_evidence,
        "execution_evidence_sha256": hashlib.sha256(
            execution_evidence_bytes
        ).hexdigest(),
        "trainer_lineage": evidence.lineage,
    }
    artifacts.append(
        _write_artifact(
            roots.artifacts,
            "training_lineage",
            "training_lineage.json",
            _canonical_json(lineage),
        )
    )
    artifacts.append(
        _write_artifact(
            roots.artifacts,
            "training_metrics",
            "training_metrics.json",
            _canonical_json(metrics),
        )
    )
    artifacts.append(
        _archive_artifact(
            roots.artifacts,
            role="final_model",
            filename="final_model.tar",
            source=evidence.final_model_dir,
            artifact_kind="model",
            locked_model_ref=workload.document["configuration"]["document"]["model"]["ref"],
        )
    )
    artifacts.append(
        _archive_artifact(
            roots.artifacts,
            role="tokenizer",
            filename="tokenizer.tar",
            source=evidence.tokenizer_dir,
            artifact_kind="tokenizer",
        )
    )
    inventory = {
        "schema_version": "synaptic-artifact-inventory/v1",
        "workload_fingerprint": workload.fingerprint,
        "artifacts": artifacts,
    }
    inventory_path = roots.state / "runtime-v1-inventory.json"
    _write_exclusive(inventory_path, _canonical_json(inventory))
    return RuntimeResult(workload.fingerprint, inventory_path, tuple(artifacts))


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeV1Error("runtime value cannot be canonically encoded") from exc


def _normalize_metrics(values: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise RuntimeV1Error("trainer metric key is invalid")
        if value is None or isinstance(value, (str, bool, int)):
            result[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[key] = value
        else:
            raise RuntimeV1Error("trainer metrics must be finite JSON scalars")
    return result


def _expected_trainer_projection(
    workload: object,
    *,
    dataset_path: Path,
    run_dir: Path,
    final_model_dir: Path,
) -> dict[str, object]:
    document = workload.document
    config = document["configuration"]["document"]
    model = config["model"]
    dataset = config["dataset"]
    sft = config["sft"]
    return {
        "schema_version": "synaptic-sft-trainer-projection/v1",
        "workload_fingerprint": workload.fingerprint,
        "configuration_revision": document["configuration"]["revision"],
        "model": {
            "ref": model["ref"],
            "revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
            "load_in_4bit": model["load_in_4bit"],
        },
        "dataset": {
            "resolved_path": str(dataset_path.resolve()),
            "revision": dataset["revision"],
            "content_digest": dataset["content_digest"],
        },
        "training": {
            "batch_size": sft["batch_size"],
            "gradient_accumulation_steps": sft["gradient_accumulation_steps"],
            "learning_rate": float(sft["learning_rate"]),
            "max_steps": sft.get("max_steps", -1),
            "num_epochs": float(sft.get("num_epochs", 1)),
            "max_seq_length": sft["max_seq_length"],
            "seed": sft["seed"],
            "save_steps": sft["save_steps"],
            "save_total_limit": sft["save_total_limit"],
            "split_dataset": sft["split_dataset"],
        },
        "lora": {
            "rank": sft["lora_rank"],
            "alpha": sft["lora_alpha"],
            "dropout": float(sft["lora_dropout"]),
            "target_modules": sft["lora_target_modules"],
            "use_dora": sft["use_dora"],
            "use_rslora": sft["use_rslora"],
            "init_lora_weights": sft["init_lora_weights"],
        },
        "outputs": {
            "run_dir": str(run_dir.resolve()),
            "final_model_dir": str(final_model_dir.resolve()),
        },
        "status": "completed",
    }


def _validate_trainer_evidence(
    evidence: TrainerEvidence,
    invocation: TrainerInvocation,
    workload: object,
) -> None:
    if (
        evidence.final_model_dir != invocation.final_model_dir
        or evidence.tokenizer_dir != invocation.tokenizer_dir
    ):
        raise RuntimeV1Error("trainer evidence returned an unexpected output path")
    lineage = evidence.lineage
    projection = evidence.projection
    if not isinstance(lineage, Mapping) or not isinstance(projection, Mapping):
        raise RuntimeV1Error("trainer lineage is missing")
    if not _json_type_equal(projection, invocation.expected_projection):
        raise RuntimeV1Error("trainer lineage does not bind the accepted invocation")
    if not _json_type_equal(lineage.get("synaptic_runtime_projection"), projection):
        raise RuntimeV1Error("trainer lineage does not contain the accepted projection")
    _canonical_json(lineage)
    _canonical_json(projection)


def _build_execution_evidence(
    workload: object,
    invocation: TrainerInvocation,
    evidence: TrainerEvidence,
) -> dict[str, object]:
    config = workload.document["configuration"]["document"]
    dataset_index = invocation.argv.index("--local-file") + 1
    return {
        "schema_version": _EXECUTION_EVIDENCE_SCHEMA,
        "workload_fingerprint": workload.fingerprint,
        "configuration_revision": workload.document["configuration"]["revision"],
        "model": {
            "ref": config["model"]["ref"],
            "revision": config["model"]["revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "load_in_4bit": config["model"]["load_in_4bit"],
        },
        "dataset": {
            "ref": config["dataset"]["ref"],
            "resolved_path": invocation.argv[dataset_index],
            "revision": config["dataset"]["revision"],
            "content_digest": config["dataset"]["content_digest"],
        },
        "sft": config["sft"],
        "argv": list(invocation.argv),
        "environment": dict(invocation.environment),
        "cwd": str(invocation.cwd),
        "outputs": {
            "run_dir": str(invocation.run_dir),
            "final_model_dir": str(invocation.final_model_dir),
            "tokenizer_dir": str(invocation.tokenizer_dir),
            "lineage_path": str(invocation.lineage_path),
        },
        "result": {"exit_code": evidence.exit_code, "status": "completed"},
    }


def _validate_artifact_directory(path: Path, state_root: Path) -> None:
    _assert_no_redirected_components(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeV1Error("trainer artifact directory is missing") from exc
    if not resolved.is_dir() or state_root not in resolved.parents or _is_redirect(path):
        raise RuntimeV1Error("trainer artifact directory escapes writable state")
    for item in resolved.iterdir():
        _assert_no_redirected_components(item)
        try:
            info = item.lstat()
        except OSError as exc:
            raise RuntimeV1Error("trainer artifact entry is unavailable") from exc
        if _is_redirect(item) or not stat.S_ISREG(info.st_mode):
            raise RuntimeV1Error(
                "trainer artifacts contain a nested, redirected, or special entry"
            )


def _write_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise RuntimeV1Error("runtime artifact write failed") from exc


def _write_artifact(root: Path, role: str, filename: str, content: bytes) -> dict[str, object]:
    path = root / filename
    _write_exclusive(path, content)
    return {
        "role": role,
        "path": filename,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def _archive_artifact(
    root: Path,
    *,
    role: str,
    filename: str,
    source: Path,
    artifact_kind: str,
    locked_model_ref: str | None = None,
) -> dict[str, object]:
    members = _select_artifact_members(source, artifact_kind, locked_model_ref=locked_model_ref)
    destination = root / filename
    try:
        with destination.open("xb") as raw:
            with tarfile.open(fileobj=raw, mode="w") as archive:
                for path in members:
                    relative = path.relative_to(source).as_posix()
                    _add_stable_archive_member(archive, path, relative)
            raw.flush()
            os.fsync(raw.fileno())
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeV1Error("runtime artifact archive failed") from exc
    content_digest = hashlib.sha256()
    size = 0
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            content_digest.update(chunk)
            size += len(chunk)
    return {"role": role, "path": filename, "sha256": content_digest.hexdigest(), "size": size}


def _select_artifact_members(
    source: Path, artifact_kind: str, *, locked_model_ref: str | None = None
) -> tuple[Path, ...]:
    try:
        files = {
            item.name: item
            for item in source.iterdir()
            if stat.S_ISREG(item.lstat().st_mode)
        }
    except OSError as exc:
        raise RuntimeV1Error("trainer artifact layout is unreadable") from exc
    validated_tokenizer = _validate_tokenizer_files(files)
    if artifact_kind == "model":
        adapter = "adapter_config.json" in files
        full = "config.json" in files
        if adapter == full:
            raise RuntimeV1Error("trainer model output must contain exactly one model family")
        family = "adapter_model" if adapter else "model"
        config_name = "adapter_config.json" if adapter else "config.json"
        payloads = sorted(name for name in files if (match := _MODEL_PAYLOADS.fullmatch(name)) and match.group(1) == family)
        opposite = [name for name in files if (match := _MODEL_PAYLOADS.fullmatch(name)) and match.group(1) != family]
        if opposite:
            raise RuntimeV1Error("trainer model output mixes model families")
        index_name = f"{family}.safetensors.index.json"
        selected = [config_name] + payloads
        if index_name in files:
            selected.append(index_name)
        selected += sorted(_MODEL_OPTIONAL & files.keys())
        configs = [config_name]
    elif artifact_kind == "tokenizer":
        configs = sorted(_TOKENIZER_CONFIGS & files.keys())
        payloads = sorted(_TOKENIZER_PAYLOADS & files.keys())
        selected = configs + payloads + sorted(_TOKENIZER_OPTIONAL & files.keys())
    else:
        raise RuntimeV1Error("unsupported runtime artifact kind")
    if not configs or not payloads:
        raise RuntimeV1Error(
            f"trainer {artifact_kind} output lacks recognizable config or payload"
        )
    if any(
        not 0 < files[name].lstat().st_size <= _MAX_ARCHIVE_MEMBER_BYTES
        for name in selected
    ):
        raise RuntimeV1Error(
            f"trainer {artifact_kind} artifact contains an empty or oversized file"
        )
    if artifact_kind == "model":
        for name in configs:
            _validate_model_config(files[name], name, locked_model_ref=locked_model_ref)
        tensor_info = {name: _validate_safetensors_file(files[name]) for name in payloads}
        _validate_model_shards(payloads, files, tensor_info, family)
        if "generation_config.json" in files:
            generation = _read_json_file(files["generation_config.json"], maximum=4 * 1024 * 1024)
            if not isinstance(generation, Mapping) or not _bounded_json_tree(generation):
                raise RuntimeV1Error("generation_config.json is malformed")
        if "README.md" in files and not _read_utf8_text(files["README.md"], maximum=4 * 1024 * 1024).strip():
            raise RuntimeV1Error("README.md is empty")
        unknown = set(files) - set(selected) - _KNOWN_IGNORED - validated_tokenizer
        if unknown:
            raise RuntimeV1Error("trainer model output contains an unsupported file")
        if "training_args.bin" in files and not 0 < files["training_args.bin"].stat().st_size <= 16 * 1024 * 1024:
            raise RuntimeV1Error("training_args.bin is empty or oversized")
    else:
        if set(selected) != validated_tokenizer:
            raise RuntimeV1Error("tokenizer selection does not match validated files")
    return tuple(files[name] for name in selected)


def _validate_model_config(path: Path, name: str, *, locked_model_ref: str | None) -> None:
    document = _read_json_file(path, maximum=4 * 1024 * 1024)
    if name == "adapter_config.json":
        if (
            document.get("peft_type") != "LORA"
            or not isinstance(document.get("base_model_name_or_path"), str)
            or document["base_model_name_or_path"] != locked_model_ref
        ):
            raise RuntimeV1Error("trainer adapter config is not recognizable LoRA")
    elif (
        not isinstance(document.get("model_type"), str)
        or not document["model_type"]
    ):
        raise RuntimeV1Error("trainer model config is not recognizable")


def _validate_tokenizer_config(path: Path) -> None:
    document = _read_json_file(path, maximum=4 * 1024 * 1024)
    if (
        not isinstance(document.get("tokenizer_class"), str)
        or not document["tokenizer_class"]
    ):
        raise RuntimeV1Error("trainer tokenizer config is not recognizable")


def _validate_tokenizer_json(path: Path) -> None:
    document = _read_json_file(path, maximum=512 * 1024 * 1024)
    model = document.get("model")
    if (
        not isinstance(document.get("version"), str)
        or not document["version"]
        or not isinstance(model, Mapping)
        or not isinstance(model.get("type"), str)
        or not model["type"]
        or not isinstance(model.get("vocab"), (Mapping, list))
        or not model["vocab"]
    ):
        raise RuntimeV1Error("trainer tokenizer JSON is not recognizable")


def _validate_tokenizer_files(files: Mapping[str, Path]) -> set[str]:
    present = (_TOKENIZER_CONFIGS | _TOKENIZER_PAYLOADS | _TOKENIZER_OPTIONAL) & set(files)
    for name in present:
        path = files[name]
        if not 0 < path.lstat().st_size <= _MAX_ARCHIVE_MEMBER_BYTES:
            raise RuntimeV1Error("tokenizer sidecar is empty or oversized")
        if name == "tokenizer_config.json":
            _validate_tokenizer_config(path)
        elif name == "tokenizer.json":
            _validate_tokenizer_json(path)
        elif name in {"vocab.json", "added_tokens.json"}:
            document = _read_json_file(path, maximum=512 * 1024 * 1024)
            if not isinstance(document, Mapping) or not document or len(document) > 2_000_000 or any(
                not isinstance(token, str) or not token or not isinstance(index, int)
                or isinstance(index, bool) or index < 0
                for token, index in document.items()
            ):
                raise RuntimeV1Error(f"{name} is not a recognizable token mapping")
        elif name == "special_tokens_map.json":
            document = _read_json_file(path, maximum=4 * 1024 * 1024)
            if not isinstance(document, Mapping) or len(document) > 4096 or not _bounded_json_tree(document):
                raise RuntimeV1Error("special_tokens_map.json is malformed")
        elif name == "merges.txt":
            text = _read_utf8_text(path, maximum=512 * 1024 * 1024)
            lines = [line for line in text.splitlines() if line and not line.startswith("#")]
            if not lines or any(len(line.split()) != 2 for line in lines):
                raise RuntimeV1Error("merges.txt is malformed")
        elif name == "chat_template.jinja":
            text = _read_utf8_text(path, maximum=4 * 1024 * 1024)
            if not text.strip():
                raise RuntimeV1Error("chat_template.jinja is empty")
    return present


def _read_utf8_text(path: Path, *, maximum: int) -> str:
    raw = _read_regular(path, maximum=maximum)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeV1Error("tokenizer text sidecar is not UTF-8") from exc


def _bounded_json_tree(value: object, *, depth: int = 0) -> bool:
    if depth > 12:
        return False
    if value is None or isinstance(value, (str, bool, int)):
        return not isinstance(value, str) or len(value) <= 1_000_000
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return len(value) <= 100_000 and all(_bounded_json_tree(item, depth=depth + 1) for item in value)
    if isinstance(value, Mapping):
        return len(value) <= 100_000 and all(
            isinstance(key, str) and len(key) <= 4096 and _bounded_json_tree(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def _validate_model_shards(
    payloads: list[str],
    files: Mapping[str, Path],
    tensor_info: Mapping[str, tuple[frozenset[str], int]],
    family: str,
) -> None:
    matches = [_MODEL_PAYLOADS.fullmatch(name) for name in payloads]
    sharded = [match for match in matches if match is not None and match.group(2) is not None]
    index_name = f"{family}.safetensors.index.json"
    if sharded:
        totals = {int(match.group(3)) for match in sharded}
        if len(totals) != 1:
            raise RuntimeV1Error("model shards declare inconsistent totals")
        total = totals.pop()
        numbers = {int(match.group(2)) for match in sharded}
        if not 1 <= total <= _MAX_SHARDS or numbers != set(range(1, total + 1)) or len(payloads) != total:
            raise RuntimeV1Error("model shard set is incomplete")
        if index_name not in files:
            raise RuntimeV1Error("sharded model output lacks its index")
        index = _read_json_file(files[index_name], maximum=_MAX_INDEX_BYTES)
        if set(index) != {"metadata", "weight_map"} or not isinstance(index["metadata"], Mapping) or not isinstance(index["weight_map"], Mapping):
            raise RuntimeV1Error("model shard index is malformed")
        metadata = index["metadata"]
        if set(metadata) - {"total_size"} or (
            "total_size" in metadata and (
                not isinstance(metadata["total_size"], int)
                or isinstance(metadata["total_size"], bool)
                or metadata["total_size"] != sum(info[1] for info in tensor_info.values())
            )
        ):
            raise RuntimeV1Error("model shard index metadata is invalid")
        weight_map = index["weight_map"]
        if not 0 < len(weight_map) <= _MAX_TENSORS or any(
            not isinstance(name, str) or not name or shard not in payloads
            for name, shard in weight_map.items()
        ):
            raise RuntimeV1Error("model shard index weight map is invalid")
        indexed = {shard: set() for shard in payloads}
        for name, shard in weight_map.items():
            indexed[shard].add(name)
        if any(indexed[name] != set(tensor_info[name][0]) for name in payloads):
            raise RuntimeV1Error("model shard index does not exactly describe payload tensors")
    else:
        if len(payloads) != 1 or payloads[0] != f"{family}.safetensors" or index_name in files:
            raise RuntimeV1Error("unsharded model output is inconsistent")


def _validate_safetensors_file(path: Path) -> tuple[frozenset[str], int]:
    _assert_no_redirected_components(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise RuntimeV1Error("safetensors header is truncated")
            header_size = int.from_bytes(prefix, "little", signed=False)
            if not 0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise RuntimeV1Error("safetensors header length is invalid")
            header = stream.read(header_size)
            if len(header) != header_size:
                raise RuntimeV1Error("safetensors header is truncated")
            data_read = 0
            has_nonzero_data = False
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                data_read += len(chunk)
                has_nonzero_data = has_nonzero_data or chunk.count(0) != len(chunk)
    except OSError as exc:
        raise RuntimeV1Error("safetensors payload is unreadable") from exc
    document = _strict_json_bytes(header, label="safetensors header")
    if not isinstance(document, Mapping):
        raise RuntimeV1Error("safetensors header must be an object")
    _validate_safetensors_index(document, info.st_size - 8 - header_size)
    if data_read != info.st_size - 8 - header_size or not has_nonzero_data:
        raise RuntimeV1Error("safetensors tensor payload is empty or all-zero")
    return frozenset(name for name in document if name != "__metadata__"), data_read


def _validate_safetensors_index(
    document: Mapping[str, object], data_size: int
) -> None:
    metadata = document.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, Mapping)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())
    ):
        raise RuntimeV1Error("safetensors metadata is invalid")
    intervals: list[tuple[int, int]] = []
    for name, descriptor in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(descriptor, Mapping):
            raise RuntimeV1Error("safetensors tensor descriptor is invalid")
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
            raise RuntimeV1Error("safetensors tensor descriptor is invalid")
        start, end = offsets
        elements = math.prod(shape)
        if end <= start or end - start != elements * _SAFETENSORS_DTYPES[dtype]:
            raise RuntimeV1Error("safetensors tensor span is invalid")
        intervals.append((start, end))
    if not intervals:
        raise RuntimeV1Error("safetensors payload has no tensors")
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start != cursor:
            raise RuntimeV1Error("safetensors tensor offsets are not contiguous")
        cursor = end
    if cursor != data_size:
        raise RuntimeV1Error("safetensors file length does not match declared tensors")


def _add_stable_archive_member(
    archive: tarfile.TarFile,
    path: Path,
    relative: str,
) -> None:
    _assert_no_redirected_components(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as content:
        before = os.fstat(content.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_ARCHIVE_MEMBER_BYTES:
            raise RuntimeV1Error("archive member must be bounded, nonempty, and regular")
        info = tarfile.TarInfo(relative)
        info.size = before.st_size
        info.mode = 0o644
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        digest = hashlib.sha256()
        archive.addfile(info, _HashingReader(content, digest))
        after = os.fstat(content.fileno())
        content.seek(0)
        confirm = hashlib.file_digest(content, "sha256").hexdigest()
    current = path.lstat()
    if (
        _file_identity(before) != _file_identity(after)
        or _stable_path_identity(after) != _stable_path_identity(current)
        or digest.hexdigest() != confirm
    ):
        raise RuntimeV1Error("trainer artifact changed during archival")


class _HashingReader:
    def __init__(self, stream: BinaryIO, digest: object) -> None:
        self._stream = stream
        self._digest = digest

    def read(self, size: int = -1) -> bytes:
        content = self._stream.read(size)
        self._digest.update(content)
        return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical SFT runtime v1")
    parser.add_argument("--canonical-workload-stdin", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        payload = read_bounded_workload(sys.stdin.buffer)
        result = execute_runtime(
            payload,
            environment=os.environ,
            runner=SubprocessTrainerRunner(),
        )
    except RuntimeV1Error:
        print("SFT_RUNTIME_V1_REJECTED", file=sys.stderr)
        return 2
    print(
        _canonical_json(
            {
                "workload_fingerprint": result.workload_fingerprint,
                "inventory_path": str(result.inventory_path),
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
