"""Engine-root process dispatch with explicit writable capabilities."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol, TypeAlias, runtime_checkable

from synaptic_tuner.api.v1.training import (
    ArtifactPolicy,
    CanonicalDocument,
    ResourceSpec,
    RuntimeSpec,
    TrainingPlan,
)
from tuner.cloud.runtime_layout import CloudRuntimeLayout
from tuner.project.execution_source import ExecutionSourceV1
from tuner.training import default_recipe_registry
from tuner.training.recipes import (
    MAX_WORKLOAD_BYTES,
    CompiledWorkload,
    canonical_json_bytes,
)
from tuner.runtime.offline_sft_worker import (
    OFFLINE_SFT_MANIFEST_NAME,
    OfflineSFTWorkerManifestV1,
    load_packaged_offline_sft_worker_manifest,
)


WORKER_INVOCATION_SCHEMA = "synaptic-worker-invocation/v1"
WORKER_BUNDLE_MATERIALIZATION_SCHEMA = "synaptic-worker-bundle-materialization/v1"
_WORKLOAD_FINGERPRINT_DOMAIN = b"synaptic-training-workload/v1\0"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT_NAMES = ("engine", "project", "artifacts", "state", "tracking", "cache", "tmp")
_ISSUANCE_KEY = secrets.token_bytes(32)
_FORBIDDEN_WORKER_ENVIRONMENT = frozenset(
    {"PYTHONHOME", "PYTHONUSERBASE", "HF_TOKEN"}
)


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _absolute_runtime_path(value: PurePosixPath | str, label: str) -> PurePosixPath:
    raw = os.fspath(value)
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or raw != path.as_posix()
        or "//" in raw
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    return path


@dataclass(frozen=True, slots=True)
class CanonicalWorkloadFileLocationV1:
    control_root: PurePosixPath

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "control_root",
            _absolute_runtime_path(self.control_root, "workload control root"),
        )


@dataclass(frozen=True, slots=True)
class WorkerControlLocationV1:
    control_root: PurePosixPath

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "control_root",
            _absolute_runtime_path(self.control_root, "worker control root"),
        )

    @property
    def manifest_path(self) -> PurePosixPath:
        return self.control_root / OFFLINE_SFT_MANIFEST_NAME


@dataclass(frozen=True, slots=True, init=False)
class CanonicalWorkloadBytesV1:
    payload: bytes
    byte_count: int
    sha256: str
    workload_fingerprint: str
    _issuance_seal: str

    def __new__(cls, *args: object, **kwargs: object) -> "CanonicalWorkloadBytesV1":
        raise TypeError("canonical workload transports are factory-issued")


@dataclass(frozen=True, slots=True, init=False)
class CanonicalWorkloadFileV1:
    control_root: PurePosixPath
    path: PurePosixPath
    byte_count: int
    sha256: str
    workload_fingerprint: str
    logical_name: str = "workload.json"
    read_only: bool = True
    _issuance_seal: str = ""

    def __new__(cls, *args: object, **kwargs: object) -> "CanonicalWorkloadFileV1":
        raise TypeError("canonical workload transports are factory-issued")


CanonicalWorkloadTransportV1: TypeAlias = (
    CanonicalWorkloadBytesV1 | CanonicalWorkloadFileV1
)


@dataclass(frozen=True, slots=True, init=False)
class WorkerInvocationV1:
    plan_fingerprint: str
    workload_fingerprint: str
    entrypoint: PurePosixPath
    roots: tuple[tuple[str, PurePosixPath], ...]
    environment: tuple[tuple[str, str], ...]
    interpreter: str
    transport: CanonicalWorkloadTransportV1
    control_location: WorkerControlLocationV1
    closure_manifest: OfflineSFTWorkerManifestV1
    schema_version: str = WORKER_INVOCATION_SCHEMA
    _issuance_seal: str = ""
    _plan: TrainingPlan | None = None
    _layout: CloudRuntimeLayout | None = None
    _file_location: CanonicalWorkloadFileLocationV1 | None = None

    def __new__(cls, *args: object, **kwargs: object) -> "WorkerInvocationV1":
        raise TypeError("worker invocations are factory-issued")

    @property
    def roots_map(self) -> Mapping[str, PurePosixPath]:
        return dict(self.roots)


def _worker_environment(
    roots: Mapping[str, PurePosixPath],
    workload_fingerprint: str,
    model_snapshot: PurePosixPath,
    control_location: WorkerControlLocationV1,
    closure_manifest: OfflineSFTWorkerManifestV1,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    base = {} if base_environment is None else dict(base_environment)
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in base.items()):
        raise TypeError("worker environment must contain only text keys and values")
    if _FORBIDDEN_WORKER_ENVIRONMENT & set(base):
        raise ValueError("worker environment contains a forbidden ambient variable")
    return {
        **base,
        "SYNAPTIC_ENGINE_ROOT": roots["engine"].as_posix(),
        "SYNAPTIC_PROJECT_ROOT": roots["project"].as_posix(),
        "SYNAPTIC_ARTIFACT_ROOT": roots["artifacts"].as_posix(),
        "SYNAPTIC_STATE_ROOT": roots["state"].as_posix(),
        "SYNAPTIC_TRACKING_ROOT": roots["tracking"].as_posix(),
        "SYNAPTIC_CACHE_ROOT": roots["cache"].as_posix(),
        "SYNAPTIC_TMP_ROOT": roots["tmp"].as_posix(),
        "SYNAPTIC_WORKLOAD_FINGERPRINT": workload_fingerprint,
        "SYNAPTIC_MODEL_SNAPSHOT": model_snapshot.as_posix(),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SYNAPTIC_WORKER_CLOSURE_MANIFEST": control_location.manifest_path.as_posix(),
        "SYNAPTIC_WORKER_CLOSURE_DIGEST": closure_manifest.closure.closure_digest,
    }


def _model_snapshot_path(
    cache_root: PurePosixPath, workload: CompiledWorkload
) -> PurePosixPath:
    configuration = workload.document.get("configuration")
    document = configuration.get("document") if type(configuration) is dict else None
    model = document.get("model") if type(document) is dict else None
    model_ref = model.get("ref") if type(model) is dict else None
    revision = model.get("revision") if type(model) is dict else None
    if (
        type(model_ref) is not str
        or type(revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", revision) is None
    ):
        raise ValueError("compiled workload model identity is malformed")
    parts = model_ref.split("/")
    if len(parts) not in (1, 2) or any(
        re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", part) is None
        or "--" in part
        or ".." in part
        for part in parts
    ):
        raise ValueError("compiled workload model ref has no canonical snapshot layout")
    repository_folder = "models--" + "--".join(parts)
    return cache_root / "model" / repository_folder / "snapshots" / revision


def _canonical_seal(domain: bytes, value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hmac.new(_ISSUANCE_KEY, domain + b"\0" + payload, hashlib.sha256).hexdigest()


def _transport_projection(
    transport: CanonicalWorkloadTransportV1,
) -> dict[str, object]:
    common = {
        "byte_count": transport.byte_count,
        "sha256": transport.sha256,
        "workload_fingerprint": transport.workload_fingerprint,
    }
    if type(transport) is CanonicalWorkloadBytesV1:
        return {"kind": "bytes", **common}
    if type(transport) is CanonicalWorkloadFileV1:
        return {
            "kind": "file",
            **common,
            "control_root": transport.control_root.as_posix(),
            "path": transport.path.as_posix(),
            "logical_name": transport.logical_name,
            "read_only": transport.read_only,
        }
    raise TypeError("worker transport has an unsupported concrete type")


def _issue_transport(
    workload: CompiledWorkload,
    location: CanonicalWorkloadFileLocationV1 | None,
) -> CanonicalWorkloadTransportV1:
    payload = workload.canonical_bytes
    values: dict[str, object] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "workload_fingerprint": workload.fingerprint,
    }
    if location is None:
        transport = object.__new__(CanonicalWorkloadBytesV1)
        object.__setattr__(transport, "payload", payload)
    else:
        transport = object.__new__(CanonicalWorkloadFileV1)
        object.__setattr__(transport, "control_root", location.control_root)
        object.__setattr__(transport, "path", location.control_root / "workload.json")
        object.__setattr__(transport, "logical_name", "workload.json")
        object.__setattr__(transport, "read_only", True)
    for name, value in values.items():
        object.__setattr__(transport, name, value)
    projection = _transport_projection(transport)
    object.__setattr__(
        transport,
        "_issuance_seal",
        _canonical_seal(b"synaptic-workload-transport/v1", projection),
    )
    return transport


def _require_issued_transport(transport: CanonicalWorkloadTransportV1) -> None:
    try:
        projection = _transport_projection(transport)
        _digest(transport.sha256, "transport sha256")
        _digest(transport.workload_fingerprint, "transport workload_fingerprint")
        if (
            type(transport.byte_count) is not int
            or isinstance(transport.byte_count, bool)
            or not 1 <= transport.byte_count <= MAX_WORKLOAD_BYTES
        ):
            raise ValueError
        if type(transport) is CanonicalWorkloadBytesV1:
            if (
                type(transport.payload) is not bytes
                or len(transport.payload) != transport.byte_count
                or hashlib.sha256(transport.payload).hexdigest() != transport.sha256
                or hashlib.sha256(
                    _WORKLOAD_FINGERPRINT_DOMAIN + transport.payload
                ).hexdigest()
                != transport.workload_fingerprint
            ):
                raise ValueError
            document = json.loads(transport.payload.decode("utf-8", errors="strict"))
            if (
                type(document) is not dict
                or canonical_json_bytes(document) != transport.payload
            ):
                raise ValueError
        else:
            root = _absolute_runtime_path(
                transport.control_root, "workload control root"
            )
            path = _absolute_runtime_path(transport.path, "workload file path")
            if (
                transport.logical_name != "workload.json"
                or transport.read_only is not True
                or path != root / "workload.json"
            ):
                raise ValueError
        expected = _canonical_seal(b"synaptic-workload-transport/v1", projection)
        valid_seal = type(transport._issuance_seal) is str and hmac.compare_digest(
            transport._issuance_seal, expected
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        valid_seal = False
    if not valid_seal:
        raise ValueError("worker transport is not an authentic factory issuance")


def _locked_runtime_binding(
    workload: CompiledWorkload,
) -> tuple[dict[str, str], str]:
    source = workload.document.get("execution_source")
    runtime = source.get("runtime") if type(source) is dict else None
    roots = runtime.get("roots") if type(runtime) is dict else None
    interpreter = runtime.get("interpreter") if type(runtime) is dict else None
    if (
        type(runtime) is not dict
        or runtime.get("schema_version") != "synaptic-training-runtime/v1"
        or type(roots) is not dict
        or set(roots) != set(_ROOT_NAMES)
        or any(type(value) is not str for value in roots.values())
        or type(interpreter) is not dict
        or type(interpreter.get("executable")) is not str
    ):
        raise ValueError("compiled workload runtime binding is malformed")
    return dict(roots), interpreter["executable"]


def _require_exact_plan(plan: TrainingPlan) -> None:
    if type(plan) is not TrainingPlan:
        raise TypeError("plan must be an exact canonical TrainingPlan")
    expected = (
        (plan.execution_source, ExecutionSourceV1, "execution_source"),
        (plan.execution_context, CanonicalDocument, "execution_context"),
        (plan.resolved_config, CanonicalDocument, "resolved_config"),
        (plan.workload, CanonicalDocument, "workload"),
        (plan.runtime, RuntimeSpec, "runtime"),
        (plan.resources, ResourceSpec, "resources"),
        (plan.artifact_policy, ArtifactPolicy, "artifact_policy"),
    )
    if any(type(value) is not expected_type for value, expected_type, _ in expected):
        raise TypeError("plan contains an ambiguous noncanonical component type")


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_disjoint_roots(roots: tuple[tuple[str, PurePosixPath], ...]) -> None:
    for index, (name, path) in enumerate(roots):
        for other_name, other in roots[index + 1 :]:
            if _paths_overlap(path, other):
                raise ValueError(f"worker roots {name!r} and {other_name!r} overlap")


def _compile_plan_workload(plan: TrainingPlan) -> CompiledWorkload:
    config = plan.resolved_config.to_dict()
    method = config.get("method")
    if type(method) is not str or not method:
        raise ValueError("canonical training plan has no exact method")
    compiled = (
        default_recipe_registry()
        .resolve(method)
        .compile(
            resolved_config=plan.resolved_config,
            execution_source=plan.execution_source,
        )
    )
    if type(compiled) is not CompiledWorkload:
        raise TypeError("default recipe returned an ambiguous workload type")
    planned_bytes = plan.workload.canonical_json.encode("utf-8")
    if compiled.canonical_bytes != planned_bytes:
        raise ValueError("plan workload differs from deterministic recompilation")
    document = compiled.document
    configuration = document.get("configuration")
    if (
        type(configuration) is not dict
        or configuration.get("document") != config
        or document.get("execution_source") != plan.execution_source.to_dict()
    ):
        raise ValueError("plan config or execution source differs from its workload")
    artifacts = document.get("artifacts")
    requirements = artifacts.get("requirements") if type(artifacts) is dict else None
    if type(requirements) is not list or any(
        type(item) is not dict for item in requirements
    ):
        raise ValueError("plan workload artifact requirements are malformed")
    roles = tuple(item.get("role") for item in requirements)
    if (
        any(type(role) is not str for role in roles)
        or len(set(roles)) != len(roles)
        or not set(plan.artifact_policy.required_kinds).issubset(set(roles))
    ):
        raise ValueError("plan artifact roles differ from the workload contract")
    return compiled


def _derive_worker_fields(
    plan: TrainingPlan,
    layout: CloudRuntimeLayout,
    file_location: CanonicalWorkloadFileLocationV1 | None,
    control_location: WorkerControlLocationV1,
) -> tuple[
    str,
    CompiledWorkload,
    PurePosixPath,
    tuple[tuple[str, PurePosixPath], ...],
    tuple[tuple[str, str], ...],
    str,
    CanonicalWorkloadTransportV1,
    OfflineSFTWorkerManifestV1,
]:
    _require_exact_plan(plan)
    if type(layout) is not CloudRuntimeLayout:
        raise TypeError("layout must be an exact CloudRuntimeLayout")
    if (
        file_location is not None
        and type(file_location) is not CanonicalWorkloadFileLocationV1
    ):
        raise TypeError("file location must be exact CanonicalWorkloadFileLocationV1")
    if type(control_location) is not WorkerControlLocationV1:
        raise TypeError("control location must be exact WorkerControlLocationV1")
    before_fingerprint = plan.fingerprint
    workload = _compile_plan_workload(plan)
    entrypoint = PurePosixPath(workload.entrypoint)
    if entrypoint.is_absolute() or any(
        part in {"", ".", ".."} for part in entrypoint.parts
    ):
        raise ValueError("worker entrypoint is not contained engine-relative")
    writable = layout.writable_by_name
    roots = (
        ("engine", layout.engine.target),
        ("project", layout.project.target),
        *((name, writable[name].target) for name in _ROOT_NAMES[2:]),
    )
    roots = tuple(
        (name, _absolute_runtime_path(path, f"worker root {name}"))
        for name, path in roots
    )
    _require_disjoint_roots(roots)
    locked_roots, interpreter = _locked_runtime_binding(workload)
    if {name: path.as_posix() for name, path in roots} != locked_roots:
        raise ValueError("runtime layout does not match the compiled workload roots")
    if file_location is not None and any(
        _paths_overlap(file_location.control_root, path) for _, path in roots
    ):
        raise ValueError(
            "workload control root must be disjoint from all runtime roots"
        )
    if any(_paths_overlap(control_location.control_root, path) for _, path in roots):
        raise ValueError("worker control root must be disjoint from all runtime roots")
    _require_safe_staged_entrypoint(layout.engine.source, entrypoint)
    model_snapshot = _model_snapshot_path(dict(roots)["cache"], workload)
    closure_manifest = load_packaged_offline_sft_worker_manifest()
    planned_environment = dict(plan.execution_source.environment)
    if planned_environment.get("PYTHONPATH") != dict(roots)["engine"].as_posix():
        raise ValueError("source PYTHONPATH does not bind the authenticated engine root")
    planned_environment.pop("PYTHONPATH")
    environment = tuple(sorted(_worker_environment(
        dict(roots), workload.fingerprint, model_snapshot, control_location,
        closure_manifest, planned_environment,
    ).items()))
    interpreter = _absolute_runtime_path(interpreter, "worker interpreter").as_posix()
    transport = _issue_transport(workload, file_location)
    after_fingerprint = plan.fingerprint
    if before_fingerprint != after_fingerprint:
        raise ValueError("canonical training plan changed during worker derivation")
    return (
        before_fingerprint,
        workload,
        entrypoint,
        roots,
        environment,
        interpreter,
        transport,
        closure_manifest,
    )


@dataclass(frozen=True, slots=True)
class DispatchInvocation:
    argv: tuple[str, ...]
    cwd: PurePosixPath
    environment: tuple[tuple[str, str], ...]
    stdin: bytes

    @property
    def environment_map(self) -> Mapping[str, str]:
        return dict(self.environment)


@dataclass(frozen=True, slots=True, init=False)
class WorkerBundleMaterializationV1:
    plan_fingerprint: str
    workload_fingerprint: str
    canonical_workload_bytes: bytes
    workload_byte_count: int
    workload_sha256: str
    closure_manifest_bytes: bytes
    closure_manifest_source: str
    closure_manifest_byte_count: int
    closure_manifest_sha256: str
    closure_digest: str
    closure_manifest_runtime_path: PurePosixPath
    dispatch: DispatchInvocation
    canonical_projection_bytes: bytes
    projection_sha256: str
    schema_version: str = WORKER_BUNDLE_MATERIALIZATION_SCHEMA
    _issuance_seal: str = ""

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "WorkerBundleMaterializationV1":
        raise TypeError("worker bundle materializations are factory-issued")


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("exit_code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("process output must be text")


def _worker_projection(worker: WorkerInvocationV1) -> dict[str, object]:
    return {
        "schema_version": worker.schema_version,
        "plan_fingerprint": worker.plan_fingerprint,
        "workload_fingerprint": worker.workload_fingerprint,
        "entrypoint": worker.entrypoint.as_posix(),
        "roots": [[name, path.as_posix()] for name, path in worker.roots],
        "environment": [list(item) for item in worker.environment],
        "interpreter": worker.interpreter,
        "transport": _transport_projection(worker.transport),
        "control_root": worker.control_location.control_root.as_posix(),
        "closure_manifest_source": worker.closure_manifest.source_ref,
        "closure_manifest_sha256": worker.closure_manifest.sha256,
        "closure_digest": worker.closure_manifest.closure.closure_digest,
    }


def build_worker_invocation(
    plan: TrainingPlan,
    layout: CloudRuntimeLayout,
    control_location: WorkerControlLocationV1,
    file_location: CanonicalWorkloadFileLocationV1 | None = None,
) -> WorkerInvocationV1:
    (
        plan_fingerprint,
        workload,
        entrypoint,
        roots,
        environment,
        interpreter,
        transport,
        closure_manifest,
    ) = _derive_worker_fields(plan, layout, file_location, control_location)
    worker = object.__new__(WorkerInvocationV1)
    values = {
        "schema_version": WORKER_INVOCATION_SCHEMA,
        "plan_fingerprint": plan_fingerprint,
        "workload_fingerprint": workload.fingerprint,
        "entrypoint": entrypoint,
        "roots": roots,
        "environment": environment,
        "interpreter": interpreter,
        "transport": transport,
        "control_location": control_location,
        "closure_manifest": closure_manifest,
        "_plan": plan,
        "_layout": layout,
        "_file_location": file_location,
    }
    for name, value in values.items():
        object.__setattr__(worker, name, value)
    object.__setattr__(
        worker,
        "_issuance_seal",
        _canonical_seal(b"synaptic-worker-invocation/v1", _worker_projection(worker)),
    )
    return worker


def build_source_worker_invocation(
    plan: TrainingPlan,
    control_location: WorkerControlLocationV1,
) -> WorkerInvocationV1:
    """Issue a worker from an authenticated source lock without host path probing."""

    _require_exact_plan(plan)
    if type(control_location) is not WorkerControlLocationV1:
        raise TypeError("control location must be exact WorkerControlLocationV1")
    workload = _compile_plan_workload(plan)
    locked_roots, interpreter = _locked_runtime_binding(workload)
    roots = tuple(
        (name, _absolute_runtime_path(locked_roots[name], f"worker root {name}"))
        for name in _ROOT_NAMES
    )
    _require_disjoint_roots(roots)
    if any(_paths_overlap(control_location.control_root, path) for _, path in roots):
        raise ValueError("worker control root must be disjoint from all runtime roots")
    entrypoint = PurePosixPath(workload.entrypoint)
    if entrypoint.is_absolute() or any(part in {"", ".", ".."} for part in entrypoint.parts):
        raise ValueError("worker entrypoint is not contained engine-relative")
    manifest = load_packaged_offline_sft_worker_manifest()
    planned_environment = dict(plan.execution_source.environment)
    if planned_environment.get("PYTHONPATH") != dict(roots)["engine"].as_posix():
        raise ValueError("source PYTHONPATH does not bind the authenticated engine root")
    planned_environment.pop("PYTHONPATH")
    environment = tuple(sorted(_worker_environment(
        dict(roots), workload.fingerprint,
        _model_snapshot_path(dict(roots)["cache"], workload),
        control_location, manifest, planned_environment,
    ).items()))
    worker = object.__new__(WorkerInvocationV1)
    values = {
        "schema_version": WORKER_INVOCATION_SCHEMA,
        "plan_fingerprint": plan.fingerprint,
        "workload_fingerprint": workload.fingerprint,
        "entrypoint": entrypoint,
        "roots": roots,
        "environment": environment,
        "interpreter": _absolute_runtime_path(interpreter, "worker interpreter").as_posix(),
        "transport": _issue_transport(workload, None),
        "control_location": control_location,
        "closure_manifest": manifest,
        "_plan": plan,
        "_layout": None,
        "_file_location": None,
    }
    for name, value in values.items():
        object.__setattr__(worker, name, value)
    object.__setattr__(
        worker, "_issuance_seal",
        _canonical_seal(b"synaptic-worker-invocation/v1", _worker_projection(worker)),
    )
    return worker


def build_dispatch_invocation(
    plan: TrainingPlan, layout: CloudRuntimeLayout, control_location: WorkerControlLocationV1
) -> DispatchInvocation:
    return materialize_worker_invocation(build_worker_invocation(plan, layout, control_location))


def _require_canonical_worker(worker: WorkerInvocationV1) -> None:
    if type(worker) is not WorkerInvocationV1:
        raise TypeError("worker must be an exact WorkerInvocationV1")
    try:
        _require_issued_transport(worker.transport)
        current_projection = _worker_projection(worker)
        expected_seal = _canonical_seal(
            b"synaptic-worker-invocation/v1", current_projection
        )
        authentic = type(worker._issuance_seal) is str and hmac.compare_digest(
            worker._issuance_seal, expected_seal
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        authentic = False
    if not authentic:
        raise ValueError("worker invocation is not an authentic factory issuance")
    expected = (
        build_source_worker_invocation(worker._plan, worker.control_location)
        if worker._layout is None
        else build_worker_invocation(
            worker._plan, worker._layout, worker.control_location, worker._file_location
        )
    )
    if current_projection != _worker_projection(expected):
        raise ValueError("worker invocation differs from its canonical plan projection")


def _dispatch_invocation(worker: WorkerInvocationV1) -> DispatchInvocation:
    roots = worker.roots_map
    entrypoint = (roots["engine"] / worker.entrypoint).as_posix()
    if type(worker.transport) is CanonicalWorkloadBytesV1:
        argv = (worker.interpreter, entrypoint, "--canonical-workload-stdin")
        stdin = worker.transport.payload
    else:
        transport = worker.transport
        argv = (
            worker.interpreter,
            entrypoint,
            "--canonical-workload-file",
            transport.path.as_posix(),
            "--canonical-workload-control-root",
            transport.control_root.as_posix(),
            "--canonical-workload-byte-count",
            str(transport.byte_count),
            "--canonical-workload-sha256",
            transport.sha256,
            "--canonical-workload-fingerprint",
            transport.workload_fingerprint,
        )
        stdin = b""
    return DispatchInvocation(
        argv=argv,
        cwd=roots["tmp"],
        environment=worker.environment,
        stdin=stdin,
    )


def _dispatch_projection(dispatch: DispatchInvocation) -> dict[str, object]:
    return {
        "argv": list(dispatch.argv),
        "cwd": dispatch.cwd.as_posix(),
        "environment": [list(item) for item in dispatch.environment],
        "stdin_base64": base64.b64encode(dispatch.stdin).decode("ascii"),
    }


def materialize_worker_bundle(
    worker: WorkerInvocationV1,
) -> WorkerBundleMaterializationV1:
    _require_canonical_worker(worker)
    workload = _compile_plan_workload(worker._plan)
    payload = workload.canonical_bytes
    byte_count = len(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    if (
        workload.fingerprint != worker.workload_fingerprint
        or byte_count != worker.transport.byte_count
        or sha256 != worker.transport.sha256
    ):
        raise ValueError("worker workload differs from its authenticated transport")
    dispatch = _dispatch_invocation(worker)
    projection = {
        "schema_version": WORKER_BUNDLE_MATERIALIZATION_SCHEMA,
        "plan_fingerprint": worker.plan_fingerprint,
        "workload_fingerprint": worker.workload_fingerprint,
        "workload": {
            "payload_base64": base64.b64encode(payload).decode("ascii"),
            "byte_count": byte_count,
            "sha256": sha256,
        },
        "closure_manifest": {
            "payload_base64": base64.b64encode(worker.closure_manifest.canonical_bytes).decode("ascii"),
            "source": worker.closure_manifest.source_ref,
            "byte_count": worker.closure_manifest.byte_count,
            "sha256": worker.closure_manifest.sha256,
            "closure_digest": worker.closure_manifest.closure.closure_digest,
            "runtime_path": worker.control_location.manifest_path.as_posix(),
        },
        "dispatch": _dispatch_projection(dispatch),
    }
    projection_bytes = canonical_json_bytes(projection)
    projection_sha256 = hashlib.sha256(projection_bytes).hexdigest()
    bundle = object.__new__(WorkerBundleMaterializationV1)
    values = {
        "schema_version": WORKER_BUNDLE_MATERIALIZATION_SCHEMA,
        "plan_fingerprint": worker.plan_fingerprint,
        "workload_fingerprint": worker.workload_fingerprint,
        "canonical_workload_bytes": payload,
        "workload_byte_count": byte_count,
        "workload_sha256": sha256,
        "closure_manifest_bytes": worker.closure_manifest.canonical_bytes,
        "closure_manifest_source": worker.closure_manifest.source_ref,
        "closure_manifest_byte_count": worker.closure_manifest.byte_count,
        "closure_manifest_sha256": worker.closure_manifest.sha256,
        "closure_digest": worker.closure_manifest.closure.closure_digest,
        "closure_manifest_runtime_path": worker.control_location.manifest_path,
        "dispatch": dispatch,
        "canonical_projection_bytes": projection_bytes,
        "projection_sha256": projection_sha256,
    }
    for name, value in values.items():
        object.__setattr__(bundle, name, value)
    object.__setattr__(
        bundle,
        "_issuance_seal",
        _canonical_seal(b"synaptic-worker-bundle-materialization/v1", projection),
    )
    return bundle


def materialize_worker_invocation(worker: WorkerInvocationV1) -> DispatchInvocation:
    return materialize_worker_bundle(worker).dispatch


def _require_safe_staged_entrypoint(
    engine_source: Path, entrypoint: PurePosixPath
) -> None:
    root = engine_source.absolute()
    candidate = root.joinpath(*entrypoint.parts)
    current = Path(root.anchor)
    for part in (*root.parts[1:], *entrypoint.parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ValueError(
                "dispatch entrypoint is absent from the staged engine"
            ) from exc
        redirected = (
            current.is_symlink()
            or (hasattr(os.path, "isjunction") and os.path.isjunction(current))
            or bool(
                getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        )
        if redirected:
            raise ValueError("dispatch entrypoint traverses redirected staging")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            "dispatch entrypoint is absent from the staged engine"
        ) from exc
    if (
        resolved_root != root
        or resolved_root not in resolved.parents
        or not stat.S_ISREG(candidate.lstat().st_mode)
    ):
        raise ValueError("dispatch entrypoint is not a contained regular file")


@runtime_checkable
class ProcessRunner(Protocol):
    def run(self, invocation: DispatchInvocation) -> ProcessResult: ...


class SubprocessRunner:
    """Concrete local/container runner; provider submission is not its concern."""

    def __init__(self, *, base_environment: Mapping[str, str] | None = None) -> None:
        self._base_environment = dict(
            os.environ if base_environment is None else base_environment
        )

    def run(self, invocation: DispatchInvocation) -> ProcessResult:
        child_environment = {
            key: value
            for key, value in self._base_environment.items()
            if key not in (_FORBIDDEN_WORKER_ENVIRONMENT | {"PYTHONPATH"})
        }
        child_environment.update(invocation.environment_map)
        completed = subprocess.run(
            invocation.argv,
            cwd=str(invocation.cwd),
            env=child_environment,
            input=invocation.stdin,
            capture_output=True,
            check=False,
        )
        return ProcessResult(
            exit_code=completed.returncode,
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
        )


class EngineDispatcher:
    def __init__(self, runner: ProcessRunner) -> None:
        if not isinstance(runner, ProcessRunner):
            raise TypeError("runner must implement ProcessRunner")
        self._runner = runner

    def dispatch(
        self, plan: TrainingPlan, layout: CloudRuntimeLayout,
        control_location: WorkerControlLocationV1,
    ) -> ProcessResult:
        return self._runner.run(build_dispatch_invocation(plan, layout, control_location))


__all__ = [
    "CanonicalWorkloadBytesV1",
    "CanonicalWorkloadFileV1",
    "CanonicalWorkloadFileLocationV1",
    "CanonicalWorkloadTransportV1",
    "DispatchInvocation",
    "EngineDispatcher",
    "ProcessResult",
    "ProcessRunner",
    "SubprocessRunner",
    "WORKER_BUNDLE_MATERIALIZATION_SCHEMA",
    "WORKER_INVOCATION_SCHEMA",
    "WorkerBundleMaterializationV1",
    "WorkerControlLocationV1",
    "WorkerInvocationV1",
    "build_dispatch_invocation",
    "build_worker_invocation",
    "build_source_worker_invocation",
    "materialize_worker_bundle",
    "materialize_worker_invocation",
]
